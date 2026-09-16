use ndarray::{Array1, Array2};
use ndarray_linalg::{AllocatedArray, Lapack, Norm, UPLO};
use smolprng::{JsfLarge, PRNG};
use sprs::CsMat;

pub fn get_Q_norm(Q: &CsMat<f64>) -> f64 {
    // compute the l1 norm of Q
    let mut c = Array1::<f64>::zeros(Q.shape().0);
    for (q_ij, (i, _)) in Q.iter() {
        c[i] += q_ij.abs();
    }
    *c.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap()
}

pub fn obj(Q: &CsMat<f64>, V: &Array2<f64>) -> f64 {
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        if i == j {
            trace += q_ij;
        }
        if i < j {
            trace += 2.0 * q_ij * V.row(i).dot(&V.row(j));
        }
    }

    trace
}

pub fn obj_from_qv(qv: &Array2<f64>, v: &Array2<f64>) -> f64 {
    let mut trace = 0.0;
    for i in 0..v.nrows() {
        trace += qv.row(i).dot(&v.row(i));
    }
    trace
}

pub fn obj_rounded(Q: &CsMat<f64>, x_0: &Array1<f64>) -> f64 {
    Q.iter()
        .map(|(q_ij, (i, j)): (&f64, (usize, usize))| -> f64 {
            if i == j {
                return *q_ij;
            }
            if i < j {
                2.0 * q_ij * x_0[i] * x_0[j]
            } else {
                0.0
            }
        })
        .sum()
}

pub fn grad(Q: &CsMat<f64>, V: &Array2<f64>) -> Array2<f64> {
    2.0 * (Q * V)
}

pub fn dual_variables(Q: &CsMat<f64>, V: &Array2<f64>) -> Array1<f64> {
    // based on equation 8 of https://arxiv.org/pdf/0807.4423, much more efficient than the naive implementation
    // l_i = tr(V.T* E_ii * X * V)/ tr(V.T*E_ii*E_ii*V)
    let mut dual = Array1::<f64>::zeros(Q.shape().0);

    let G = Q * V;

    for i in 0..Q.shape().0 {
        dual[i] = G.row(i).dot(&V.row(i));
    }

    dual
}

pub fn dual_variables_with_QV(QV: &Array2<f64>, V: &Array2<f64>) -> Array1<f64> {
    let mut dual = Array1::<f64>::zeros(QV.shape()[0]);

    for i in 0..QV.shape()[0] {
        dual[i] = QV.row(i).dot(&V.row(i));
    }

    dual
}

pub fn dual_bound(Q: &CsMat<f64>, V: &Array2<f64>) -> f64 {
    let y = dual_variables(Q, V);
    dual_bound_from_variables(Q, &y)
}

/// A cheap bound using |X_ij| <= 1 when X is PSD and diag(X) = 1.
pub fn entrywise_lower_bound(q: &CsMat<f64>) -> f64 {
    let mut bound = 0.0;
    let mut magnitude = 0.0;
    for (&value, (i, j)) in q {
        bound += if i == j { value } else { -value.abs() };
        magnitude += value.abs();
    }
    bound - 8.0 * f64::EPSILON * (q.nnz() as f64 + 1.0) * magnitude
}

/// Repair a candidate dual using the smallest slack eigenvalue.
/// Includes a floating-point safety margin, not an interval-arithmetic proof.
pub fn dual_bound_from_variables(q: &CsMat<f64>, y: &Array1<f64>) -> f64 {
    assert_eq!(q.rows(), y.len());
    if y.is_empty() {
        return 0.0;
    }
    let n = q.rows() as f64;
    let mut slack = q.to_dense();
    for i in 0..q.rows() {
        slack[[i, i]] -= y[i];
    }
    let slack_norm = slack
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let margin = 64.0 * f64::EPSILON * n * slack_norm;
    // ndarray-linalg 0.18.1's eigvalsh still calls eigh(true). Explicitly disable
    // eigenvectors at the safe LAPACK layer instead of paying for discarded vectors.
    let layout = slack.square_layout().expect("slack must be square");
    let Ok(eigenvalues) = f64::eigh(
        false,
        layout,
        UPLO::Upper,
        slack
            .as_slice_memory_order_mut()
            .expect("dense slack must be contiguous"),
    ) else {
        return entrywise_lower_bound(q);
    };
    let min_eigenvalue = eigenvalues[0];
    if !min_eigenvalue.is_finite() {
        return entrywise_lower_bound(q);
    }
    let shift = (min_eigenvalue - margin).min(0.0);
    let sum_margin =
        8.0 * f64::EPSILON * n * (y.iter().map(|value| value.abs()).sum::<f64>() + n * shift.abs());
    shift.mul_add(n, y.sum()) - sum_margin
}

pub fn compute_rounded_sol(Q: &CsMat<f64>, V: &Array2<f64>, iters: usize) -> (Array1<f64>, f64) {
    // instantiate a PRNG
    let mut prng = PRNG {
        generator: JsfLarge::default(),
    };

    // create a tracker for the best solution
    let mut best_sol = Array1::zeros(V.shape()[0]);
    let mut best_obj = f64::MAX;

    // create scratch space for the rounded solution and random arrays we are making
    let mut x_scratch = Array1::zeros(V.shape()[0]);
    let mut r_scratch = Array1::zeros(V.shape()[1]);

    for _ in 0..iters {
        // generate a random vector on the n sphere
        r_scratch.mapv_inplace(|_| prng.normal());
        r_scratch /= r_scratch.norm_l2();

        // compute the rounded solution
        x_scratch.assign(&V.dot(&r_scratch));
        x_scratch.mapv_inplace(|x| if x > 0.0 { 1.0 } else { -1.0 });

        // compute the objective value
        let obj_rounded = obj_rounded(Q, &x_scratch);

        // if this is a better solution than the best solution (so far), update the best solution
        if obj_rounded < best_obj {
            best_obj = obj_rounded;
            best_sol.assign(&x_scratch);
        }
    }

    // return the best solution and its objective value
    (best_sol, best_obj)
}

pub(crate) fn compute_rounded_sols(V: &Array2<f64>, k: usize) -> Vec<Array1<f64>> {
    // instantiate a PRNG
    let mut prng = PRNG {
        generator: JsfLarge::default(),
    };

    let mut rounded_sols = Vec::new();

    // create scratch space for the rounded solution and random arrays we are making
    let mut x_scratch = Array1::zeros(V.shape()[0]);
    let mut r_scratch = Array1::zeros(V.shape()[1]);

    for _ in 0..k {
        // generate a random vector on the n sphere
        r_scratch.mapv_inplace(|_| prng.normal());
        r_scratch /= r_scratch.norm_l2();

        // compute the rounded solution
        x_scratch.assign(&V.dot(&r_scratch));
        x_scratch.mapv_inplace(|x| if x > 0.0 { 1.0 } else { -1.0 });

        // push the rounded solution to the vector
        rounded_sols.push(x_scratch.clone());
    }

    // return the best solution and its objective value
    rounded_sols
}

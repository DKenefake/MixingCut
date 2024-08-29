use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Norm, UPLO};
use smolprng::{JsfLarge, PRNG};
use sprs::CsMat;

pub fn get_Q_norm(Q: &CsMat<f64>) -> f64 {
    // compute the l1 norm of Q
    let mut c = Array1::<f64>::zeros(Q.shape().0);
    for (q_ij, (i, _)) in Q.iter() {
        c[i] += q_ij.abs();
    }
    c.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap().clone()
}

pub fn obj(Q: &CsMat<f64>, V:&Array2<f64>) -> f64{
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        if i == j{
            trace += q_ij * V.row(i).dot(&V.row(j));
        }
        if i < j{
            trace += 2.0 * q_ij * V.row(i).dot(&V.row(j));
        }
    }

    trace
}

pub fn obj_rounded(Q: &CsMat<f64>, x_0: &Array1<f64>) -> f64{
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        if i == j{
            trace += q_ij * x_0[i] * x_0[i];
        }
        if i < j{
            trace += 2.0 * q_ij * x_0[i] * x_0[j];
        }
    }

    trace
}

pub fn grad(Q: &CsMat<f64>, V:&Array2<f64>) -> Array2<f64>{
    2.0 * (Q * V)
}

pub fn dual_variables(Q: &CsMat<f64>, V: &Array2<f64>) -> Array1<f64>{

    // based on equation 8 of https://arxiv.org/pdf/0807.4423, much more efficient than the naive implementation
    // l_i = tr(V.T* E_ii * X * V)/ tr(V.T*E_ii*E_ii*V)
    let mut dual = Array1::<f64>::zeros(Q.shape().0);

    let G = Q * V;

    for i in 0..Q.shape().0{
        dual[i] = G.row(i).dot(&V.row(i));
    }

    dual
}


pub fn dual_bound(Q: &CsMat<f64>, V: &Array2<f64>) -> f64{
    // this is very expensive to compute O(n^3) no matter what

    let n = Q.shape().0 as f64;
    let y = dual_variables(Q, V);
    let y_sum = y.iter().sum::<f64>();

    // start the S matrix from the dense version of Q
    let mut S = Q.to_dense();

    // subtract the dual variables from the diagonal
    for i in 0..Q.shape().0{
        S[[i, i]] -= y[i];
    }

    // compute the eigenvalues
    let (eigs, _) = S.eigh(UPLO::Upper).unwrap();

    // find the lowest eigenvalue
    let min_eig = eigs.iter().fold(f64::INFINITY, |acc, &x| x.min(acc));

    // return the dual bound
    y_sum + min_eig * n
}

pub(crate) fn compute_rounded_sol(Q: &CsMat<f64>, V: &Array2<f64>, iters: usize) -> (Array1<f64>, f64){

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

    for _ in 0..iters{

        // generate a random vector on the n sphere
        r_scratch.mapv_inplace(|_| prng.normal());
        r_scratch /= r_scratch.norm_l2();

        // compute the rounded solution
        x_scratch.assign(&V.dot(&r_scratch));
        x_scratch.mapv_inplace(|x| if x > 0.0 {1.0} else {-1.0});

        // compute the objective value
        let obj_rounded = obj_rounded(Q, &x_scratch);

        // if this is a better solution than the best solution (so far), update the best solution
        if obj_rounded < best_obj{
            best_obj = obj_rounded;
            best_sol.assign(&x_scratch);
        }
    }

    // return the best solution and its objective value
    (best_sol, best_obj)
}

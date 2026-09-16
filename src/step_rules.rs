use crate::maxcut_oracle::{grad, obj};
use crate::sdp_project::project;
use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use sprs::CsMat;

#[derive(Clone, Copy)]
pub enum StepRule {
    Grad(f64),
    GradAdv(f64),
    Coord(f64),
    CoordNoStep,
    /// Normalize (1 + beta) * exact_update - beta * old_row, with 0 <= beta < 1.
    CoordMomentum(f64),
}

pub fn generate_step_rule(step_rule: &str, alpha: f64) -> StepRule {
    match step_rule {
        "grad" => StepRule::Grad(alpha),
        "grad_adv" => StepRule::GradAdv(alpha),
        "coord" => StepRule::Coord(alpha),
        "coord_no_step" => StepRule::CoordNoStep,
        "coord_momentum" => StepRule::CoordMomentum(alpha),
        _ => StepRule::Grad(alpha),
    }
}

pub fn apply_step(Q: &CsMat<f64>, V: Array2<f64>, step_rule: StepRule) -> Array2<f64> {
    match step_rule {
        StepRule::Grad(alpha) => make_step(Q, V, alpha),
        StepRule::GradAdv(alpha) => make_step_adv(Q, V, alpha),
        StepRule::Coord(alpha) => make_step_coord(Q, V, alpha),
        StepRule::CoordNoStep => make_step_coord_no_step(Q, V),
        StepRule::CoordMomentum(beta) => make_step_coord_momentum(Q, V, beta),
    }
}

fn l2_norm(values: &Array1<f64>) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn make_step(Q: &CsMat<f64>, V: Array2<f64>, alpha_safe: f64) -> Array2<f64> {
    // compute gradient
    let grad = grad(Q, &V);
    // take gradient step and project
    project(V - alpha_safe * grad)
}

pub fn make_step_adv(Q: &CsMat<f64>, V: Array2<f64>, alpha_safe: f64) -> Array2<f64> {
    // compute gradient
    let grad = grad(Q, &V);

    // take the objective value at f(alpha = 0), f(alpha = alpha_safe), f(alpha = -alpha_safe)
    let f_0 = obj(Q, &V);
    let x = obj(Q, &(&V + alpha_safe * &grad)) - f_0;
    let y = obj(Q, &(&V - alpha_safe * &grad)) - f_0;

    // compute the step size based on the quadratic approximation
    let mut alpha = (0.5 * (y - x) * alpha_safe) / (x + y);

    // take a step
    let proposed_step_val = obj(Q, &(&V - alpha * &grad));

    // if the step is not a descent, take the safe step size
    if proposed_step_val > f_0 {
        alpha = alpha_safe;
    }

    // take the step and project
    project(V - alpha * grad)
}

pub fn make_step_coord(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64) -> Array2<f64> {
    // apply coordinate descent with a step size
    for i in 0..Q.shape().0 {
        // take a view of the i-th row of Q
        let Q_i = Q.outer_view(i).unwrap();

        // make a scratch space for the gradient
        let mut g_i = Array1::<f64>::zeros(V.shape()[1]);

        // compute g_i
        for (k, &v) in Q_i.iter() {
            if k != i {
                g_i = g_i + v * &V.row(k);
            }
        }

        // normalize g_i
        g_i = &V.row(i) - alpha_safe * g_i;
        g_i /= g_i.norm_l2();

        // update the i-th row of V
        V.row_mut(i).assign(&g_i);
    }

    V
}

pub fn make_step_coord_no_step(Q: &CsMat<f64>, V: Array2<f64>) -> Array2<f64> {
    make_step_coord_momentum(Q, V, 0.0)
}

pub fn make_step_coord_momentum(Q: &CsMat<f64>, mut V: Array2<f64>, beta: f64) -> Array2<f64> {
    if !V.is_standard_layout() {
        V = V.as_standard_layout().into_owned();
    }
    let mut scratch = vec![0.0; V.ncols()];
    make_step_coord_in_place(Q, &mut V, &mut scratch, beta);
    V
}

pub(crate) fn make_step_coord_no_step_in_place(
    Q: &CsMat<f64>,
    V: &mut Array2<f64>,
    g_i: &mut [f64],
) {
    make_step_coord_in_place(Q, V, g_i, 0.0);
}

pub(crate) fn validate_momentum(beta: f64) {
    assert!(
        beta.is_finite() && (0.0..1.0).contains(&beta),
        "coordinate momentum must be finite and in [0, 1)"
    );
}

pub(crate) fn make_step_coord_in_place(
    Q: &CsMat<f64>,
    V: &mut Array2<f64>,
    scratch: &mut [f64],
    beta: f64,
) {
    validate_momentum(beta);
    if beta == 0.0 {
        dispatch_coordinate_sweep::<false>(Q, V, scratch, beta);
    } else {
        dispatch_coordinate_sweep::<true>(Q, V, scratch, beta);
    }
}

fn dispatch_coordinate_sweep<const MOMENTUM: bool>(
    q: &CsMat<f64>,
    v: &mut Array2<f64>,
    scratch: &mut [f64],
    beta: f64,
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the runtime check verifies AVX2 support, including OS vector state.
        // All indexing and arithmetic in the specialized kernel use safe Rust.
        unsafe {
            coordinate_sweep_avx2::<MOMENTUM>(q, v, scratch, beta);
        }
        return;
    }
    coordinate_sweep::<MOMENTUM>(q, v, scratch, beta);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn coordinate_sweep_avx2<const MOMENTUM: bool>(
    q: &CsMat<f64>,
    v: &mut Array2<f64>,
    scratch: &mut [f64],
    beta: f64,
) {
    coordinate_sweep::<MOMENTUM>(q, v, scratch, beta);
}

#[inline(always)]
fn squared_norm(values: &[f64]) -> f64 {
    let mut sums = [0.0; 4];
    let mut chunks = values.chunks_exact(4);
    for c in &mut chunks {
        for j in 0..4 {
            sums[j] += c[j] * c[j];
        }
    }
    let mut norm = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for &value in chunks.remainder() {
        norm += value * value;
    }
    norm
}

#[inline(always)]
fn gradient_block<const WIDTH: usize>(
    row: &sprs::CsVecView<'_, f64>,
    values: &[f64],
    vertex: usize,
    rank: usize,
    offset: usize,
) -> [f64; WIDTH] {
    let mut sums = [0.0; WIDTH];
    for (k, &weight) in row.iter() {
        if k != vertex {
            let source = &values[k * rank + offset..k * rank + offset + WIDTH];
            for j in 0..WIDTH {
                sums[j] -= weight * source[j];
            }
        }
    }
    sums
}

#[inline(always)]
fn coordinate_sweep<const MOMENTUM: bool>(
    Q: &CsMat<f64>,
    V: &mut Array2<f64>,
    g_i: &mut [f64],
    beta: f64,
) {
    let rank = V.ncols();
    assert_eq!(g_i.len(), rank);
    let values = V
        .as_slice_mut()
        .expect("coordinate factor must be row-major");
    g_i.fill(0.0);
    for i in 0..Q.shape().0 {
        let Q_i = Q.outer_view(i).unwrap();
        let mut blocks = g_i.chunks_exact_mut(16);
        for (block, gradient) in blocks.by_ref().enumerate() {
            gradient.copy_from_slice(&gradient_block::<16>(&Q_i, values, i, rank, block * 16));
        }
        let remainder = blocks.into_remainder();
        let offset = rank - remainder.len();
        let mut blocks = remainder.chunks_exact_mut(4);
        for (block, gradient) in blocks.by_ref().enumerate() {
            gradient.copy_from_slice(&gradient_block::<4>(
                &Q_i,
                values,
                i,
                rank,
                offset + block * 4,
            ));
        }
        let tail = blocks.into_remainder();
        let offset = rank - tail.len();
        match tail.len() {
            1 => tail.copy_from_slice(&gradient_block::<1>(&Q_i, values, i, rank, offset)),
            2 => tail.copy_from_slice(&gradient_block::<2>(&Q_i, values, i, rank, offset)),
            3 => tail.copy_from_slice(&gradient_block::<3>(&Q_i, values, i, rank, offset)),
            _ => {}
        }

        let norm = squared_norm(g_i).sqrt();
        if norm >= 1E-24 {
            let row = &mut values[i * rank..(i + 1) * rank];
            let inv_norm = norm.recip();
            for (value, &gradient) in row.iter_mut().zip(g_i.iter()) {
                let exact = gradient * inv_norm;
                *value = if MOMENTUM {
                    (1.0 + beta) * exact - beta * *value
                } else {
                    exact
                };
            }
            if MOMENTUM {
                let inv_norm = squared_norm(row).sqrt().recip();
                for value in row {
                    *value *= inv_norm;
                }
            }
        }

        g_i.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialize::make_random_matrix;
    use sprs::TriMat;

    #[test]
    fn portable_and_dispatched_kernels_match_across_vector_tails() {
        let mut tri = TriMat::new((8, 8));
        for i in 0..7 {
            for j in i + 1..7 {
                let value = ((3 * i + j) % 5) as f64 - 2.0;
                tri.add_triplet(i, j, value);
                tri.add_triplet(j, i, value);
            }
            tri.add_triplet(i, i, 1000.0);
        }
        let q = tri.to_csr();
        for rank in [1, 2, 3, 4, 5, 7, 8, 9, 16, 21, 33, 41, 65] {
            for beta in [0.0, 0.2, 0.5, 0.8, 0.99] {
                let mut portable = make_random_matrix(8, rank, Some(71));
                let isolated_row = portable.row(7).to_owned();
                let mut dispatched = portable.clone();
                let mut scratch = vec![0.0; rank];
                let mut previous = obj(&q, &portable);
                for _ in 0..15 {
                    if beta == 0.0 {
                        coordinate_sweep::<false>(&q, &mut portable, &mut scratch, beta);
                    } else {
                        coordinate_sweep::<true>(&q, &mut portable, &mut scratch, beta);
                    }
                    make_step_coord_in_place(&q, &mut dispatched, &mut scratch, beta);
                    assert!((&portable - &dispatched).iter().all(|v| v.abs() < 1e-12));
                    assert_eq!(dispatched.row(7), isolated_row.view());
                    for row in dispatched.rows() {
                        assert!((row.dot(&row) - 1.0).abs() < 1e-13);
                    }
                    let current = obj(&q, &dispatched);
                    assert!(current <= previous + 1e-10);
                    previous = current;
                }
            }
        }
    }

    #[test]
    fn invalid_momentum_is_rejected() {
        for beta in [-0.1, 1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(std::panic::catch_unwind(|| validate_momentum(beta)).is_err());
        }
    }
}

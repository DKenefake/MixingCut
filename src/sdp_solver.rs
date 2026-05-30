use crate::initialize::make_random_matrix;
use crate::maxcut_oracle::{dual_variables, dual_variables_with_QV, get_Q_norm};
use crate::sdp_project;
use crate::step_rules::{apply_step, StepRule};
use ndarray::Array1;
use ndarray_linalg::Norm;
use sprs::CsMat;

// simple implementation that computes the perturbation vector, y, for a hessian matrix, Q
pub fn compute_approx_perturbation(
    Q: &CsMat<f64>,
    rank: Option<usize>,
    seed: Option<u64>,
    iters: Option<usize>,
    stat_tol: Option<f64>,
    step_rule: Option<StepRule>,
    verbose: bool,
) -> Array1<f64> {
    let use_rank = rank.map_or_else(|| (2 * Q.cols()).isqrt() + 1, |r| r);

    let norm_Q = get_Q_norm(Q);

    let use_iters = iters.unwrap_or(1000);

    let use_stat_tol = stat_tol.unwrap_or(1e-4);

    let use_step_rule = step_rule.unwrap_or(StepRule::CoordNoStep);

    let mut V = make_random_matrix(Q.rows(), use_rank, seed);

    for i in 0..use_iters {
        // take a single step of the coordinate descent

        V = apply_step(Q, V, use_step_rule);
        V = sdp_project::project(V);

        if i % 100 == 0 {
            let mut QV = Q * &V;
            let y = dual_variables_with_QV(&QV, &V);

            // compute ||QV - y * V||_2^2
            QV = QV - &y.view().insert_axis(ndarray::Axis(1)) * &V;
            let norm_diff = QV.norm_l2().powi(2);

            if verbose {
                println!("Iteration {}: ||QV - y * V||_2^2 = {}", i, norm_diff,);
            }

            if norm_diff < use_stat_tol * norm_Q {
                return y;
            }
        }
    }

    dual_variables(Q, &V)
}

use crate::initialize::make_random_matrix;
use crate::maxcut_oracle::{dual_bound_from_variables, obj};
use crate::sdp_solver::{
    default_sdp_rank, eval_qubo_objective, solve_maxcut_sdp, solve_qubo_sdp_subproblem,
    SolveOptions, SolveStatus, WarmStart,
};
use crate::step_rules::{make_step_coord_no_step, StepRule};
use ndarray::{array, Array1, Array2};
use ndarray_linalg::{EigValsh, UPLO};
use sprs::{CsMat, TriMat};

fn graph(n: usize, edges: &[(usize, usize, f64)], diagonal: f64) -> CsMat<f64> {
    let mut q = TriMat::new((n, n));
    for &(i, j, value) in edges {
        q.add_triplet(i, j, value);
        q.add_triplet(j, i, value);
    }
    for i in 0..n {
        if diagonal != 0.0 {
            q.add_triplet(i, i, diagonal);
        }
    }
    q.to_csr()
}

fn options() -> SolveOptions {
    SolveOptions {
        seed: Some(7),
        max_iterations: 400,
        min_stationarity_iterations: 0,
        stationarity_tolerance: 1e-8,
        compute_rounding: false,
        compute_dual_bound: true,
        ..SolveOptions::default()
    }
}

#[test]
fn rank_capacity_and_factor_rank_inference() {
    assert_eq!(default_sdp_rank(487), 33);
    assert_eq!(default_sdp_rank(1), 1);
    let q = graph(4, &[(0, 1, 1.0)], 0.0);
    let factor = make_random_matrix(4, 2, Some(3));
    let result = solve_maxcut_sdp(
        &q,
        &SolveOptions {
            max_iterations: 0,
            warm_start: WarmStart::Factor(factor.clone()),
            ..options()
        },
    );
    assert_eq!(result.rank, 2);
    assert!((&result.factor_matrix - &factor)
        .iter()
        .all(|v| v.abs() < 1e-14));
    let singleton = solve_maxcut_sdp(&graph(1, &[], 3.0), &options());
    assert_eq!(singleton.rank, 1);
    assert_eq!(singleton.relaxed_objective, 3.0);
    assert!(singleton.dual_bound.unwrap() <= 3.0);
}

#[test]
fn diagonal_constants_do_not_change_stationarity_or_trajectory() {
    let edges = [
        (0, 1, 1.0),
        (1, 2, 0.7),
        (2, 3, -0.4),
        (0, 3, 0.6),
        (0, 2, 0.2),
    ];
    let q = graph(4, &edges, 0.0);
    let base = solve_maxcut_sdp(&q, &options());
    for shift in [1e8, -1e8] {
        let shifted = solve_maxcut_sdp(&graph(4, &edges, shift), &options());
        assert_eq!(base.iterations, shifted.iterations);
        assert_eq!(base.status, shifted.status);
        assert_eq!(base.factor_matrix, shifted.factor_matrix);
        assert!((shifted.relaxed_objective - 4.0 * shift - base.relaxed_objective).abs() < 1e-7);
        for i in 0..4 {
            assert!((shifted.dual_variables[i] - shift - base.dual_variables[i]).abs() < 1e-7);
        }
        assert!(
            (shifted.dual_bound.unwrap() - 4.0 * shift - base.dual_bound.unwrap()).abs() < 1e-5
        );
    }
}

#[test]
fn sign_start_can_escape_a_rank_one_stationary_point() {
    let q = graph(3, &[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)], 0.0);
    let result = solve_maxcut_sdp(
        &q,
        &SolveOptions {
            warm_start: WarmStart::Signs(array![1.0, 1.0, -1.0]),
            ..options()
        },
    );
    assert_eq!(result.status, SolveStatus::ObjectiveTolerance);
    assert!((result.relaxed_objective + 3.0).abs() < 1e-8);
    assert!(result.dual_bound.unwrap() <= -3.0 + 1e-12);
    assert!(result.dual_bound.unwrap() >= -3.0 - 1e-6);
}

#[test]
fn qubo_bound_is_not_the_primal_value_even_at_bad_stationary_points() {
    let base = graph(3, &[(0, 1, -3.0)], 2.0);
    let mut h = TriMat::new((3, 3));
    for (&value, (i, j)) in &base {
        h.add_triplet(i, j, value);
    }
    h.add_triplet(2, 2, 18.0);
    let h = h.to_csr();
    let c = Array1::zeros(3);
    let mut factor = Array2::zeros((4, 4));
    factor.column_mut(0).fill(1.0);
    for compute_dual_bound in [false, true] {
        for max_iterations in [0, 1] {
            let result = solve_qubo_sdp_subproblem(
                &h,
                &c,
                &SolveOptions {
                    max_iterations,
                    compute_dual_bound,
                    warm_start: WarmStart::Factor(factor.clone()),
                    ..options()
                },
            );
            assert!(result.relaxed_objective.abs() < 1e-12);
            assert!(result.qubo_lower_bound <= -1.0);
            assert_eq!(result.dual_bound.is_some(), compute_dual_bound);
            if let Some(bound) = result.dual_bound {
                assert_eq!(bound, result.qubo_lower_bound);
            }
        }
    }
}

#[test]
fn interrupted_qubo_bounds_respect_exhaustive_binary_optima() {
    for seed in 0..12 {
        let n = 5;
        let dense = make_random_matrix(n, n, Some(seed));
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            for j in 0..n {
                tri.add_triplet(i, j, dense[[i, j]] + dense[[j, i]]);
            }
        }
        let h = tri.to_csr();
        let c = Array1::from_shape_fn(n, |i| dense[[i, i]]);
        let exact = (0..1 << n)
            .map(|mask| {
                let x = Array1::from_shape_fn(n, |i| ((mask >> i) & 1) as f64);
                eval_qubo_objective(&h, &c, &x)
            })
            .fold(f64::INFINITY, f64::min);
        for (compute_dual_bound, step_rule) in [
            (false, StepRule::CoordNoStep),
            (true, StepRule::CoordNoStep),
            (false, StepRule::CoordMomentum(0.5)),
            (true, StepRule::CoordMomentum(0.5)),
            (false, StepRule::CoordMomentum(0.8)),
            (true, StepRule::CoordMomentum(0.8)),
        ] {
            for max_iterations in [0, 1, 100] {
                let result = solve_qubo_sdp_subproblem(
                    &h,
                    &c,
                    &SolveOptions {
                        seed: Some(seed),
                        max_iterations,
                        compute_dual_bound,
                        step_rule,
                        ..options()
                    },
                );
                assert!(result.qubo_lower_bound <= exact + 1e-12);
            }
        }
    }
}

#[test]
fn result_matches_factor_on_every_exit_path() {
    let q = graph(4, &[(0, 1, 1.0), (1, 2, 0.7), (2, 3, -0.4)], 3.0);
    for max_iterations in [0, 1, 7, 100] {
        for step_rule in [
            StepRule::CoordNoStep,
            StepRule::Grad(-1.0),
            StepRule::Grad(0.01),
        ] {
            let result = solve_maxcut_sdp(
                &q,
                &SolveOptions {
                    max_iterations,
                    step_rule,
                    ..options()
                },
            );
            assert!((obj(&q, &result.factor_matrix) - result.relaxed_objective).abs() < 1e-10);
        }
    }
}

#[test]
fn verbose_and_sparse_storage_do_not_change_solves() {
    let q = graph(4, &[(0, 1, 1.0), (1, 2, 0.7), (2, 3, -0.4)], 3.0);
    let baseline = solve_maxcut_sdp(&q, &options());
    let verbose = solve_maxcut_sdp(
        &q,
        &SolveOptions {
            verbose: true,
            ..options()
        },
    );
    let csc = solve_maxcut_sdp(&q.to_csc(), &options());
    for result in [verbose, csc] {
        assert_eq!(baseline.factor_matrix, result.factor_matrix);
        assert_eq!(baseline.iterations, result.iterations);
        assert_eq!(baseline.dual_bound, result.dual_bound);
    }
}

#[test]
fn coordinate_kernel_matches_scalar_reference_and_accepts_column_major() {
    let q = graph(7, &[(0, 1, 1.0), (1, 2, -0.7), (2, 4, 0.3)], 5.0);
    let mut reference = make_random_matrix(3, 7, Some(9)).reversed_axes();
    let mut actual = reference.clone();
    for _ in 0..12 {
        for i in 0..q.rows() {
            let mut gradient = vec![0.0; 3];
            for (k, &weight) in q.outer_view(i).unwrap().iter() {
                if k != i {
                    for j in 0..3 {
                        gradient[j] -= weight * reference[[k, j]];
                    }
                }
            }
            let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm >= 1e-24 {
                for j in 0..3 {
                    reference[[i, j]] = gradient[j] / norm;
                }
            }
        }
        actual = make_step_coord_no_step(&q, actual);
        assert!((&actual - &reference).iter().all(|v| v.abs() < 1e-13));
    }
}

#[test]
fn eigenvalue_repair_makes_a_dual_feasible() {
    let q = graph(5, &[(0, 1, 1.0), (1, 2, -0.7), (2, 4, 0.3)], 5.0);
    let y = array![9.0, -4.0, 2.0, 8.0, -1.0];
    let bound = dual_bound_from_variables(&q, &y);
    let shift = (bound - y.sum()) / 5.0;
    let mut slack = q.to_dense();
    for i in 0..5 {
        slack[[i, i]] -= y[i] + shift;
    }
    assert!(slack.eigvalsh(UPLO::Upper).unwrap()[0] >= -1e-12);
}

#[test]
fn momentum_converges_to_triangle_optimum_with_a_valid_bound() {
    let q = graph(3, &[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)], 0.0);
    for beta in [0.0, 0.2, 0.5, 0.8] {
        let result = solve_maxcut_sdp(
            &q,
            &SolveOptions {
                step_rule: StepRule::CoordMomentum(beta),
                ..options()
            },
        );
        assert_eq!(result.status, SolveStatus::ObjectiveTolerance);
        assert!((result.relaxed_objective + 3.0).abs() < 1e-9);
        assert!(result.dual_bound.unwrap() <= -3.0 + 1e-12);
        assert!(result.dual_bound.unwrap() >= -3.0 - 1e-6);
        let qv = &q * &result.factor_matrix;
        let mut residual = 0.0;
        for i in 0..q.rows() {
            let y = qv.row(i).dot(&result.factor_matrix.row(i));
            for j in 0..result.rank {
                residual += (qv[[i, j]] - y * result.factor_matrix[[i, j]]).powi(2);
            }
        }
        let norm = qv.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(residual.sqrt() <= options().stationarity_tolerance * norm.max(1.0));
    }
}

#[test]
fn momentum_zero_is_the_standard_update_and_does_not_use_objective_stopping() {
    let q = graph(
        7,
        &[(0, 1, 1.0), (1, 2, -0.7), (2, 4, 0.3), (0, 4, 0.8)],
        0.0,
    );
    let config = SolveOptions {
        max_iterations: 4,
        stationarity_tolerance: 0.0,
        objective_tolerance: 1e10,
        ..options()
    };
    let base = solve_maxcut_sdp(&q, &config);
    let zero = solve_maxcut_sdp(
        &q,
        &SolveOptions {
            step_rule: StepRule::CoordMomentum(0.0),
            ..config.clone()
        },
    );
    assert_eq!(base.factor_matrix, zero.factor_matrix);
    assert_eq!(base.dual_bound, zero.dual_bound);
    for beta in [0.2, 0.5, 0.8] {
        let result = solve_maxcut_sdp(
            &q,
            &SolveOptions {
                step_rule: StepRule::CoordMomentum(beta),
                ..config.clone()
            },
        );
        assert_eq!(result.iterations, 4);
        assert_eq!(result.status, SolveStatus::MaxIterations);
    }
}

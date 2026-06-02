use crate::initialize::make_random_matrix;
use crate::maxcut_oracle::{
    compute_rounded_sol, dual_bound, dual_variables, dual_variables_with_QV, get_Q_norm, obj,
    obj_from_qv,
};
use crate::sdp_local_search::beam_search;
use crate::sdp_project;
use crate::step_rules::{apply_step, StepRule};
use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use sprs::{CsMat, TriMat};
use std::time::Instant;

#[derive(Clone)]
pub enum WarmStart {
    Random,
    Factor(Array2<f64>),
    Signs(Array1<f64>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolveStatus {
    MaxIterations,
    ObjectiveTolerance,
    ObjectiveIncreased,
}

#[derive(Clone)]
pub struct SolveOptions {
    pub rank: Option<usize>,
    pub seed: Option<u64>,
    pub max_iterations: usize,
    pub min_stationarity_iterations: usize,
    pub objective_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub rounding_iterations: usize,
    pub beam_width: Option<usize>,
    pub compute_dual_bound: bool,
    pub compute_rounding: bool,
    pub step_rule: StepRule,
    pub verbose: bool,
    pub warm_start: WarmStart,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            rank: None,
            seed: None,
            max_iterations: 1000,
            min_stationarity_iterations: 20,
            objective_tolerance: 1e-2,
            stationarity_tolerance: 1e-4,
            rounding_iterations: 100,
            beam_width: Some(128),
            compute_dual_bound: false,
            compute_rounding: true,
            step_rule: StepRule::CoordNoStep,
            verbose: false,
            warm_start: WarmStart::Random,
        }
    }
}

pub struct SolveResult {
    pub factor_matrix: Array2<f64>,
    pub relaxed_objective: f64,
    pub dual_variables: Array1<f64>,
    pub dual_bound: Option<f64>,
    pub rounded_solution: Option<Array1<f64>>,
    pub rounded_objective: Option<f64>,
    pub locally_improved_solution: Option<Array1<f64>>,
    pub locally_improved_objective: Option<f64>,
    pub iterations: usize,
    pub elapsed_seconds: f64,
    pub rank: usize,
    pub status: SolveStatus,
}

pub struct ProfiledSolveResult {
    pub solve_result: SolveResult,
    pub iteration_seconds: f64,
    pub finalization_seconds: f64,
}

pub struct ReducedProblem {
    pub reduced_q: CsMat<f64>,
    pub original_size: usize,
    pub variable_map: Vec<usize>,
    pub fixed_assignments: Vec<(usize, f64)>,
}

pub struct ReducedSolveResult {
    pub reduced_result: SolveResult,
    pub lifted_rounded_solution: Array1<f64>,
    pub lifted_locally_improved_solution: Option<Array1<f64>>,
}

pub struct QuboSdpResult {
    pub qubo_lower_bound: f64,
    pub factor_matrix: Array2<f64>,
    pub dual_variables: Array1<f64>,
    pub dual_bound: Option<f64>,
    pub iterations: usize,
    pub elapsed_seconds: f64,
    pub rank: usize,
    pub status: SolveStatus,
}

#[must_use]
pub fn absorb_linear_terms_into_hessian(q: &CsMat<f64>, linear: &Array1<f64>) -> CsMat<f64> {
    assert_eq!(
        q.rows(),
        linear.len(),
        "linear term length must match Hessian dimension"
    );
    assert_eq!(q.rows(), q.cols(), "Hessian must be square");

    let mut tri_q = TriMat::<f64>::new((q.rows(), q.cols()));

    for (&q_ij, (i, j)) in q {
        tri_q.add_triplet(i, j, q_ij);
    }

    for i in 0..linear.len() {
        if linear[i] != 0.0 {
            tri_q.add_triplet(i, i, 2.0 * linear[i]);
        }
    }

    tri_q.to_csr()
}

#[must_use]
pub fn qubo_hessian_to_sign_matrix(hessian: &CsMat<f64>) -> CsMat<f64> {
    assert_eq!(hessian.rows(), hessian.cols(), "Hessian must be square");

    let n = hessian.rows();
    let mut row_sums = Array1::<f64>::zeros(n);
    let mut constant = 0.0;

    for (&h_ij, (i, j)) in hessian {
        if i == j {
            constant += 0.125 * h_ij;
            row_sums[i] += h_ij;
        } else if i < j {
            constant += 0.25 * h_ij;
            row_sums[i] += h_ij;
            row_sums[j] += h_ij;
        }
    }

    let mut tri_q = TriMat::<f64>::new((n + 1, n + 1));
    for (&h_ij, (i, j)) in hessian {
        let value = 0.125 * h_ij;
        tri_q.add_triplet(i, j, value);
    }

    for i in 0..n {
        let linear_coeff = -0.25 * row_sums[i];
        if linear_coeff != 0.0 {
            let edge_value = 0.5 * linear_coeff;
            tri_q.add_triplet(i, n, edge_value);
            tri_q.add_triplet(n, i, edge_value);
        }
    }

    tri_q.add_triplet(n, n, constant);
    tri_q.to_csr()
}

#[must_use]
pub fn qubo_binary_from_sign_solution(sign_solution: &Array1<f64>) -> Array1<f64> {
    assert!(
        !sign_solution.is_empty(),
        "sign solution must include anchor variable"
    );

    let anchor = sign_solution[sign_solution.len() - 1];
    let mut binary = Array1::<f64>::zeros(sign_solution.len() - 1);

    for i in 0..binary.len() {
        let effective_sign = sign_solution[i] * anchor;
        binary[i] = 0.5 * (1.0 - effective_sign);
    }

    binary
}

#[must_use]
pub fn eval_qubo_objective(
    quadratic: &CsMat<f64>,
    linear: &Array1<f64>,
    binary_solution: &Array1<f64>,
) -> f64 {
    0.5 * binary_solution.dot(&(quadratic * binary_solution)) + linear.dot(binary_solution)
}

impl ReducedProblem {
    #[must_use]
    pub fn from_principal_submatrix(
        q: &CsMat<f64>,
        variable_map: Vec<usize>,
        fixed_assignments: Vec<(usize, f64)>,
    ) -> Self {
        let mut reverse_map = vec![None; q.rows()];
        for (reduced_index, &original_index) in variable_map.iter().enumerate() {
            reverse_map[original_index] = Some(reduced_index);
        }

        let mut tri_q = TriMat::<f64>::new((variable_map.len(), variable_map.len()));
        for (&q_ij, (i, j)) in q {
            let Some(i_new) = reverse_map[i] else {
                continue;
            };
            let Some(j_new) = reverse_map[j] else {
                continue;
            };
            tri_q.add_triplet(i_new, j_new, q_ij);
        }

        Self {
            reduced_q: tri_q.to_csr(),
            original_size: q.rows(),
            variable_map,
            fixed_assignments,
        }
    }

    #[must_use]
    pub fn from_qubo_subproblem(
        quadratic: &CsMat<f64>,
        linear: &Array1<f64>,
        variable_map: Vec<usize>,
        fixed_assignments: Vec<(usize, f64)>,
    ) -> Self {
        assert_eq!(
            quadratic.rows(),
            linear.len(),
            "linear term length must match reduced quadratic dimension"
        );
        assert_eq!(
            quadratic.rows(),
            variable_map.len(),
            "variable map length must match reduced quadratic dimension"
        );

        let original_size = variable_map
            .iter()
            .copied()
            .chain(fixed_assignments.iter().map(|(index, _)| *index))
            .max()
            .map_or(0, |max_index| max_index + 1);

        Self {
            reduced_q: absorb_linear_terms_into_hessian(quadratic, linear),
            original_size,
            variable_map,
            fixed_assignments,
        }
    }

    #[must_use]
    pub fn lift_solution(&self, reduced_solution: &Array1<f64>) -> Array1<f64> {
        let mut lifted = Array1::<f64>::zeros(self.original_size);

        for (index, value) in &self.fixed_assignments {
            lifted[*index] = *value;
        }

        for (reduced_index, &original_index) in self.variable_map.iter().enumerate() {
            lifted[original_index] = reduced_solution[reduced_index];
        }

        lifted
    }
}

fn resolve_rank(q: &CsMat<f64>, rank: Option<usize>) -> usize {
    rank.map_or_else(|| 2 * (q.rows() as f64).log2() as usize, |value| value)
}

fn prepare_initial_factor(
    q: &CsMat<f64>,
    rank: usize,
    seed: Option<u64>,
    warm_start: &WarmStart,
) -> Array2<f64> {
    match warm_start {
        WarmStart::Random => make_random_matrix(q.rows(), rank, seed),
        WarmStart::Factor(v) => {
            assert_eq!(v.nrows(), q.rows(), "warm-start factor row count mismatch");
            assert_eq!(v.ncols(), rank, "warm-start factor rank mismatch");
            sdp_project::project(v.clone())
        }
        WarmStart::Signs(x) => {
            assert_eq!(x.len(), q.rows(), "warm-start sign vector length mismatch");
            let mut v = Array2::<f64>::zeros((q.rows(), rank));
            for i in 0..q.rows() {
                v[[i, 0]] = if x[i] >= 0.0 { 1.0 } else { -1.0 };
            }
            if rank > 1 {
                for i in 0..q.rows() {
                    for j in 1..rank {
                        v[[i, j]] = 0.0;
                    }
                }
            }
            sdp_project::project(v)
        }
    }
}

fn stationarity_residual_from_qv_with_dual(
    qv: &Array2<f64>,
    v: &Array2<f64>,
    dual: &Array1<f64>,
) -> f64 {
    let mut residual_sq = 0.0;

    for i in 0..qv.nrows() {
        for j in 0..qv.ncols() {
            let diff = qv[[i, j]] - dual[i] * v[[i, j]];
            residual_sq += diff * diff;
        }
    }

    residual_sq.sqrt()
}

fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[must_use]
pub fn solve_maxcut_sdp(q: &CsMat<f64>, options: &SolveOptions) -> SolveResult {
    solve_maxcut_sdp_profiled(q, options).solve_result
}

#[must_use]
pub fn solve_maxcut_sdp_profiled(q: &CsMat<f64>, options: &SolveOptions) -> ProfiledSolveResult {
    let rank = resolve_rank(q, options.rank);
    let start = Instant::now();
    let mut v = prepare_initial_factor(q, rank, options.seed, &options.warm_start);
    let mut obj_val = obj(q, &v);
    let mut status = SolveStatus::MaxIterations;
    let mut iterations = 0;
    let objective_check_frequency = if options.verbose { 10 } else { 20 };
    for i in 0..options.max_iterations {
        v = apply_step(q, v, options.step_rule);
        iterations = i + 1;
        let should_check_objective = iterations == 1
            || iterations == options.max_iterations
            || iterations % objective_check_frequency == 0;

        if should_check_objective {
            let qv = q * &v;
            let dual = dual_variables_with_QV(&qv, &v);
            let residual = stationarity_residual_from_qv_with_dual(&qv, &v, &dual);
            let qv_norm = frobenius_norm(&qv);
            let stationarity_threshold = options.stationarity_tolerance * qv_norm.max(1.0);
            let new_obj_val = obj_from_qv(&qv, &v);

            if iterations >= options.min_stationarity_iterations
                && residual <= stationarity_threshold
            {
                obj_val = new_obj_val;
                status = SolveStatus::ObjectiveTolerance;
                break;
            }

            if options.verbose && i % 10 == 0 {
                println!("Iteration {i}: objective {new_obj_val}, stationarity {residual}");
            }

            if !matches!(options.step_rule, StepRule::CoordNoStep)
                && (new_obj_val - obj_val).abs() < options.objective_tolerance
            {
                obj_val = new_obj_val;
                status = SolveStatus::ObjectiveTolerance;
                break;
            }

            if !matches!(options.step_rule, StepRule::CoordNoStep) && new_obj_val > obj_val {
                status = SolveStatus::ObjectiveIncreased;
                break;
            }

            obj_val = new_obj_val;
        }
    }

    let iteration_seconds = start.elapsed().as_secs_f64();
    let finalization_start = Instant::now();
    let qv = q * &v;
    let dual_variables = dual_variables_with_QV(&qv, &v);
    let dual_bound_value = options.compute_dual_bound.then(|| dual_bound(q, &v));
    let (
        rounded_solution,
        rounded_objective,
        locally_improved_solution,
        locally_improved_objective,
    ) = if options.compute_rounding {
        let (rounded_solution, rounded_objective) =
            compute_rounded_sol(q, &v, options.rounding_iterations);

        let (locally_improved_solution, locally_improved_objective) =
            options.beam_width.map_or((None, None), |beam_width| {
                if beam_width == 0 {
                    return (None, None);
                }

                let (best_obj, best_sol) =
                    beam_search(q, beam_width, vec![rounded_solution.clone()]);
                (Some(best_sol), Some(best_obj))
            });

        (
            Some(rounded_solution),
            Some(rounded_objective),
            locally_improved_solution,
            locally_improved_objective,
        )
    } else {
        (None, None, None, None)
    };

    let finalization_seconds = finalization_start.elapsed().as_secs_f64();

    ProfiledSolveResult {
        solve_result: SolveResult {
            factor_matrix: v,
            relaxed_objective: obj_val,
            dual_variables,
            dual_bound: dual_bound_value,
            rounded_solution,
            rounded_objective,
            locally_improved_solution,
            locally_improved_objective,
            iterations,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            rank,
            status,
        },
        iteration_seconds,
        finalization_seconds,
    }
}

#[must_use]
pub fn solve_reduced_problem(
    problem: &ReducedProblem,
    options: &SolveOptions,
) -> ReducedSolveResult {
    let reduced_result = solve_maxcut_sdp(&problem.reduced_q, options);
    let lifted_rounded_solution = reduced_result
        .rounded_solution
        .as_ref()
        .map(|solution| problem.lift_solution(solution))
        .unwrap_or_else(|| Array1::<f64>::zeros(problem.original_size));
    let lifted_locally_improved_solution = reduced_result
        .locally_improved_solution
        .as_ref()
        .map(|solution| problem.lift_solution(solution));

    ReducedSolveResult {
        reduced_result,
        lifted_rounded_solution,
        lifted_locally_improved_solution,
    }
}

#[must_use]
pub fn solve_qubo_sdp_subproblem(
    quadratic: &CsMat<f64>,
    linear: &Array1<f64>,
    options: &SolveOptions,
) -> QuboSdpResult {
    let effective_hessian = absorb_linear_terms_into_hessian(quadratic, linear);
    let sign_matrix = qubo_hessian_to_sign_matrix(&effective_hessian);
    let sdp_result = solve_maxcut_sdp(&sign_matrix, options);

    QuboSdpResult {
        qubo_lower_bound: sdp_result.relaxed_objective,
        factor_matrix: sdp_result.factor_matrix,
        dual_variables: sdp_result.dual_variables,
        dual_bound: sdp_result.dual_bound,
        iterations: sdp_result.iterations,
        elapsed_seconds: sdp_result.elapsed_seconds,
        rank: sdp_result.rank,
        status: sdp_result.status,
    }
}

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

#[cfg(test)]
mod tests {
    use super::{
        absorb_linear_terms_into_hessian, eval_qubo_objective, qubo_binary_from_sign_solution,
        qubo_hessian_to_sign_matrix, solve_maxcut_sdp, solve_qubo_sdp_subproblem,
        solve_reduced_problem, ReducedProblem, SolveOptions, SolveStatus, WarmStart,
    };
    use crate::maxcut_oracle::obj_rounded;
    use crate::step_rules::StepRule;
    use herculesabqp::matrix::QuadraticMatrix;
    use herculesabqp::solver::{PreparedSolver, SolverOptions as ABQPSolverOptions};
    use ndarray::{array, Array1, Array2};
    use sprs::{CsMat, TriMat};

    fn test_q() -> CsMat<f64> {
        let mut q = TriMat::<f64>::new((4, 4));
        q.add_triplet(0, 1, -0.5);
        q.add_triplet(1, 0, -0.5);
        q.add_triplet(1, 2, -0.75);
        q.add_triplet(2, 1, -0.75);
        q.add_triplet(2, 3, -0.25);
        q.add_triplet(3, 2, -0.25);
        q.to_csr()
    }

    fn brute_force_qubo_optimum(quadratic: &CsMat<f64>, linear: &Array1<f64>) -> f64 {
        let n = quadratic.rows();
        let mut best = f64::INFINITY;
        for mask in 0..(1usize << n) {
            let mut x = Array1::<f64>::zeros(n);
            for i in 0..n {
                x[i] = if (mask >> i) & 1 == 1 { 1.0 } else { 0.0 };
            }
            best = best.min(eval_qubo_objective(quadratic, linear, &x));
        }
        best
    }

    fn with_structural_diagonal(matrix: &CsMat<f64>) -> CsMat<f64> {
        let n = matrix.rows();
        let mut tri = TriMat::<f64>::with_capacity((n, n), matrix.nnz() + n);

        for (i, row) in matrix.outer_iterator().enumerate() {
            let mut has_diag = false;
            for (j, value) in row.iter() {
                if i == j {
                    has_diag = true;
                }
                tri.add_triplet(i, j, *value);
            }

            if !has_diag {
                tri.add_triplet(i, i, 0.0);
            }
        }

        tri.to_csr()
    }

    fn solve_abqp_box_relaxation(
        quadratic: &CsMat<f64>,
        linear: &Array1<f64>,
    ) -> herculesabqp::solver::SolverResult {
        let q = QuadraticMatrix::sparse(with_structural_diagonal(quadratic));
        let c = linear.to_vec();
        let lb = vec![0.0; quadratic.rows()];
        let ub = vec![1.0; quadratic.rows()];
        let mut options = ABQPSolverOptions {
            assume_symmetric: true,
            ..Default::default()
        };
        options.logging.verbose = false;
        options.stopping.dual_certification = true;
        options.polish.enabled = true;

        let prepared_solver =
            PreparedSolver::new(&q, &c, &options).expect("ABQP prepare should succeed");
        prepared_solver
            .solve(&lb, &ub, &options)
            .expect("ABQP solve should succeed")
    }

    #[test]
    fn solve_api_returns_well_formed_result() {
        let q = test_q();
        let result = solve_maxcut_sdp(
            &q,
            &SolveOptions {
                rank: Some(2),
                seed: Some(7),
                max_iterations: 25,
                min_stationarity_iterations: 1,
                objective_tolerance: 1e-8,
                stationarity_tolerance: 1e-8,
                rounding_iterations: 8,
                beam_width: Some(4),
                compute_dual_bound: false,
                compute_rounding: true,
                step_rule: StepRule::CoordNoStep,
                verbose: false,
                warm_start: WarmStart::Random,
            },
        );

        assert_eq!(result.factor_matrix.nrows(), q.rows());
        assert_eq!(result.factor_matrix.ncols(), 2);
        assert_eq!(result.rounded_solution.as_ref().unwrap().len(), q.rows());
        assert_eq!(result.dual_variables.len(), q.rows());
        assert!(result.elapsed_seconds >= 0.0);
        assert!(matches!(
            result.status,
            SolveStatus::MaxIterations
                | SolveStatus::ObjectiveTolerance
                | SolveStatus::ObjectiveIncreased
        ));
    }

    #[test]
    fn factor_warm_start_is_used_with_zero_iterations() {
        let q = test_q();
        let warm_start =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).unwrap();

        let result = solve_maxcut_sdp(
            &q,
            &SolveOptions {
                rank: Some(2),
                seed: None,
                max_iterations: 0,
                min_stationarity_iterations: 0,
                objective_tolerance: 1e-12,
                stationarity_tolerance: 1e-12,
                rounding_iterations: 4,
                beam_width: Some(0),
                compute_dual_bound: false,
                compute_rounding: true,
                step_rule: StepRule::CoordNoStep,
                verbose: false,
                warm_start: WarmStart::Factor(warm_start.clone()),
            },
        );

        assert_eq!(result.iterations, 0);
        assert_eq!(result.factor_matrix.nrows(), warm_start.nrows());
        assert_eq!(result.factor_matrix.ncols(), warm_start.ncols());
        for i in 0..result.factor_matrix.nrows() {
            let norm = result
                .factor_matrix
                .row(i)
                .dot(&result.factor_matrix.row(i));
            assert!((norm - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn sign_warm_start_produces_rank_one_embedding() {
        let q = test_q();
        let signs = array![1.0, -1.0, 1.0, -1.0];

        let result = solve_maxcut_sdp(
            &q,
            &SolveOptions {
                rank: Some(3),
                seed: None,
                max_iterations: 0,
                min_stationarity_iterations: 0,
                objective_tolerance: 1e-12,
                stationarity_tolerance: 1e-12,
                rounding_iterations: 4,
                beam_width: Some(0),
                compute_dual_bound: false,
                compute_rounding: true,
                step_rule: StepRule::CoordNoStep,
                verbose: false,
                warm_start: WarmStart::Signs(signs.clone()),
            },
        );

        for i in 0..signs.len() {
            assert_eq!(result.factor_matrix[[i, 0]], signs[i]);
            assert_eq!(result.factor_matrix[[i, 1]], 0.0);
            assert_eq!(result.factor_matrix[[i, 2]], 0.0);
        }
    }

    #[test]
    fn reduced_problem_lifts_solution_back_to_original_space() {
        let q = test_q();
        let problem =
            ReducedProblem::from_principal_submatrix(&q, vec![1, 3], vec![(0, 1.0), (2, -1.0)]);

        let result = solve_reduced_problem(
            &problem,
            &SolveOptions {
                rank: Some(2),
                seed: Some(1),
                max_iterations: 10,
                min_stationarity_iterations: 1,
                objective_tolerance: 1e-8,
                stationarity_tolerance: 1e-8,
                rounding_iterations: 4,
                beam_width: Some(0),
                compute_dual_bound: false,
                compute_rounding: true,
                step_rule: StepRule::CoordNoStep,
                verbose: false,
                warm_start: WarmStart::Random,
            },
        );

        assert_eq!(result.lifted_rounded_solution.len(), 4);
        assert_eq!(result.lifted_rounded_solution[0], 1.0);
        assert_eq!(result.lifted_rounded_solution[2], -1.0);
        assert_eq!(
            result.lifted_rounded_solution[1],
            result.reduced_result.rounded_solution.as_ref().unwrap()[0]
        );
        assert_eq!(
            result.lifted_rounded_solution[3],
            result.reduced_result.rounded_solution.as_ref().unwrap()[1]
        );
    }

    #[test]
    fn reduced_problem_lift_preserves_reduced_objective_on_submatrix() {
        let q = test_q();
        let problem = ReducedProblem::from_principal_submatrix(&q, vec![0, 2, 3], vec![(1, 1.0)]);
        let reduced_solution = Array1::from_vec(vec![1.0, -1.0, 1.0]);
        let lifted = problem.lift_solution(&reduced_solution);

        let reduced_obj = obj_rounded(&problem.reduced_q, &reduced_solution);
        let lifted_free = Array1::from_vec(vec![lifted[0], lifted[2], lifted[3]]);
        let lifted_obj = obj_rounded(&problem.reduced_q, &lifted_free);

        assert!((reduced_obj - lifted_obj).abs() < 1e-8);
    }

    #[test]
    fn absorbing_linear_terms_is_exact_on_binary_points() {
        let mut q = TriMat::<f64>::new((3, 3));
        q.add_triplet(0, 0, 1.0);
        q.add_triplet(0, 1, -0.5);
        q.add_triplet(1, 0, -0.5);
        q.add_triplet(1, 2, 0.75);
        q.add_triplet(2, 1, 0.75);
        let q = q.to_csr();
        let linear = array![0.5, -1.25, 0.75];
        let q_eff = absorb_linear_terms_into_hessian(&q, &linear);

        let binary_points = [
            array![0.0, 0.0, 0.0],
            array![1.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
            array![1.0, 1.0, 0.0],
            array![1.0, 0.0, 1.0],
            array![0.0, 1.0, 1.0],
            array![1.0, 1.0, 1.0],
        ];

        for x in binary_points {
            let lhs = 0.5 * x.dot(&(&q * &x)) + linear.dot(&x);
            let rhs = 0.5 * x.dot(&(&q_eff * &x));
            assert!((lhs - rhs).abs() < 1e-8);
        }
    }

    #[test]
    fn reduced_problem_from_qubo_subproblem_absorbs_linear_terms() {
        let mut quadratic = TriMat::<f64>::new((2, 2));
        quadratic.add_triplet(0, 1, -0.5);
        quadratic.add_triplet(1, 0, -0.5);
        let quadratic = quadratic.to_csr();
        let linear = array![1.0, -0.5];

        let problem = ReducedProblem::from_qubo_subproblem(
            &quadratic,
            &linear,
            vec![1, 3],
            vec![(0, 1.0), (2, 0.0)],
        );

        assert_eq!(problem.reduced_q.get(0, 0), Some(&2.0));
        assert_eq!(problem.reduced_q.get(1, 1), Some(&-1.0));
        assert_eq!(problem.reduced_q.get(0, 1), Some(&-0.5));
        assert_eq!(problem.reduced_q.get(1, 0), Some(&-0.5));
        assert_eq!(problem.original_size, 4);
    }

    #[test]
    fn sign_matrix_matches_qubo_objective_on_binary_points() {
        let mut quadratic = TriMat::<f64>::new((2, 2));
        quadratic.add_triplet(0, 0, 1.0);
        quadratic.add_triplet(0, 1, -0.5);
        quadratic.add_triplet(1, 0, -0.5);
        quadratic.add_triplet(1, 1, 0.75);
        let quadratic = quadratic.to_csr();
        let linear = array![-0.25, 0.5];
        let effective_hessian = absorb_linear_terms_into_hessian(&quadratic, &linear);
        let sign_matrix = qubo_hessian_to_sign_matrix(&effective_hessian);

        let binary_points = [
            array![0.0, 0.0],
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
        ];

        for x in binary_points {
            let sign = Array1::from_vec(vec![1.0 - 2.0 * x[0], 1.0 - 2.0 * x[1], 1.0]);
            let qubo_obj = eval_qubo_objective(&quadratic, &linear, &x);
            let sign_obj = obj_rounded(&sign_matrix, &sign);
            assert!((qubo_obj - sign_obj).abs() < 1e-8);
        }
    }

    #[test]
    fn sign_solution_maps_back_to_binary_solution() {
        let sign_solution = Array1::from_vec(vec![-1.0, 1.0, -1.0]);
        let binary = qubo_binary_from_sign_solution(&sign_solution);

        assert_eq!(binary[0], 0.0);
        assert_eq!(binary[1], 1.0);
    }

    #[test]
    fn mixingcut_and_abqp_bounds_are_valid_on_small_qubo() {
        let mut quadratic = TriMat::<f64>::new((3, 3));
        quadratic.add_triplet(0, 0, 1.5);
        quadratic.add_triplet(0, 1, -0.5);
        quadratic.add_triplet(1, 0, -0.5);
        quadratic.add_triplet(1, 1, 2.0);
        quadratic.add_triplet(1, 2, 0.25);
        quadratic.add_triplet(2, 1, 0.25);
        quadratic.add_triplet(2, 2, 1.0);
        let quadratic = quadratic.to_csr();
        let linear = array![-0.75, 0.5, -0.25];

        let mixingcut_result = solve_qubo_sdp_subproblem(
            &quadratic,
            &linear,
            &SolveOptions {
                rank: Some(2),
                seed: Some(5),
                max_iterations: 50,
                min_stationarity_iterations: 1,
                objective_tolerance: 1e-8,
                stationarity_tolerance: 1e-8,
                rounding_iterations: 0,
                beam_width: Some(0),
                compute_dual_bound: false,
                compute_rounding: false,
                step_rule: StepRule::CoordNoStep,
                verbose: false,
                warm_start: WarmStart::Random,
            },
        );
        let abqp_result = solve_abqp_box_relaxation(&quadratic, &linear);
        let exact_optimum = brute_force_qubo_optimum(&quadratic, &linear);

        assert!(abqp_result.objective.is_finite());
        assert!(abqp_result.quality.certified_lower_bound.is_finite());
        assert!(mixingcut_result.qubo_lower_bound.is_finite());
        assert_eq!(mixingcut_result.factor_matrix.nrows(), quadratic.rows() + 1);
        assert_eq!(mixingcut_result.dual_variables.len(), quadratic.rows() + 1);
        assert!(matches!(
            mixingcut_result.status,
            SolveStatus::MaxIterations
                | SolveStatus::ObjectiveTolerance
                | SolveStatus::ObjectiveIncreased
        ));

        assert!(
            abqp_result.quality.certified_lower_bound <= exact_optimum + 1e-8,
            "ABQP lower bound should not beat the exact binary optimum"
        );
        assert!(
            mixingcut_result.qubo_lower_bound <= exact_optimum + 1e-8,
            "MixingCut lower bound should not beat the exact binary optimum"
        );
    }

    #[test]
    fn mixingcut_beats_abqp_on_a_nonconvex_exact_case() {
        let mut quadratic = TriMat::<f64>::new((3, 3));
        quadratic.add_triplet(0, 0, 1.5);
        quadratic.add_triplet(0, 1, -0.5);
        quadratic.add_triplet(1, 0, -0.5);
        quadratic.add_triplet(1, 1, 2.0);
        quadratic.add_triplet(1, 2, 0.25);
        quadratic.add_triplet(2, 1, 0.25);
        quadratic.add_triplet(2, 2, 1.0);
        let quadratic = quadratic.to_csr();
        let linear = array![-0.75, 0.5, -0.25];

        let abqp_result = solve_abqp_box_relaxation(&quadratic, &linear);
        let mixingcut_result = solve_qubo_sdp_subproblem(
            &quadratic,
            &linear,
            &SolveOptions {
                rank: Some(2),
                seed: Some(5),
                max_iterations: 50,
                min_stationarity_iterations: 1,
                objective_tolerance: 1e-8,
                stationarity_tolerance: 1e-8,
                rounding_iterations: 0,
                beam_width: Some(0),
                compute_dual_bound: false,
                compute_rounding: false,
                step_rule: StepRule::CoordNoStep,
                verbose: false,
                warm_start: WarmStart::Random,
            },
        );

        assert!(
            mixingcut_result.qubo_lower_bound >= abqp_result.quality.certified_lower_bound + 0.1,
            "expected MixingCut SDP bound to be meaningfully tighter on this nonconvex case"
        );
    }

    #[test]
    fn qubo_sdp_lower_bound_stays_valid_on_small_deterministic_instances() {
        let cases = [
            (
                3usize,
                vec![
                    (0usize, 0usize, 2.0),
                    (0, 1, -3.0),
                    (1, 0, -3.0),
                    (1, 1, 1.0),
                    (1, 2, 2.5),
                    (2, 1, 2.5),
                    (2, 2, -0.5),
                ],
                vec![-0.5, 1.25, -0.75],
            ),
            (
                3usize,
                vec![
                    (0usize, 0usize, -1.0),
                    (0, 2, 4.0),
                    (2, 0, 4.0),
                    (1, 1, 3.0),
                    (1, 2, -2.0),
                    (2, 1, -2.0),
                    (2, 2, 1.5),
                ],
                vec![0.75, -1.0, 0.5],
            ),
            (
                4usize,
                vec![
                    (0usize, 1usize, -1.5),
                    (1, 0, -1.5),
                    (0, 2, 1.0),
                    (2, 0, 1.0),
                    (1, 2, -2.25),
                    (2, 1, -2.25),
                    (0, 0, 0.5),
                    (1, 1, 0.25),
                    (2, 2, 2.0),
                    (3, 3, -1.0),
                    (2, 3, 1.25),
                    (3, 2, 1.25),
                ],
                vec![0.0, -0.5, 1.0, -0.25],
            ),
            (
                4usize,
                vec![
                    (0usize, 0usize, 3.5),
                    (0, 1, -2.5),
                    (1, 0, -2.5),
                    (1, 1, -1.0),
                    (1, 3, 2.0),
                    (3, 1, 2.0),
                    (2, 2, 1.0),
                    (2, 3, -3.5),
                    (3, 2, -3.5),
                    (3, 3, 2.5),
                ],
                vec![-1.0, 0.75, -0.5, 0.25],
            ),
        ];

        for (n, entries, linear_entries) in cases {
            let mut tri = TriMat::<f64>::new((n, n));
            for (i, j, value) in entries {
                tri.add_triplet(i, j, value);
            }
            let quadratic = tri.to_csr();
            let linear = Array1::from_vec(linear_entries);
            let exact_optimum = brute_force_qubo_optimum(&quadratic, &linear);

            let result = solve_qubo_sdp_subproblem(
                &quadratic,
                &linear,
                &SolveOptions {
                    rank: Some(((2.0 * n as f64).sqrt().ceil() as usize).max(2)),
                    seed: Some(7),
                    max_iterations: 75,
                    min_stationarity_iterations: 1,
                    objective_tolerance: 1e-8,
                    stationarity_tolerance: 1e-8,
                    rounding_iterations: 0,
                    beam_width: Some(0),
                    compute_dual_bound: false,
                    compute_rounding: false,
                    step_rule: StepRule::CoordNoStep,
                    verbose: false,
                    warm_start: WarmStart::Random,
                },
            );

            assert!(
                result.qubo_lower_bound <= exact_optimum + 1e-8,
                "SDP lower bound {} should not beat exact optimum {}",
                result.qubo_lower_bound,
                exact_optimum
            );
        }
    }
}

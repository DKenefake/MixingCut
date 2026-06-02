#![allow(non_snake_case)] // reasoning: The code is based on linear algebra notation (X is a matrix, x is a vector)

use clap::Parser;
use mixingcut::io_operations;
use mixingcut::io_operations::write_solution_matrix;
use mixingcut::maxcut_oracle::get_Q_norm;
use mixingcut::sdp_solver::{solve_maxcut_sdp, SolveOptions, WarmStart};
use mixingcut::step_rules::generate_step_rule;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Name of the input file
    #[clap(short, long)]
    input_path: String,

    // Name of the output file
    #[clap(short, long, default_value = "output.txt")]
    output_path: String,

    #[clap(short, long, default_value = "0")]
    rank: usize,

    // The stopping tolerance
    #[clap(short, long, default_value = "1e-2")]
    tolerance: f64,

    // Number of iterations
    #[clap(short, long, default_value = "1000")]
    max_iters: usize,

    // Step Rule
    #[clap(short, long, default_value = "coord_no_step")]
    step_rule: String,

    // index correction
    #[clap(long, default_value = "1")]
    index_correction: usize,

    // compute dual bound
    #[clap(short, long, default_value = "0")]
    dual_bound: usize,

    // verbosity
    #[clap(short, long, default_value = "1")]
    verbose: usize,

    // rounding iters
    #[clap(long, default_value = "100")]
    rounding_iters: usize,

    // beam search width
    #[clap(long, default_value = "128")]
    beam_width: usize,
}

fn main() {
    let args: Args = Args::parse();

    let index_correction = args.index_correction;

    // read in the graph
    let Q = io_operations::read_graph_matrix(&args.input_path, index_correction);

    let Q_norm = get_Q_norm(&Q);

    // compute the safe step size
    let alpha_safe = 1.0 / Q_norm;

    let n = Q.shape().0;

    let step_rule = generate_step_rule(&args.step_rule, alpha_safe);

    let verbose = args.verbose;

    // print the mixing cut vanity header if verbose
    if verbose == 1 {
        println!("------------------------------------------------------------------");
        println!("               MixingCut v0.1.4 - MAX CUT SDP Solver              ");
        println!("         (c) Dustin Kenefake, Texas A&M University, 2024          ");
        println!("------------------------------------------------------------------");
    }

    // set up the rank size of the problem
    let k = match args.rank {
        0 => 2 * (n as f64).log2() as usize,
        1 => (2.0 * n as f64).sqrt() as usize,
        _ => args.rank,
    };

    // print problem statistics if verbose
    if verbose == 1 {
        println!("Problem Statistics:");
        println!("Size of Q {} {}", n, n);
        println!("NNZ(Q) {}", Q.nnz());
        println!("Q norm {}", Q_norm);
        println!("Rank {}", k);
        println!("------------------------------------------------------------------");
        println!(
            "{0: <20} | {1: <20} | {2: <20}",
            "Iteration", "Primal Value", "Time(sec)"
        );
    }

    let result = solve_maxcut_sdp(
        &Q,
        &SolveOptions {
            rank: Some(k),
            seed: None,
            max_iterations: args.max_iters,
            min_stationarity_iterations: 20,
            objective_tolerance: args.tolerance,
            stationarity_tolerance: 1e-4,
            rounding_iterations: args.rounding_iters,
            beam_width: Some(args.beam_width),
            compute_dual_bound: args.dual_bound == 1,
            compute_rounding: true,
            step_rule,
            verbose: verbose == 1,
            warm_start: WarmStart::Random,
        },
    );

    if verbose == 1 {
        println!("------------------------------------------------------------------")
    }

    if verbose == 1 {
        // print the rounded solution
        if let (Some(rounded_objective), Some(rounded_solution)) =
            (result.rounded_objective, result.rounded_solution.as_ref())
        {
            println!(
                "Rounded solution: {:?} {:?}",
                rounded_objective, rounded_solution
            );
        }

        if let (Some(best_obj), Some(best_sol)) = (
            result.locally_improved_objective,
            result.locally_improved_solution.as_ref(),
        ) {
            println!(
                "Rounded solution with local search: {:?} {:?}",
                best_obj, best_sol
            );
        }
    }

    // print the dual bound
    if let Some(dual_bound) = result.dual_bound {
        if verbose == 1 {
            println!("Dual bound: {:?}", dual_bound);
        }
    }

    // write the solution to a file
    let rounded_solution = result
        .rounded_solution
        .expect("CLI solve should always compute a rounded solution");
    let rounded_objective = result
        .rounded_objective
        .expect("CLI solve should always compute a rounded objective");
    write_solution_matrix(
        &args.output_path,
        rounded_solution,
        rounded_objective,
        result.relaxed_objective,
    );
}

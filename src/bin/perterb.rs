#![allow(non_snake_case)] // reasoning: The code is based on linear algebra notation (X is a matrix, x is a vector)

use clap::Parser;
use mixingcut::io_operations;
use mixingcut::sdp_solver::compute_approx_perturbation;
use std::time::{SystemTime, UNIX_EPOCH};

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

fn current_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

fn main() {
    let args: Args = Args::parse();

    let index_correction = args.index_correction;

    // read in the graph
    let mut Q = io_operations::read_graph_matrix(&args.input_path, index_correction);

    let n = Q.shape().0;

    let max_iters = args.max_iters;

    let verbose = args.verbose;

    let start = current_time();

    let tolerance = args.tolerance;

    // set up the rank size of the problem
    let k = match args.rank {
        0 => 2 * (n as f64).log2() as usize,
        1 => (2.0 * n as f64).sqrt() as usize,
        _ => args.rank,
    };

    let y_sol = compute_approx_perturbation(&Q, Some(k), None, Some(max_iters), Some(tolerance));

    let end = current_time();

    // print the perturbation solution
    if verbose > 0 {
        println!("Perturbation solution: {:?}", y_sol);
        println!("Perturbation solution norm: {}", y_sol.sum());
        println!("Solved in {:.2} seconds", end - start);
    }
}

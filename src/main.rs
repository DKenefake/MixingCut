#![allow(non_snake_case)] // reasoning: The code is based on linear algebra notation (X is a matrix, x is a vector)

mod read_graph;
mod sdp_project;
mod step_rules;
mod maxcut_oracle;
mod initialize;

use std::time::{SystemTime, UNIX_EPOCH};
use clap::Parser;
use crate::initialize::make_random_matrix;
use crate::maxcut_oracle::{compute_rounded_sol, get_Q_norm, obj};
use crate::read_graph::write_solution_matrix;
use crate::step_rules::generate_step_rule;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args{
    // Name of the input file
    #[clap(short, long)]
    input_path: String,

    // Name of the output file
    #[clap(short, long, default_value = "output.txt")]
    output_path: String,

    #[clap(short, long, default_value = "0")]
    rank: usize,

    // The stopping tolerance
    #[clap(short, long, default_value = "1e-6")]
    tolerance: f64,

    // Number of iterations
    #[clap(short, long, default_value = "1000")]
    max_iters: usize,

    // Step Rule
    #[clap(short, long, default_value = "grad")]
    step_rule: String,

}


fn current_time() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros()
}

fn main() {

    // parse the arguments
    let args: Args = Args::parse();

    // read in the graph
    let Q = read_graph::read_graph_matrix(&args.input_path);

    // get the norm of Q
    let Q_norm = get_Q_norm(&Q);

    // compute the safe step size
    let alpha_safe = 1.0 / Q_norm;

    // get the size of the problem
    let n = Q.shape().0;

    // generate the step rule from the argument
    let step_rule = generate_step_rule(&args.step_rule, alpha_safe);

    // read in the number of iterations
    let max_iters = args.max_iters;

    // set up the rank size of the problem
    let k = match args.rank {
        0 => 2 * (n as f64).log2() as usize,
        1 => (2.0 * n as f64).sqrt() as usize,
        _ => args.rank,
    };

    // generate random initial point
    let mut V = make_random_matrix(n, k);

    // print problem statistics
    println!("Size of Q {:?}", Q.shape());
    println!("NNZ(Q) {:?}", Q.nnz());
    println!("Q norm {:?}", Q_norm);
    println!("Size of V {:?}", V.shape());

    // get current time
    let start = current_time();

    // iterate over the number of iterations
    for i in 0..max_iters{

        // apply the step rule
        V = step_rules::apply_step(&Q, V, step_rule);

        // every 100 iterations, print the objective value
        if i % 100 == 0{
            println!("{} {} {}" , i, obj(&Q, &V), current_time() - start);
        }
    }

    // compute the rounded solution
    let (x_0, obj_rounded) = compute_rounded_sol(&Q, &V, 1000);

    // print the rounded solution
    println!("Rounded solution: {:?} {:?}", obj_rounded, x_0);

    // write the solution to a file
    write_solution_matrix(&args.output_path, x_0, obj_rounded);

}

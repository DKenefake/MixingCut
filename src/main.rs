#![allow(non_snake_case)] // reasoning: The code is based on linear algebra notation (X is a matrix, x is a vector)

mod io_operations;
mod sdp_project;
mod step_rules;
mod maxcut_oracle;
mod initialize;

use std::time::{SystemTime, UNIX_EPOCH};
use clap::Parser;
use crate::initialize::make_random_matrix;
use crate::maxcut_oracle::{compute_rounded_sol, get_Q_norm, obj};
use crate::io_operations::write_solution_matrix;
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
    verbose: usize
}


fn current_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn main() {

    // parse the arguments
    let args: Args = Args::parse();

    // find the index correction
    let index_correction = args.index_correction;

    // read in the graph
    let Q = io_operations::read_graph_matrix(&args.input_path, index_correction);

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

    // get verbosity
    let verbose = args.verbose;

    // print the mixing cut vanity header if verbose
    if verbose == 1{
        println!("------------------------------------------------------------------
	     MixingCut v0.0.1 - MAX CUT SDP Solver
	(c) Dustin Kenefake, Texas A&M University, 2024
------------------------------------------------------------------");
    }

    // set up the rank size of the problem
    let k = match args.rank {
        0 => 2 * (n as f64).log2() as usize,
        1 => (2.0 * n as f64).sqrt() as usize,
        _ => args.rank,
    };

    // generate random initial point
    let mut V = make_random_matrix(n, k);

    // print problem statistics if verbose
    if verbose == 1{
        println!("Size of Q {:?}", Q.shape());
        println!("NNZ(Q) {:?}", Q.nnz());
        println!("Q norm {:?}", Q_norm);
        println!("Size of V {:?}", V.shape());
        println!("------------------------------------------------------------------")
    }

    // get current time
    let start = current_time();

    // get the objective value
    let mut obj_val = obj(&Q, &V);

    // iterate over the number of iterations
    for i in 0..max_iters{

        // apply the step rule
        V = step_rules::apply_step(&Q, V, step_rule);

        // compute the objective value
        let new_obj_val = obj(&Q, &V);

        // if the objective value is not changing, break
        if (new_obj_val - obj_val).abs()  < args.tolerance{
            if verbose == 1 {
                println!("{} {} {}", i, obj(&Q, &V), current_time() - start);
            }
            break;
        }

        // if the objective value is increasing, break
        if new_obj_val > obj_val{
            if verbose == 1{
                println!("Objective value is increasing");
            }
            break;
        }

        obj_val = new_obj_val;

        // every 10 iterations, print the objective value
        if verbose == 1 && i % 10 == 0{
            println!("{} {} {}" , i, obj_val, current_time() - start);
        }
    }

    if verbose == 1{
        println!("------------------------------------------------------------------")
    }

    // compute the rounded solution
    let (x_0, obj_rounded) = compute_rounded_sol(&Q, &V, 1000);

    if verbose == 1{
        // print the rounded solution
        println!("Rounded solution: {:?} {:?}", obj_rounded, x_0);
    }

    // print the dual bound
    if args.dual_bound == 1{
        let dual_bound = maxcut_oracle::dual_bound(&Q, &V);
        if verbose == 1{
            println!("Dual bound: {:?}", dual_bound);
        }
    }

    // write the solution to a file
    write_solution_matrix(&args.output_path, x_0, obj_rounded, obj(&Q, &V));

}

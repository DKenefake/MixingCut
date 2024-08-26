mod read_graph;
mod sdp_project;
mod step_rules;
mod maxcut_oracle;
mod initialize;

use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::{Array1, Array2};
use ndarray::linalg::Dot;
use ndarray_linalg::{Norm, Trace};
use smolprng::PRNG;
use sprs::{CsMat};
use smolprng::Algorithm;
use smolprng::*;
use clap::Parser;
use crate::initialize::make_random_matrix;
use crate::maxcut_oracle::{compute_rounded_sol, get_Q_norm, obj};
use crate::step_rules::make_step_coord_no_step;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args{
    // Name of the filepath
    #[clap(short, long)]
    path: String,

    // Number of iterations
    #[clap(short, long)]
    iters: usize,

    #[clap(short, long)]
    rank: usize,
}


fn current_time() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros()
}

fn main() {

    let args: Args = Args::parse();

    let Q = read_graph::read_graph_matrix(&args.path);
    let Q_norm = get_Q_norm(&Q);
    let n = Q.shape().0;

    let max_iters = args.iters;


    let k = match args.rank {
        0 => 2 * (n as f64).log2() as usize,
        1 => (2.0 * n as f64).sqrt() as usize,
        _ => args.rank,
    };

    let mut V = make_random_matrix(n, k);

    println!("Size of Q {:?}", Q.shape());
    println!("NNZ(Q) {:?}", Q.nnz());
    println!("Q norm {:?}", Q_norm);
    println!("Size of V {:?}", V.shape());
    let alpha_safe = 1.0 / Q_norm;

    let start = current_time();

    for i in 0..max_iters{

        V = make_step_coord_no_step(&Q, V, alpha_safe);

        if i % 100 == 0{
            println!("{} {} {}" , i, obj(&Q, &V), current_time() - start);
        }
    }

    let (x_0, obj_rounded) = compute_rounded_sol(&Q, &V, 1000, 0);

    println!("Rounded solution: {:?} {:?}", obj_rounded, x_0);

}

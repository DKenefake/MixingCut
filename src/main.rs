mod read_graph;
mod sdp_project;

use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::{Array1, Array2};
use ndarray::linalg::Dot;
use ndarray_linalg::{Norm, Trace};
use smolprng::PRNG;
use sprs::{CsMat};
use smolprng::Algorithm;
use smolprng::*;
use clap::Parser;

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

fn obj(Q: &CsMat<f64>, V:&Array2<f64>) -> f64{
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        if i == j{
            trace += q_ij * V.row(i).dot(&V.row(j));
        }
        else{
            trace += 2.0 * q_ij * V.row(i).dot(&V.row(j));
        }
    }

    trace
}

fn obj_rounded(Q: &CsMat<f64>, x_0: &Array1<f64>) -> f64{
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        if i == j{
            trace += q_ij * x_0[i] * x_0[i];
        }
        else{
            trace += 2.0 * q_ij * x_0[i] * x_0[j];
        }
    }

    trace
}

fn grad(Q: &CsMat<f64>, V:&Array2<f64>) -> Array2<f64>{
    2.0 * (Q * V)
}

fn make_step(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{
    let grad = grad(&Q, &V);
    sdp_project::project(V - alpha_safe * grad)
}

fn make_step_adv(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{
    let grad = grad(&Q, &V);

    let f_0 = obj(&Q, &V);
    let x = obj(&Q, &(&V + alpha_safe * &grad)) - f_0;
    let y = obj(&Q, &(&V - alpha_safe * &grad)) - f_0;

    let mut alpha = (0.5*(y - x)* alpha_safe)/(x + y);

    let proposed_step_val = obj(&Q, &(&V - alpha * &grad));

    if proposed_step_val > f_0{
        alpha = alpha_safe;
    }

    sdp_project::project(V - alpha * grad)
}

fn make_step_coord(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{

    for i in 0..Q.shape().0{

        let Q_i= Q.outer_view(i).unwrap();

        let mut g_i = Array1::<f64>::zeros(V.shape()[1]);

        // compute g_i
        for (k, &v) in Q_i.iter() {
            g_i = g_i + v * &V.row(k);
        }

        // normalize g_i
        g_i = &V.row(i) - alpha_safe * g_i;
        g_i /= g_i.norm_l2();

        V.row_mut(i).assign(&g_i);

    }

    V
}

fn make_step_coord_no_step(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{

    for i in 0..Q.shape().0{

        let Q_i= Q.outer_view(i).unwrap();

        let mut g_i = Array1::<f64>::zeros(V.shape()[1]);

        // compute g_i
        for (k, &v) in Q_i.iter() {
            g_i = g_i - v * &V.row(k);
        }

        // normalize g_i
        g_i /= g_i.norm_l2();

        V.row_mut(i).assign(&g_i);

    }

    V
}

fn make_random_matrix(n:usize, k:usize) -> Array2<f64>{
    let mut V = Array2::zeros((n, k));

    let mut prng = PRNG {
        generator: JsfLarge::default(),
    };

    for i in 0..n{
        for j in 0..k{
            V[[i, j]] = prng.normal();
        }
    }

    sdp_project::project(V)
}

fn compute_rounded_sol(Q: &CsMat<f64>, V: &Array2<f64>, iters: usize, seed: usize) -> (Array1<f64>, f64){

    let mut prng = PRNG {
        generator: JsfLarge::default(),
    };


    let mut best_sol = Array1::zeros(V.shape()[0]);
    let mut best_obj = f64::MAX;
    let mut x_scratch = Array1::zeros(V.shape()[0]);
    let mut r_scratch = Array1::zeros(V.shape()[1]);

    for _ in 0..iters{

        r_scratch.mapv_inplace(|x| prng.normal());
        x_scratch.assign(&V.dot(&r_scratch));
        x_scratch.mapv_inplace(|x| if x > 0.0 {1.0} else {-1.0});

        let obj_rounded = obj_rounded(&Q, &x_scratch);

        if obj_rounded < best_obj{
            best_obj = obj_rounded;
            best_sol.assign(&x_scratch);
        }
    }

    (best_sol, best_obj)
}

fn get_Q_norm(Q: &CsMat<f64>) -> f64 {
    // compute the l1 norm of Q
    let mut c = Array1::<f64>::zeros(Q.shape().0);
    for (q_ij, (i, j)) in Q.iter() {
        c[i] += q_ij.abs();
    }
    c.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap().clone()
}

fn main() {

    let args: Args = Args::parse();

    let Q = read_graph::read_graph_matrix(&args.path);
    let max_iters = args.iters;


    let k = match args.rank {
        0 => 2 * (Q.shape().0 as f64).log2() as usize,
        1 => 2 * (Q.shape().0 as f64).sqrt() as usize,
        _ => args.rank,
    };

    let Q_norm = get_Q_norm(&Q);

    let n = Q.shape().0;


    let mut V = make_random_matrix(n, k);

    println!("Size of Q {:?}", Q.shape());
    println!("NNZ(Q) {:?}", Q.nnz());
    println!("Q norm {:?}", Q_norm);
    println!("Size of V {:?}", V.shape());
    let alpha_safe = 2.0 / Q_norm;

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

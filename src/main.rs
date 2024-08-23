mod read_graph;
mod sdp_project;

use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use ndarray::linalg::Dot;
use ndarray_linalg::{Norm, Trace};
use smolprng::PRNG;
use sprs::{CsMat};
use smolprng::Algorithm;
use smolprng::*;

fn current_time() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros()
}

fn obj(Q: &CsMat<f64>, V:&Array2<f64>) -> f64{
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        trace += q_ij * V.row(i).dot(&V.row(j));
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

fn get_Q_norm(Q: &CsMat<f64>) -> f64 {
    // compute the l1 norm of Q
    let mut c = Array1::<f64>::zeros(Q.shape().0);
    for (q_ij, (i, j)) in Q.iter() {
        c[i] += q_ij.abs();
    }
    c.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap().clone()
}

fn main() {

    let Q = read_graph::read_graph_matrix("graphs/G_35.graph");

    let Q_norm = get_Q_norm(&Q);

    let n = Q.shape().0;
    // let k = (2.0*n as f64 + 1.0).sqrt() as usize;
    let k = 5;

    let mut V = make_random_matrix(n, k);

    println!("Size of Q {:?}", Q.shape());
    println!("NNZ(Q) {:?}", Q.nnz());
    println!("Q norm {:?}", Q_norm);
    println!("Size of V {:?}", V.shape());
    let alpha_safe = 2.0 / Q_norm;

    let start = current_time();

    for i in 0..10000{

        V = make_step_coord_no_step(&Q, V, alpha_safe);

        if i % 100 == 0{
            println!("{} {} {}" , i, obj(&Q, &V), current_time() - start);
        }
    }


}

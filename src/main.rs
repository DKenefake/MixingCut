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

fn make_step(Q: &CsMat<f64>, V: &Array2<f64>, alpha_safe: f64 ) -> (Array2<f64>, f64, f64){
    let grad = grad(&Q, &V);
    let V_new =  sdp_project::project(V - alpha_safe * grad);
    let obj_new = obj(&Q, &V_new);
    (V_new, obj_new,alpha_safe)
}

fn make_step_adv(Q: &CsMat<f64>, V: &Array2<f64>, alpha_safe: f64 ) -> (Array2<f64>, f64, f64){
    let grad = grad(&Q, &V);

    // let V_abs = V.mapv(|x| x.abs());

    let f_0 = obj(&Q, V);
    let x = obj(&Q, &(V + alpha_safe * &grad)) - f_0;
    let y = obj(&Q, &(V - alpha_safe * &grad)) - f_0;

    let mut alpha = (0.5*(y - x)* alpha_safe)/(x + y);

    if alpha < 0.0{
        alpha = -alpha;
    }

    let V_new =  sdp_project::project(V - alpha * grad);
    let obj_new = obj(&Q, &V_new);
    (V_new, obj_new, alpha)
}

fn make_step_coord(Q: &CsMat<f64>, V: &Array2<f64>, alpha_safe: f64 ) -> (Array2<f64>, f64, f64){

    let mut V_new = V.clone();

    for i in 0..Q.shape().0{

        let Q_i= Q.outer_view(i).unwrap();

        // compute g_i
        let mut g_i = Q_i.iter().map(|(k, &v)| -v * &V.row(k)).fold( Array1::<f64>::zeros(V.shape()[1]), |acc, x| acc + x);

        // normalize g_i
        // let norm = g_i.dot(&g_i).sqrt();
        g_i /= g_i.norm_l2();

        V_new.row_mut(i).assign(&g_i);

    }

    let new_obj = obj(Q, &V_new);

    (V_new, new_obj, alpha_safe)
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

    V
}

fn get_Q_norm(Q: &CsMat<f64>) -> f64 {
    // compute the l1 norm of Q
    let mut c = Array1::<f64>::zeros(Q.shape().0);
    for (q_ij, (i, j)) in Q.iter() {
        c[i] += q_ij.abs();
        c[j] += q_ij.abs();
    }
    c.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap().clone()
}

fn main() {

    let Q = read_graph::read_graph_matrix("graphs/G_35.graph");

    let Q_norm = get_Q_norm(&Q);

    let n = Q.shape().0;
    let k = (2.0*n as f64 + 1.0).sqrt() as usize;
    // let k = 5;

    let mut V = make_random_matrix(n, k);

    println!("Size of Q {:?}", Q.shape());
    println!("NNZ(Q) {:?}", Q.nnz());
    println!("Q norm {:?}", Q_norm);
    println!("Size of V {:?}", V.shape());

    for i in 0..1000000{
        let alpha_safe = 1.0 / Q_norm;

        let (V_new, obj_new, step_size) = make_step(&Q, &V, alpha_safe);

        V = V_new;
        if i % 1000 == 0{
            println!("{} {}" , i, obj_new);
        }
    }


}


use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::Array2;
use rayon::prelude::*;
use ndarray::linalg::Dot;
use ndarray_linalg::Trace;
use sprs::{CsMat, TriMat};

fn current_time() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros()
}

fn make_sparse_hess_matrix(n: usize) -> CsMat<f64>{

    let mut temp = TriMat::new((n, n));

    for i in 0..n {
        temp.add_triplet(i, i, 1.0);
    }

    temp.to_csr()
}


fn make_dense_hess_matrix(n: usize) -> Array2<f64>{
    Array2::eye(n)
}

fn make_dense_matrix(n:usize, k:usize) -> Array2<f64>{
    Array2::ones((n, k))
}

fn trial_1(Q: &CsMat<f64>, V:&Array2<f64>) -> (f64, u128){
    let start = current_time();
    let trace = Q.dot(&V.dot(&V.t())).trace().unwrap();
    let end = current_time();
    (trace, end - start)
}

fn trial_2(Q: &Array2<f64>, V:&Array2<f64>) -> (f64, u128){
    let start = current_time();
    let trace = Q.dot(&V.dot(&V.t())).trace().unwrap();
    let end = current_time();
    (trace, end - start)
}

fn trial_3(Q: &CsMat<f64>, V:&Array2<f64>) -> (f64, u128){
    let start = current_time();
    let mut trace = 0.0;

    for (q_ij, (i, j)) in Q.iter() {
        trace += q_ij * V.row(i).dot(&V.row(j));
    }

    let end = current_time();
    (trace, end - start)
}

fn trial_3_par(Q: &CsMat<f64>, V:&Array2<f64>) -> (f64, u128){

    let start = current_time();

    let trace = Q.iter().par_bridge().map(|(&q_ij, (i, j))|
        q_ij * &V.row(i).dot(&V.row(j))).sum::<f64>();

    let end = current_time();
    (trace, end - start)
}

fn main() {


    let n = 25600;
    let k = (n as f64).sqrt() as usize;

    let Q_s = make_sparse_hess_matrix(n);
    // let Q_d = make_dense_hess_matrix(n);
    let V = make_dense_matrix(n, k);

    // let t1 = trial_1(&Q_s, &V);
    // let t2 = trial_2(&Q_d, &V);
    let t3 = trial_3_par(&Q_s, &V);

    // println!("Trial 1 Q_s(V*V'): {} {}", t1.0, t1.1);
    // println!("Trial 2 Q_d(V*V'): {} {}", t2.0, t2.1);
    println!("Trial 3 Sum(Q_ij (V*V')_ij): {} {}", t3.0, t3.1);
}

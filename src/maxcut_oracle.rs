use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use smolprng::{JsfLarge, PRNG};
use sprs::CsMat;

pub fn get_Q_norm(Q: &CsMat<f64>) -> f64 {
    // compute the l1 norm of Q
    let mut c = Array1::<f64>::zeros(Q.shape().0);
    for (q_ij, (i, j)) in Q.iter() {
        c[i] += q_ij.abs();
    }
    c.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap().clone()
}

pub fn obj(Q: &CsMat<f64>, V:&Array2<f64>) -> f64{
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

pub fn obj_rounded(Q: &CsMat<f64>, x_0: &Array1<f64>) -> f64{
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

pub fn grad(Q: &CsMat<f64>, V:&Array2<f64>) -> Array2<f64>{
    2.0 * (Q * V)
}

pub(crate) fn compute_rounded_sol(Q: &CsMat<f64>, V: &Array2<f64>, iters: usize, seed: usize) -> (Array1<f64>, f64){

    let mut prng = PRNG {
        generator: JsfLarge::default(),
    };


    let mut best_sol = Array1::zeros(V.shape()[0]);
    let mut best_obj = f64::MAX;
    let mut x_scratch = Array1::zeros(V.shape()[0]);
    let mut r_scratch = Array1::zeros(V.shape()[1]);

    for _ in 0..iters{

        r_scratch.mapv_inplace(|x| prng.normal());
        r_scratch /= r_scratch.norm_l2();

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

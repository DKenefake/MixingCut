use ndarray::Array2;
use smolprng::{JsfLarge, PRNG};
use crate::sdp_project;

pub fn make_random_matrix(n:usize, k:usize) -> Array2<f64>{
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


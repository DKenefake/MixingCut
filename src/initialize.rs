use crate::sdp_project;
use ndarray::Array2;
use smolprng::{JsfLarge, PRNG};

pub fn make_random_matrix(n: usize, k: usize, seed: Option<u64>) -> Array2<f64> {
    // generate a zeroed matrix V of size n x k
    let mut V = Array2::zeros((n, k));

    // instantiate a PRNG
    let mut prng = seed.map_or_else(|| PRNG {
        generator: JsfLarge::default(),
    }, |real| PRNG {
        generator: JsfLarge::from(real),
    });

    // fill V with random values
    for i in 0..n {
        for j in 0..k {
            V[[i, j]] = prng.normal();
        }
    }

    // project V into the feasible space and return it
    sdp_project::project(V)
}

use ndarray::{Array1};
use sprs::{CsMat};
use crate::maxcut_oracle;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;


pub fn get_improving_children(x_0: &Array1<f64>, Q: &CsMat<f64>) -> Vec<(f64, Array1<f64>)> {

    // returns only the improving children of the current rounded solution

    let mut fused_pair = Vec::new();

    let d = -2.0*x_0;

    let mut obj_delta = Q*x_0 ;
    let rhs = obj_delta.clone();
    let original_obj =x_0.dot(&rhs);

    for i in 0..Q.rows(){
        obj_delta[i] = 2.0*d[i]*obj_delta[i] + d[i]*d[i]*Q.get(i, i).unwrap_or(&0.0);
    }

    for i in 0..Q.rows(){
        if obj_delta[i] < -1E-5{
            let mut x = x_0.clone();
            x[i] = -x[i];
            fused_pair.push((obj_delta[i] + original_obj, x));
        }
    }

    fused_pair
}

pub fn beam_search(Q: &CsMat<f64>, beta: usize, candidates: Vec<Array1<f64>>) -> (f64, Array1<f64>) {
    // modified beam search algorithm we are basically doing a more greedy version of beam search via
    // hill climbing

    let mut best_solution: Array1<f64> = Array1::<f64>::zeros(Q.shape().0);
    let mut best_obj = f64::INFINITY;

    for x in candidates.iter(){
        let obj = maxcut_oracle::obj_rounded(Q, x);
        if obj < best_obj{
            best_obj = obj;
            best_solution = x.clone();
        }
    }

    let mut beam_candidates: Vec<_> = candidates.iter().flat_map(|x| get_improving_children(&x, Q)).collect();

    while beam_candidates.len() != 0 {

        for (obj, sol) in beam_candidates.iter(){
            if obj < &best_obj{
                best_obj = *obj;
                best_solution = sol.clone();
            }
        }

        beam_candidates = beam_candidates.par_iter().flat_map(|(_, x)| get_improving_children(x, Q)).collect();

        println!("Beam search: {:?}", beam_candidates.len());

        // sort the beam candidates
        beam_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // select the best beta candidates
        beam_candidates = beam_candidates.iter().take(beta).map(|x| x.clone()).collect();

        println!("Beam search obj: {:?}", best_obj);
    }

    (best_obj, best_solution)
}
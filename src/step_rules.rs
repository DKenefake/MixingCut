use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use smolprng::{JsfLarge, PRNG};
use sprs::CsMat;
use crate::{obj, sdp_project};
use crate::maxcut_oracle::{grad, obj_rounded};

pub fn make_step(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{
    let grad = grad(&Q, &V);
    sdp_project::project(V - alpha_safe * grad)
}

pub fn make_step_adv(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{
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

pub fn make_step_coord(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{

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

pub fn make_step_coord_no_step(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{

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



use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use sprs::CsMat;
use crate::{obj, sdp_project};
use crate::maxcut_oracle::grad;

#[derive(Clone, Copy)]
pub enum StepRule{
    Grad(f64),
    GradAdv(f64),
    Coord(f64),
    CoordNoStep
}

pub fn generate_step_rule(step_rule: &str, alpha: f64) -> StepRule{
    match step_rule{
        "grad" => StepRule::Grad(alpha),
        "grad_adv" => StepRule::GradAdv(alpha),
        "coord" => StepRule::Coord(alpha),
        "coord_no_step" => StepRule::CoordNoStep,
        _ => StepRule::Grad(alpha)
    }
}

pub fn apply_step(Q: &CsMat<f64>, V: Array2<f64>, step_rule: StepRule) -> Array2<f64>{
    match step_rule{
        StepRule::Grad(alpha) => make_step(Q, V, alpha),
        StepRule::GradAdv(alpha) => make_step_adv(Q, V, alpha),
        StepRule::Coord(alpha) => make_step_coord(Q, V, alpha),
        StepRule::CoordNoStep => make_step_coord_no_step(Q, V),
    }
}


pub fn make_step(Q: &CsMat<f64>, V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{
    // compute gradient
    let grad = grad(&Q, &V);
    // take gradient step and project
    sdp_project::project(V - alpha_safe * grad)
}

pub fn make_step_adv(Q: &CsMat<f64>, V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{
    // compute gradient
    let grad = grad(&Q, &V);

    // take the objective value at f(alpha = 0), f(alpha = alpha_safe), f(alpha = -alpha_safe)
    let f_0 = obj(&Q, &V);
    let x = obj(&Q, &(&V + alpha_safe * &grad)) - f_0;
    let y = obj(&Q, &(&V - alpha_safe * &grad)) - f_0;

    // compute the step size based on the quadratic approximation
    let mut alpha = (0.5*(y - x)* alpha_safe)/(x + y);

    // take a step
    let proposed_step_val = obj(&Q, &(&V - alpha * &grad));

    // if the step is not a descent, take the safe step size
    if proposed_step_val > f_0{
        alpha = alpha_safe;
    }

    // take the step and project
    sdp_project::project(V - alpha * grad)
}

pub fn make_step_coord(Q: &CsMat<f64>, mut V: Array2<f64>, alpha_safe: f64 ) -> Array2<f64>{

    // apply coordinate descent with a step size
    for i in 0..Q.shape().0{

        // take a view of the i-th row of Q
        let Q_i= Q.outer_view(i).unwrap();

        // make a scratch space for the gradient
        let mut g_i = Array1::<f64>::zeros(V.shape()[1]);

        // compute g_i
        for (k, &v) in Q_i.iter() {
            g_i = g_i + v * &V.row(k);
        }

        // normalize g_i
        g_i = &V.row(i) - alpha_safe * g_i;
        g_i /= g_i.norm_l2();

        // update the i-th row of V
        V.row_mut(i).assign(&g_i);

    }

    V
}

pub fn make_step_coord_no_step(Q: &CsMat<f64>, mut V: Array2<f64>) -> Array2<f64>{

    // apply coordinate descent without a step size
    for i in 0..Q.shape().0{

        // take a view of the i-th row of Q
        let Q_i= Q.outer_view(i).unwrap();

        // make a scratch space for the gradient
        let mut g_i = Array1::<f64>::zeros(V.shape()[1]);

        // compute g_i
        for (k, &v) in Q_i.iter() {
            g_i = g_i - v * &V.row(k);
        }

        if g_i.norm_l2() == 0.0{
            continue;
        }

        // normalize g_i
        g_i /= g_i.norm_l2();

        // update the i-th row of V
        V.row_mut(i).assign(&g_i);

    }

    V
}

#[cfg(test)]
mod tests{
    #[test]
    fn is_true(){
        // simple test to get things working
        assert_eq!(1, 1);
    }

}

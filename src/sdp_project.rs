use ndarray::Array2;
use ndarray_linalg::norm::normalize;
use ndarray_linalg::NormalizeAxis;

pub(crate) fn project(V: Array2<f64>) -> Array2<f64> {
    // project V into the feasible space
    normalize(V, NormalizeAxis::Row).0
}
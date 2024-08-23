use ndarray::Array2;

pub(crate) fn project(mut V: Array2<f64>) -> Array2<f64> {
    // normallize all rows to have unit norm
    for i in 0..V.shape()[0] {
        let norm = V.row(i).dot(&V.row(i)).sqrt();
        V.row_mut(i).mapv_inplace(|x| x / norm);
    }
    V
}
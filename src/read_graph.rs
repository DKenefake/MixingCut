use std::io::BufRead;
use std::io::Write;

use ndarray::Array1;
use sprs::{CsMat, TriMat};

pub(crate) fn read_graph_matrix(path: &str, index_correction: usize) -> CsMat<f64>{

    // open the file and create a reader
    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    // get the number of variables
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let num_x = line.trim().parse::<usize>().unwrap();

    // reset the line
    line = String::new();

    // set up the sparse matrix and dense vector
    let mut q = TriMat::<f64>::new((num_x, num_x));

    // read the file
    while reader.read_line(&mut line).unwrap() > 0 {
        let row_data: Vec<_> = line.split_whitespace().collect();

        let (i, j , value) = match row_data.len(){
            3 => (row_data[0].parse::<usize>().unwrap() - index_correction, row_data[1].parse::<usize>().unwrap() - index_correction, row_data[2].parse::<f64>().unwrap()),
            2 => (row_data[0].parse::<usize>().unwrap() - index_correction, row_data[1].parse::<usize>().unwrap() - index_correction, 1.0),
            _ => (0, 0, 0.0)
        };

        if i == j{
            q.add_triplet(i, j, value);
        }
        else{
            q.add_triplet(i, j, 0.5 * value);
            q.add_triplet(j, i, 0.5 * value);
        }

        // reset the line
        line = String::new();
    }
    // convert from triplet to crs format
    q.to_csr()
}

pub(crate) fn write_solution_matrix(path: &str, x_0: Array1<f64>, obj: f64) -> () {

    // open the file and create a writer
    let file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    // write the objective value
    writeln!(writer, "{}", obj).unwrap();

    // write the solution vector
    for (_, &x) in x_0.iter().enumerate(){
        writeln!(writer, "{}", x).unwrap();
    }

}
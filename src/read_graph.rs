use std::io::BufRead;
use sprs::{CsMat, TriMat};

pub(crate) fn read_graph_matrix(path: &str) -> CsMat<f64>{

    let file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    // get the number of variables
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let num_x = line.trim().parse::<usize>().unwrap();

    line = String::new();
    // set up the sparse matrix and dense vector
    let mut q = TriMat::<f64>::new((num_x, num_x));

    // read the file
    while reader.read_line(&mut line).unwrap() > 0 {
        let row_data: Vec<_> = line.split_whitespace().collect();

        if row_data.len() == 3 {
            let i = row_data[0].parse::<usize>().unwrap() - 1;
            let j = row_data[1].parse::<usize>().unwrap() - 1;
            let value = row_data[2].parse::<f64>().unwrap();
            q.add_triplet(i, j, value);
            q.add_triplet(j, i, value);
        }

        // reset the line
        line = String::new();
    }

    q.to_csc()
}
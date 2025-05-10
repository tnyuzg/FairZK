use algebra::{DenseMultilinearExtension, Field};
use algebra::Goldilocks;
use std::fs::File;
use std::io::{BufRead, BufReader};
type F = Goldilocks;

use std::rc::Rc;

pub fn read_matrix(file_name: &str) -> (usize, usize, Vec<Vec<f64>>) {
    let file = match File::open(file_name) {
        Ok(file) => file,
        Err(why) => panic!("couldn't open file: {}", why),
    };
    let mut reader = BufReader::new(file);
    // first line is row number and column number
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut iter = line.split_whitespace();
    let row_num: usize = iter.next().unwrap().parse().unwrap();
    let col_num: usize = iter.next().unwrap().parse().unwrap();
    println!("row_num: {}, col_num: {}", row_num, col_num);
    line.clear();

    let mut matrix: Vec<Vec<f64>> = Vec::new();
    // next col_num lines are the matrix
    for _ in 0..col_num {
        reader.read_line(&mut line).unwrap();
        let col_: Vec<f64> = line
            .split(' ')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        matrix.push(col_);
        line.clear();
    }
    (row_num, col_num, matrix)
}

pub fn read_vector(file_name: &str) -> (usize, Vec<f64>) {
    let file = match File::open(file_name) {
        Ok(file) => file,
        Err(why) => panic!("couldn't open file: {}", why),
    };
    let mut reader = BufReader::new(file);
    // first line is length of the vector
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let len: usize = line.trim().parse().unwrap();
    line.clear();
    // next line is the vector
    reader.read_line(&mut line).unwrap();
    let vector: Vec<f64> = line
        .split(' ')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    (len, vector)
}

pub fn read_number(file_name: &str) -> f64 {
    let file = match File::open(file_name) {
        Ok(file) => file,
        Err(why) => panic!("couldn't open file: {}", why),
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let number: f64 = line.trim().parse().unwrap();
    number
}

// input matrix in form of cols
pub fn pad_matrix_to<T: Clone + Default>(
    row_num_padded: usize,
    col_num_padded: usize,
    input: Vec<Vec<T>>,
    element: T,
) -> Vec<Vec<T>> {
    let mut padded: Vec<Vec<T>> = input
        .iter()
        .map(|col| {
            let mut col_padded = col.clone();
            col_padded.resize(row_num_padded, element.clone()); // pad columns to row_num_padded length
            col_padded
        })
        .collect();
    // w_padded.len is the original col_num
    if padded.len() < col_num_padded {
        let pad_col_num = col_num_padded - padded.len();
        for _ in 0..pad_col_num {
            // pad with 0
            padded.push(vec![element.clone(); row_num_padded]);
        }
    }
    padded
}

pub fn pad_vector_to<T: Clone + Default>(len_padded: usize, input: Vec<T>, element: T) -> Vec<T> {
    let mut padded = input.clone();
    padded.resize(len_padded, element.clone());
    padded
}

pub fn quantize_from_f64(x: f64, frac_bits: usize, int_bits: usize) -> u64 {
    assert!(x.abs() < 2_f64.powi(int_bits as i32));
    let p = F::MODULUS_VALUE;
    let rounded = (x.abs() * 2_f64.powi(frac_bits as i32)).round() as u64;
    if x < 0.0 {
        (p - rounded) as u64
    } else {
        rounded
    }
}

pub struct MatrixInfo {
    pub row_num: usize,
    pub col_num: usize,
    pub frac_bits: usize,
    pub int_bits: usize,
    pub matrix: Vec<Vec<f64>>,
    pub padded_row_num: usize,
    pub padded_col_num: usize,
    pub padded_matrix: Vec<Vec<u64>>,
    pub mle: Rc<DenseMultilinearExtension<F>>,
}

pub struct VectorInfo {
    pub len: usize,
    pub frac_bits: usize,
    pub int_bits: usize,
    pub vector: Vec<f64>,
    pub padded_len: usize,
    pub padded_vector: Vec<u64>,
    pub mle: Rc<DenseMultilinearExtension<F>>,
}

pub fn gen_matrix_from_params(
    file_name: &str,
    frac_bits: usize,
    int_bits: usize,
    pad_with: u64,
) -> MatrixInfo {
    // read matrix
    let (row_num, col_num, matrix) = read_matrix(file_name);
    // println!("row_num: {}, col_num: {}", row_num, col_num);
    // println!("{:?}", matrix);

    // quantize matrix
    let matrix_quantized = matrix
        .iter()
        .map(|col| {
            col.iter()
                .map(|x| quantize_from_f64(*x, frac_bits, int_bits))
                .collect()
        })
        .collect();

    // pad matrix
    let padded_row_num = row_num.next_power_of_two();
    let padded_col_num = col_num.next_power_of_two();
    let matrix_padded = pad_matrix_to(padded_row_num, padded_col_num, matrix_quantized, pad_with);

    // construct multilinear extension
    let mltext = Rc::new(DenseMultilinearExtension::from_named_vec(
        &file_name.to_string(),
        (padded_col_num.ilog2() as usize) + (padded_row_num.ilog2() as usize),
        matrix_padded.iter().flatten().map(|x| F::new(*x)).collect(),
    ));

    MatrixInfo {
        row_num,
        col_num,
        frac_bits,
        int_bits,
        matrix,
        padded_row_num,
        padded_col_num,
        padded_matrix: matrix_padded,
        mle: mltext,
    }
}

pub fn gen_vector_from_params(
    file_name: &str,
    frac_bits: usize,
    int_bits: usize,
    pad_with: u64,
) -> VectorInfo {
    // read vector
    let (len, vector) = read_vector(file_name);
    // println!("len: {}", len);
    // println!("{:?}", vector);

    // quantize vector
    let vector_quantized = vector
        .iter()
        .map(|x| quantize_from_f64(*x, frac_bits, int_bits))
        .collect();

    // pad vector
    let padded_len = len.next_power_of_two();
    let vector_padded = pad_vector_to(padded_len, vector_quantized, pad_with);

    // construct multilinear extension
    let mle = Rc::new(DenseMultilinearExtension::from_named_vec(
        &file_name.to_string(),
        padded_len.ilog2() as usize,
        vector_padded.iter().map(|x| F::new(*x)).collect(),
    ));

    VectorInfo {
        len,
        frac_bits,
        int_bits,
        vector,
        padded_len,
        padded_vector: vector_padded,
        mle,
    }
}

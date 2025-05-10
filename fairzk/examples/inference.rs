use algebra::{
    utils::Transcript, DenseMultilinearExtension,  Goldilocks, GoldilocksExtension,
};
use fairzk::read_params::gen_matrix_from_params;
use fairzk::utils::{i128_to_field, random_absolute_u64_vec, u64_to_field, vec_to_named_poly};
use fairzk::{pcs::OracleTable, piop::InferenceInstance, snark::InferenceSnark};
use nalgebra::DMatrix;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::prelude::*;
use sha2::Sha256;
use std::rc::Rc;
use std::vec;

type F = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 64;

//cargo run --example inference --release
fn main() {
   
    let i_bit = 5;
    let f_bit: usize = 5;

    // generate random instance
    let mut _instance = random_instance(i_bit, f_bit);
    // generate real instance
    let mut instance = real_instance(i_bit, f_bit);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let mut oracle_table = OracleTable::new(&code_spec);
    let mut snark: InferenceSnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> = Default::default();

    let mut trans_p = Transcript::new();

    snark.prove_timing(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();

    let decision = snark.verify_timing(&oracle_table, &mut trans_v);

    if decision {
        println!("inference success");
    } else {
        println!("inference fail");
    }
}

fn random_instance(i_bit: usize, f_bit: usize) -> InferenceInstance<F> {
    let num_layers = 1;
    let num_vars_data = 5;
    let num_vars_feats = vec![5, 0];


    let mut w = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        w.push(random_w(
            &format!("w{}", i),
            num_vars_feats[i],
            num_vars_feats[i + 1],
            1,
            f_bit,
        ));
    }

    let data = vec_to_named_poly(
        &"data".to_string(),
        &u64_to_field::<F>(&random_absolute_u64_vec::<F>(
            1 << (num_vars_data + num_vars_feats[0]),
            1 << 3,
        )),
    );

    InferenceInstance::new_essential(
        num_layers,
        num_vars_data,
        &num_vars_feats,
        i_bit,
        f_bit,
        &data,
        &w,
    )
}

fn real_instance(i_bit: usize, f_bit: usize) -> InferenceInstance<F> {
    let path = "models/quantized";

    let num_layers = 2;
    let num_vars_data = 13;
    let num_vars_feats = vec![6, 7, 0];
    let data_name = path.to_owned() + "/german_data_oath.txt";
    let w0_name = path.to_owned() + "/german_layer0_W.txt";
    let w1_name = path.to_owned() + "/german_layer1_W.txt";
    let w0 = gen_matrix_from_params(&w0_name, f_bit, i_bit, 0).mle;
    let w1 = gen_matrix_from_params(&w1_name, f_bit, i_bit, 0).mle;
    let w = vec![w0, w1];

    let data_info = gen_matrix_from_params(&data_name, f_bit, i_bit, 0);
    let data = data_info.mle;
    let data = vec_to_named_poly(
        &"one data".to_string(),
        &data
            .iter()
            .take(1 << (num_vars_data + num_vars_feats[0]))
            .cloned()
            .collect::<Vec<_>>(),
    );

    InferenceInstance::new_essential(
        num_layers,
        num_vars_data,
        &num_vars_feats,
        i_bit,
        f_bit,
        &data,
        &w,
    )
        
}

fn random_w(
    name: &String,
    num_vars_row: usize,
    num_vars_col: usize,
    i_bit: usize,
    f_bit: usize,
) -> Rc<DenseMultilinearExtension<F>> {
    let mut rng = rand::thread_rng();
    let range = 1 << (i_bit + f_bit);
    let f_shift = 1 << f_bit;
    let w = DMatrix::from_fn(1 << num_vars_row, 1 << num_vars_col, |_, _| {
        rng.gen_range(-(range as i64)..(range as i64)) as f64 / f_shift as f64
    });
    //let w = spectral_normalization(w, 2.0);
    let w = w.map(|x| (x * f_shift as f64).round() as i128);
    let w: Vec<i128> = w.iter().cloned().collect();
    let w = vec_to_named_poly(name, &i128_to_field::<F>(&w));
    w
}
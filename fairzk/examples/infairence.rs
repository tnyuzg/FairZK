use algebra::{
    utils::Transcript, DenseMultilinearExtension, Goldilocks, GoldilocksExtension,
};
use fairzk::piop::MetaDisparityInstance;
use fairzk::read_params::{gen_matrix_from_params, gen_vector_from_params};
use fairzk::utils::i128_to_field;
use fairzk::utils::{u64_to_field, vec_to_named_poly};
use fairzk::{pcs::OracleTable, piop::InfairenceInstance, snark::InfairenceSnark};
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


//cargo run --example infairence --release

fn main() {
    let i_bit = 5;
    let f_bit = 5;
    
    // generate random instance
    // let mut instance = random_instance(i_bit, f_bit);
    // generate real instance
    let mut instance = real_instance(i_bit, f_bit);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let mut oracle_table = OracleTable::new(&code_spec);
    let mut snark: InfairenceSnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> =
        Default::default();

    let mut trans_p = Transcript::new();
    snark.prove(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();
    let decision = snark.verify(&oracle_table, &mut trans_v);


    let proof_size_sumcheck = snark.proof_size();
    let proof_size_oracle = oracle_table.proof_size();

    println!("proof size sumcheck {} bytes", proof_size_sumcheck);
    println!("proof size oracle {} bytes", proof_size_oracle);
    println!(
        "proof size total {}",
        (proof_size_sumcheck + proof_size_oracle)
    );


    if decision {
        println!("infairence success");
    } else {
        println!("infairence fail");
    }

    let save_path = "infairence.csv";
    oracle_table.statistics_detailed(save_path);
}


fn random_instance(i_bit:usize, f_bit:usize) -> InfairenceInstance<F> {
    let num_layers = 4;
    let num_vars_feats = vec![6, 5, 5, 5, 0];
    

    let w_vec: Vec<Rc<DenseMultilinearExtension<F>>> = num_vars_feats
    .windows(2)
    .take(num_layers)
    .enumerate()
    .map(|(i, shape)| small_w(&format!("w{}", i), shape[1], shape[0]))
    .collect();

    let max_deviation = vec_to_named_poly(
        &"max deviation".to_string(),
        &u64_to_field(&random_vec(1 << num_vars_feats[0], 1 << 2)),
    );

    InfairenceInstance::new_overflow(
        num_layers,
        &num_vars_feats,
        i_bit,
        f_bit,
        &max_deviation,
        &w_vec,
    )

}


fn real_instance(i_bit: usize, f_bit:usize) -> InfairenceInstance<F> {
    let path = "models/quantized/";

    // let num_layers = 1;
    // let num_vars_data = 16;
    // let num_vars_feats = vec![6, 0];
    // let data_name = path.to_owned() + "/adult_data.txt";
    // let w_name = path.to_owned() + "/adult_lr_w1.txt";
    // let group_name = path.to_owned() + "/adult_sensitive_label.txt";

    // let num_layers = 1;
    // let num_vars_data = 10;
    // let num_vars_feats = vec![6, 0];
    // let data_name = path.to_owned() + "/german_data.txt";
    // let w_name = path.to_owned() +  "/german_lr_w1.txt";
    // let group_name = path.to_owned() + "/german_sensitive_label.txt";

    // let num_layers = 1;
    // let num_vars_data = 13;
    // let num_vars_feats = vec![4, 0];
    // let data_name = path.to_owned() + "/compas_data.txt";
    // let w_name = path.to_owned() + "/compas_lr_w1.txt";
    // let group_name = path.to_owned() + "/compas_sensitive_label.txt";

    // let w = gen_vector_from_params(&w_name, f_bit, i_bit, 0).mle;
    // let w = vec![w];

    // let data = gen_matrix_from_params(&data_name, f_bit, i_bit, 0).mle;

    // let group = gen_vector_from_params(&group_name, 0, i_bit, 0).mle;


    // ---------------------------------------------- medium weights ----------------------------------------------

    let path2 =
        "models/quantized/medium_weights/wout_SP_fair_layer";
    let num_layers = 8;
    let num_vars_data = 16;
    let num_vars_feats = vec![6, 11, 11, 11, 11, 11, 11, 11, 0];
    let data_name = path.to_owned() + "/adult_data.txt";
    let w0_name = path2.to_owned() + "0_W.txt";
    let w1_name = path2.to_owned() + "1_W.txt";
    let w2_name = path2.to_owned() + "2_W.txt";
    let w3_name = path2.to_owned() + "3_W.txt";
    let w4_name = path2.to_owned() + "4_W.txt";
    let w5_name = path2.to_owned() + "5_W.txt";
    let w6_name = path2.to_owned() + "6_W.txt";
    let w7_name = path2.to_owned() + "7_W.txt";
    let group_name = path.to_owned() + "/adult_sensitive_label.txt";
    let w0 = gen_matrix_from_params(&w0_name, f_bit, i_bit, 0).mle;
    let w0 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w0".to_string(),
        w0.num_vars,
        &w0.evaluations,
    ));
    dbg!(&w0);

    let w1 = gen_matrix_from_params(&w1_name, f_bit, i_bit, 0).mle;
    let w1 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w1".to_string(),
        w1.num_vars,
        &w1.evaluations,
    ));

    let w2 = gen_matrix_from_params(&w2_name, f_bit, i_bit, 0).mle;
    let w2 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w2".to_string(),
        w2.num_vars,
        &w2.evaluations,
    ));

    let w3 = gen_matrix_from_params(&w3_name, f_bit, i_bit, 0).mle;
    let w3 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w3".to_string(),
        w3.num_vars,
        &w3.evaluations,
    ));

    let w4 = gen_matrix_from_params(&w4_name, f_bit, i_bit, 0).mle;
    let w4 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w4".to_string(),
        w4.num_vars,
        &w4.evaluations,
    ));

    let w5 = gen_matrix_from_params(&w5_name, f_bit, i_bit, 0).mle;
    let w5 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w5".to_string(),
        w5.num_vars,
        &w5.evaluations,
    ));

    let w6 = gen_matrix_from_params(&w6_name, f_bit, i_bit, 0).mle;
    let w6 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w6".to_string(),
        w6.num_vars,
        &w6.evaluations,
    ));

    let w7 = gen_matrix_from_params(&w7_name, f_bit, i_bit, 0).mle;
    let w7 = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"w7".to_string(),
        w7.num_vars,
        &w7.evaluations,
    ));

    let w = vec![w0, w1, w2, w3, w4, w5, w6, w7];
    let data_info = gen_matrix_from_params(&data_name, f_bit, i_bit, 0);
    let data = data_info.mle;
    let data = vec_to_named_poly(
        &"data".to_string(),
        &data
            .iter()
            .take(1 << (num_vars_data + num_vars_feats[0]))
            .cloned()
            .collect::<Vec<_>>(),
    );
    let group = gen_vector_from_params(&group_name, 0, i_bit, 0).mle;
    let group = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"group".to_string(),
        group.num_vars,
        &group.evaluations,
    ));

    let instance =
        MetaDisparityInstance::new(num_vars_feats[0], num_vars_data, i_bit, f_bit, data, group);

    let max_deviation = instance.max_deviation;


    InfairenceInstance::new_overflow(
        num_layers,
        &num_vars_feats,
        i_bit,
        f_bit,
        &max_deviation,
        &w,
    )

}


fn random_vec(length: usize, range: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..length).map(|_| rng.gen_range(0..range)).collect()
}

fn small_w(
    name: &String,
    num_vars_row: usize,
    num_vars_col: usize,
) -> Rc<DenseMultilinearExtension<F>> {
    let mut rng = rand::thread_rng();
    let range = 1 << 3;
    let w = DMatrix::from_fn(1 << num_vars_row, 1 << num_vars_col, |_, _| {
        rng.gen_range(-(range as i128)..(range as i128))
    });
    let w: Vec<i128> = w.iter().cloned().collect();
    let w = vec_to_named_poly(name, &i128_to_field::<F>(&w));
    w
}
use std::rc::Rc;
use algebra::utils::Transcript;
use algebra::{DenseMultilinearExtension, Goldilocks, GoldilocksExtension};
use fairzk::read_params::{gen_matrix_from_params, gen_vector_from_params};
use fairzk::utils::{
    random_absolute_u64_vec, random_positive_u64_vec, u64_to_field, vec_to_named_poly,
};
use fairzk::{pcs::OracleTable, piop::MetaDisparityInstance, snark::MetaDisparitySnark};
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use sha2::Sha256;

type F = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 64;

//cargo run --example meta_disparity --release

fn main() {
    let i_bit = 5;
    let f_bit: usize = 5;

    // generate random instance
    // let mut instance = random_instance(i_bit, f_bit);

    // generate real instance
    let path = "models/quantized";
    let mut instance = real_instance(path, i_bit, f_bit);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 4);
    let mut oracle_table = OracleTable::new(&code_spec);

    let mut snark: MetaDisparitySnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> =
        Default::default();

    let mut trans_p = Transcript::<EF>::new();
    snark.prove(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();
    let decision = snark.verify(&oracle_table, &mut trans_v);

    if decision {
        println!("meta disparity norm snark success");
    } else {
        println!("meata disparity snark fail");
    }

    let proof_size_sumcheck = snark.proof_size();
    let proof_size_oracle = oracle_table.proof_size();

    println!("proof size sumcheck: {}MB", proof_size_sumcheck >> 20);
    println!("proof size oracle: {}MB", proof_size_oracle >> 20);
    println!(
        "proof size total: {}MB",
        (proof_size_sumcheck + proof_size_oracle) >> 20
    );

    let save_path = "pcs_meta_disparity.csv";
    oracle_table.statistics_detailed(save_path);
}

fn random_instance(i_bit: usize, f_bit: usize) -> MetaDisparityInstance<F> {
    let num_vars_data = 7;
    let num_vars_feat = 6;

    let data = vec_to_named_poly(
        &"data".to_string(),
        &u64_to_field::<F>(&random_absolute_u64_vec::<F>(
            1 << (num_vars_data + num_vars_feat),
            1 << (i_bit + f_bit),
        )),
    );

    let group = vec_to_named_poly(
        &"group".to_string(),
        &u64_to_field::<F>(&random_positive_u64_vec(1 << num_vars_data, 2)),
    );

    let instance =
        MetaDisparityInstance::new(num_vars_feat, num_vars_data, i_bit, f_bit, data, group);

    instance
}

fn real_instance(path: &str, i_bit: usize, f_bit: usize) -> MetaDisparityInstance<F> {
    // let num_vars_data = 16;
    // let num_vars_feats = vec![6, 0];
    // let data_name = path.to_owned() + "/adult_data.txt";
    // let group_name = path.to_owned() + "/adult_sensitive_label.txt";

    // let num_vars_data = 10;
    // let num_vars_feats = vec![6, 0];
    // let data_name = path.to_owned() + "/german_data.txt";
    // let group_name = path.to_owned() + "/german_sensitive_label.txt";

    let num_vars_data = 13;
    let num_vars_feats = vec![4, 0];
    let data_name = path.to_owned() + "/compas_data.txt";
    let group_name = path.to_owned() + "/compas_sensitive_label.txt";

    let data_info = gen_matrix_from_params(&data_name, f_bit, i_bit, 0);
    let data = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"data".to_string(),
        data_info.mle.num_vars,
        &data_info.mle.evaluations,
    ));

    let group_info = gen_vector_from_params(&group_name, 0, i_bit, 0);
    let group = Rc::new(DenseMultilinearExtension::from_named_slice(
        &"group".to_string(),
        group_info.mle.num_vars,
        &group_info.mle.evaluations,
    ));

    MetaDisparityInstance::new(num_vars_feats[0], num_vars_data, i_bit, f_bit, data, group)
}

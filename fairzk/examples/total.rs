use algebra::utils::Transcript;
use algebra::{Goldilocks, GoldilocksExtension};
use fairzk::pcs::OracleTable;
use fairzk::piop::total::TotalInstance;
use fairzk::read_params::{gen_matrix_from_params, gen_vector_from_params};
use fairzk::snark::total_snark::TotalSnark;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use sha2::Sha256;

type F = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 63;


// RUST_BACKTRACE=FULL cargo run --example total --release
fn main() {
    let i_bit = 5;
    let f_bit: usize = 5;

    // generate real instance
    let mut instance = real_instance(i_bit, f_bit);


    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let mut oracle_table = OracleTable::new(&code_spec);

    let mut snark: TotalSnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> = Default::default();

    let mut trans_p = Transcript::new();
    snark.prove(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();
    let decision = snark.verify(&oracle_table, &mut trans_v);

    let proof_size_sumcheck = snark.proof_size();
    let proof_size_oracle = oracle_table.proof_size();

    println!(
        "proof size total {} MB\n",
        (proof_size_sumcheck + proof_size_oracle) >> 20
    );

    if decision {
        println!("total success");
    } else {
        println!("total fail");
    }
}

fn real_instance(i_bit: usize, f_bit: usize) -> TotalInstance<F> {
    let path = "models/quantized/";

    let num_layers = 3;
    let num_vars_data = 16;
    let num_vars_feats = vec![6, 7, 7, 0];
    let data_name = path.to_owned() + "/adult_data.txt";
    let w0_name = path.to_owned() + "/adult_layer0_W.txt";
    let w1_name = path.to_owned() + "/adult_layer1_W.txt";
    let w2_name = path.to_owned() + "/adult_layer2_W.txt";
    let group_name = path.to_owned() + "/adult_sensitive_label.txt";
    let w0 = gen_matrix_from_params(&w0_name, f_bit, i_bit, 0).mle;
    let w1 = gen_matrix_from_params(&w1_name, f_bit, i_bit, 0).mle;
    let w2 = gen_matrix_from_params(&w2_name, f_bit, i_bit, 0).mle;
    let w = vec![w0, w1, w2];

    let data_info = gen_matrix_from_params(&data_name, f_bit, i_bit, 0);
    let data = data_info.mle;

    let group_info = gen_vector_from_params(&group_name, 0, i_bit, 0);
    let group = group_info.mle;

    TotalInstance::new(
        num_layers,
        num_vars_data,
        &num_vars_feats,
        i_bit,
        f_bit,
        w,
        data,
        group,
    )

}
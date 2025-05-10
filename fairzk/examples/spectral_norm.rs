use algebra::utils::Transcript;
use algebra::{Goldilocks, GoldilocksExtension};
use fairzk::read_params::gen_matrix_from_params;
use fairzk::utils::random_normalized_w;
use fairzk::{pcs::OracleTable, piop::SpectralNormInstance, snark::SpectralNormSnark};
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use sha2::Sha256;

type F = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 63;

// cargo run --example spectral_norm --release
fn main() {
    let i_bit = 5;
    let f_bit = 5;

    println!("constructing instance ..");

    // generate random instance
    // let mut instance = random_instance(i_bit, f_bit);

    // generate real instance
    let path = "models/quantized/";
    let mut instance = real_instance(path, i_bit, f_bit);

    let code_spec: ExpanderCodeSpec =
        ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let mut oracle_table = OracleTable::new(&code_spec);

    let mut snark: SpectralNormSnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> =
        Default::default();

    let mut trans_p = Transcript::<EF>::new();
    snark.prove(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();
    let decision = snark.verify(&oracle_table, &mut trans_v);

    let (
        proof_size_sumcheck_square,
        proof_size_sumcheck_eigen,
        proof_size_rangecheck_err,
        proof_size_rangecheck_eig,
    ) = snark.proof_size();
    let proof_size_oracle = oracle_table.proof_size();
    let proof_size = proof_size_sumcheck_square
        + proof_size_sumcheck_eigen
        + proof_size_rangecheck_err
        + proof_size_rangecheck_eig
        + proof_size_oracle;
    println!("proof size sumcheck square: {}", proof_size_sumcheck_square);
    println!("proof size sumcheck eigen: {}", proof_size_sumcheck_eigen);
    println!("proof size rangecheck err: {}", proof_size_rangecheck_err);
    println!("proof size rangecheck eig: {}", proof_size_rangecheck_eig);
    println!("proof size oracle: {}", proof_size_oracle);
    println!("proof size: {}", proof_size);

    if decision {
        println!("spectral norm snark success");
    } else {
        println!("spectral norm snark fail");
    }

    let save_path = "pcs_spectral_norm.csv";
    oracle_table.statistics_detailed(save_path);
}

fn random_instance(i_bit: usize, f_bit: usize) -> SpectralNormInstance<F> {
    let num_vars_row = 10;
    let num_vars_col = 10;
    let num_row = 1 << num_vars_row;
    let num_col = 1 << num_vars_col;
    println!("sampling w ...");
    let w = random_normalized_w(&"w".to_string(), num_row, num_col, i_bit, f_bit, 1.5);

    let instance = SpectralNormInstance::new(num_vars_row, num_vars_col, i_bit, f_bit, &w);
    assert!(instance.is_valid());

    instance
}

fn real_instance(path: &str, i_bit: usize, f_bit: usize) -> SpectralNormInstance<F> {

    let num_vars_feats = vec![4, 6, 0];
    let w0_name = path.to_owned() + "compas_layer0_W.txt";
    let w1_name = path.to_owned() + "compas_layer1_W.txt";

    let w0_info = gen_matrix_from_params(&w0_name, f_bit, i_bit, 0);
    let w0 = w0_info.mle;

    let w1_info = gen_matrix_from_params(&w1_name, f_bit, i_bit, 0);
    let w1 = w1_info.mle;

    let _w = vec![w0.clone(), w1.clone()];

    let num_vars_row = num_vars_feats[0];
    let num_vars_col = num_vars_feats[1];

    let instance = SpectralNormInstance::new(num_vars_row, num_vars_col, i_bit, f_bit, &w0);
    instance.is_valid();

    instance
}

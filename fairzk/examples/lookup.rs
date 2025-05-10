use algebra::{utils::Transcript, Field, Goldilocks, GoldilocksExtension};
use fairzk::{
    pcs::OracleTable,
    piop::LookupInstance,
    snark::LookupSnark,
    timing,
    utils::{field_range_tables, random_absolute_u64_vec, u64_to_field, vec_to_named_polys},
};
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use sha2::Sha256;

type F = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 64;

// RUST_BACKTRACE=FULL cargo run --example lookup --release
fn main() {
    let mut instance = random_instance();

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let mut snark: LookupSnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> = LookupSnark::new();
    let mut oracle_table = OracleTable::new(&code_spec);

    let mut trans_p = Transcript::new();
    snark.prove(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();
    let decision = snark.verify(&mut oracle_table, &mut trans_v);

    if decision {
        println!("lookup success");
    } else {
        println!("lookup fail");
    }
}

fn random_instance() -> LookupInstance<F, F, EF> {
    let num_vars = 10;
    let block_size = 2;
    let f_len = 8;

    let range = 1 << (num_vars - 2);

    let mut u64_vec = random_absolute_u64_vec::<F>(f_len * (1 << num_vars), range);

    u64_vec.resize(f_len * (1 << num_vars), 0);
    let field_vec: Vec<F> = u64_to_field(&u64_vec);
    let f_name = (0..f_len)
        .map(|i| format!("f{}", i))
        .collect::<Vec<String>>();
    let f_vec = vec_to_named_polys(num_vars, &field_vec, &f_name);

    let min = -F::new(range + 1);
    let max = F::new(range + 1);
    let t_vec = field_range_tables(num_vars, min, max);

    let instance: LookupInstance<F, F, EF> = timing!(
        "constructing the lookup instance",
        LookupInstance::new(
            &f_vec,
            &t_vec,
            "lookup".to_string(),
            "range".to_string(),
            block_size,
        )
    );
    instance
}

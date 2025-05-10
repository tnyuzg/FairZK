use algebra::{utils::Transcript, AsFrom, Field, Goldilocks, GoldilocksExtension, PrimeField};
use fairzk::{
    pcs::OracleTable,
    piop::VectorLookupInstance,
    snark::VectorLookupSnark,
    utils::{
        i128_to_field, random_absolute_u64_vec, u64_to_field, vec_to_named_polys, vec_to_polys,
    },
};
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use sha2::Sha256;

type F = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;

fn main() {
    let num_vars = 10;
    let block_size = 1;
    let f_len = 1;
    let poly_size = 1 << num_vars;

    let i = 5;
    let f = 5;
    let _q_y = i + f;
    let q_x = i + 2 * f;
    let modulus: u64 = F::MODULUS_VALUE.into();
    dbg!(modulus);

    let x: Vec<F> = u64_to_field(&random_absolute_u64_vec::<F>(
        f_len * (1 << num_vars),
        1 << q_x,
    ));
    let y: Vec<F> = x.iter().map(|&x| relu(x, i, f)).collect::<Vec<F>>();

    let mut domain: Vec<F> = i128_to_field(&(-(1 << q_x) + 1..(1 << q_x)).collect::<Vec<i128>>());
    let pad_size = (poly_size - (domain.len() % poly_size)) % poly_size;
    domain.resize(domain.len() + pad_size, F::new(0));

    let image: Vec<F> = domain.iter().map(|&x| relu(x, i, f)).collect();

    let fx_name = (0..f_len)
        .map(|i| format!("fx{}", i))
        .collect::<Vec<String>>();
    let fy_name = (0..f_len)
        .map(|i| format!("fy{}", i))
        .collect::<Vec<String>>();

    let fx_vec = vec_to_named_polys(num_vars, &x, &fx_name);
    let fy_vec = vec_to_named_polys(num_vars, &y, &fy_name);
    let tx_vec = vec_to_polys(num_vars, &domain);
    let ty_vec = vec_to_polys(num_vars, &image);

    let mut instance: VectorLookupInstance<F, F, EF> = VectorLookupInstance::new(
        num_vars,
        &fx_vec,
        &fy_vec,
        &tx_vec,
        &ty_vec,
        &"example relu".to_string(),
        &"domain of relu".to_string(),
        &"image of relu".to_string(),
        block_size,
    );

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, F::MODULUS_VALUE as usize, 10);
    let mut snark: VectorLookupSnark<F, EF, Hash, ExpanderCode<F>, ExpanderCodeSpec> =
        VectorLookupSnark::new();
    let mut oracle_table = OracleTable::new(&code_spec);

    let mut trans_p = Transcript::new();
    snark.prove(&mut instance, &mut oracle_table, &mut trans_p);

    let mut trans_v = Transcript::new();
    let decision = snark.verify(&mut oracle_table, &mut trans_v);

    assert!(decision);
    if decision {
        println!("lookup success");
    } else {
        println!("lookup fail");
    }
}

fn relu<F: PrimeField>(x: F, i: usize, f: usize) -> F {
    let x: u64 = x.value().into();
    let p: u64 = F::MODULUS_VALUE.into();
    if x < p / 2 {
        let y = x >> f;
        debug_assert!(y < (1 << (i + f)));
        F::new(F::Value::as_from(y))
    } else {
        F::zero()
    }
}

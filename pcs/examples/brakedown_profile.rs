use std::time::Instant;

use algebra::{
    utils::Transcript, DenseMultilinearExtension, FieldUniformSampler,
    Goldilocks, GoldilocksExtension, MultilinearExtension,
};
use csv::WriterBuilder;
use pcs::{
    multilinear::brakedown::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
    PolynomialCommitmentScheme,
};
use rand::Rng;
use sha2::Sha256;
use std::fs::OpenOptions;

// type FF = BabyBear;
// type EF = BabyBearExtension;
type FF = Goldilocks;
type EF = GoldilocksExtension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 64;

pub fn run_brakedown_benchmark_csv(
    output_path: &str,
    num_vars_range: std::ops::RangeInclusive<usize>,
) -> std::io::Result<()> {
    let file_exists = std::path::Path::new(output_path).exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;
    let mut wtr = WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);

    if !file_exists {
        wtr.write_record(&[
            "num_vars",
            "setup_ms",
            "commit_ms",
            "open_ms",
            "verify_ms",
            "proof_size_MB",
        ])?;
    }

    for &num_vars in num_vars_range.clone().collect::<Vec<_>>().iter() {
        let code_spec: ExpanderCodeSpec =
            ExpanderCodeSpec::new(0.1195, 0.0284, 1.9, BASE_FIELD_BITS, 10);

        let evaluations: Vec<FF> = rand::thread_rng()
            .sample_iter(FieldUniformSampler::new())
            .take(1 << num_vars)
            .collect();
        let poly = DenseMultilinearExtension::from_vec(num_vars, evaluations);

        let now = Instant::now();
        let pp = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::setup(
            num_vars,
            Some(code_spec),
        );
        let setup_ms = now.elapsed().as_millis();

        let mut trans = Transcript::<EF>::new();

        let now = Instant::now();
        let (comm, state) =
            BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::commit(&pp, &poly);
        let commit_ms = now.elapsed().as_millis();

        let point: Vec<EF> = rand::thread_rng()
            .sample_iter(FieldUniformSampler::new())
            .take(num_vars)
            .collect();

        let now = Instant::now();
        let proof = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::open(
            &pp, &comm, &state, &point, &mut trans,
        );
        let open_ms = now.elapsed().as_millis();

        let eval = poly.evaluate_ext(&point);

        let mut trans = Transcript::<EF>::new();

        let now = Instant::now();
        let check = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::verify(
            &pp, &comm, &point, eval, &proof, &mut trans,
        );
        let verify_ms = now.elapsed().as_millis();

        assert!(check);

        let proof_size_bytes = proof.to_bytes().unwrap().len();

        wtr.write_record(&[
            num_vars.to_string(),
            setup_ms.to_string(),
            commit_ms.to_string(),
            open_ms.to_string(),
            verify_ms.to_string(),
            format!("{:.2}", proof_size_bytes as f64 / 1024.0 / 1024.0),
        ])?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn run_brakedown_ef_benchmark_csv(
    output_path: &str,
    num_vars_range: std::ops::RangeInclusive<usize>,
) -> std::io::Result<()> {
    let file_exists = std::path::Path::new(output_path).exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;
    let mut wtr = WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);

    if !file_exists {
        wtr.write_record(&[
            "num_vars",
            "setup_ms",
            "commit_ms",
            "open_ms",
            "verify_ms",
            "proof_size_MB",
        ])?;
    }

    for &num_vars in num_vars_range.clone().collect::<Vec<_>>().iter() {
        let code_spec: ExpanderCodeSpec =
            ExpanderCodeSpec::new(0.1195, 0.0284, 1.9, BASE_FIELD_BITS, 10);

        let evaluations: Vec<EF> = rand::thread_rng()
            .sample_iter(FieldUniformSampler::new())
            .take(1 << num_vars)
            .collect();
        let poly = DenseMultilinearExtension::from_vec(num_vars, evaluations);

        let now = Instant::now();
        let pp = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::setup(
            num_vars,
            Some(code_spec),
        );
        let setup_ms = now.elapsed().as_millis();

        let mut trans = Transcript::<EF>::new();

        let now = Instant::now();
        let (comm, state) =
            BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::commit_ef(&pp, &poly);
        let commit_ms = now.elapsed().as_millis();

        let point: Vec<EF> = rand::thread_rng()
            .sample_iter(FieldUniformSampler::new())
            .take(num_vars)
            .collect();

        let now = Instant::now();
        let proof = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::open_ef(
            &pp, &comm, &state, &point, &mut trans,
        );
        let open_ms = now.elapsed().as_millis();

        let eval = poly.evaluate(&point);

        let mut trans = Transcript::<EF>::new();

        let now = Instant::now();
        let check = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::verify_ef(
            &pp, &comm, &point, eval, &proof, &mut trans,
        );
        let verify_ms = now.elapsed().as_millis();

        assert!(check);

        let proof_size_bytes = proof.to_bytes().unwrap().len();

        wtr.write_record(&[
            num_vars.to_string(),
            setup_ms.to_string(),
            commit_ms.to_string(),
            open_ms.to_string(),
            verify_ms.to_string(),
            format!("{:.2}", proof_size_bytes as f64 / 1024.0 / 1024.0),
        ])?;
        wtr.flush()?;
    }

    Ok(())
}

fn main() {
    run_brakedown_ef_benchmark_csv("brakedown_ef_bench1.csv", 2..=26).unwrap();
}

use crate::utils::{combine_suboracle_evals, polys_to_vec, vec_to_poly};
use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    MultilinearExtension,
};
use pcs::{
    multilinear::brakedown::BrakedownPCS,
    utils::{
        code::{LinearCode, LinearCodeSpec},
        hash::Hash,
    },
    PolynomialCommitmentScheme,
};
use rand::thread_rng;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use std::rc::Rc;
use std::{collections::HashMap, mem};

pub type Parameters<F, H, C, S, EF> =
    <BrakedownPCS<F, H, C, S, EF> as PolynomialCommitmentScheme<F, EF, S>>::Parameters;

pub type Commitment<F, H, C, S, EF> =
    <BrakedownPCS<F, H, C, S, EF> as PolynomialCommitmentScheme<F, EF, S>>::Commitment;

pub type CommitmentState<F, H, C, S, EF> =
    <BrakedownPCS<F, H, C, S, EF> as PolynomialCommitmentScheme<F, EF, S>>::CommitmentState;
pub type CommitmentStateEF<F, H, C, S, EF> =
    <BrakedownPCS<F, H, C, S, EF> as PolynomialCommitmentScheme<F, EF, S>>::CommitmentStateEF;

pub type CommitResult<F, H, C, S, EF> = (
    Parameters<F, H, C, S, EF>,
    Commitment<F, H, C, S, EF>,
    CommitmentState<F, H, C, S, EF>,
);

pub type CommitResultEF<F, H, C, S, EF> = (
    Parameters<F, H, C, S, EF>,
    Commitment<F, H, C, S, EF>,
    CommitmentStateEF<F, H, C, S, EF>,
);

pub type OpenProof<F, H, C, S, EF> =
    <BrakedownPCS<F, H, C, S, EF> as PolynomialCommitmentScheme<F, EF, S>>::Proof;
pub type OpenProofEF<F, H, C, S, EF> =
    <BrakedownPCS<F, H, C, S, EF> as PolynomialCommitmentScheme<F, EF, S>>::ProofEF;

#[derive(Default)]
pub struct Oracle<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    poly: Rc<DenseMultilinearExtension<F>>,
    commitment: CommitResult<F, H, C, S, EF>,
    queries: HashMap<Vec<EF>, (Vec<EF>, OpenProof<F, H, C, S, EF>)>,

    poly_ef: Rc<DenseMultilinearExtension<EF>>,
    commitment_ef: CommitResultEF<F, H, C, S, EF>,
    queries_ef: HashMap<Vec<EF>, (Vec<EF>, OpenProofEF<F, H, C, S, EF>)>,

    num_suboracle: usize,
    virtual_oracle: bool,
}

impl<F, EF, H, C, S> Oracle<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn new(poly: Rc<DenseMultilinearExtension<F>>, code_spec: S) -> Self {
        let num_vars = poly.num_vars;
        let pp = BrakedownPCS::<F, H, C, S, EF>::setup(num_vars, Some(code_spec.clone()));
        let (com, com_state) = BrakedownPCS::<F, H, C, S, EF>::commit(&pp, &poly);
        Oracle {
            poly,
            commitment: (pp, com, com_state),
            queries: HashMap::new(),
            num_suboracle: 1,
            poly_ef: Default::default(),
            commitment_ef: Default::default(),
            queries_ef: HashMap::new(),
            virtual_oracle: false,
        }
    }

    pub fn new_super_oracle(
        poly: Rc<DenseMultilinearExtension<F>>,
        num_suboracle: usize,
        code_spec: S,
    ) -> Self {
        let num_vars = poly.num_vars;
        let pp = BrakedownPCS::<F, H, C, S, EF>::setup(num_vars, Some(code_spec.clone()));
        let (com, com_state) = BrakedownPCS::<F, H, C, S, EF>::commit(&pp, &poly);
        Oracle {
            poly,
            commitment: (pp, com, com_state),
            queries: HashMap::new(),
            num_suboracle,
            poly_ef: Default::default(),
            commitment_ef: Default::default(),
            queries_ef: HashMap::new(),
            virtual_oracle: false,
        }
    }

    pub fn new_virtual(poly: Rc<DenseMultilinearExtension<F>>, code_spec: S) -> Self {
        let num_vars = poly.num_vars;
        let pp = BrakedownPCS::<F, H, C, S, EF>::setup(num_vars, Some(code_spec.clone()));
        let (com, com_state) = BrakedownPCS::<F, H, C, S, EF>::commit(&pp, &poly);
        Oracle {
            poly,
            commitment: (pp, com, com_state),
            queries: HashMap::new(),
            num_suboracle: 1,
            poly_ef: Default::default(),
            commitment_ef: Default::default(),
            queries_ef: HashMap::new(),
            virtual_oracle: true,
        }
    }

    pub fn clear(&mut self) {
        self.poly = Rc::new(DenseMultilinearExtension::new_empty(
            &self.poly.name,
            self.poly.num_vars,
        ));
        self.poly_ef = Rc::new(DenseMultilinearExtension::new_empty(
            &self.poly_ef.name,
            self.poly_ef.num_vars,
        ));
    }

    pub fn add_point(&mut self, point: &[EF]) {
        let num_vars_sup = self.poly.num_vars;
        let num_vars_idx = self.num_suboracle.next_power_of_two().ilog2() as usize;
        let num_vars_sub = num_vars_sup - num_vars_idx;

        debug_assert!(
            num_vars_sup == point.len(),
            "num vars super {} not equal to point len {} with num vars sub {} at {}",
            num_vars_sup,
            point.len(),
            num_vars_sub,
            self.poly.name,
        );

        self.queries.entry(point.to_vec()).or_insert_with(|| {
            let subpoint = &point[..num_vars_sub];

            let evals = (0..self.num_suboracle)
                .map(|i| {
                    let poly = vec_to_poly(
                        &self.poly.evaluations
                            [i * (1 << num_vars_sub)..(i + 1) * (1 << num_vars_sub)],
                    );
                    poly.evaluate_ext(subpoint)
                })
                .collect::<Vec<EF>>();

            (evals, Default::default())
        });
    }

    pub fn prove(&mut self, trans: &mut Transcript<EF>) {
        self.queries.iter_mut().for_each(|(point, (_eval, proof))| {
            *proof = BrakedownPCS::<F, H, C, S, EF>::open(
                &self.commitment.0,
                &self.commitment.1,
                &self.commitment.2,
                point,
                trans,
            );
        });
    }

    pub fn verify(&self, trans: &mut Transcript<EF>) -> bool {
        if self.poly == Default::default() {
            return true;
        }
        let num_vars_sup = self.poly.num_vars;
        let num_vars_idx = self.num_suboracle.next_power_of_two().ilog2() as usize;
        let num_vars_sub = num_vars_sup - num_vars_idx;

        let result = self.queries.iter().all(|(point, (eval, proof))| {
            BrakedownPCS::<F, H, C, S, EF>::verify(
                &self.commitment.0,
                &self.commitment.1,
                point,
                combine_suboracle_evals(eval, &point[num_vars_sub..]),
                proof,
                trans,
            )
        });
        if !result {
            println!("{}", self.poly.name)
        };
        result
    }

    pub fn new_ef(poly_ef: Rc<DenseMultilinearExtension<EF>>, code_spec: S) -> Self {
        let num_vars = poly_ef.num_vars;
        let pp = BrakedownPCS::<F, H, C, S, EF>::setup(num_vars, Some(code_spec.clone()));
        let (com, com_state_ef) = BrakedownPCS::<F, H, C, S, EF>::commit_ef(&pp, &poly_ef);
        Oracle {
            poly_ef,
            commitment_ef: (pp, com, com_state_ef),
            queries: HashMap::new(),
            num_suboracle: 1,
            poly: Default::default(),
            commitment: Default::default(),
            queries_ef: HashMap::new(),
            virtual_oracle: false,
        }
    }

    pub fn new_super_oracle_ef(
        poly_ef: Rc<DenseMultilinearExtension<EF>>,
        num_suboracle: usize,
        code_spec: S,
    ) -> Self {
        let num_vars = poly_ef.num_vars;
        let pp = BrakedownPCS::<F, H, C, S, EF>::setup(num_vars, Some(code_spec.clone()));
        let (com, com_state) = BrakedownPCS::<F, H, C, S, EF>::commit_ef(&pp, &poly_ef);
        Oracle {
            poly_ef: poly_ef,
            commitment_ef: (pp, com, com_state),
            queries: HashMap::new(),
            num_suboracle,
            poly: Default::default(),
            commitment: Default::default(),
            queries_ef: HashMap::new(),
            virtual_oracle: false,
        }
    }

    pub fn add_point_ef(&mut self, point: &[EF]) {
        let num_vars_sup = self.poly_ef.num_vars;
        let num_vars_idx = self.num_suboracle.next_power_of_two().ilog2() as usize;
        let num_vars_sub = num_vars_sup - num_vars_idx;

        debug_assert!(
            num_vars_sup == point.len(),
            "num vars super {} not equal to point len {}",
            num_vars_sup,
            point.len()
        );

        self.queries_ef.entry(point.to_vec()).or_insert_with(|| {
            let subpoint = &point[..num_vars_sub];

            let evals = (0..self.num_suboracle)
                .map(|i| {
                    let poly = vec_to_poly(
                        &self.poly_ef.evaluations
                            [i * (1 << num_vars_sub)..(i + 1) * (1 << num_vars_sub)],
                    );
                    poly.evaluate(subpoint)
                })
                .collect::<Vec<EF>>();

            (evals, Default::default())
        });
    }

    pub fn prove_ef(&mut self, trans: &mut Transcript<EF>) {
        self.queries_ef
            .iter_mut()
            .for_each(|(point, (_eval, proof))| {
                *proof = BrakedownPCS::<F, H, C, S, EF>::open_ef(
                    &self.commitment_ef.0,
                    &self.commitment_ef.1,
                    &self.commitment_ef.2,
                    point,
                    trans,
                );
            });
    }

    pub fn verify_ef(&self, trans: &mut Transcript<EF>) -> bool {
        if self.poly_ef == Default::default() {
            return true;
        }
        let num_vars_sup = self.poly_ef.num_vars;
        let num_vars_idx = self.num_suboracle.next_power_of_two().ilog2() as usize;
        let num_vars_sub = num_vars_sup - num_vars_idx;

        self.queries_ef.iter().all(|(point, (eval, proof))| {
            BrakedownPCS::<F, H, C, S, EF>::verify_ef(
                &self.commitment_ef.0,
                &self.commitment_ef.1,
                point,
                combine_suboracle_evals(eval, &point[num_vars_sub..]),
                proof,
                trans,
            )
        })
    }
}

#[derive(Default)]
pub struct OracleTable<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    oracles: HashMap<String, Oracle<F, EF, H, C, S>>,
    code_spec: S,
}

impl<F, EF, H, C, S> OracleTable<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn proof_size(&mut self) -> usize {
        self.clear();

        let mut total_size = mem::size_of_val(self);

        for (oracle_name, oracle) in self.oracles.iter() {
            if oracle_name.starts_with("intermediate oracle") {
                continue;
            }

            // total_size += mem::size_of_val(oracle_name);
            // total_size += mem::size_of_val(oracle);

            for (_point, (value, proof)) in oracle.queries.iter() {
                // Add memory for each key-value pair in the inner HashMap
                //total_size += mem::size_of_val(point);
                total_size += mem::size_of_val(value);
                total_size += match proof.to_bytes() {
                    Ok(bytes) => bytes.len(),
                    Err(_) => 0,
                };
            }

            for (_point, (value, proof)) in oracle.queries_ef.iter() {
                // Add memory for each key-value pair in the inner HashMap
                //total_size += mem::size_of_val(point);
                total_size += mem::size_of_val(value);
                total_size += match proof.to_bytes() {
                    Ok(bytes) => bytes.len(),
                    Err(_) => 0,
                };
            }
        }
        total_size
    }

    pub fn new(code_spec: &S) -> Self {
        OracleTable {
            oracles: HashMap::new(),
            code_spec: code_spec.clone(),
        }
    }

    pub fn clear(&mut self) {
        self.oracles.values_mut().for_each(|oracle| {
            oracle.clear();
        });
    }

    pub fn add_oracle(&mut self, name: &String, poly: Rc<DenseMultilinearExtension<F>>) {
        self.oracles
            .entry(name.clone())
            .or_insert_with(|| Oracle::new(poly, self.code_spec.clone()));
    }

    pub fn add_named_oracle(&mut self, poly: Rc<DenseMultilinearExtension<F>>) {
        self.oracles
            .entry(poly.name.clone())
            .or_insert_with(|| Oracle::new(poly, self.code_spec.clone()));
    }

    pub fn add_super_oracle(&mut self, name: &String, polys: &[Rc<DenseMultilinearExtension<F>>]) {
        let num_vars = polys[0].num_vars;
        let num_suboracle = polys.len();
        let num_vars_index = num_suboracle.next_power_of_two().ilog2() as usize;
        debug_assert!(polys.iter().all(|poly| poly.num_vars == num_vars));

        let mut evaluations = polys_to_vec(&polys);
        evaluations.resize(1 << (num_vars_index + num_vars), F::zero());
        let poly = vec_to_poly(&evaluations);

        self.oracles.entry(name.clone()).or_insert_with(|| {
            Oracle::new_super_oracle(poly, num_suboracle, self.code_spec.clone())
        });
    }

    pub fn add_point(&mut self, name: &String, point: &[EF]) {
        let oracle = self
            .oracles
            .get_mut(name)
            .expect(&format!("oracle '{}' not found in oracles table", name));
        oracle.add_point(point);
    }

    pub fn get_eval(&self, name: &String, point: &[EF]) -> &[EF] {
        let oracle = self
            .oracles
            .get(name)
            .expect(&format!("oracle {} not found", name));

        &oracle
            .queries
            .get(point)
            .expect(&format!("point {:?} not found in oracle {}", point, name))
            .0
    }

    // to chenage
    pub fn prove(&mut self, trans: &mut Transcript<EF>) {
        for oracle in self.oracles.values_mut() {
            // let now = Instant::now();
            if oracle.virtual_oracle {
                continue;
            }

            oracle.prove(trans);
            oracle.prove_ef(trans);

            // let open_ms = now.elapsed().as_millis();
            // let name = oracle.poly.name.clone();
            // println!("{name} {open_ms}");
        }
    }

    pub fn verify(&self, trans: &mut Transcript<EF>) -> bool {
        for oracle in self.oracles.values() {
            if oracle.virtual_oracle {
                continue;
            }

            if !oracle.verify(trans) || !oracle.verify_ef(trans) {
                return false;
            }
        }
        true
    }

    pub fn add_oracle_ef(&mut self, name: &String, poly: Rc<DenseMultilinearExtension<EF>>) {
        self.oracles
            .entry(name.clone())
            .or_insert_with(|| Oracle::new_ef(poly, self.code_spec.clone()));
    }

    pub fn add_super_oracle_ef(
        &mut self,
        name: &String,
        polys: &[Rc<DenseMultilinearExtension<EF>>],
    ) {
        let num_vars = polys[0].num_vars;
        let num_suboracle = polys.len();
        let num_vars_index = num_suboracle.next_power_of_two().ilog2() as usize;
        debug_assert!(polys.iter().all(|poly| poly.num_vars == num_vars));

        let mut evaluations = polys_to_vec(&polys);
        evaluations.resize(1 << (num_vars_index + num_vars), EF::zero());
        let poly = vec_to_poly(&evaluations);
        self.oracles.entry(name.clone()).or_insert_with(|| {
            Oracle::new_super_oracle_ef(poly, num_suboracle, self.code_spec.clone())
        });
    }

    pub fn add_point_ef(&mut self, name: &String, point: &[EF]) {
        let oracle = self
            .oracles
            .get_mut(name)
            .expect(&format!("oracle '{}' not found in oracles table", name));
        oracle.add_point_ef(point);
    }

    pub fn get_eval_ef(&self, name: &String, point: &[EF]) -> &[EF] {
        let oracle = self
            .oracles
            .get(name)
            .expect(&format!("oracle {} not found", name));

        &oracle
            .queries_ef
            .get(point)
            .expect(&format!("point {:?} not found in oracle {}", point, name))
            .0
    }

    pub fn statistics(&self) -> HashMap<(usize, usize, bool), usize> {
        let mut stats: HashMap<(usize, usize, bool), usize> = HashMap::new();

        for oracle in self.oracles.values() {
            let (num_vars, num_queries, poly_status) = if oracle.poly != Default::default() {
                (oracle.poly.num_vars(), oracle.queries.len(), true) // poly is set
            } else {
                (oracle.poly_ef.num_vars(), oracle.queries_ef.len(), false) // poly_ef is set
            };

            *stats
                .entry((num_vars, num_queries, poly_status))
                .or_insert(0) += 1;
        }

        // for ((num_vars, num_queries, poly_status), count) in &stats {
        //     println!("num_vars: {}, num_queries: {}, poly_status: {}, count: {}", num_vars, num_queries, poly_status, count);
        // }

        let mut stats_vec: Vec<_> = stats.iter().collect();

        stats_vec.sort_by(
            |((a_vars, a_queries, a_status), _), ((b_vars, b_queries, b_status), _)| {
                (a_vars, a_queries, a_status).cmp(&(b_vars, b_queries, b_status))
            },
        );

        for ((num_vars, num_queries, poly_status), count) in stats_vec {
            println!(
                "num_vars: {}, num_queries: {}, poly_status: {}, count: {}",
                num_vars, num_queries, poly_status, count
            );
        }

        stats
    }

    pub fn statistics_detailed(
        &mut self,
        csv_save_path: &str,
    ) -> HashMap<(usize, usize, usize, bool), usize> {
        self.clear();

        let mut stats: HashMap<(usize, usize, usize, bool), (usize, usize, Vec<String>)> =
            HashMap::new();

        for (name, oracle) in self.oracles.iter() {
            let (mut num_vars, num_queries, batch_size, extension_field, proof_size) =
                if oracle.poly != Default::default() {
                    let num_vars = oracle.poly.num_vars();
                    let num_queries = oracle.queries.len();
                    let num_suboracle = oracle.num_suboracle;
                    let extension_field = false;

                    let mut proof_size = 0;
                    for (_point, (value, proof)) in oracle.queries.iter() {
                        //proof_size += std::mem::size_of_val(point);
                        proof_size += std::mem::size_of_val(value);
                        proof_size += match proof.to_bytes() {
                            Ok(bytes) => bytes.len(),
                            Err(_) => 0,
                        };
                    }

                    (
                        num_vars,
                        num_queries,
                        num_suboracle,
                        extension_field,
                        proof_size,
                    )
                } else {
                    let num_vars = oracle.poly_ef.num_vars();
                    let num_queries = oracle.queries_ef.len();
                    let num_suboracle = oracle.num_suboracle;
                    let extension_field = true;

                    let mut proof_size = 0;
                    for (_point, (value, proof)) in oracle.queries_ef.iter() {
                        //proof_size += std::mem::size_of_val(point);
                        proof_size += std::mem::size_of_val(value);
                        proof_size += match proof.to_bytes() {
                            Ok(bytes) => bytes.len(),
                            Err(_) => 0,
                        };
                    }

                    (
                        num_vars,
                        num_queries,
                        num_suboracle,
                        extension_field,
                        proof_size,
                    )
                };

            if batch_size > 1 {
                num_vars -= batch_size.ilog2() as usize;
            }

            let entry = stats
                .entry((num_vars, num_queries, batch_size, extension_field))
                .or_insert((0, 0, Vec::new()));
            entry.0 += 1;
            entry.1 += proof_size;
            entry.2.push(name.clone());
        }

        let mut stats_vec: Vec<_> = stats.into_iter().collect();
        stats_vec.sort_by(|a, b| a.0.cmp(&b.0));

        let mut pcs_proof_size = 0;
        let mut wtr = csv::Writer::from_path(csv_save_path).unwrap();

        wtr.write_record(&[
            "num_vars",
            "num_queries",
            "batch_size",
            "extension_field",
            "unit_proof_size_MB",
            "total_proof_size_MB",
            "count",
            "oracle_names",
        ])
        .unwrap();

        for (
            (num_vars, num_queries, batch_size, extension_field),
            (count, total_proof_size, names),
        ) in stats_vec.iter()
        {
            let unit_proof_size = if *count > 0 {
                *total_proof_size as f64 / *count as f64
            } else {
                0.0
            };

            // 打印
            println!(
                "num_vars (adjusted): {}, num_queries: {}, batch_size: {}, extension_field: {}, count: {}, total_proof_size: {:.3} MB, unit_proof_size: {:.3} MB",
                num_vars,
                num_queries,
                batch_size,
                extension_field,
                count,
                *total_proof_size as f64 / 1024.0 / 1024.0,
                unit_proof_size / 1024.0 / 1024.0
            );
            for name in names {
                println!("    - oracle: {}", name);
            }

            // 写入 CSV
            wtr.write_record(&[
                num_vars.to_string(),
                num_queries.to_string(),
                batch_size.to_string(),
                extension_field.to_string(),
                format!("{:.2}", unit_proof_size / 1024.0 / 1024.0),
                format!("{:.2}", *total_proof_size as f64 / 1024.0 / 1024.0),
                count.to_string(),
                names.join(";   "),
            ])
            .unwrap();

            pcs_proof_size += *total_proof_size;
        }

        wtr.flush().unwrap();

        println!(
            "\n[Total proof size across all oracles]: {:.3} MB",
            pcs_proof_size as f64 / 1024.0 / 1024.0
        );

        stats_vec
            .into_iter()
            .map(|(k, (count, _, _))| (k, count))
            .collect()
    }
}

pub fn sample_and_measure<F, EF, H, C, S>(l: usize, k: usize, code_spec: S)
where
    F: Field + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    let mut table = OracleTable::<F, EF, H, C, S>::new(&code_spec);
    let mut rng = thread_rng();

    let configs = vec![
        (l, 1, 4, false),
        (l + k, 2, 1, false),
        (l + k, 1, 3, false),
        (l + 1, 1, 2, true),
        (2 * l, 1, 8, false),
        (2 * l + k, 5, 1, false),
        (2 * l + k, 4, 1, false),
        (2 * l + k, 2, 1, false),
        (2 * l + 1, 1, 4, true),
    ];

    let mut name_idx = 0;

    for (num_vars, num_queries, count, is_ef) in configs {
        for _ in 0..count {
            let name = format!("oracle_{name_idx}");
            name_idx += 1;

            if is_ef {
                let evals = (0..(1 << num_vars))
                    .map(|_| EF::random(&mut rng))
                    .collect::<Vec<_>>();
                let poly = Rc::new(DenseMultilinearExtension::from_named_slice(
                    &name, num_vars, &evals,
                ));
                table.add_oracle_ef(&name, poly.clone());

                for _ in 0..num_queries {
                    let point = (0..num_vars)
                        .map(|_| EF::random(&mut rng))
                        .collect::<Vec<_>>();
                    table.add_point_ef(&name, &point);
                }
            } else {
                let evals = (0..(1 << num_vars))
                    .map(|_| F::random(&mut rng))
                    .collect::<Vec<_>>();
                let poly = Rc::new(DenseMultilinearExtension::from_named_slice(
                    &name, num_vars, &evals,
                ));
                table.add_oracle(&name, poly.clone());

                for _ in 0..num_queries {
                    let point = (0..num_vars)
                        .map(|_| EF::random(&mut rng))
                        .collect::<Vec<_>>();
                    table.add_point(&name, &point);
                }
            }
        }
    }

    let mut transcript = Transcript::new();
    let start = Instant::now();
    table.prove(&mut transcript);
    let prove_time = start.elapsed().as_millis();

    let mut transcript = Transcript::new();
    let start = Instant::now();
    let is_valid = table.verify(&mut transcript);
    let verify_time = start.elapsed().as_millis();

    let proof_size_bytes = table.proof_size();

    println!("Prove time: {} ms", prove_time);
    println!("Verify time: {} ms", verify_time);
    assert!(is_valid);
    println!(
        "Proof size: {:.3} MB",
        proof_size_bytes as f64 / 1024.0 / 1024.0
    );
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use algebra::{Goldilocks, GoldilocksExtension};
//     use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
//     use sha2::Sha256;

//     type F = Goldilocks;
//     type EF = GoldilocksExtension;
//     type Hash = Sha256;
//     type Code = ExpanderCode<F>;
//     type CodeSpec = ExpanderCodeSpec;

//     #[test]
//     fn test_sample_and_measure() {
//         let base_field_bits = 64;
//         let code_spec = CodeSpec::new(0.1195, 0.0248, 1.9, base_field_bits, 10);
//         let l = 12;
//         let k = 3;
//         sample_and_measure::<F, EF, Hash, Code, CodeSpec>(l, k, code_spec);
//     }
// }

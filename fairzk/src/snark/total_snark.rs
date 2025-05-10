use crate::piop::total::TotalInstance;
use crate::piop::{BoundIOP, InfairenceIOP, LookupIOP};
use crate::piop::{LookupInstance, MetaDisparityIOP, SpectralNormIOP};

use crate::pcs::OracleTable;
use crate::piop::L2NormIOP;

use algebra::PrimeField;
use algebra::{utils::Transcript, AbstractExtensionField};
use pcs::utils::{
    code::{LinearCode, LinearCodeSpec},
    hash::Hash,
};

use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct TotalSnark<F, EF, H, C, S>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub lookups: Vec<LookupIOP<F, EF, H, C, S>>,
    pub meta_disparity_iop: MetaDisparityIOP<F, EF, H, C, S>,
    pub infairence_iop: InfairenceIOP<F, EF, H, C, S>,
    pub spectral_norm_iops: Vec<SpectralNormIOP<F, EF, H, C, S>>,
    pub l2_norm_iops: Vec<L2NormIOP<F, EF, H, C, S>>,
    pub bound_iop: BoundIOP<F, EF, H, C, S>,
}

impl<F, EF, H, C, S> TotalSnark<F, EF, H, C, S>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn proof_size(&self) -> usize {
        let size_meta_disparity = bincode::serialize(&self.meta_disparity_iop)
            .map(|v| v.len())
            .unwrap_or(0);

        let size_infairence = bincode::serialize(&self.infairence_iop)
            .map(|v| v.len())
            .unwrap_or(0);

        let mut size_spectral_norm = 0;
        self.spectral_norm_iops.iter().for_each(|x| {
            size_spectral_norm += bincode::serialize(x).map(|v| v.len()).unwrap_or(0);
        });

        let mut size_lookups = 0;
        self.lookups.iter().for_each(|x| {
            size_lookups += bincode::serialize(x).map(|v| v.len()).unwrap_or(0);
        });

        // dbg!(size_meta_disparity);
        // dbg!(size_infairence);
        // dbg!(size_spectral_norm);
        // dbg!(size_lookups);

        size_meta_disparity + size_infairence + size_spectral_norm + size_lookups
    }

    pub fn prove(
        &mut self,
        instance: &mut TotalInstance<F>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        let (lookup_instances, time_construct_lookup) =
            time!("construct lookup instance", { instance.lookups() });

        let (_, time_commit_oralces) = time!("commit oracles", {
            instance.construct_oracles(oracle_table);
        });

        let (_, time_commit_lookup_first_oracles) = time!("commit lookup first oracles", {
            lookup_instances
                .iter()
                .for_each(|lookup| lookup.construct_first_oracles(oracle_table));
        });

        let mut lookup_instances_ef: Vec<LookupInstance<EF, F, EF>> = lookup_instances
            .iter()
            .map(|lookup| lookup.to_ef())
            .collect();

        let (_, time_commit_lookup_second_oracles) = time!("commit lookup second oracles", {
            lookup_instances_ef
                .iter_mut()
                .for_each(|lookup| lookup.construct_second_oracle(oracle_table, trans));
        });

        let mut instance_ef = instance.to_ef();

        let (_, time_prove_infairence_sumcheck) = time!("prove infairence sumcheck", {
            self.infairence_iop
                .prove(&mut instance_ef.infairence_instance, oracle_table, trans);
        });

        let (_, time_prove_spectral_norm_sumcheck) = time!("prove spectral norm sumcheck", {
            instance_ef
                .spectral_norm_instances
                .iter_mut()
                .for_each(|sn| {
                    let mut sn_iop = SpectralNormIOP::default();
                    sn_iop.prove(sn, oracle_table, trans);
                    self.spectral_norm_iops.push(sn_iop);
                });
        });

        let (_, time_prove_l2_norm_sumcheck) = time!("prove l2 norm sumcheck", {
            instance_ef.l2_norms.iter_mut().for_each(|l2| {
                let mut l2_iop = L2NormIOP::default();
                l2_iop.prove(l2, oracle_table, trans);
                self.l2_norm_iops.push(l2_iop);
            });
        });

        let (_, time_prove_bound_sumcheck) = time!("prove bound sumcheck", {
            self.bound_iop
                .prove(&mut instance_ef.bound, oracle_table, trans);
        });

        let (_, time_prove_lookup_sumcheck) = time!("prove total snark lookup sumcheck", {
            lookup_instances_ef.iter().for_each(|lookup_instance| {
                let mut lookup = LookupIOP::default();
                lookup.prove(lookup_instance, oracle_table, trans);
                self.lookups.push(lookup);
            });
        });

        let (_, time_prove_oracles) = time!("prove oracles", {
            oracle_table.prove(trans);
        });

        let time_construct = time_construct_lookup;
        let time_commit_oracles = time_commit_oralces
            + time_commit_lookup_first_oracles
            + time_commit_lookup_second_oracles;
        let time_prove_oracles = time_prove_oracles;
        let time_prove_sumcheck = time_prove_bound_sumcheck
            + time_prove_infairence_sumcheck
            + time_prove_l2_norm_sumcheck
            + time_prove_lookup_sumcheck
            + time_prove_spectral_norm_sumcheck;

        println!("\n== Prove Timing (in ms) ==");
        println!("Construct:          {} ms", time_construct);
        println!("Commit Oracles:     {} ms", time_commit_oracles);
        println!("Prove Oracles:      {} ms", time_prove_oracles);
        println!("Prove Sumcheck:     {} ms\n", time_prove_sumcheck);
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        let result = self
            .lookups
            .iter()
            .all(|lookup| lookup.pre_randomness(trans));
        assert!(result);

        let (result, time_verify_infairence_sumcheck) = time!("verify infairence sumcheck", {
            self.infairence_iop.verify(oracle_table, trans)
        });
        assert!(&result);

        let (result, time_verify_spectral_norm_sumcheck) =
            time!("verify spectral norm sumcheck", {
                self.spectral_norm_iops
                    .iter()
                    .all(|sn| sn.verify(oracle_table, trans))
            });
        assert!(&result);

        let (result, time_verify_l2_norm_sumcheck) = time!("verify l2 norm sumcheck", {
            self.l2_norm_iops
                .iter()
                .all(|l2| l2.verify(oracle_table, trans))
        });
        assert!(&result);

        let (result, time_verify_bound_sumcheck) = time!("verify bound sumcheck", {
            self.bound_iop.verify(oracle_table, trans)
        });
        assert!(result);

        let (result, time_verify_lookup_sumcheck) = time!("verify lookup sumcheck", {
            self.lookups
                .iter()
                .all(|lookup| lookup.verify(oracle_table, trans))
        });
        assert!(&result);

        let (result, time_verify_oracles) = time!("verify oracle", { oracle_table.verify(trans) });
        assert!(result);

        let time_verify_oracles = time_verify_oracles;
        let time_verify_sumcheck = time_verify_bound_sumcheck
            + time_verify_infairence_sumcheck
            + time_verify_l2_norm_sumcheck
            + time_verify_lookup_sumcheck
            + time_verify_spectral_norm_sumcheck;

        println!("\n== Verify Timing (in ms) ==");
        println!("Verify Oracles:     {} ms", time_verify_oracles);
        println!("Verify Sumcheck:    {} ms\n", time_verify_sumcheck);

        true
    }
}

use crate::piop::{BoundIOP, BoundInstance, LookupIOP};

use crate::pcs::OracleTable;

use algebra::{utils::Transcript, AbstractExtensionField, Field};
use pcs::utils::{
    code::{LinearCode, LinearCodeSpec},
    hash::Hash,
};

use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct BoundSnark<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub bound_iop: BoundIOP<F, EF, H, C, S>,
    pub lookups: Vec<LookupIOP<F, EF, H, C, S>>,
}

impl<F, EF, H, C, S> BoundSnark<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn prove(
        &mut self,
        instance: &mut BoundInstance<F>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) {
        // let lookup_instances = timing!("construct lookup instance", { instance.lookup_instances() });

        timing!("commit spectral norm oracles", {
            instance.construct_oracles(oracle_table);
        });

        // timing!("commit lookup first oracles", {
        //     lookup_instances
        //         .iter()
        //         .for_each(|lookup| lookup.construct_first_oracles(oracle_table));
        // });

        // let mut lookup_instances_ef: Vec<LookupInstance<EF, F, EF>> = lookup_instances
        //     .iter()
        //     .map(|lookup| lookup.to_ef())
        //     .collect();

        // timing!("commit lookup second oracles", {
        //     lookup_instances_ef
        //         .iter_mut()
        //         .for_each(|lookup| lookup.construct_second_oracle(oracle_table, transcript));
        // });

        let mut instance_ef = instance.to_ef();

        timing!("prove spectral norm sumcheck", {
            self.bound_iop
                .prove(&mut instance_ef, oracle_table, transcript);
        });

        // timing!("prove lookup sumcheck", {
        //     lookup_instances_ef.iter().for_each(|lookup_instance| {
        //         let mut lookup = LookupIOP::default();
        //         lookup.prove(lookup_instance, oracle_table, transcript);
        //         self.lookups.push(lookup);
        //     });
        // });

        timing!("prove oracles", {
            oracle_table.prove(transcript);
        });
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) -> bool {
        // self.lookups
        //     .iter()
        //     .all(|lookup| lookup.pre_randomness(transcript))
        //     &&
        timing!("verify spectral norm sumcheck", {
                self.bound_iop.verify(oracle_table, transcript)
            })
            // && timing!("verify lookup sumcheck", {
            //     self.lookups
            //         .iter()
            //         .all(|lookup| lookup.verify(oracle_table, transcript))
            // })
            && timing!("verify oracle", { oracle_table.verify(transcript) })
    }
}

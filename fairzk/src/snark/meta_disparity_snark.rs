use crate::piop::LookupInstance;
use crate::piop::{LookupIOP, MetaDisparityIOP, MetaDisparityInstance};

use crate::pcs::OracleTable;

use algebra::{utils::Transcript, AbstractExtensionField, Field, PrimeField};
use pcs::utils::{
    code::{LinearCode, LinearCodeSpec},
    hash::Hash,
};

use serde::{Deserialize, Serialize};

#[derive(Default, Serialize)]
pub struct MetaDisparitySnark<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub meta_disparity_iop: MetaDisparityIOP<F, EF, H, C, S>,
    pub lookups: Vec<LookupIOP<F, EF, H, C, S>>,
}

impl<F, EF, H, C, S> MetaDisparitySnark<F, EF, H, C, S>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn proof_size(&self) -> usize {
        let mut proof_size = 0;
        proof_size += bincode::serialize(&self.meta_disparity_iop).unwrap().len();
        self.lookups.iter().for_each(|x| {
            proof_size += bincode::serialize(x).unwrap().len();
        });
        proof_size
    }

    pub fn prove(
        &mut self,
        instance: &mut MetaDisparityInstance<F>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        let lookup_instances =
            timing!("construct lookup instance", { instance.lookup_instances() });

        timing!("commit meta disparity oracles", {
            instance.construct_oracles(oracle_table);
        });

        timing!("commit lookup first oracles", {
            lookup_instances
                .iter()
                .for_each(|lookup| lookup.construct_first_oracles(oracle_table));
        });

        let mut lookup_instances_ef: Vec<LookupInstance<EF, F, EF>> = lookup_instances
            .iter()
            .map(|lookup| lookup.to_ef())
            .collect();

        timing!("commit lookup second oracles", {
            lookup_instances_ef
                .iter_mut()
                .for_each(|lookup| lookup.construct_second_oracle(oracle_table, trans));
        });

        let mut instance_ef = instance.to_ef();

        timing!("prove meta disparity sumcheck", {
            self.meta_disparity_iop
                .prove(&mut instance_ef, oracle_table, trans);
        });

        timing!("prove lookup sumcheck", {
            lookup_instances_ef.iter().for_each(|lookup_instance| {
                let mut lookup = LookupIOP::default();
                lookup.prove(lookup_instance, oracle_table, trans);
                self.lookups.push(lookup);
            });
        });

        timing!("prove oracles", {
            oracle_table.prove(trans);
        });
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        self.lookups
            .iter()
            .all(|lookup| lookup.pre_randomness(trans))
            && timing!("verify infairence sumcheck", {
                self.meta_disparity_iop.verify(oracle_table, trans)
            })
            && timing!("verify lookup sumcheck", {
                self.lookups
                    .iter()
                    .all(|lookup| lookup.verify(oracle_table, trans))
            })
            && timing!("verify oracle", { oracle_table.verify(trans) })
    }
}

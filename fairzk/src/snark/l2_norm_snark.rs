use crate::piop::{L2NormIOP, L2NormInstance};

use crate::pcs::OracleTable;

use algebra::{utils::Transcript, AbstractExtensionField, PrimeField};
use pcs::utils::{
    code::{LinearCode, LinearCodeSpec},
    hash::Hash,
};

use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct L2NormSnark<F, EF, H, C, S>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub l2_norm_iop: L2NormIOP<F, EF, H, C, S>,
}

impl<F, EF, H, C, S> L2NormSnark<F, EF, H, C, S>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn prove(
        &mut self,
        instance: &mut L2NormInstance<F>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) {
        timing!("commit l2 norm oracles", {
            instance.construct_oracles(oracle_table);
        });

        let mut instance_ef = instance.to_ef();
        timing!("prove l2 norm sumcheck", {
            self.l2_norm_iop
                .prove(&mut instance_ef, oracle_table, transcript);
        });

        timing!("prove oracles", {
            oracle_table.prove(transcript);
        });
    }

    pub fn verify(
        &mut self,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) -> bool {
        timing!("verify l2 norm sumcheck", {
            self.l2_norm_iop.verify(oracle_table, transcript)
        }) && timing!("verify oracles", { oracle_table.verify(transcript) })
    }
}

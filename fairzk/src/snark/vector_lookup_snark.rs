use crate::piop::{vector_lookup::VectorLookupIOP, vector_lookup::VectorLookupInstance};

use crate::pcs::OracleTable;
use algebra::{utils::Transcript, AbstractExtensionField, Field, PrimeField};

use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};
use serde::{Deserialize, Serialize};

// SNARKs for lookup compiled with PCS
#[derive(Default)]
pub struct VectorLookupSnark<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    iop: VectorLookupIOP<F, EF, H, C, S>,
    _marker: std::marker::PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> VectorLookupSnark<F, EF, H, C, S>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn new() -> Self {
        Self {
            iop: VectorLookupIOP::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn prove(
        &mut self,
        instance: &mut VectorLookupInstance<F, F, EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        timing!(
            "commit first oracles",
            instance.construct_first_oracles(oracle_table)
        );

        let mut instance_ef = instance.to_ef();

        timing!(
            "commit second oracles",
            instance_ef.construct_second_oracle(oracle_table, trans)
        );

        timing!(
            "prove sumcheck",
            self.iop.prove(&mut instance_ef, oracle_table, trans)
        );

        timing!("prove oralce", oracle_table.prove(trans));
    }

    pub fn verify(
        &self,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        self.iop.pre_randomness(trans);

        timing!("verify sumcheck", self.iop.verify(oracle_table, trans))
            && timing!("verify oracle", oracle_table.verify(trans))
    }
}

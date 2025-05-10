use crate::pcs::OracleTable;
use crate::sumcheck::{MLSumcheck, SumcheckProof};
use crate::utils::{
    eval_identity_poly, field_to_i128, gen_identity_poly,
    i128_to_field, vec_to_named_poly,
};

use algebra::PrimeField;

use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials, MultilinearExtension,
};

use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};

use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, vec};

#[derive(Default)]
pub struct L2NormInstance<F>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
{
    // length of the vector
    pub num_vars_len: usize,

    // vector, quantization 2^f
    pub vector: Rc<DenseMultilinearExtension<F>>,
    // square of every element in the vector, 2^{2f}
    pub square: Rc<DenseMultilinearExtension<F>>,

    // l2 norm, quantization 2^f
    pub value: F,
    // error because of truncation
    // values^2 + error = \sum_h square(h)
    pub error: F,
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> L2NormInstance<F> {
    #[inline]
    pub fn new_essential(num_vars_len: usize, vector: &Rc<DenseMultilinearExtension<F>>) -> Self {
        assert!(vector.num_vars == num_vars_len);
        let vector_i128 = field_to_i128(&vector.evaluations);
        let square_i128: Vec<i128> = vector_i128.iter().map(|x| x * x).collect();
        let square_sum_i128 = square_i128.iter().sum::<i128>();
        // compute value = sqrt(square_sum)
        let value_i128 = (square_sum_i128 as f64).sqrt() as i128;
        let error_i128 = -value_i128 * value_i128 + square_sum_i128;
        Self {
            num_vars_len,
            vector: vector.clone(),
            square: vec_to_named_poly(
                &format!("square_of_{}", vector.name),
                &i128_to_field(&square_i128),
            ),
            value: i128_to_field(&vec![value_i128])[0],
            error: i128_to_field(&vec![error_i128])[0],
        }
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> L2NormInstance<F> {
    #[inline]
    pub fn new(
        num_vars_len: usize,
        vector: &Rc<DenseMultilinearExtension<F>>,
        square: &Rc<DenseMultilinearExtension<F>>,
        value: F,
        error: F,
    ) -> Self {
        debug_assert_eq!(vector.num_vars, num_vars_len);
        debug_assert_eq!(square.num_vars, num_vars_len);
        L2NormInstance {
            num_vars_len,
            vector: vector.clone(),
            square: square.clone(),
            value,
            error,
        }
    }

    // transform this instance on base field to an instance on extension field
    pub fn to_ef<EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>>(
        &self,
    ) -> L2NormInstance<EF> {
        L2NormInstance {
            num_vars_len: self.num_vars_len,
            vector: Rc::new(self.vector.to_ef()),
            square: Rc::new(self.square.to_ef()),
            value: EF::from(self.value),
            error: EF::from(self.error),
        }
    }

    pub fn construct_oracles<EF, H, C, S>(&self, table: &mut OracleTable<F, EF, H, C, S>)
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        table.add_named_oracle(self.vector.clone());
        table.add_named_oracle(self.square.clone());
    }
}

#[derive(Default, Serialize)]
pub struct L2NormIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub num_vars_len: usize,
    pub value: EF,
    pub error: EF,
    pub vector: String,
    pub square: String,
    pub sumcheck_squre: SumcheckProof<EF>,
    pub sumcheck_l2_norm: SumcheckProof<EF>,

    _marker: PhantomData<(F, H, C, S)>,
}

impl<F, EF, H, C, S> L2NormIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn info(&mut self, instance: &L2NormInstance<EF>) {
        self.num_vars_len = instance.num_vars_len;
        self.value = instance.value.clone();
        self.error = instance.error.clone();
        self.vector = instance.vector.name.clone();
        self.square = instance.square.name.clone();
    }

    pub fn prove(
        &mut self,
        instance: &mut L2NormInstance<EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) {
        self.info(&instance);

        // \sum_h eq(r, h) * vector(h) * vector(h) = square(r)
        // \sum_h square(h) = value * value + error
        // combined to \sum l0 * eq(r, h) * vector(h) * vector(h) + square(h) = l0 * square(r) + value * value + error

        let r0 = transcript
            .get_vec_challenge(b"random point in schwartz-zippel lemma", self.num_vars_len);
        let l0 = transcript.get_challenge(b"random combine");

        let eq_h_r0 = Rc::new(gen_identity_poly(&r0));

        let mut poly = ListOfProductsOfPolynomials::<EF>::new(self.num_vars_len);
        poly.add_product(
            vec![
                eq_h_r0.clone(),
                instance.vector.clone(),
                instance.vector.clone(),
            ],
            l0,
        );
        poly.add_product(vec![instance.square.clone()], EF::one());

        let (proof, state) =
            <MLSumcheck<EF>>::prove(transcript, &poly).expect("Proof generated in L2Norm");

        let sumcheck_point = state.randomness;

        self.sumcheck_squre = SumcheckProof {
            proof: proof,
            info: poly.info(),
            claimed_sum: l0 * instance.square.evaluate(&r0)
                + instance.value * instance.value
                + instance.error,
        };

        oracle_table.add_point(&instance.vector.name, &sumcheck_point);
        oracle_table.add_point(&instance.square.name, &sumcheck_point);
        oracle_table.add_point(&instance.square.name, &r0);
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) -> bool {
        let r0 = transcript
            .get_vec_challenge(b"random point in schwartz-zippel lemma", self.num_vars_len);
        let l0 = transcript.get_challenge(b"random combine");

        let sumcheck_oracle = <MLSumcheck<EF>>::verify(transcript, &self.sumcheck_squre).ok()
        .expect("Verification failed in L2Norm");
        

            let sumcheck_point = sumcheck_oracle.sumcheck_point;

            let eq_s_r0 = eval_identity_poly(&r0, &sumcheck_point);
            let square_r = oracle_table.get_eval(&self.square, &r0)[0];
            let square_s = oracle_table.get_eval(&self.square, &sumcheck_point)[0];
            let vector_s = oracle_table.get_eval(&self.vector, &sumcheck_point)[0];
            let mut verify_value =
                l0 * eq_s_r0 * vector_s * vector_s + square_s == sumcheck_oracle.oracle_eval;
            verify_value &=
                self.sumcheck_squre.claimed_sum == l0 * square_r + self.value * self.value + self.error;

            debug_assert!(verify_value, "Verification failed in L2Norm");

            verify_value
    }
}

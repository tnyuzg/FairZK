use super::LookupInstance;
use crate::pcs::OracleTable;
use crate::sumcheck::{MLSumcheck, SumcheckProof};
use crate::utils::{eval_identity_poly, field_range_tables, gen_identity_poly};
use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};

use crate::utils::{
    field_to_i128, i128_to_field, vec_to_named_poly,
};
use algebra::PrimeField;

use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials, MultilinearExtension,
};

use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, vec};

#[derive(Default)]
pub struct BoundInstance<F>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
{
    // number of layers
    pub num_vars_layers: usize,
    // qunatization 2^f
    pub error_range: F,
    pub power_2_f: F,

    // delta_h_lhs, with 2^{2f + lipschitz} quantization
    // delta_h layer 1 to L
    pub delta_h_lhs: Rc<DenseMultilinearExtension<F>>,

    // delta_h_rhs, with 2^{f} quantization
    // delta_h layer 0 to L-1
    // delta_h_lhs = delta_h_rhs * 2^f + delta_error
    pub delta_h_rhs: Rc<DenseMultilinearExtension<F>>,

    // eigen_values, with 2^{f} quantization
    // weight layer 0 to L-1
    pub eigen_values: Rc<DenseMultilinearExtension<F>>,

    // spectral norms, with 2^{f} quantization
    // spectral_norm ^2 = 2^f * \sum_h eigen_values(h) + spectral_error
    // these error should be small than 2^{f}
    pub spectral_norms: Rc<DenseMultilinearExtension<F>>,
    pub spectral_errors: Rc<DenseMultilinearExtension<F>>,

    // ||delta_z||_2, with 2^{f} quantization
    // delta_z layer 1 to L, from infairence inner product
    pub delta_z_l2_norms: Rc<DenseMultilinearExtension<F>>,

    // Lipschitz constant, which is also quantized
    // so to compensate, the lhs must be quantized by its quantization
    pub lipschitz_c: F,
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> BoundInstance<F> {
    #[inline]
    pub fn new_essential(
        num_vars_layers: usize,
        quantize: usize,
        eigen_values: &Rc<DenseMultilinearExtension<F>>,
        delta_z_l2_norms: &Rc<DenseMultilinearExtension<F>>,
        delta_0: F, // with quantization 2^{f + lipschitz_quantize}
        error_range: F,
        lipschitz_c: F, // quantized
    ) -> Self {
        assert!(num_vars_layers == eigen_values.num_vars);
        assert!(num_vars_layers == delta_z_l2_norms.num_vars);
        let layer_number = 1 << num_vars_layers as i128;
        let power_2_f = 1 << quantize as i128;

        let eigenvalues_i128 = field_to_i128(&eigen_values.evaluations);
        // compute square root
        let spectral_norms_i128: Vec<i128> = eigenvalues_i128
            .iter()
            .map(|x| ((*x * power_2_f) as f64).sqrt() as i128)
            .collect();
        let spectral_errors_i128: Vec<i128> = eigenvalues_i128
            .iter()
            .zip(spectral_norms_i128.iter())
            .map(|(x, y)| (*y) * (*y) - (*x) * power_2_f)
            .collect();

        // // manually check spectral_norms_i128 ^ 2 = 2^f * eigen_values_i128 + spectral_errors_i128
        // for i in 0..layer_number {
        //     let spectral_norm_i128 = spectral_norms_i128[i];
        //     let spectral_error_i128 = spectral_errors_i128[i];
        //     let eigen_value_i128 = eigenvalues_i128[i];
        //     let check = spectral_norm_i128 * spectral_norm_i128 == power_2_f * eigen_value_i128 + spectral_error_i128;
        //     assert!(check);
        // }

        // delta_h_lhs = lipschitz_c * spectral_norms * delta_h_rhs + 2 * lipschitz_c * delta_z_l2_norms * 2^f
        let mut delta_h_lhs_i128 = Vec::<i128>::new(); // 2^{2f + lipschitz_quantize}
        let mut delta_h_rhs_i128 = Vec::<i128>::new(); // 2^{f + lipschitz_quantize}
        delta_h_rhs_i128.push((delta_0.value().into() as u64) as i128);
        let lipschitz_i128 = (lipschitz_c.value().into() as u64) as i128;
        let delta_z_l2_norms_i128 = field_to_i128(&delta_z_l2_norms.evaluations);
        for i in 0..layer_number {
            delta_h_lhs_i128.push(
                lipschitz_i128 * spectral_norms_i128[i] * delta_h_rhs_i128[i]
                    + 2 * lipschitz_i128 * delta_z_l2_norms_i128[i] * power_2_f,
            );
            if i < layer_number - 1 {
                delta_h_rhs_i128.push(delta_h_lhs_i128[i] / power_2_f);
            }
        }

        // // manually check delta_h_lhs = lipschitz_c * spectral_norms * delta_h_rhs + 2 * lipschitz_c * delta_z_l2_norms * 2^f
        // for i in 0..layer_number {
        //     let delta_h_lhs_i128 = delta_h_lhs_i128[i];
        //     let delta_h_rhs_i128 = delta_h_rhs_i128[i];
        //     let spectral_norm_i128 = spectral_norms_i128[i];
        //     let delta_z_l2_norm_i128 = delta_z_l2_norms_i128[i];
        //     let check = delta_h_lhs_i128 == lipschitz_i128 * spectral_norm_i128 * delta_h_rhs_i128 + 2 * lipschitz_i128 * delta_z_l2_norm_i128 * power_2_f;
        //     assert!(check);
        // }


        Self {
            num_vars_layers,
            delta_h_lhs: vec_to_named_poly(
                &"delta_h_lhs".to_string(),
                &i128_to_field(&delta_h_lhs_i128),
            ),
            delta_h_rhs: vec_to_named_poly(
                &"delta_h_rhs".to_string(),
                &i128_to_field(&delta_h_rhs_i128),
            ),
            eigen_values: eigen_values.clone(),
            spectral_norms: vec_to_named_poly(
                &"spectral_norms".to_string(),
                &i128_to_field(&spectral_norms_i128),
            ),
            spectral_errors: vec_to_named_poly(
                &"spectral_errors".to_string(),
                &i128_to_field(&spectral_errors_i128),
            ),
            delta_z_l2_norms: delta_z_l2_norms.clone(),
            error_range,
            power_2_f: i128_to_field(&[power_2_f])[0],
            lipschitz_c,
        }
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> BoundInstance<F> {
    #[inline]
    pub fn new(
        num_vars_layers: usize,
        delta_h_lhs: &Rc<DenseMultilinearExtension<F>>,
        delta_h_rhs: &Rc<DenseMultilinearExtension<F>>,
        eigen_values: &Rc<DenseMultilinearExtension<F>>,
        spectral_norms: &Rc<DenseMultilinearExtension<F>>,
        spectral_errors: &Rc<DenseMultilinearExtension<F>>,
        delta_z_l2_norms: &Rc<DenseMultilinearExtension<F>>,
        error_range: F,
        power_2_f: F,
        lipschitz_c: F,
    ) -> Self {
        debug_assert_eq!(delta_h_lhs.num_vars, num_vars_layers);
        debug_assert_eq!(delta_h_rhs.num_vars, num_vars_layers);
        debug_assert_eq!(eigen_values.num_vars, num_vars_layers);
        debug_assert_eq!(spectral_norms.num_vars, num_vars_layers);
        debug_assert_eq!(spectral_errors.num_vars, num_vars_layers);
        debug_assert_eq!(delta_z_l2_norms.num_vars, num_vars_layers);
        BoundInstance {
            num_vars_layers,
            delta_h_lhs: delta_h_lhs.clone(),
            delta_h_rhs: delta_h_rhs.clone(),
            eigen_values: eigen_values.clone(),
            spectral_norms: spectral_norms.clone(),
            spectral_errors: spectral_errors.clone(),
            delta_z_l2_norms: delta_z_l2_norms.clone(),
            error_range,
            power_2_f,
            lipschitz_c,
        }
    }

    pub fn to_ef<EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>>(
        &self,
    ) -> BoundInstance<EF> {
        BoundInstance {
            num_vars_layers: self.num_vars_layers,
            error_range: EF::from(self.error_range),
            power_2_f: EF::from(self.power_2_f),
            delta_h_lhs: Rc::new(self.delta_h_lhs.to_ef()),
            delta_h_rhs: Rc::new(self.delta_h_rhs.to_ef()),
            eigen_values: Rc::new(self.eigen_values.to_ef()),
            spectral_norms: Rc::new(self.spectral_norms.to_ef()),
            spectral_errors: Rc::new(self.spectral_errors.to_ef()),
            delta_z_l2_norms: Rc::new(self.delta_z_l2_norms.to_ef()),
            lipschitz_c: EF::from(self.lipschitz_c),
        }
    }

    pub fn construct_oracles<EF, H, C, S>(&self, table: &mut OracleTable<F, EF, H, C, S>)
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        table.add_named_oracle(self.delta_h_lhs.clone());
        table.add_named_oracle(self.delta_h_rhs.clone());
        table.add_named_oracle(self.eigen_values.clone());
        table.add_named_oracle(self.spectral_norms.clone());
        table.add_named_oracle(self.spectral_errors.clone());
        table.add_named_oracle(self.delta_z_l2_norms.clone());
    }
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> BoundInstance<F> {
    pub fn lookup_instances<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let tables_error =
            field_range_tables(self.num_vars_layers, -self.error_range, self.error_range);
        let rangecheck_error = LookupInstance::new(
            &vec![self.spectral_errors.clone()],
            &tables_error,
            "error_range bound for bound bound".to_string(),
            "error_range".to_string(),
            2,
        );

        vec![rangecheck_error]
    }
}

#[derive(Default, Serialize)]
pub struct BoundIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub num_vars_layers: usize,
    pub error_range: EF,
    pub power_2_f: EF,
    pub delta_h_lhs: String,
    pub delta_h_rhs: String,
    pub eigen_values: String,
    pub spectral_norms: String,
    pub spectral_errors: String,
    pub delta_z_l2_norms: String,
    pub lipschitz_c: EF,

    pub sumcheck: SumcheckProof<EF>,

    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> BoundIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn info(&mut self, instance: &BoundInstance<EF>) {
        self.num_vars_layers = instance.num_vars_layers;
        self.error_range = instance.error_range;
        self.power_2_f = instance.power_2_f.clone();
        self.delta_h_lhs = instance.delta_h_lhs.name.clone();
        self.delta_h_rhs = instance.delta_h_rhs.name.clone();
        self.eigen_values = instance.eigen_values.name.clone();
        self.spectral_norms = instance.spectral_norms.name.clone();
        self.spectral_errors = instance.spectral_errors.name.clone();
        self.delta_z_l2_norms = instance.delta_z_l2_norms.name.clone();
        self.lipschitz_c = instance.lipschitz_c.clone();
    }

    pub fn prove(
        &mut self,
        instance: &mut BoundInstance<EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) {
        self.info(&instance);

        // spectral_norms(i) * spectral_norms(i) = 2^f * eigen_values(i) + spectral_errors(i) for all i
        // delta_h_lhs(i) = lipschitz_c * spectral_norms(i) * delta_h_rhs(i) + 2 * lipschitz_c * delta_z_l2_norms(i) * 2^f for all i
        // \sum_h eq(r, h) * spectral_norms(h) * spectral_norms(h) = 2^f * eigen_values(r) + spectral_errors(r)
        // \sum_h lipschitz_c * eq(r, h) * delta_h_rhs(h) * spectral_norms(h) + 2 * lipschitz_c * eq(r, h) * delta_z_l2_norms(h) * 2^f = delta_h_lhs(r)
        // combined with random linear l0

        let r0 = transcript.get_vec_challenge(
            b"random point in schwartz-zippel lemma",
            self.num_vars_layers,
        );
        let l0 = transcript.get_challenge(b"random combine");

        let eq_h_r0 = Rc::new(gen_identity_poly(&r0));
        let mut poly = ListOfProductsOfPolynomials::<EF>::new(self.num_vars_layers);
        poly.add_product(
            vec![
                eq_h_r0.clone(),
                instance.spectral_norms.clone(),
                instance.spectral_norms.clone(),
            ],
            l0,
        );
        poly.add_product(
            vec![
                eq_h_r0.clone(),
                instance.delta_h_rhs.clone(),
                instance.spectral_norms.clone(),
            ],
            instance.lipschitz_c,
        );
        poly.add_product(
            vec![eq_h_r0.clone(), instance.delta_z_l2_norms.clone()],
            instance.lipschitz_c * instance.power_2_f,
        );
        poly.add_product(
            vec![eq_h_r0.clone(), instance.delta_z_l2_norms.clone()],
            instance.lipschitz_c * instance.power_2_f,
        );

        let (proof, state) =
            <MLSumcheck<EF>>::prove(transcript, &poly).expect("Proof generated in L2Norm");
        let sumcheck_point = state.randomness;

        self.sumcheck = SumcheckProof {
            proof,
            info: poly.info(),
            claimed_sum: l0
                * (instance.eigen_values.evaluate(&r0) * (instance.power_2_f)
                    + instance.spectral_errors.evaluate(&r0))
                + instance.delta_h_lhs.evaluate(&r0),
            // claimed_sum: l0 * (instance.eigen_values.evaluate(&r0) * (instance.power_2_f) + instance.spectral_errors.evaluate(&r0)),
            // claimed_sum: instance.delta_h_lhs.evaluate(&r0),
        };

        oracle_table.add_point(&instance.spectral_norms.name, &sumcheck_point);
        oracle_table.add_point(&instance.delta_h_rhs.name, &sumcheck_point);
        oracle_table.add_point(&instance.delta_z_l2_norms.name, &sumcheck_point);
        oracle_table.add_point(&instance.eigen_values.name, &r0);
        oracle_table.add_point(&instance.spectral_errors.name, &r0);
        oracle_table.add_point(&instance.delta_h_lhs.name, &r0);
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        transcript: &mut Transcript<EF>,
    ) -> bool {
        let r0 = transcript.get_vec_challenge(
            b"random point in schwartz-zippel lemma",
            self.num_vars_layers,
        );
        let l0 = transcript.get_challenge(b"random combine");

        let sumcheck_oracle = <MLSumcheck<EF>>::verify(transcript, &self.sumcheck)
            .expect("Verify sumcheck in BoundIOP");

        let sumcheck_point = sumcheck_oracle.sumcheck_point;
        let spectral_norms_s = oracle_table.get_eval(&self.spectral_norms, &sumcheck_point)[0];
        let delta_h_rhs_s = oracle_table.get_eval(&self.delta_h_rhs, &sumcheck_point)[0];
        let delta_z_l2_norms_s = oracle_table.get_eval(&self.delta_z_l2_norms, &sumcheck_point)[0];
        let eigen_values_r = oracle_table.get_eval(&self.eigen_values, &r0)[0];
        let spectral_error_r = oracle_table.get_eval(&self.spectral_errors, &r0)[0];
        let delta_h_lhs_r = oracle_table.get_eval(&self.delta_h_lhs, &r0)[0];
        let eq_s_r0 = eval_identity_poly(&r0, &sumcheck_point);

        let mut verify_sumcheck = l0 * eq_s_r0 * spectral_norms_s * spectral_norms_s
            + self.lipschitz_c * eq_s_r0 * delta_h_rhs_s * spectral_norms_s
            + self.lipschitz_c * eq_s_r0 * delta_z_l2_norms_s * self.power_2_f
            + self.lipschitz_c * eq_s_r0 * delta_z_l2_norms_s * self.power_2_f
            == sumcheck_oracle.oracle_eval;
        verify_sumcheck &= l0 * (eigen_values_r * self.power_2_f + spectral_error_r)
            + delta_h_lhs_r
            == self.sumcheck.claimed_sum;

        debug_assert!(verify_sumcheck);

        verify_sumcheck
    }
}

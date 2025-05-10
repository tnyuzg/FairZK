use super::{LookupInstance, VectorLookupInstance};
use crate::pcs::OracleTable;
use crate::sumcheck::{MLSumcheck, SumcheckProof};
use crate::utils::{
    absolute_range_tables, compute_a_b, concat_slices, field_to_i128, i128_to_field, is_valid,
    relu_i128, vec_to_named_poly, vec_to_polys,
};
use algebra::PrimeField;
use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};

use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials, MultilinearExtension,
};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, vec};

// x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
pub struct InferenceInstance<F: Field + Serialize + for<'de> Deserialize<'de>> {
    pub num_layers: usize,
    pub num_vars_data: usize,
    pub num_vars_feats: Vec<usize>,
    pub i_bit: usize,
    pub f_bit: usize,
    pub w: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub x: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub y: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> InferenceInstance<F> {
    // x w y | x w y | x w y | x w y | x
    #[inline]
    pub fn new_essential(
        num_layers: usize,
        num_vars_data: usize,
        num_vars_feats: &[usize],
        i_bit: usize,
        f_bit: usize,
        data: &Rc<DenseMultilinearExtension<F>>,
        w: &[Rc<DenseMultilinearExtension<F>>],
    ) -> Self {
        debug_assert_eq!(w.len(), num_layers);
        debug_assert_eq!(num_vars_feats.len(), num_layers + 1);
        debug_assert!(data.num_vars == num_vars_data + num_vars_feats[0]);
        for i in 0..num_layers {
            debug_assert_eq!(w[i].num_vars, num_vars_feats[i] + num_vars_feats[i + 1]);
        }

        data.iter().all(|x| is_valid(*x, i_bit, f_bit));
        w.iter()
            .flat_map(|w| w.iter())
            .all(|x| is_valid(*x, i_bit, f_bit));

        let num_data = 1 << num_vars_data;

        let mut x: Vec<DMatrix<i128>> = Vec::with_capacity(num_layers); // let mut x: Vec<DMatrix<i128>> = Vec::with_capacity(num_layers+1);

        let mut y: Vec<DMatrix<i128>> = Vec::with_capacity(num_layers);

        let data_i128 = DMatrix::from_column_slice(
            num_data,
            1 << num_vars_feats[0],
            &field_to_i128(&data.evaluations),
        );
        x.push(data_i128.clone());

        for i in 0..num_layers {
            let num_row = 1 << num_vars_feats[i];
            let num_col = 1 << num_vars_feats[i + 1];
            let w_matrix =
                DMatrix::from_column_slice(num_row, num_col, &field_to_i128(&w[i].evaluations));
            let x_matrix = x[i].clone();
            let y_matrix = x_matrix.clone() * w_matrix;

            y.push(y_matrix.clone());

            let next_x_matrix = y_matrix.map(|y| relu_i128(y, i_bit, f_bit));
            x.push(next_x_matrix);
        }

        let x = x
            .iter()
            .enumerate()
            .map(|(i, x_matrix)| {
                vec_to_named_poly(
                    &format!("x {} inferred", i),
                    &i128_to_field(&x_matrix.as_slice().to_vec()),
                )
            })
            .collect::<Vec<_>>();

        let y = y
            .iter()
            .enumerate()
            .map(|(i, y_matrix)| {
                vec_to_named_poly(
                    &format!("y {} inferred", i),
                    &i128_to_field(&y_matrix.as_slice().to_vec()),
                )
            })
            .collect::<Vec<_>>();

        debug_assert!((0..num_layers).all(|i| {
            let product = compute_a_b(
                &x[i].evaluations,
                &w[i].evaluations,
                1 << (num_vars_data),
                1 << num_vars_feats[i],
                1 << (num_vars_feats[i + 1]),
            );
            assert!(product == y[i].evaluations);
            true
        }));

        debug_assert!(x.len() == num_layers + 1);
        debug_assert!(y.len() == num_layers);
        debug_assert!(x
            .iter()
            .enumerate()
            .all(|(i, x)| x.num_vars == num_vars_data + num_vars_feats[i]));
        debug_assert!(y
            .iter()
            .enumerate()
            .all(|(i, y)| y.num_vars == num_vars_data + num_vars_feats[i + 1]));

        Self {
            num_layers,
            num_vars_data,
            num_vars_feats: num_vars_feats.to_vec(),
            i_bit,
            f_bit,
            w: w.to_vec(),
            x,
            y,
        }
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> InferenceInstance<F> {
    pub fn to_ef<EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>>(
        &self,
    ) -> InferenceInstance<EF> {
        InferenceInstance::<EF> {
            num_layers: self.num_layers,
            num_vars_data: self.num_vars_data,
            num_vars_feats: self.num_vars_feats.clone(),
            i_bit: self.i_bit,
            f_bit: self.f_bit,
            w: self.w.iter().map(|w| Rc::new(w.to_ef())).collect(),
            x: self.x.iter().map(|x| Rc::new(x.to_ef())).collect(),
            y: self.y.iter().map(|y| Rc::new(y.to_ef())).collect(),
        }
    }

    // oracles: x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
    pub fn construct_oracles<EF, H, C, S>(&self, table: &mut OracleTable<F, EF, H, C, S>)
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        for i in 0..self.num_layers {
            table.add_oracle(&self.x[i].name, self.x[i].clone());
            table.add_oracle(&self.w[i].name, self.w[i].clone());
            table.add_oracle(&self.y[i].name, self.y[i].clone());
        }
    }
}
impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> InferenceInstance<F> {
    pub fn lookup_instances<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let mut lookup_instances = Vec::new();
        lookup_instances.push(self.rangecheck_data());
        lookup_instances.extend(self.rangecheck_w());
        lookup_instances
    }

    pub fn vector_lookup_instances<EF>(&self) -> Vec<VectorLookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        self.lookup_xy()
    }

    pub fn rangecheck_data<EF>(&self) -> LookupInstance<F, F, EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let block_size = 1;
        let range = 1 << (self.i_bit + self.f_bit);
        let num_vars = self.num_vars_data + self.num_vars_feats[0];
        let t = absolute_range_tables(num_vars, range);
        LookupInstance::new(
            &[self.x[0].clone()],
            &t,
            format!("rangecheck {}", self.x[0].name),
            format!(
                "rangecheck ({}, {}) num vars {}",
                1 - range as i128,
                range,
                num_vars
            ),
            block_size,
        )
    }

    pub fn rangecheck_w<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let block_size = 1;
        let range = 1 << (self.i_bit + self.f_bit);
        let mut lookups = Vec::with_capacity(self.num_layers);

        for i in 0..self.num_layers {
            let num_vars = self.num_vars_feats[i] + self.num_vars_feats[i + 1];
            let f = &[self.w[i].clone()];
            let t = absolute_range_tables(num_vars, range);
            lookups.push(LookupInstance::new(
                f,
                &t,
                format!("rangecheck {}", self.w[i].name),
                format!("fraction bit num vars {} {}", num_vars, self.f_bit),
                block_size,
            ));
        }
        lookups
    }

    pub fn lookup_xy<EF>(&self) -> Vec<VectorLookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let block_size = 1;

        let mut lookups = Vec::with_capacity(self.num_layers - 1);

        let q_domain = self.i_bit + 2 * self.f_bit;

        for i in 0..self.num_layers {
            let num_vars = self.num_vars_data + self.num_vars_feats[i + 1];
            let mut domain: Vec<i128> =
                (-(1 << q_domain) + 1..(1 << q_domain)).collect::<Vec<i128>>();

            let poly_size = 1 << num_vars;
            let pad_size = (poly_size - (domain.len() % poly_size)) % poly_size;
            domain.resize(domain.len() + pad_size, 0);

            let image: Vec<i128> = domain
                .iter()
                .map(|&x| relu_i128(x, self.i_bit, self.f_bit))
                .collect();

            let domain = i128_to_field(&domain);
            let image = &i128_to_field(&image);

            let tx_vec = vec_to_polys(num_vars, &domain);
            let ty_vec = vec_to_polys(num_vars, &image);

            debug_assert_eq!(self.y[i].num_vars, num_vars);
            debug_assert_eq!(self.x[i + 1].num_vars, num_vars);

            lookups.push(VectorLookupInstance::new(
                num_vars,
                &vec![self.y[i].clone()],
                &vec![self.x[i + 1].clone()],
                &tx_vec,
                &ty_vec,
                &format!("relu from {} to {} ", self.y[i].name, self.x[i + 1].name),
                &format!("domain relu x{}", num_vars),
                &format!("image relu y{}", num_vars),
                block_size,
            ));
        }
        lookups
    }
}

#[derive(Default, Serialize)]
pub struct InferenceIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub num_layers: usize,
    pub num_vars_data: usize,
    pub num_vars_feats: Vec<usize>,
    pub i: usize,
    pub f: usize,
    pub w: Vec<String>,
    pub x: Vec<String>,
    pub y: Vec<String>,

    pub sumcheck_proofs: Vec<SumcheckProof<EF>>,
    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> InferenceIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn proof_size(&self) -> usize {
        bincode::serialize(&self).unwrap().len()
    }

    pub fn info(&mut self, instance: &InferenceInstance<EF>) {
        self.num_layers = instance.num_layers;
        self.num_vars_data = instance.num_vars_data;
        self.num_vars_feats = instance.num_vars_feats.clone();
        self.i = instance.i_bit;
        self.f = instance.f_bit;
        self.w = instance.w.iter().map(|w| w.name.clone()).collect();
        self.x = instance.x.iter().map(|x| x.name.clone()).collect();
        self.y = instance.y.iter().map(|y| y.name.clone()).collect();
    }

    // oracles: x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
    // relations:  \sum_h x(r_l, h) wi(h, r_r) = yi(r_l, r_r)
    pub fn prove(
        &mut self,
        instance: &mut InferenceInstance<EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        self.info(&instance);
        let mut sumcheck_proofs = Vec::with_capacity(instance.num_layers);

        for i in 0..instance.num_layers {
            let num_vars_l = instance.num_vars_data;
            let num_vars_m = instance.num_vars_feats[i];
            let num_vars_r = instance.num_vars_feats[i + 1];

            let random_point_l =
                trans.get_vec_challenge(b"random point in schwartz-zippel lemma", num_vars_l);

            let random_point_r =
                trans.get_vec_challenge(b"random point in schwartz-zippel lemma", num_vars_r);

            let mut poly = ListOfProductsOfPolynomials::<EF>::new(num_vars_m);

            let w_h_r = Rc::new(instance.w[i].fix_variables_back(&random_point_r));
            let x_l_h = Rc::new(instance.x[i].fix_variables_front(&random_point_l));

            let product = vec![x_l_h, w_h_r];
            poly.add_product(product, EF::one());

            let (sumcheck_proof, sumcheck_state) =
                MLSumcheck::prove(trans, &poly).expect("fail to prove the sumcheck protocol");

            let sumcheck_point = sumcheck_state.randomness;

            let sumcheck_proof = SumcheckProof {
                proof: sumcheck_proof,
                info: poly.info(),
                claimed_sum: instance.y[i]
                    .evaluate(&concat_slices(&[&random_point_l, &random_point_r])),
            };

            oracle_table.add_point(
                &self.x[i],
                &concat_slices(&[&random_point_l, &sumcheck_point]),
            );

            oracle_table.add_point(
                &self.w[i],
                &concat_slices(&[&sumcheck_point, &random_point_r]),
            );

            oracle_table.add_point(
                &self.y[i],
                &concat_slices(&[&random_point_l, &random_point_r]),
            );

            sumcheck_proofs.push(sumcheck_proof);
        }

        self.sumcheck_proofs = sumcheck_proofs;
    }

    // oracles: x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
    // relations: yi(r) = \sum_h wi(r, h) xi(h)
    //            yi(r) = 2^f x{i+1}(r) + ri(r)
    // get randomness
    // verify the sumcheck protocol
    // verify sumcheck.final_sum = wi(r,s) xi(s)
    // verify yi(r) = 2^f x{i+1}(r) + ri(r)
    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        let sumcheck_proofs = &self.sumcheck_proofs;

        (0..self.num_layers).all(|i| {
            let num_vars_l = self.num_vars_data;
            let num_vars_r = self.num_vars_feats[i + 1];

            let random_point_l =
                trans.get_vec_challenge(b"random point in schwartz-zippel lemma", num_vars_l);

            let random_point_r =
                trans.get_vec_challenge(b"random point in schwartz-zippel lemma", num_vars_r);

            let subclaim = MLSumcheck::verify(trans, &sumcheck_proofs[i])
                .expect("fail to verify the course of sumcheck protocol in inference");

            let sumcheck_point = subclaim.sumcheck_point;

            let xi_s = oracle_table.get_eval(
                &self.x[i],
                &concat_slices(&[&random_point_l, &sumcheck_point]),
            )[0];
            let wi_rs = oracle_table.get_eval(
                &self.w[i],
                &concat_slices(&[&sumcheck_point, &random_point_r]),
            )[0];
            let yi_r = oracle_table.get_eval(
                &self.y[i],
                &concat_slices(&[&random_point_l, &random_point_r]),
            )[0];

            debug_assert!(subclaim.oracle_eval == wi_rs * xi_s);
            debug_assert!(sumcheck_proofs[i].claimed_sum == yi_r);

            (subclaim.oracle_eval == wi_rs * xi_s) && sumcheck_proofs[i].claimed_sum == yi_r
        })
    }
}

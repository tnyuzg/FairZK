use super::LookupInstance;
use crate::pcs::OracleTable;
use crate::sumcheck::{MLSumcheck, SumcheckProof};
use crate::utils::{
    concat_slices, field_to_i128, field_to_u64, mat_vec_mul_column_major, range_tables,
    u64_to_field, vec_to_named_poly,
};
use algebra::PrimeField;
use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};

use algebra::{
    utils::Transcript, AbstractExtensionField, AsFrom, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials, MultilinearExtension,
};

use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, vec};

// x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
pub struct InfairenceInstance<F: Field + Serialize + for<'de> Deserialize<'de>> {
    pub num_layers: usize,
    pub num_vars_feats: Vec<usize>,
    pub i_bit: usize,
    pub f_bit: usize,
    pub w_abs: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub x: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub y: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub r: Vec<Rc<DenseMultilinearExtension<F>>>,
}

fn abs_values(input: Vec<i128>) -> Vec<u64> {
    input.into_iter().map(|x| x.abs() as u64).collect()
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> InfairenceInstance<F> {
    #[inline]
    pub fn new_overflow(
        num_layers: usize,
        num_vars_feats: &[usize],
        i_bit: usize,
        f_bit: usize,
        max_deviation: &Rc<DenseMultilinearExtension<F>>,
        w: &[Rc<DenseMultilinearExtension<F>>],
    ) -> Self {
        debug_assert_eq!(w.len(), num_layers);
        debug_assert_eq!(num_vars_feats.len(), num_layers + 1);

        debug_assert_eq!(max_deviation.num_vars, num_vars_feats[0]);
        for i in 0..num_layers {
            debug_assert_eq!(w[i].num_vars, num_vars_feats[i] + num_vars_feats[i + 1]);
        }

        let mut x: Vec<Vec<u64>> = Vec::with_capacity(num_layers);
        let mut y: Vec<Vec<u64>> = Vec::with_capacity(num_layers);
        let mut r: Vec<Vec<u64>> = Vec::with_capacity(num_layers - 1);

        let w_abs_u64: Vec<Vec<u64>> = w
            .iter()
            .map(|w| abs_values(field_to_i128(&w.evaluations)))
            .collect::<Vec<_>>();
        let x_0 = field_to_u64(&max_deviation.evaluations);

        x.push(x_0);

        for (i, w) in w_abs_u64.iter().enumerate() {
            let y_layer = mat_vec_mul_column_major(&w, &x[i]);
            y.push(y_layer.clone());

            if i == num_layers - 1 {
                break;
            }

            let x_next: Vec<u64> = y_layer.iter().map(|&val| val >> f_bit).collect();
            let r_next: Vec<u64> = y_layer
                .iter()
                .zip(&x_next)
                .map(|(&y_val, &x_val)| y_val - (x_val << f_bit))
                .collect();

            x.push(x_next);
            r.push(r_next);
        }

        let w_abs = w_abs_u64
            .iter()
            .zip(w.iter())
            .map(|(vec_u64, w)| {
                vec_to_named_poly(&format!("absolute of {}", w.name), &u64_to_field(&vec_u64))
            })
            .collect::<Vec<_>>();
        let x = x
            .iter()
            .enumerate()
            .map(|(i, vec_u64)| {
                vec_to_named_poly(
                    &format!("x infaired disparity at layer {}", i),
                    &u64_to_field(&vec_u64),
                )
            })
            .collect::<Vec<_>>();
        let y = y
            .iter()
            .enumerate()
            .map(|(i, vec_u64)| {
                vec_to_named_poly(
                    &format!("y infaired disparity at layer {}", i),
                    &u64_to_field(&vec_u64),
                )
            })
            .collect::<Vec<_>>();
        let r = r
            .iter()
            .enumerate()
            .map(|(i, vec_u64)| {
                vec_to_named_poly(
                    &format!("r infaired disparity at layer {}", i),
                    &u64_to_field(&vec_u64),
                )
            })
            .collect::<Vec<_>>();

        Self {
            num_layers,
            num_vars_feats: num_vars_feats.to_vec(),
            i_bit,
            f_bit,
            w_abs,
            x,
            y,
            r,
        }
    }

}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> InfairenceInstance<F> {
    pub fn lookups_instances<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let mut lookup_instances = Vec::new();
        lookup_instances.extend(self.rangecheck_r());
        lookup_instances.extend(self.rangecheck_w_abs());
        lookup_instances
    }

    pub fn rangecheck_r<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let block_size = 1;
        let range = 1 << self.f_bit;
        let mut lookups = Vec::with_capacity(self.num_layers - 1);

        for i in 0..self.num_layers - 1 {
            let num_vars = self.num_vars_feats[i + 1];
            let tables = range_tables(num_vars, range);
            let t_name = format!(
                "t [infair rangecheck r ({}, {}) num vars {}",
                0, range, num_vars
            );
            lookups.push(LookupInstance::new(
                &[self.r[i].clone()],
                &tables,
                format!(
                    "infair rangecheck r {} in {}",
                    self.r[i].name,
                    t_name.clone()
                ),
                t_name,
                block_size,
            ));
        }
        lookups
    }

    pub fn rangecheck_w_abs<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let block_size = 1;
        let range = 1 << (self.i_bit + self.f_bit);
        let mut lookups = Vec::with_capacity(self.num_layers);

        for i in 0..self.num_layers {
            let num_vars = self.num_vars_feats[i] + self.num_vars_feats[i + 1];
            let f = &[self.w_abs[i].clone()];
            let tables = range_tables(num_vars, range);
            let t_name = format!(
                "t [infairence w_abs rangecheck ({}, {}) num vars {}",
                0, range, num_vars
            );
            lookups.push(LookupInstance::new(
                f,
                &tables,
                format!(
                    "infairence w_abs rangecheck {} in {}",
                    self.w_abs[i].name,
                    t_name.clone()
                ),
                t_name,
                block_size,
            ));
        }
        lookups
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> InfairenceInstance<F> {
    #[inline]
    pub fn new_from_witness(
        num_layers: usize,
        num_vars_feat: Vec<usize>,
        f_bit: usize,
        i_bit: usize,
        x: Vec<Rc<DenseMultilinearExtension<F>>>,
        w: Vec<Rc<DenseMultilinearExtension<F>>>,
        y: Vec<Rc<DenseMultilinearExtension<F>>>,
        r: Vec<Rc<DenseMultilinearExtension<F>>>,
    ) -> Self {
        debug_assert_eq!(x.len(), num_layers);
        debug_assert_eq!(w.len(), num_layers);
        debug_assert_eq!(y.len(), num_layers);
        debug_assert_eq!(r.len(), num_layers - 1);
        debug_assert_eq!(num_vars_feat.len(), num_layers + 1);

        for i in 0..num_layers {
            debug_assert_eq!(x[i].num_vars, num_vars_feat[i]);
            if i != num_layers - 1 {
                debug_assert_eq!(r[i].num_vars, num_vars_feat[i + 1]);
            }
            debug_assert_eq!(w[i].num_vars, num_vars_feat[i] + num_vars_feat[i + 1]);
            debug_assert_eq!(y[i].num_vars, num_vars_feat[i + 1]);
        }

        InfairenceInstance {
            num_layers,
            num_vars_feats: num_vars_feat,
            i_bit,
            f_bit,
            x,
            w_abs: w,
            y,
            r,
        }
    }

    pub fn to_ef<EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>>(
        &self,
    ) -> InfairenceInstance<EF> {
        InfairenceInstance::<EF> {
            num_layers: self.num_layers,
            num_vars_feats: self.num_vars_feats.clone(),
            i_bit: self.i_bit,
            f_bit: self.f_bit,
            w_abs: self.w_abs.iter().map(|w| Rc::new(w.to_ef())).collect(),
            x: self.x.iter().map(|x| Rc::new(x.to_ef())).collect(),
            y: self.y.iter().map(|y| Rc::new(y.to_ef())).collect(),
            r: self.r.iter().map(|r| Rc::new(r.to_ef())).collect(),
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
            table.add_oracle(&self.w_abs[i].name, self.w_abs[i].clone());
            table.add_oracle(&self.y[i].name, self.y[i].clone());
            if i != self.num_layers - 1 {
                table.add_oracle(&self.r[i].name, self.r[i].clone());
            }
        }
    }
}

#[derive(Default, Serialize)]
pub struct InfairenceIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub num_layers: usize,
    pub num_vars_feats: Vec<usize>,
    pub f: usize,
    pub w_abs: Vec<String>,
    pub x: Vec<String>,
    pub y: Vec<String>,
    pub r: Vec<String>,

    pub sumcheck_proofs: Vec<SumcheckProof<EF>>,
    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> InfairenceIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn info(&mut self, instance: &InfairenceInstance<EF>) {
        self.num_layers = instance.num_layers;
        self.num_vars_feats = instance.num_vars_feats.clone();
        self.f = instance.f_bit;
        self.w_abs = instance.w_abs.iter().map(|w| w.name.clone()).collect();
        self.x = instance.x.iter().map(|x| x.name.clone()).collect();
        self.y = instance.y.iter().map(|y| y.name.clone()).collect();
        self.r = instance.r.iter().map(|r| r.name.clone()).collect();
    }

    // oracles: x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
    // relations: yi(r) = \sum_h wi(r, h) xi(h)
    //            yi(r) = 2^f x{i+1}(r) + ri(r)
    pub fn prove(
        &mut self,
        instance: &mut InfairenceInstance<EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        self.info(&instance);
        let mut sumcheck_proofs = Vec::with_capacity(instance.num_layers);
        //dbg!(&instance.num_vars_feats);
        for i in 0..instance.num_layers {
            let num_vars_row = instance.num_vars_feats[i + 1];
            let num_vars_col = instance.num_vars_feats[i];

            let random_point =
                trans.get_vec_challenge(b"random point in schwartz-zippel lemma", num_vars_row);

            let mut poly = ListOfProductsOfPolynomials::<EF>::new(num_vars_col);

            let w_r = Rc::new(instance.w_abs[i].fix_variables_front(&random_point));
            let product = vec![w_r, instance.x[i].clone()];
            poly.add_product(product, EF::one());

            let (sumcheck_proof, sumcheck_state) =
                MLSumcheck::prove(trans, &poly).expect("fail to prove the sumcheck protocol");

            let sumcheck_point = sumcheck_state.randomness;

            let sumcheck_proof = SumcheckProof {
                proof: sumcheck_proof,
                info: poly.info(),
                claimed_sum: instance.y[i].evaluate(&random_point),
            };

            oracle_table.add_point(&self.x[i], &sumcheck_point);

            oracle_table.add_point(
                &self.w_abs[i],
                &concat_slices(&[&random_point, &sumcheck_point]),
            );

            oracle_table.add_point(&self.y[i], &random_point);

            if i != instance.num_layers - 1 {
                oracle_table.add_point(&self.r[i], &random_point);
                oracle_table.add_point(&self.x[i + 1], &random_point);
            }

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
            // num row col for y
            let num_vars_row = self.num_vars_feats[i + 1];

            let random_point =
                trans.get_vec_challenge(b"random point in schwartz-zippel lemma", num_vars_row);

            let subclaim = MLSumcheck::verify(trans, &sumcheck_proofs[i])
                .expect("fail to verify the course of sumcheck protocol in infairence");

            let sumcheck_point = subclaim.sumcheck_point;

            let xi_s = oracle_table.get_eval(&self.x[i], &sumcheck_point)[0];
            let wi_rs = oracle_table.get_eval(
                &self.w_abs[i],
                &concat_slices(&[&random_point, &sumcheck_point]),
            )[0];
            let yi_r = oracle_table.get_eval(&self.y[i], &random_point)[0];

            assert!(subclaim.oracle_eval == wi_rs * xi_s);
            assert!(sumcheck_proofs[i].claimed_sum == yi_r);

            if i != self.num_layers - 1 {
                let xi1_r = oracle_table.get_eval(&self.x[i + 1], &random_point)[0];
                let ri_r = oracle_table.get_eval(&self.r[i], &random_point)[0];
                yi_r == xi1_r * EF::new(<EF as Field>::Value::as_from((1 << self.f) as f64)) + ri_r
            } else {
                true
            }
        })
    }
}

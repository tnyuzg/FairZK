//! PIOP for range check
//! The prover wants to convince that lookups f are all in range
//!
//! <==> \forall x \in H_f, \forall i \in [lookup_num], f_i(x) \in [range]
//!
//! <==> \forall x in H_f, \forall i \in [lookup_num], f_i(x) \in {t(x) | x \in H_t} := {0, 1, 2, ..., range - 1}  
//!      where |H_f| is the size of one lookup and |H_t| is the size of table / range
//!
//! <==> \exists m s.t. \forall y, \sum_{i} \sum_{x \in H_f} 1 / f_i(x) - y = \sum_{x \in H_t} m(x) / t(x) - y
//!
//! <==> \sum_{i} \sum_{x \in H_f} 1 / f_i(x) - r = \sum_{x \in H_t} m(x) / t(x) - r
//!      where r is a random challenge from verifier (a single random element since y is a single variable)
//!
//! <==> \sum_{x \in H_f} \sum_{i \in [block_num]} h_i(x) = \sum_{x \in H_t} h_t(x)
//!      \forall i \in [block_num] \forall x \in H_f, h(x) * \prod_{j \in [block_size]}(f_j(x) - r) = \sum_{i \in [block_size]} \prod_{j \in [block_size], j != i} (f_j(x) - r)
//!      \forall x \in H_t, h_t(x) * (t(x) - r) = m(x)
//!
//! <==> \sum_{x \in H_f} \sum_{i \in [block_num]} h_i(x) = c_sum
//!      \sum_{x \in H_t} h_t(x) = c_sum
//!      \sum_{x \in H_f} \sum_{i \in [block_num]} eq(x, u) * (h(x) * \prod_{j \in [block_size]}(f_j(x) - r) - r * \sum_{i \in [block_size]} \prod_{j \in [block_size], j != i} (f_j(x) - r)) = 0
//!      \sum_{x \in H_t} eq(x, u) * (h_t(x) * (t(x) - r) - m(x)) = 0
//!      where u is a random challenge given from verifier (a vector of random element) and c_sum is some constant
//!
//! <==> \sum_{x \in H_f} \sum_{i \in [block_num]} h_i(x)
//!                     + \sum_{i \in [block_num]} eq(x, u) * (h(x) * \prod_{j \in [block_size]}(f_j(x) - r) - r * \sum_{i \in [block_size]} \prod_{j \in [block_size], j != i} (f_j(x) - r))
//!                     = c_sum
//!      \sum_{x \in H_t} h_t(x)
//!                     + eq(x, u) * (h_t(x) * (t(x) - r) - m(x))
//!                     = c_sum
//!      where u is a random challenge given from verifier (a vector of random element) and c_sum is some constant

use crate::pcs::OracleTable;
use crate::sumcheck::{verifier::SumcheckOracle, MLSumcheck, SumcheckProof};
use crate::utils::{
    batch_inverse, concat_slices, eval_identity_poly, gen_identity_poly, vec_to_polys,
    xy_multiplicity,
};

use algebra::PrimeField;
use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials
};
use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Clone, Default)]
pub struct VectorLookupInstance<F: Field, BF: Field, EF: AbstractExtensionField<BF>> {
    pub num_vars: usize,
    pub block_num: usize,
    pub block_size: usize,

    pub fx_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub fy_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub tx_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub ty_vec: Vec<Rc<DenseMultilinearExtension<F>>>,

    pub m_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub h_vec: Vec<Rc<DenseMultilinearExtension<F>>>,

    pub name: String,
    pub tx_name: String,
    pub ty_name: String,
    pub m_name: String,
    pub h_name: String,

    pub random_values: Vec<F>,
    _marker: PhantomData<(BF, EF)>,
}

impl<F, EF> VectorLookupInstance<F, F, EF>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
{
    #[inline]
    pub fn new(
        num_vars: usize,
        fx_vec: &[Rc<DenseMultilinearExtension<F>>],
        fy_vec: &[Rc<DenseMultilinearExtension<F>>],
        tx_vec: &[Rc<DenseMultilinearExtension<F>>],
        ty_vec: &[Rc<DenseMultilinearExtension<F>>],
        lookup_name: &String,
        tx_name: &String,
        ty_name: &String,
        block_size: usize,
    ) -> Self {
        let mut fx_vec = fx_vec.to_vec();
        let mut fy_vec = fy_vec.to_vec();
        let tx_vec = tx_vec.to_vec();
        let ty_vec = ty_vec.to_vec();
        let f_len = fx_vec.len();
        let t_len = tx_vec.len();

        debug_assert!(f_len == fy_vec.len() && t_len == ty_vec.len());
        debug_assert!(
            fx_vec.iter().all(|x| x.num_vars == num_vars)
                && fy_vec.iter().all(|x| x.num_vars == num_vars)
        );

        debug_assert!(
            tx_vec.iter().all(|x| x.num_vars == num_vars)
                && ty_vec.iter().all(|x| x.num_vars == num_vars)
        );

        let pad_size = block_size - ((f_len + t_len) % block_size);
        let f_len = f_len + pad_size;

        let f_pad = Rc::new(DenseMultilinearExtension::from_named_vec(
            &format!("zero_{}", num_vars),
            num_vars,
            vec![F::zero(); 1 << num_vars],
        ));

        fx_vec.resize(f_len, f_pad.clone());
        fy_vec.resize(f_len, f_pad);

        let m_vec = xy_multiplicity(&fx_vec, &fy_vec, &tx_vec, &ty_vec);

        let m_name = format!("m of {}", lookup_name);
        let h_name = format!("h of {}", lookup_name);

        debug_assert!(fx_vec
            .iter()
            .flat_map(|x| x.iter())
            .zip(fy_vec.iter().flat_map(|y| y.iter()))
            .all(|(x, y)| {
                if !tx_vec
                    .iter()
                    .flat_map(|tx| tx.iter())
                    .zip(ty_vec.iter().flat_map(|ty| ty.iter()))
                    .any(|(tx, ty)| *tx == *x && *ty == *y)
                {
                    dbg!(x);
                    dbg!(y);
                    assert!(false);
                }
                true
            }));

        debug_assert!(m_vec.len() == tx_vec.len());
        debug_assert!(m_vec.len() == ty_vec.len());

        Self {
            num_vars,
            block_num: (f_len + t_len) / block_size,
            block_size,
            fx_vec,
            fy_vec,
            tx_vec,
            ty_vec,
            m_vec,
            h_vec: Default::default(),
            random_values: Default::default(),
            name: lookup_name.clone(),
            tx_name: tx_name.clone(),
            ty_name: ty_name.clone(),
            m_name,
            h_name,
            _marker: PhantomData,
        }
    }

    pub fn to_ef(&self) -> VectorLookupInstance<EF, F, EF> {
        VectorLookupInstance::<EF, F, EF> {
            num_vars: self.num_vars,
            block_num: self.block_num,
            block_size: self.block_size,
            fx_vec: self.fx_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            fy_vec: self.fy_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            tx_vec: self.tx_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            ty_vec: self.ty_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            m_vec: self.m_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            h_vec: self.h_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            random_values: self
                .random_values
                .iter()
                .map(|x| EF::from_base(*x))
                .collect(),
            name: self.name.clone(),
            tx_name: self.tx_name.clone(),
            ty_name: self.ty_name.clone(),
            m_name: self.m_name.clone(),
            h_name: self.h_name.clone(),
            _marker: PhantomData,
        }
    }

    pub fn construct_first_oracles<H, C, S>(&self, oracle_table: &mut OracleTable<F, EF, H, C, S>)
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        self.fx_vec.iter().for_each(|fx| {
            oracle_table.add_named_oracle(fx.clone());
        });
        self.fy_vec.iter().for_each(|fy| {
            oracle_table.add_named_oracle(fy.clone());
        });

        oracle_table.add_super_oracle(&self.tx_name, &self.tx_vec);
        oracle_table.add_super_oracle(&self.ty_name, &self.ty_vec);
        oracle_table.add_super_oracle(&self.m_name, &self.m_vec);
    }
}

impl<F, EF> VectorLookupInstance<EF, F, EF>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
{
    pub fn construct_second_oracle<H, C, S>(
        &mut self,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        let random_value =
            trans.get_vec_challenge(b"random point used to generate the second oracle", 2);
        self.compute_h_vec(random_value);
        oracle_table.add_super_oracle_ef(&self.h_name, &self.h_vec);
    }

    /// receive random value
    pub fn compute_h_vec(&mut self, random_values: Vec<EF>) {
        self.random_values = random_values.to_vec();
        let num_vars = self.num_vars;

        let f: Vec<EF> = self
            .fx_vec
            .iter()
            .flat_map(|fx| fx.iter())
            .zip(self.fy_vec.iter().flat_map(|fy| fy.iter()))
            .map(|(fx, fy)| *fx + random_values[1] * fy)
            .collect();

        let t: Vec<EF> = self
            .tx_vec
            .iter()
            .flat_map(|tx| tx.iter())
            .zip(self.ty_vec.iter().flat_map(|ty| ty.iter()))
            .map(|(tx, ty)| *tx + random_values[1] * ty)
            .collect();

        debug_assert!(f.iter().all(|f| {
            assert!(t.iter().any(|t| *t == *f), "f {}", f);
            true
        }));

        let mut col_vec: Vec<EF> = f.iter().chain(t.iter()).cloned().collect();

        let mut m_vec = vec![-EF::one(); self.fx_vec.len() * (1 << num_vars)];
        m_vec.extend(self.m_vec.iter().flat_map(|m| m.iter()).cloned());

        debug_assert_eq!(col_vec.len(), m_vec.len());

        col_vec.iter_mut().for_each(|x| {
            *x -= random_values[0];
            debug_assert!(*x != EF::zero());
        });

        // inverse
        col_vec = batch_inverse(&col_vec);

        col_vec.iter_mut().zip(m_vec.iter()).for_each(|(col, m)| {
            *col *= *m;
        });

        let mut h_vec = vec![EF::zero(); self.block_num * (1 << num_vars)];
        for (i, col) in col_vec.iter().enumerate() {
            let i_col = i / (1 << num_vars);
            let i_block = i_col / self.block_size;
            let j = i % (1 << num_vars);
            h_vec[i_block * (1 << num_vars) + j] += col;
        }

        let h_vec = h_vec
            .chunks_exact(1 << num_vars)
            .map(|evals: &[EF]| Rc::new(DenseMultilinearExtension::from_slice(num_vars, evals)))
            .collect();

        self.h_vec = h_vec;
    }
}

#[derive(Default, Serialize)]
pub struct VectorLookupIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub num_vars: usize,
    pub f_len: usize,
    pub t_len: usize,
    pub block_size: usize,
    pub block_num: usize,

    pub fx: Vec<String>,
    pub fy: Vec<String>,
    pub tx: String,
    pub ty: String,

    pub f: Vec<String>,
    pub t: String,
    pub m: String,
    pub h: String,

    pub random_values: Vec<EF>,
    pub sumcheck_proof: SumcheckProof<EF>,
    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> VectorLookupIOP<F, EF, H, C, S>
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

    pub fn info(&mut self, instance: &VectorLookupInstance<EF, F, EF>) {
        self.num_vars = instance.num_vars;
        self.f_len = instance.fx_vec.len();
        self.t_len = instance.tx_vec.len();
        self.block_size = instance.block_size;
        self.block_num = instance.block_num;

        self.fx = instance.fx_vec.iter().map(|x| x.name.clone()).collect();
        self.fy = instance.fy_vec.iter().map(|x| x.name.clone()).collect();
        self.tx = instance.tx_name.clone();
        self.ty = instance.ty_name.clone();

        self.m = instance.m_name.clone();
        self.h = instance.h_name.clone();
    }

    #[inline]
    pub fn pre_randomness(&self, trans: &mut Transcript<EF>) -> bool {
        let random_values =
            trans.get_vec_challenge(b"random point used to generate the second oracle", 2);
        random_values == self.random_values
    }

    pub fn prove(
        &mut self,
        instance: &VectorLookupInstance<EF, F, EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        self.info(instance);

        debug_assert!(instance.h_vec.len() > 0);

        self.random_values = instance.random_values.clone();

        let random_point =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", instance.num_vars);

        let random_sumcheck_combine = trans.get_vec_challenge(
            b"randomness to combine sumcheck protocols",
            instance.block_num,
        );

        let random_oracle_tx_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            self.t_len.next_power_of_two().ilog2() as usize,
        );

        let random_oracle_ty_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            self.t_len.next_power_of_two().ilog2() as usize,
        );

        let random_oracle_m_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            instance.m_vec.len().next_power_of_two().ilog2() as usize,
        );

        let random_oracle_h_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            instance.h_vec.len().next_power_of_two().ilog2() as usize,
        );

        let eq = Rc::new(gen_identity_poly(&random_point));
        let f_vec = instance
            .fx_vec
            .iter()
            .flat_map(|x| x.iter())
            .zip(instance.fy_vec.iter().flat_map(|y| y.iter()))
            .map(|(fx, fy)| *fx + instance.random_values[1] * fy)
            .collect::<Vec<EF>>();
        let t_vec = instance
            .tx_vec
            .iter()
            .flat_map(|tx| tx.iter())
            .zip(instance.ty_vec.iter().flat_map(|ty| ty.iter()))
            .map(|(tx, ty)| *tx + instance.random_values[1] * ty)
            .collect::<Vec<EF>>();

        debug_assert!(f_vec.iter().all(|f| {
            assert!(t_vec.iter().any(|t| *t == *f), "f {}", f);
            true
        }));

        let f_vec = vec_to_polys(self.num_vars, &f_vec);
        let t_vec = vec_to_polys(self.num_vars, &t_vec);

        let mut poly = ListOfProductsOfPolynomials::<EF>::new(instance.num_vars);

        VectorLookupIOP::<F, EF, H, C, S>::construct_sumcheck(
            &self,
            &random_sumcheck_combine,
            &mut poly,
            &f_vec,
            &t_vec,
            &instance.m_vec,
            &instance.h_vec,
            &eq,
        );

        let (proof, state) =
            <MLSumcheck<EF>>::prove(trans, &poly).expect("Proof generated in VectorLookup");

        let sumcheck_point = state.randomness;

        instance.fx_vec.iter().for_each(|fx| {
            oracle_table.add_point(&fx.name, &sumcheck_point);
        });

        instance.fy_vec.iter().for_each(|fy| {
            oracle_table.add_point(&fy.name, &sumcheck_point);
        });

        oracle_table.add_point(
            &instance.tx_name,
            &concat_slices(&[&sumcheck_point, &random_oracle_tx_combine]),
        );

        oracle_table.add_point(
            &instance.ty_name,
            &concat_slices(&[&sumcheck_point, &random_oracle_ty_combine]),
        );

        oracle_table.add_point(
            &instance.m_name,
            &concat_slices(&[&sumcheck_point, &random_oracle_m_combine]),
        );

        oracle_table.add_point_ef(
            &instance.h_name,
            &concat_slices(&[&sumcheck_point, &random_oracle_h_combine]),
        );

        self.sumcheck_proof = SumcheckProof {
            proof,
            info: poly.info(),
            claimed_sum: EF::zero(),
        };
    }

    pub fn construct_sumcheck(
        &self,
        random_combine: &[EF],
        poly: &mut ListOfProductsOfPolynomials<EF>,
        f_vec: &[Rc<DenseMultilinearExtension<EF>>],
        t_vec: &[Rc<DenseMultilinearExtension<EF>>],
        m_vec: &[Rc<DenseMultilinearExtension<EF>>],
        h_vec: &[Rc<DenseMultilinearExtension<EF>>],
        eq: &Rc<DenseMultilinearExtension<EF>>,
    ) {
        let random_value = self.random_values[0];

        let mut ft_vec = Vec::with_capacity(f_vec.len() + t_vec.len());
        ft_vec.extend(f_vec);
        ft_vec.extend(t_vec);

        // construct shifted columns: (f(x) - r)
        let shifted_ft_vec: Vec<Rc<DenseMultilinearExtension<EF>>> = ft_vec
            .iter()
            .map(|f| {
                let evaluations = f.evaluations.iter().map(|x| *x - random_value).collect();
                Rc::new(DenseMultilinearExtension::from_vec(
                    self.num_vars,
                    evaluations,
                ))
            })
            .collect();

        // construct poly
        for (i, (h, l)) in h_vec.iter().zip(random_combine.iter()).enumerate() {
            let product = vec![h.clone()];
            poly.add_product(product, EF::one());

            let block = &shifted_ft_vec[i * self.block_size..(i + 1) * self.block_size];

            let mut product = block.to_vec();
            product.extend(vec![eq.clone(), h.clone()]);
            poly.add_product(product, *l);

            for j in 0..self.block_size {
                let mut product = block.to_vec();
                product[j] = eq.clone();

                let idx = i * self.block_size + j;
                if idx >= f_vec.len() {
                    product.push(m_vec[idx - f_vec.len()].clone());
                    poly.add_product(product, -*l);
                } else {
                    poly.add_product(product, *l);
                }
            }
        }
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        let proof = &self.sumcheck_proof;

        let random_values = self.random_values.clone();

        let random_point =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars);

        let random_combine =
            trans.get_vec_challenge(b"randomness to combine sumcheck protocols", self.block_num);

        let random_oracle_tx_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            self.t_len.next_power_of_two().ilog2() as usize,
        );

        let random_oracle_ty_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            self.t_len.next_power_of_two().ilog2() as usize,
        );

        let random_oracle_m_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            self.t_len.next_power_of_two().ilog2() as usize,
        );

        let random_oracle_h_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            self.block_num.next_power_of_two().ilog2() as usize,
        );

        let mut subclaim = MLSumcheck::verify(trans, &proof)
            .expect("fail to verify the course of sumcheck protocol in vector lookup");

        let sumcheck_point = subclaim.sumcheck_point.clone();
        let eq = eval_identity_poly(&random_point, &subclaim.sumcheck_point);

        let fx_vec: Vec<EF> = self
            .fx
            .iter()
            .map(|fx| oracle_table.get_eval(fx, &sumcheck_point)[0])
            .collect();

        let fy_vec: Vec<EF> = self
            .fy
            .iter()
            .map(|fy| oracle_table.get_eval(fy, &sumcheck_point)[0])
            .collect();

        let tx_vec = &oracle_table.get_eval(
            &self.tx,
            &concat_slices(&[&sumcheck_point, &random_oracle_tx_combine]),
        );

        let ty_vec = &oracle_table.get_eval(
            &self.ty,
            &concat_slices(&[&sumcheck_point, &random_oracle_ty_combine]),
        );

        let m_vec = &oracle_table.get_eval(
            &self.m,
            &concat_slices(&[&sumcheck_point, &random_oracle_m_combine]),
        );
        let h_vec = &oracle_table.get_eval_ef(
            &self.h,
            &concat_slices(&[&sumcheck_point, &random_oracle_h_combine]),
        );

        let f_vec = fx_vec
            .iter()
            .zip(fy_vec.iter())
            .map(|(fx, fy)| *fx + random_values[1] * fy)
            .collect::<Vec<EF>>();

        let t_vec = tx_vec
            .iter()
            .zip(ty_vec.iter())
            .map(|(tx, ty)| *tx + random_values[1] * ty)
            .collect::<Vec<EF>>();

        VectorLookupIOP::<F, EF, H, C, S>::evaluate_sumcheck(
            &self,
            &random_combine,
            random_values[0],
            &mut subclaim,
            &f_vec,
            &t_vec,
            m_vec,
            h_vec,
            eq,
        );

        debug_assert!(subclaim.oracle_eval == EF::zero());
        subclaim.oracle_eval == EF::zero()
    }

    #[inline]
    pub fn evaluate_sumcheck(
        &self,
        random_combine: &[EF],
        random_value: EF,
        subclaim: &mut SumcheckOracle<EF>,
        f_vec: &[EF],
        t_vec: &[EF],
        m_vec: &[EF],
        h_vec: &[EF],
        eq: EF,
    ) {
        let mut ft_vec = Vec::with_capacity(f_vec.len() + t_vec.len());
        ft_vec.extend_from_slice(f_vec);
        ft_vec.extend_from_slice(t_vec);

        let mut m = vec![-EF::one(); self.f_len];
        m.extend(m_vec);
        let m_vec = m;

        // shift
        let ft_vec: Vec<EF> = ft_vec.iter().map(|f| *f - random_value).collect();
        // inverse
        let ft_vec_inverse = batch_inverse(&ft_vec);

        for ((((l, h), m_block), ft_block), ft_inverse_block) in random_combine
            .iter()
            .zip(h_vec.iter())
            .zip(m_vec.chunks_exact(self.block_size))
            .zip(ft_vec.chunks_exact(self.block_size))
            .zip(ft_vec_inverse.chunks_exact(self.block_size))
        {
            let product = ft_block.iter().fold(EF::one(), |acc, x| acc * x);
            let sum = m_block
                .iter()
                .zip(ft_inverse_block.iter())
                .fold(EF::zero(), |acc, (m, ft_inverse)| acc + *m * ft_inverse);

            subclaim.oracle_eval -= *h + *l * eq * product * (*h - sum);
        }
    }
}

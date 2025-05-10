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
    batch_inverse, concat_slices, eval_identity_poly, gen_identity_poly, multiplicity,
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
pub struct LookupInstance<F: Field, BF: Field, EF: AbstractExtensionField<BF>> {
    pub num_vars: usize,
    pub block_num: usize,
    pub block_size: usize,

    pub f_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub t_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub m_vec: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub h_vec: Vec<Rc<DenseMultilinearExtension<F>>>,

    pub name: String,
    pub t_name: String,
    pub m_name: String,
    pub h_name: String,

    pub random_value: F,
    _marker: PhantomData<(BF, EF)>,
}

impl<F, EF> LookupInstance<F, F, EF>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
{
    pub fn to_ef(&self) -> LookupInstance<EF, F, EF> {
        LookupInstance::<EF, F, EF> {
            num_vars: self.num_vars,
            block_num: self.block_num,
            block_size: self.block_size,
            f_vec: self.f_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            t_vec: self.t_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            name: self.name.clone(),
            t_name: self.t_name.clone(),
            m_name: self.m_name.clone(),
            m_vec: self.m_vec.iter().map(|x| Rc::new(x.to_ef())).collect(),
            h_vec: Default::default(),
            h_name: self.h_name.clone(),
            random_value: EF::from_base(self.random_value),
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
        self.f_vec.iter().for_each(|f| {
            oracle_table.add_named_oracle(f.clone());
        });

        oracle_table.add_super_oracle(&self.t_name, &self.t_vec);
        oracle_table.add_super_oracle(&self.m_name, &self.m_vec);
    }
}

impl<F, EF> LookupInstance<F, F, EF>
where
    F: PrimeField + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
{
    #[inline]
    pub fn new(
        f_vec: &[Rc<DenseMultilinearExtension<F>>],
        t_vec: &[Rc<DenseMultilinearExtension<F>>],
        lookup_name: String,
        t_name: String,
        block_size: usize,
    ) -> Self {
        let num_vars = f_vec[0].num_vars;
        let mut f_vec = f_vec.to_vec();
        let t_vec = t_vec.to_vec();

        assert!(f_vec.len() != 0 && t_vec.len() != 0);
        assert!(f_vec.iter().all(|x| x.num_vars == num_vars));
        assert!(t_vec.iter().all(|x| x.num_vars == num_vars));

        let padding_size = (block_size - ((f_vec.len() + t_vec.len()) % block_size)) % block_size;

        // assume zero is in table t
        let f_pad = Rc::new(DenseMultilinearExtension::from_named_slice(
            &format!("zero_{}", num_vars),
            num_vars,
            &vec![F::zero(); 1 << num_vars],
        ));

        f_vec.resize(f_vec.len() + padding_size, f_pad);

        debug_assert!(f_vec
            .iter()
            .flat_map(|f| f.iter())
            .all(|x| { t_vec.iter().any(|t| t.evaluations.contains(x)) }));

        let m_name = format!("m of {}", lookup_name);
        let h_name = format!("h of {}", lookup_name);

        let m_vec = multiplicity(&f_vec, &t_vec);

        Self {
            num_vars,
            block_num: (f_vec.len() + t_vec.len()) / block_size,
            block_size,
            f_vec,
            t_vec,
            name: lookup_name,
            t_name,
            h_name,
            m_vec,
            m_name,
            h_vec: Default::default(),
            random_value: Default::default(),
            _marker: PhantomData,
        }
    }
}

impl<F, EF> LookupInstance<EF, F, EF>
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
        let random_value = trans.get_challenge(b"random point used to generate the second oracle");
        self.compute_h_vec(random_value);
        oracle_table.add_super_oracle_ef(&self.h_name, &self.h_vec);
    }

    /// compute_h_vec
    pub fn compute_h_vec(&mut self, random_value: EF) {
        self.random_value = random_value;
        let num_vars = self.num_vars;

        assert_eq!(self.m_vec.len(), self.t_vec.len());

        let mut col_vec: Vec<EF> = self
            .f_vec
            .iter()
            .chain(self.t_vec.iter())
            .flat_map(|ft: &Rc<DenseMultilinearExtension<EF>>| ft.iter().cloned())
            .collect();

        let mut m_vec = vec![-EF::one(); self.f_vec.len() * (1 << num_vars)];
        m_vec.extend(self.m_vec.iter().flat_map(|m| m.iter()).cloned());

        assert_eq!(col_vec.len(), m_vec.len());

        // shift
        col_vec.iter_mut().for_each(|x| *x -= random_value);

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

#[derive(Default, Serialize, Debug)]
pub struct LookupIOP<F, EF, H, C, S>
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

    pub name: String,
    pub f: Vec<String>,
    pub t: String,
    pub m: String,
    pub h: String,

    pub random_value: EF,
    pub sumcheck_proof: SumcheckProof<EF>,
    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> LookupIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn proof_size(&self) -> usize {
        bincode::serialize(&self.sumcheck_proof)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn info(&mut self, instance: &LookupInstance<EF, F, EF>) {
        self.num_vars = instance.num_vars;
        self.f_len = instance.f_vec.len();
        self.t_len = instance.t_vec.len();
        self.block_size = instance.block_size;
        self.block_num = instance.block_num;
        self.name = instance.name.clone();
        self.f = instance.f_vec.iter().map(|x| x.name.clone()).collect();
        self.t = instance.t_name.clone();
        self.m = instance.m_name.clone();
        self.h = instance.h_name.clone();
    }

    pub fn prove(
        &mut self,
        instance: &LookupInstance<EF, F, EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        self.info(instance);

        debug_assert!(instance.h_vec.len() > 0);
        debug_assert!(instance
            .f_vec
            .iter()
            .flat_map(|f| f.iter())
            .all(|x| instance.t_vec.iter().any(|t| t.evaluations.contains(x))));

        self.random_value = instance.random_value;

        let random_point =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", instance.num_vars);

        let random_sumcheck_combine = trans.get_vec_challenge(
            b"randomness to combine sumcheck protocols",
            instance.block_num,
        );

        let random_oracle_t_combine = trans.get_vec_challenge(
            b"randomness to combine oracles",
            instance.t_vec.len().next_power_of_two().ilog2() as usize,
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

        let mut poly: ListOfProductsOfPolynomials<EF> =
            ListOfProductsOfPolynomials::new(instance.num_vars);

        LookupIOP::<F, EF, H, C, S>::construct_sumcheck(
            &self,
            &random_sumcheck_combine,
            &mut poly,
            &instance.f_vec,
            &instance.t_vec,
            &instance.m_vec,
            &instance.h_vec,
            &eq,
        );

        let (proof, state) =
            <MLSumcheck<EF>>::prove(trans, &poly).expect("Proof generated in InfiniteNorm");

        let sumcheck_point = state.randomness;

        self.f.iter().for_each(|f_name| {
            oracle_table.add_point(f_name, &sumcheck_point);
        });

        oracle_table.add_point(
            &self.t,
            &concat_slices(&[&sumcheck_point, &random_oracle_t_combine]),
        );
        oracle_table.add_point(
            &self.m,
            &concat_slices(&[&sumcheck_point, &random_oracle_m_combine]),
        );
        oracle_table.add_point_ef(
            &self.h,
            &concat_slices(&[&sumcheck_point, &random_oracle_h_combine]),
        );

        self.sumcheck_proof = SumcheckProof {
            proof,
            info: poly.info(),
            claimed_sum: EF::zero(),
        }
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
        let random_value = self.random_value;

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

    #[inline]
    pub fn pre_randomness(&self, trans: &mut Transcript<EF>) -> bool {
        let random_value = trans.get_challenge(b"random point used to generate the second oracle");
        random_value == self.random_value
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        let proof = &self.sumcheck_proof;
        let random_value = self.random_value;

        let random_point =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars);

        let random_combine =
            trans.get_vec_challenge(b"randomness to combine sumcheck protocols", self.block_num);

        let random_oracle_t_combine = trans.get_vec_challenge(
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

        let mut subclaim = MLSumcheck::verify(trans, &proof).expect(&format!(
            "fail to verify the course of sumcheck protocol in lookup {:?}",
            self.name
        ));

        let sumcheck_point = subclaim.sumcheck_point.clone();
        let eq = eval_identity_poly(&random_point, &subclaim.sumcheck_point);

        let f_vec: Vec<EF> = self
            .f
            .iter()
            .map(|f| oracle_table.get_eval(f, &sumcheck_point)[0])
            .collect();

        let t_vec = &oracle_table.get_eval(
            &self.t,
            &concat_slices(&[&sumcheck_point, &random_oracle_t_combine]),
        );
        let m_vec = &oracle_table.get_eval(
            &self.m,
            &concat_slices(&[&sumcheck_point, &random_oracle_m_combine]),
        );
        let h_vec = &oracle_table.get_eval_ef(
            &self.h,
            &concat_slices(&[&sumcheck_point, &random_oracle_h_combine]),
        );

        debug_assert!(
            f_vec.len() == self.f_len
                && t_vec.len() == self.t_len
                && m_vec.len() == self.t_len
                && h_vec.len() == self.block_num
        );

        LookupIOP::<F, EF, H, C, S>::evaluate_sumcheck(
            &self,
            &random_combine,
            random_value,
            &mut subclaim,
            &f_vec,
            t_vec,
            m_vec,
            h_vec,
            eq,
        );
        if subclaim.oracle_eval != EF::zero() {
            println!("lookup sumcheck fail {}", self.name);
        }
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

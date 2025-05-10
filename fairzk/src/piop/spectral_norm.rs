use super::LookupInstance;
use crate::pcs::OracleTable;
use crate::sumcheck::{MLSumcheck, SumcheckProof};
use crate::utils::{
    absolute_range_tables, compute_wt_w, concat_slices, eval_identity_poly, field_to_i128,
    i128_to_field, is_valid, is_valid_i128, range_tables, vec_to_named_poly,
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

use nalgebra::{max, DMatrix};
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, vec};

pub struct SpectralNormInstance<F>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
{
    pub i_bit: usize,
    pub f_bit: usize,

    // specifcation of shape of weight matrix
    pub num_vars_row: usize,
    pub num_vars_col: usize,

    // error bound in [-epsilon, epsilon]
    pub error_range: usize,

    // spectral norm of weight matrix
    // i.e. the largest eigenvalue of a = w^T * w
    pub spectral_norm: usize,

    // weight matrix
    // shape [num_vars_row, num_vars_col]
    w: Rc<DenseMultilinearExtension<F>>,

    // square matrix: a = w^T * w
    // shape [num_vars_col, num_vars_col]
    a: Rc<DenseMultilinearExtension<F>>,

    // eigen vectors of a
    // shape [num_vars_col, num_vars_col]
    eig_vec: Rc<DenseMultilinearExtension<F>>,

    // eigen values of a
    // shape [num_vars_col]
    eig_val: Rc<DenseMultilinearExtension<F>>,

    // error for \sum_h eig_vec(h,r2) eig_vec(h,r3) = e(r2, r3)
    // shape [num_vars_col, num_vars_col]
    err_vec: Rc<DenseMultilinearExtension<F>>,

    // e(r2, r3) = \sum_h l * eig_vec(h,r2) eig_vec(h,r3) + a(r2,h) * eig_vec(h,r3) - eq(h, r3) eig_value(h) eig_vec(r2,h)
    // shape [num_vars_col, num_vars_col]
    err_val: Rc<DenseMultilinearExtension<F>>,

    power_2_f: F,
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> SpectralNormInstance<F> {
    #[inline]
    pub fn new(
        num_vars_row: usize,
        num_vars_col: usize,
        i_bit: usize,
        f_bit: usize,
        w: &Rc<DenseMultilinearExtension<F>>,
    ) -> Self {
        let num_row = 1 << num_vars_row;
        let num_col = 1 << num_vars_col;
        debug_assert!(w.num_vars == num_vars_row + num_vars_col);

        w.iter().all(|x| is_valid(*x, i_bit, f_bit));

        let f_shift = 1 << f_bit;

        // field -> i128
        let w_i128 = field_to_i128::<F>(&w.evaluations);
        let w_i128 = DMatrix::from_column_slice(num_row, num_col, &w_i128);
        let a_i128 = w_i128.transpose() * &w_i128;
        a_i128.iter().for_each(|x| {
            is_valid_i128(*x, i_bit, 2 * f_bit);
        });

        // i128 -> f64 to perform eigenvalue and eigenvector decomposition
        let a_f64 = a_i128.map(|x| (x as f64) / ((f_shift * f_shift) as f64));
        let sym_eigen = a_f64.clone().symmetric_eigen();
        let eig_val_f64 = sym_eigen.eigenvalues;
        let eig_vec_f64 = sym_eigen.eigenvectors;

        // f64 -> i128
        let eig_val_i128 = eig_val_f64.map(|x| (x * f_shift as f64).round() as i128);
        let eig_vec_i128 = eig_vec_f64.map(|x| (x * f_shift as f64).round() as i128);
        let eig_val_diag = DMatrix::from_diagonal(&eig_val_i128);

        let mut err_vec_i128 = eig_vec_i128.transpose() * &eig_vec_i128;
        // for diagonal elements, subtract 1's encoding, i.e. 2^{2f}
        for i in 0..err_vec_i128.nrows() {
            for j in 0..err_vec_i128.ncols() {
                if i == j {
                    err_vec_i128[(i, j)] -= f_shift * f_shift;
                }
            }
        }

        // err_val is defined by A - V * D * V^T, so A rasied by 2^f
        let err_val_i128 =
            &a_i128 * (f_shift) - &eig_vec_i128 * &eig_val_diag * &eig_vec_i128.transpose();

        // collecting bounds
        let max_eig_val = eig_val_i128.max();

        let mut max_err_vec: usize = 0;
        for i in 0..err_vec_i128.nrows() {
            for j in 0..err_vec_i128.ncols() {
                if i != j {
                    let val = err_vec_i128[(i, j)].abs() as usize;
                    if val > max_err_vec {
                        max_err_vec = val;
                    }
                }
            }
        }

        let max_err_val: usize = err_val_i128
            .iter()
            .map(|x| x.abs() as usize)
            .max()
            .unwrap_or(0);

        let max_err = max(max_err_vec, max_err_val);
        let err_range = max_err + 1;

        // Matrix ->  Vec<i128>
        let a_i128: Vec<i128> = a_i128.iter().cloned().collect();
        let eig_vec_i128: Vec<i128> = eig_vec_i128.iter().cloned().collect();
        let eig_val_i128: Vec<i128> = eig_val_i128.iter().cloned().collect();
        let err_vec_i128: Vec<i128> = err_vec_i128.iter().cloned().collect();
        let err_val_i128: Vec<i128> = err_val_i128.iter().cloned().collect();

        let a = vec_to_named_poly(&format!("a of {}", w.name), &i128_to_field::<F>(&a_i128));
        let eig_vec = vec_to_named_poly(
            &format!("eig vec of {}", a.name),
            &i128_to_field::<F>(&eig_vec_i128),
        );
        let eig_val = vec_to_named_poly(
            &format!("eig val of {}", a.name),
            &i128_to_field::<F>(&eig_val_i128),
        );
        let err_vec = vec_to_named_poly(
            &format!("err vec of {}", a.name),
            &i128_to_field::<F>(&err_vec_i128),
        );
        let err_val = vec_to_named_poly(
            &format!("err val of {}", a.name),
            &i128_to_field::<F>(&err_val_i128),
        );
        let power_2_f = i128_to_field(&[(1 << f_bit)])[0];

        let instance = Self {
            num_vars_row,
            num_vars_col,
            error_range: err_range,
            spectral_norm: max_eig_val as usize,
            i_bit,
            f_bit,
            w: w.clone(),
            a,
            eig_vec,
            eig_val,
            err_vec,
            err_val,
            power_2_f,
        };

        debug_assert!(instance.is_valid());

        instance
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        let num_row = 1 << self.num_vars_row;
        let num_col = 1 << self.num_vars_col;
        let wtw = compute_wt_w::<F>(&self.w.evaluations, num_row, num_col);
        //let wtw = reduce(&wtw);
        wtw == self.a.evaluations
    }
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> SpectralNormInstance<F> {
    pub fn lookup_instances<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let rangecheck_error = self.rangecheck_error();
        let rangecheck_max_eig_val = self.rangecheck_max_eig_val();
        vec![rangecheck_error, rangecheck_max_eig_val]
    }

    pub fn rangecheck_max_eig_val<EF>(&self) -> LookupInstance<F, F, EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let range = self.spectral_norm + 1;
        let tables_eig_val = range_tables(self.num_vars_col, range);

        LookupInstance::new(
            &vec![self.eig_val.clone()],
            &tables_eig_val,
            format!("max_eig_val_range for {}", self.eig_val.name),
            format!(
                "t of [range_check ({}, {}) num_vars {}]",
                0, range, self.eig_val.num_vars
            ),
            1,
        )
    }

    pub fn rangecheck_error<EF>(&self) -> LookupInstance<F, F, EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let range = self.error_range + 1;
        let tables_error = absolute_range_tables(2 * self.num_vars_col, range);

        let mut err_vec_no_diag = Vec::new();
        let mut counter = 0;
        self.err_vec.iter().for_each(|e| {
            for j in 0..self.num_vars_col {
                if ((counter >> j) & 1) != ((counter >> (j + self.num_vars_col)) & 1) {
                    // not diagonal element
                    err_vec_no_diag.push(*e);
                    break;
                }
            }
            counter += 1;
        });

        err_vec_no_diag.resize(1 << (2 * self.num_vars_col), F::zero());

        let err_vec_no_diag = Rc::new(DenseMultilinearExtension::from_named_vec(
            &format!(
                "intermediate oracle err_vec_no_diag of {}",
                self.err_vec.name
            ),
            2 * self.num_vars_col,
            err_vec_no_diag,
        ));

        LookupInstance::new(
            &vec![self.err_val.clone(), err_vec_no_diag.clone()],
            &tables_error,
            format!(
                "rangecheck for {} and {}",
                self.err_val.name, err_vec_no_diag.name
            ),
            format!(
                "rangecheck ({}, {}) num vars {}",
                1i128 - range as i128,
                range,
                2 * self.num_vars_col
            ),
            1,
        )
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> SpectralNormInstance<F> {
    // transform this instance on base field to an instance on extension field
    pub fn to_ef<EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>>(
        &self,
    ) -> SpectralNormInstance<EF> {
        SpectralNormInstance {
            num_vars_row: self.num_vars_row,
            num_vars_col: self.num_vars_col,
            error_range: self.error_range,
            spectral_norm: self.spectral_norm,
            i_bit: self.i_bit,
            f_bit: self.f_bit,
            w: Rc::new(self.w.to_ef()),
            a: Rc::new(self.a.to_ef()),
            eig_vec: Rc::new(self.eig_vec.to_ef()),
            eig_val: Rc::new(self.eig_val.to_ef()),
            err_vec: Rc::new(self.err_vec.to_ef()),
            err_val: Rc::new(self.err_val.to_ef()),
            power_2_f: EF::from(self.power_2_f),
        }
    }

    // w, a, eig_vec, eig_val, error
    pub fn construct_oracles<EF, H, C, S>(&self, table: &mut OracleTable<F, EF, H, C, S>)
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        table.add_named_oracle(self.w.clone());
        //table.add_named_oracle(self.a.clone());
        table.add_named_oracle(self.eig_vec.clone());
        table.add_named_oracle(self.eig_val.clone());
        table.add_named_oracle(self.err_vec.clone());
        table.add_named_oracle(self.err_val.clone());
    }
}

#[derive(Default, Serialize)]
pub struct SpectralNormIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub num_vars_row: usize,
    pub num_vars_col: usize,
    pub error_range: usize,
    pub spectral_norm: usize,
    pub w: String,
    pub a: String,
    pub eig_vec: String,
    pub eig_val: String,
    pub err_vec: String,
    pub err_val: String,

    pub sumcheck_square: SumcheckProof<EF>,
    pub sumcheck_eigen: SumcheckProof<EF>,

    pub i_bit: usize,
    pub f_bit: usize,
    pub power_2_f: EF,

    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> SpectralNormIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn proof_size(&self) -> (usize, usize) {
        let proof_size_sumcheck_square = bincode::serialize(&self.sumcheck_square)
            .map(|v| v.len())
            .unwrap_or(0);
        let proof_size_sumcheck_eigen = bincode::serialize(&self.sumcheck_eigen)
            .map(|v| v.len())
            .unwrap_or(0);
        (proof_size_sumcheck_square, proof_size_sumcheck_eigen)
    }

    pub fn info(&mut self, instance: &SpectralNormInstance<EF>) {
        self.num_vars_row = instance.num_vars_row;
        self.num_vars_col = instance.num_vars_col;
        self.error_range = instance.error_range;
        self.spectral_norm = instance.spectral_norm;
        self.w = instance.w.name.clone();
        self.a = instance.a.name.clone();
        self.eig_vec = instance.eig_vec.name.clone();
        self.eig_val = instance.eig_val.name.clone();
        self.err_vec = instance.err_vec.name.clone();
        self.err_val = instance.err_val.name.clone();
        self.i_bit = instance.i_bit;
        self.f_bit = instance.f_bit;
        self.power_2_f = instance.power_2_f;
    }

    // oracle w, a, eig_vec, eig_val, error

    // relation 1:  err_vec(r0, r1) + err_val(r0, r1) = \sum_h l * eig_vec(h,r0) eig_vec(h,r1) + a(r0,h) * eig_vec(h,r1) - 2^f * eq(h, r1) * eig_value(h) * eig_vec(r0,h)
    // relation 0:  a(r0, r1) = \sum_h w(h, r0)w(h, r1)
    pub fn prove(
        &mut self,
        instance: &mut SpectralNormInstance<EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        self.info(&instance);

        // relation 1: l * (err_vec(r0, r1) + 2^f * 2^f) + a(r0, r1) * 2^f - err_val(r0, r1) = \sum_h l * eig_vec(h,r0) eig_vec(h,r1) + eig_value(h) * eig_vec(r0,h) * eig_vec(h,r1)

        let r0 = trans.get_vec_challenge(b"random point", self.num_vars_col);
        let r1 = trans.get_vec_challenge(b"random point", self.num_vars_col);
        let l = trans.get_challenge(b"random combine");

        let eig_vec_h_r0 = Rc::new(instance.eig_vec.fix_variables_back(&r0));
        let eig_vec_h_r1 = Rc::new(instance.eig_vec.fix_variables_back(&r1));
        let eign_vec_r0_h = Rc::new(instance.eig_vec.fix_variables_front(&r0));
        let eign_vec_r1_h = Rc::new(instance.eig_vec.fix_variables_front(&r1));

        let mut poly_eigen = ListOfProductsOfPolynomials::<EF>::new(self.num_vars_col);
        poly_eigen.add_product(
            vec![eig_vec_h_r0.clone(), eig_vec_h_r1.clone()],
            l, //l * instance.power_2_f,
        );
        poly_eigen.add_product(
            vec![
                instance.eig_val.clone(),
                eign_vec_r0_h.clone(),
                eign_vec_r1_h.clone(),
            ],
            EF::one(),
        );
        let (proof_eigen, state_eigen) =
            <MLSumcheck<EF>>::prove(trans, &poly_eigen).expect("Proof generated in SpectralNorm");

        let sumcheck_point_eigen = state_eigen.randomness;

        let a_eval = instance.a.evaluate(&concat_slices(&[&r0, &r1]));
        let claimed_sum = l
            * (instance.err_vec.evaluate(&concat_slices(&[&r0, &r1]))
                + instance.power_2_f * instance.power_2_f * eval_identity_poly(&r0, &r1))
            + a_eval * instance.power_2_f
            - instance.err_val.evaluate(&concat_slices(&[&r0, &r1]));
        self.sumcheck_eigen = SumcheckProof {
            proof: proof_eigen,
            info: poly_eigen.info(),
            claimed_sum,
        };

        oracle_table.add_point(&instance.err_vec.name, &concat_slices(&[&r0, &r1]));
        oracle_table.add_point(&instance.err_val.name, &concat_slices(&[&r0, &r1]));
        // oracle_table.add_point(
        //     &instance.a.name,
        //     &concat_slices(&[&r2, &r3])
        // );
        oracle_table.add_point(&instance.eig_val.name, &sumcheck_point_eigen);
        oracle_table.add_point(
            &instance.eig_vec.name,
            &concat_slices(&[&sumcheck_point_eigen, &r0]),
        );
        oracle_table.add_point(
            &instance.eig_vec.name,
            &concat_slices(&[&sumcheck_point_eigen, &r1]),
        );
        oracle_table.add_point(
            &instance.eig_vec.name,
            &concat_slices(&[&r0, &sumcheck_point_eigen]),
        );
        oracle_table.add_point(
            &instance.eig_vec.name,
            &concat_slices(&[&r1, &sumcheck_point_eigen]),
        );

        // prove raletion 2
        // a(r0, r1) = \sum_h w(h, r0)w(h, r1)

        let w_h_r0 = Rc::new(instance.w.fix_variables_back(&r0));
        let w_h_r1 = Rc::new(instance.w.fix_variables_back(&r1));
        let mut poly_square = ListOfProductsOfPolynomials::<EF>::new(self.num_vars_row);
        poly_square.add_product(vec![w_h_r0.clone(), w_h_r1.clone()], EF::one());
        let (proof_square, state_square) =
            <MLSumcheck<EF>>::prove(trans, &poly_square).expect("Proof generated in SpectralNorm");

        let sumcheck_point_square = state_square.randomness;

        self.sumcheck_square = SumcheckProof {
            proof: proof_square,
            info: poly_square.info(),
            claimed_sum: a_eval,
        };

        oracle_table.add_point(
            &instance.w.name,
            &concat_slices(&[&sumcheck_point_square, &r0]),
        );
        oracle_table.add_point(
            &instance.w.name,
            &concat_slices(&[&sumcheck_point_square, &r1]),
        );
        //oracle_table.add_point(&instance.a.name, &concat_slices(&[&r0, &r1]));
    }

    // oracle w, a, eig_vec, eig_val, error

    // relation 1: l * (err_vec(r2, r3) + 2^f * 2^f) + a(r2, r3) * 2^f - err_val(r2, r3) = \sum_h l * eig_vec(h,r2) eig_vec(h,r3) + eig_value(h) * eig_vec(r2,h) * eig_vec(h,r3)
    // relation 2:
    // a(r0, r1) = \sum_h w(h, r0)w(h, r1)
    // => a(r0, r1) = w(s, r0)w(s, r1)
    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        let a_eval = self.sumcheck_square.claimed_sum;

        let r0 = trans.get_vec_challenge(b"random point", self.num_vars_col);
        let r1 = trans.get_vec_challenge(b"random point", self.num_vars_col);
        let l = trans.get_challenge(b"random combine");

        let sumcheck_oracle_eigen = <MLSumcheck<EF>>::verify(trans, &self.sumcheck_eigen)
            .expect("fail to verify sumcheck eigen");

        let sumcheck_point_eigen = &sumcheck_oracle_eigen.sumcheck_point;

        let eig_vec_s_r0 =
            oracle_table.get_eval(&self.eig_vec, &concat_slices(&[sumcheck_point_eigen, &r0]))[0];
        let eig_vec_s_r1 =
            oracle_table.get_eval(&self.eig_vec, &concat_slices(&[sumcheck_point_eigen, &r1]))[0];
        let eig_val_s = oracle_table.get_eval(&self.eig_val, sumcheck_point_eigen)[0];
        let eig_vec_r0_s =
            oracle_table.get_eval(&self.eig_vec, &concat_slices(&[&r0, sumcheck_point_eigen]))[0];
        let eig_vec_r1_s =
            oracle_table.get_eval(&self.eig_vec, &concat_slices(&[&r1, sumcheck_point_eigen]))[0];

        let mut verify_eigen = l * eig_vec_s_r0 * eig_vec_s_r1
            + eig_val_s * eig_vec_r0_s * eig_vec_r1_s
            == sumcheck_oracle_eigen.oracle_eval;

        assert!(verify_eigen);

        verify_eigen &= l
            * (oracle_table.get_eval(&self.err_vec, &concat_slices(&[&r0, &r1]))[0]
                + self.power_2_f * self.power_2_f * eval_identity_poly(&r0, &r1))
            + a_eval * self.power_2_f
            - oracle_table.get_eval(&self.err_val, &concat_slices(&[&r0, &r1]))[0]
            == self.sumcheck_eigen.claimed_sum;

        debug_assert!(
            verify_eigen,
            "fail to verify oracle reduced from sumcheck eigen"
        );

        let sumcheck_oracle_square = <MLSumcheck<EF>>::verify(trans, &self.sumcheck_square)
            .expect("fail to verify sumcheck square");

        let sumcheck_point_square = sumcheck_oracle_square.sumcheck_point;
        let w_s_r0 =
            oracle_table.get_eval(&self.w, &concat_slices(&[&sumcheck_point_square, &r0]))[0];
        let w_s_r1 =
            oracle_table.get_eval(&self.w, &concat_slices(&[&sumcheck_point_square, &r1]))[0];

        let verify_square = w_s_r0 * w_s_r1 == sumcheck_oracle_square.oracle_eval;

        debug_assert!(
            verify_square,
            "fail to verify oracle reduced from sumcheck square in spectral norm iop of w {} a{}",
            self.w, self.a
        );

        verify_square && verify_eigen
    }
}

use super::LookupInstance;
use crate::pcs::OracleTable;
use crate::sumcheck::{MLSumcheck, SumcheckProof};
use crate::utils::{
    absolute_range_tables, add_dummy_back, add_dummy_front, concat_slices, eval_identity_poly,
    field_range_tables, field_to_i128, gen_identity_poly, i128_to_field, is_valid,
    vec_to_named_poly, 
};
use algebra::{AsFrom, PrimeField};
use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};

use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials, MultilinearExtension,
};

use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, rc::Rc, vec};

#[derive(Default)]
pub struct MetaDisparityInstance<F>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
{
    pub i_bit: usize,
    pub f_bit: usize,
    // data_s0 and data_s1 original len
    // before padding, as a <F> for mean check
    pub data_size: usize,
    pub data0_size: F,
    pub data1_size: F,

    // data_s0 and data_s1 padded len, and padded_feature_num
    pub num_vars_feat: usize,
    pub num_vars_data: usize,

    // bound error introduced by mean
    pub err_range: F,

    // dense multilinear extension

    // data is a matrix whose each row is a feature vector of a sample
    // num_vars: num_vars_smpl + num_vars_feat
    pub data: Rc<DenseMultilinearExtension<F>>,
    // num_vars: num_vars_smpl,
    pub group: Rc<DenseMultilinearExtension<F>>,

    // num_vars: num_vars_feat
    pub mean0: Rc<DenseMultilinearExtension<F>>,
    // num_vars: num_vars_feat
    pub mean1: Rc<DenseMultilinearExtension<F>>,

    // err0[j] = sum_i (1 - group[i]) (data[i][j] - mean0[j])
    // num_vars: num_vars_feat
    pub err0: Rc<DenseMultilinearExtension<F>>,

    // err1[j] = sum_i group[i] (data[i][j] - mean1[j])
    // num_vars: num_vars_feat
    pub err1: Rc<DenseMultilinearExtension<F>>,

    // max_deviation[j] = max (max_i (|data0[i][j] - mean0[j]|), max_i (|data1[i][j] - mean1[j]|)), i.e big delta
    // num_vars: num_vars_feat
    pub max_deviation: Rc<DenseMultilinearExtension<F>>,

    //disparity = |mean1 - mean0|,
    // num_vars: num_vars_feature
    pub disparity: Rc<DenseMultilinearExtension<F>>,

    // the sign of disparity
    // num_vars: num_vars_feature
    pub sign_disparity: Rc<DenseMultilinearExtension<F>>,

    // big delta big_disparity = |data - mean|
    // its max value is max_deviation in each feature
    // num_vars: num_vars_smpl + num_vars_feat, the same as data
    pub deviation: Rc<DenseMultilinearExtension<F>>,

    // the sign of big_disparity
    // num_vars: num_vars_smpl + num_vars_feat, the same as data
    pub sign_deviation: Rc<DenseMultilinearExtension<F>>,
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> MetaDisparityInstance<F> {
    // assume data group doesn't need to pad
    #[inline]
    pub fn new(
        num_vars_feat: usize,
        num_vars_data: usize,
        i_bit: usize,
        f_bit: usize,
        data: Rc<DenseMultilinearExtension<F>>,
        group: Rc<DenseMultilinearExtension<F>>,
    ) -> Self {
        data.iter().all(|x| is_valid(*x, i_bit, f_bit));
        group.iter().all(|x| is_valid(*x, i_bit, f_bit));

        let size_feat = 1 << num_vars_feat; // F
        let size_data = 1 << num_vars_data; // N

        debug_assert_eq!(data.num_vars(), num_vars_feat + num_vars_data);
        debug_assert_eq!(group.num_vars, num_vars_data);
        debug_assert!(group
            .iter()
            .all(|group_label| *group_label == F::zero() || *group_label == F::one()));

        // data is a column major matrix
        let data_i128 = field_to_i128(&data.evaluations);
        let group_i128 = field_to_i128(&group.evaluations);

        let data0_size = group_i128
            .iter()
            .filter(|group_label| **group_label == 0)
            .count() as i128;

        let data1_size = group_i128
            .iter()
            .filter(|group_label| **group_label == 1)
            .count() as i128;

        let mut sum0 = vec![0; size_feat];
        let mut sum1 = vec![0; size_feat];

        for i in 0..size_data {
            for j in 0..size_feat {
                if group_i128[i] == 0 {
                    sum0[j] += data_i128[j * size_data + i];
                } else {
                    sum1[j] += data_i128[j * size_data + i];
                }
            }
        }

        let mut mean0 = Vec::with_capacity(1 << num_vars_feat);
        let mut mean1 = Vec::with_capacity(1 << num_vars_feat);
        let mut err0 = Vec::with_capacity(1 << num_vars_feat);
        let mut err1 = Vec::with_capacity(1 << num_vars_feat);

        sum0.iter().for_each(|sum: &i128| {
            let mean_f64 = (*sum as f64) / (data0_size as f64);
            let mean_i128: i128 = mean_f64.round() as i128;
            let err_i128: i128 = sum - data0_size * mean_i128;
            mean0.push(mean_i128);
            err0.push(err_i128);
        });

        sum1.iter().for_each(|sum: &i128| {
            let mean_f64 = (*sum as f64) / (data1_size as f64);
            let mean_i128: i128 = mean_f64.round() as i128;
            let err_i128: i128 = sum - data0_size * mean_i128;
            mean1.push(mean_i128);
            err1.push(err_i128);
        });

        let err0: Vec<i128> = sum0
            .iter()
            .zip(mean0.iter())
            .map(|(sum, mean)| sum - data0_size * mean)
            .collect();

        let err1: Vec<i128> = sum1
            .iter()
            .zip(mean1.iter())
            .map(|(sum, mean)| sum - data1_size * mean)
            .collect();

        let max_error = err0
            .iter()
            .chain(err1.iter())
            .map(|&x| x.abs())
            .max()
            .unwrap_or(0);

        let err_range = max_error + 1;

        err0.iter().for_each(|err| {
            assert!(
                *err <= err_range,
                "error {} larger than bound{}",
                &err,
                &err_range
            )
        });
        err1.iter().for_each(|err| {
            assert!(
                *err <= err_range,
                "error {} larger than bound{}",
                &err,
                &err_range
            )
        });

        let (disparity, sign_disparity): (Vec<i128>, Vec<i128>) = mean0
            .iter()
            .zip(mean1.iter())
            .map(|(mean0, mean1)| {
                let diff = mean0 - mean1;
                let sign_d = if mean0 > mean1 { 1 } else { -1 };
                (diff.abs(), sign_d)
            })
            .unzip();

        let mut max_deviation = vec![0; size_feat];
        // compute big delta big_disparity = |data - mean|
        let mut big_disparity = vec![0; size_feat * size_data];
        let mut sign_big = vec![0; size_feat * size_data];
        for i in 0..size_data {
            for j in 0..size_feat {
                if group_i128[i] == 0 {
                    let deviation = data_i128[j * size_data + i] - mean0[j];
                    big_disparity[j * size_data + i] = deviation.abs();
                    sign_big[j * size_data + i] = if deviation > 0 { 1 } else { -1 };
                    if deviation.abs() > max_deviation[j] {
                        max_deviation[j] = deviation.abs();
                    }
                } else {
                    let deviation = data_i128[j * size_data + i] - mean1[j];
                    big_disparity[j * size_data + i] = deviation.abs();
                    sign_big[j * size_data + i] = if deviation > 0 { 1 } else { -1 };
                    if deviation.abs() > max_deviation[j] {
                        max_deviation[j] = deviation.abs();
                    }
                }
            }
        }

        Self {
            i_bit,
            f_bit,
            data_size: size_data,
            data0_size: F::new(F::Value::as_from(data0_size as u64)),
            data1_size: F::new(F::Value::as_from(data1_size as u64)),
            num_vars_feat,
            num_vars_data,
            data: data.clone(),
            group: group.clone(),
            mean0: vec_to_named_poly(
                &format!("[mean0 ({}, {})]", data.name, group.name),
                &i128_to_field(&mean0),
            ),
            mean1: vec_to_named_poly(
                &format!("[mean1 ({}, {})", data.name, group.name),
                &i128_to_field(&mean1),
            ),
            disparity: vec_to_named_poly(
                &format!("[disparity ({}, {})]", data.name, group.name),
                &i128_to_field(&disparity),
            ),
            sign_disparity: vec_to_named_poly(
                &format!("[sign of disparity ({},{})]", data.name, group.name),
                &i128_to_field(&sign_disparity),
            ),
            max_deviation: vec_to_named_poly(
                &format!("[max deviation ({}, {})]", data.name, group.name),
                &i128_to_field(&max_deviation),
            ),
            deviation: vec_to_named_poly(
                &format!("[deviation of ({}, {})]", data.name, group.name),
                &i128_to_field(&big_disparity),
            ),
            sign_deviation: vec_to_named_poly(
                &format!("[sign of deviation of ({}, {})]", data.name, group.name),
                &i128_to_field(&sign_big),
            ),
            err0: vec_to_named_poly(
                &format!("[error of mean0 ({}, {})]", data.name, group.name),
                &i128_to_field(&err0),
            ),
            err1: vec_to_named_poly(
                &format!("[error of mean1 ({}, {})", data.name, group.name),
                &i128_to_field(&err1),
            ),
            err_range: F::new(F::Value::as_from(err_range as u64)),
        }
    }

    #[inline]
    pub fn new_from_polys(
        sample_len: usize,
        data_s0_len: F,
        data_s1_len: F,
        num_vars_feature: usize,
        num_vars_smpl: usize,
        data: Rc<DenseMultilinearExtension<F>>,
        group: Rc<DenseMultilinearExtension<F>>,
        data_mean_s0: Rc<DenseMultilinearExtension<F>>,
        data_mean_s1: Rc<DenseMultilinearExtension<F>>,
        delta_h_vec: Rc<DenseMultilinearExtension<F>>,
        sign_d_s0_s1: Rc<DenseMultilinearExtension<F>>,
        delta_z_vec: Rc<DenseMultilinearExtension<F>>,
        error_s0_mean: Rc<DenseMultilinearExtension<F>>,
        error_s1_mean: Rc<DenseMultilinearExtension<F>>,
        epsilon: F,
    ) -> Self {
        // check shape
        debug_assert_eq!(data.num_vars(), num_vars_feature + num_vars_smpl);
        debug_assert_eq!(data_mean_s0.num_vars(), num_vars_feature);
        debug_assert_eq!(data_mean_s1.num_vars(), num_vars_feature);
        debug_assert!(delta_h_vec.num_vars() == num_vars_feature);
        debug_assert_eq!(sign_d_s0_s1.num_vars(), num_vars_feature);
        debug_assert_eq!(delta_z_vec.num_vars(), num_vars_feature);
        debug_assert_eq!(error_s0_mean.num_vars(), num_vars_feature);
        debug_assert_eq!(error_s1_mean.num_vars(), num_vars_feature);
        debug_assert_eq!(group.num_vars, num_vars_smpl);

        // data is a column major matrix
        let data_i128 = field_to_i128(&data.evaluations);
        let group_i128 = field_to_i128(&group.evaluations);
        let data_mean_s0_i128 = field_to_i128(&data_mean_s0.evaluations);
        let data_mean_s1_i128 = field_to_i128(&data_mean_s1.evaluations);

        let mut big_disparity_i128 = vec![0; data_i128.len()];
        let mut sign_big_i128 = vec![0; data_i128.len()];
        let smpl_size = 1 << num_vars_smpl;
        let feat_size = 1 << num_vars_feature;
        for i in 0..smpl_size {
            for j in 0..feat_size {
                if group_i128[i] == 0 {
                    let deviation = data_i128[j * smpl_size + i] - data_mean_s0_i128[j];
                    big_disparity_i128[j * smpl_size + i] = deviation.abs();
                    sign_big_i128[j * smpl_size + i] = if deviation > 0 { 1 } else { -1 };
                } else {
                    let deviation = data_i128[j * smpl_size + i] - data_mean_s1_i128[j];
                    big_disparity_i128[j * smpl_size + i] = deviation.abs();
                    sign_big_i128[j * smpl_size + i] = if deviation > 0 { 1 } else { -1 };
                }
            }
        }
        let big_disparity: Rc<DenseMultilinearExtension<F>> = vec_to_named_poly(
            &format!("big_disparity of data {}, group {}", data.name, group.name),
            &i128_to_field(&big_disparity_i128),
        );
        let sign_big: Rc<DenseMultilinearExtension<F>> = vec_to_named_poly(
            &format!(
                "sign of big_disparity of data {}, group {}",
                data.name, group.name
            ),
            &i128_to_field(&sign_big_i128),
        );

        Self {
            i_bit: 0,
            f_bit: 0,
            data_size: sample_len,
            data0_size: data_s0_len,
            data1_size: data_s1_len,
            num_vars_feat: num_vars_feature,
            num_vars_data: num_vars_smpl,
            data,
            group,
            mean0: data_mean_s0,
            mean1: data_mean_s1,
            disparity: delta_h_vec,
            sign_disparity: sign_d_s0_s1,
            max_deviation: delta_z_vec,
            deviation: big_disparity,
            sign_deviation: sign_big,
            err0: error_s0_mean,
            err1: error_s1_mean,
            err_range: epsilon,
        }
    }
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> MetaDisparityInstance<F> {
    pub fn lookup_instances<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let mut lookup_instances = Vec::new();
        lookup_instances.push(self.rangecheck_err());
        lookup_instances.push(self.rangecheck_data());
        lookup_instances.push(self.rangecheck_positive());
        lookup_instances
    }

    #[inline]
    fn rangecheck_positive<EF>(&self) -> LookupInstance<F, F, EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let range_positive = field_range_tables(
            self.num_vars_feat + self.num_vars_data,
            F::zero(),
            i128_to_field(&[(1 << (self.i_bit + self.f_bit + 1))])[0],
        );
        // subtract every big_disparity from max_deviation to prove it is positive
        let max_dev_eval = &self.max_deviation.evaluations;
        let big_disparity_eval = &self.deviation.evaluations;
        let mut lookup_entries = Vec::new();
        let smpl_size = 1 << self.num_vars_data;
        let feat_size = 1 << self.num_vars_feat;
        for j in 0..feat_size {
            let max_dev_j = max_dev_eval[j].clone();
            for i in 0..smpl_size {
                let deviation_i_j = big_disparity_eval[j * smpl_size + i].clone();
                lookup_entries.push(max_dev_j - deviation_i_j);
            }
        }

        LookupInstance::new(
            &vec![vec_to_named_poly(
                &format!(
                    "[minus ({}, {}) in meta",
                    self.max_deviation.name, self.deviation.name
                ),
                &lookup_entries,
            )],
            &range_positive,
            format!("rangecheck [positive encode]"),
            format!(
                "table [num vars {} positive rangecheck]",
                self.num_vars_data + self.num_vars_feat
            ),
            1,
        )
    }

    #[inline]
    pub fn rangecheck_err<EF>(&self) -> LookupInstance<F, F, EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let err_range: u64 = self.err_range.value().into();
        let tables_error = absolute_range_tables(self.num_vars_feat, err_range as usize);

        LookupInstance::new(
            &[self.err0.clone(), self.err1.clone()],
            &tables_error,
            format!("rangecheck ere"),
            "table [error]".to_string(),
            1,
        )
    }

    #[inline]
    pub fn rangecheck_data<EF>(&self) -> LookupInstance<F, F, EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let range_i_f = absolute_range_tables(self.data.num_vars, 1 << (self.i_bit + self.f_bit));
        LookupInstance::new(
            &vec![self.data.clone()],
            &range_i_f,
            format!("lookup [(valid encode, num_vars {})]", self.data.num_vars),
            format!("t of [(valide encode, num vars {})]", self.data.num_vars,),
            1,
        )
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> MetaDisparityInstance<F> {
    // transform this instance on base field to an instance on extension field
    pub fn to_ef<EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>>(
        &self,
    ) -> MetaDisparityInstance<EF> {
        MetaDisparityInstance {
            i_bit: self.i_bit,
            f_bit: self.f_bit,
            data_size: self.data_size,
            data0_size: EF::from(self.data0_size),
            data1_size: EF::from(self.data1_size),
            num_vars_feat: self.num_vars_feat,
            num_vars_data: self.num_vars_data,
            data: Rc::new(self.data.to_ef()),
            group: Rc::new(self.group.to_ef()),
            mean0: Rc::new(self.mean0.to_ef()),
            mean1: Rc::new(self.mean1.to_ef()),
            disparity: Rc::new(self.disparity.to_ef()),
            sign_disparity: Rc::new(self.sign_disparity.to_ef()),
            max_deviation: Rc::new(self.max_deviation.to_ef()),
            deviation: Rc::new(self.deviation.to_ef()),
            sign_deviation: Rc::new(self.sign_deviation.to_ef()),
            err0: Rc::new(self.err0.to_ef()),
            err1: Rc::new(self.err1.to_ef()),
            err_range: EF::from(self.err_range),
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
        table.add_named_oracle(self.data.clone());
        table.add_named_oracle(self.group.clone());
        table.add_named_oracle(self.mean0.clone());
        table.add_named_oracle(self.mean1.clone());
        table.add_named_oracle(self.disparity.clone());
        table.add_named_oracle(self.sign_disparity.clone());
        table.add_named_oracle(self.max_deviation.clone());
        // table.add_named_oracle(self.err0.clone());
        // table.add_named_oracle(self.err1.clone());
        table.add_super_oracle(
            &"(err0, err1)".to_string(),
            &[self.err0.clone(), self.err1.clone()],
        );
        table.add_named_oracle(self.deviation.clone());
        table.add_named_oracle(self.sign_deviation.clone());
    }
}

#[derive(Default, Serialize)]
pub struct MetaDisparityIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    // data_s0 and data_s1 original len
    // before padding, as a <F> for mean check
    sample_len: usize,
    num_smpl_0: EF,
    num_smpl_1: EF,

    // data_s0 and data_s1 padded len, and padded_feature_num
    num_vars_feat: usize,
    num_vars_smpl: usize,

    // bound error introduced by mean
    epsilon: EF,

    data: String,
    group: String,

    mean0: String,
    mean1: String,
    err0: String,
    err1: String,

    disparity: String,
    sign_disparity: String,

    deviation: String,
    sign_deviation: String,

    max_deviation: String,

    pub proof_mean: SumcheckProof<EF>,

    pub proof_disparity: SumcheckProof<EF>,

    pub proof_big_disparity: SumcheckProof<EF>,

    _marker: PhantomData<(F, EF, H, C, S)>,
}

impl<F, EF, H, C, S> MetaDisparityIOP<F, EF, H, C, S>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    H: Hash + Sync + Send,
    C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
    S: LinearCodeSpec<F, Code = C> + Clone,
{
    pub fn info(&mut self, instance: &MetaDisparityInstance<EF>) {
        self.sample_len = instance.data_size;
        self.num_smpl_0 = instance.data0_size;
        self.num_smpl_1 = instance.data1_size;
        self.num_vars_feat = instance.num_vars_feat;
        self.num_vars_smpl = instance.num_vars_data;
        self.epsilon = instance.err_range;
        self.data = instance.data.name.clone();
        self.group = instance.group.name.clone();
        self.mean0 = instance.mean0.name.clone();
        self.mean1 = instance.mean1.name.clone();
        self.disparity = instance.disparity.name.clone();
        self.sign_disparity = instance.sign_disparity.name.clone();
        self.max_deviation = instance.max_deviation.name.clone();
        self.err0 = instance.err0.name.clone();
        self.err1 = instance.err1.name.clone();
        self.deviation = instance.deviation.name.clone();
        self.sign_deviation = instance.sign_deviation.name.clone();
    }

    pub fn prove(
        &mut self,
        instance: &mut MetaDisparityInstance<EF>,
        oracle_table: &mut OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) {
        self.info(&instance);

        // transcript challenge
        let r0 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat);
        // let r1 =
        //     trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat);
        let r2 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat);
        let r3 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_smpl); // row
        let r4 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat); // column
        let l = trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", 3);

        // \sum_h data(h, r0)(1 - group(h)) = n0 * mean0(r0) + err0(r0)
        // \sum_h data(h, r1) group(h) = n1 * mean1(r1) + err1(r1)
        // batch =>
        // \sum_h data(h, r0) - data(h, r0) group(h) + l0 data(h, r1) group(h) = n0 * mean0(r0) + err0(r0) + l0 * n1 * mean1(r1) + l0 * err1(r1)

        // new
        // |h| = num_vars_data, \r\ = num_vars_feat
        // \sum_h data(h, r)(1 - group(h)) = n0 * mean0(r) + err0(r)
        // \sum_h data(h, r) group(h) = n1 * mean1(r) + err1(r)
        // batch =>
        // \sum_h data(h, r) + (l0-1) data(h, r) group(h) = n0 * mean0(r) + err0(r) + l0 * n1 * mean1(r) + l0 * err1(r)
        let data_h_r0 = Rc::new(instance.data.fix_variables_back(&r0));
        //let data_h_r1 = Rc::new(instance.data.fix_variables_back(&r1));

        let mut poly_mean = ListOfProductsOfPolynomials::<EF>::new(self.num_vars_smpl);
        poly_mean.add_product([data_h_r0.clone()], EF::one());
        poly_mean.add_product(
            [data_h_r0.clone(), instance.group.clone()],
            l[0] - EF::one(),
        );
        //poly_mean.add_product([data_h_r1.clone(), instance.group.clone()], l[0]);

        let (mean_proof, mean_state) =
            MLSumcheck::prove(trans, &poly_mean).expect("fail to prove the sumcheck protocol");
        let mean_point = mean_state.randomness;

        let mean0_r0 = instance.mean0.evaluate(&r0);
        let err0_r0 = instance.err0.evaluate(&r0);
        let mean1_r0 = instance.mean1.evaluate(&r0);
        let err1_r0 = instance.err1.evaluate(&r0);

        self.proof_mean = SumcheckProof {
            proof: mean_proof,
            info: poly_mean.info(),
            claimed_sum: self.num_smpl_0 * mean0_r0
                + err0_r0
                + l[0] * (self.num_smpl_1 * mean1_r0 + err1_r0),
        };

        // data(s, r0), data(s, r1), group(s), mean0(r0), err0(r0), mean1(r1), err1(r1)
        //oracle_table.add_point(&instance.data.name, &concat_slices(&[&mean_point, &r0]));
        oracle_table.add_point(&instance.data.name, &concat_slices(&[&mean_point, &r0]));
        oracle_table.add_point(&instance.group.name, &mean_point);
        oracle_table.add_point(&instance.mean0.name, &r0);
        oracle_table.add_point(&instance.err0.name, &r0);
        oracle_table.add_point(&instance.mean1.name, &r0);
        oracle_table.add_point(&instance.err1.name, &r0);

        // \forall i, disparity(i) - sign_d(i) * (mean0(i) - mean1(i)) = 0
        // \forall i, sign_d(i) * sign_d(i) = 1
        // linear combination of the above two equations with randomness l1

        // |h| = num_vars_feat
        // \sum_h eq(h,r) * disparity(h) - eq(h,r) * sign_d(h) * mean0(h) + eq(h,r) * sign_d(h) * mean1(h) + l1 * eq(h,r) * sign_d(h) * sign_d(h) = l1
        let eq_r2 = Rc::new(gen_identity_poly(&r2));
        let mut poly_disparity = ListOfProductsOfPolynomials::<EF>::new(self.num_vars_feat);
        poly_disparity.add_product([eq_r2.clone(), instance.disparity.clone()], EF::one());
        poly_disparity.add_product(
            [
                eq_r2.clone(),
                instance.sign_disparity.clone(),
                instance.mean0.clone(),
            ],
            -EF::one(),
        );
        poly_disparity.add_product(
            [
                eq_r2.clone(),
                instance.sign_disparity.clone(),
                instance.mean1.clone(),
            ],
            EF::one(),
        );
        poly_disparity.add_product(
            [
                eq_r2.clone(),
                instance.sign_disparity.clone(),
                instance.sign_disparity.clone(),
            ],
            l[1],
        );
        let (proof_disparity, state_disparity) =
            MLSumcheck::prove(trans, &poly_disparity).expect("fail to prove the sumcheck protocol");
        let sumcheck_point_disparity = state_disparity.randomness;

        self.proof_disparity = SumcheckProof {
            proof: proof_disparity,
            info: poly_disparity.info(),
            claimed_sum: l[1],
        };

        // disparity(s), sign_d(s), mean0(s), mean1(s)
        oracle_table.add_point(&instance.disparity.name, &sumcheck_point_disparity);
        oracle_table.add_point(&instance.sign_disparity.name, &sumcheck_point_disparity);
        oracle_table.add_point(&instance.mean0.name, &sumcheck_point_disparity);
        oracle_table.add_point(&instance.mean1.name, &sumcheck_point_disparity);
        // deviation(i,j) =  sign_deviation(i,j) * (data(i,j) - group(i) * mean_s1(j) - (1-group(i)) * mean_s0(j))
        // \sum_i_N \sum_j_F eq(r3,i)*eq(r4,j) * (sign_big(i,j) * (X(i,j) - group(i)*mean_s1(j) - (1-group(i))*mean_s0(j)) - big_disparity(i,j)) + l2 * eq(r3,i)*eq(r4,j) * sign_big(i,j) * sign_big(i,j) = l2

        let eq_r3 = Rc::new(gen_identity_poly(&r3));
        let eq_r4 = Rc::new(gen_identity_poly(&r4));
        let eq_r3_pad = Rc::new(add_dummy_back(&eq_r3, self.num_vars_feat)); // r3 is for row, add col at back
        let eq_r4_pad = Rc::new(add_dummy_front(&eq_r4, self.num_vars_smpl)); // r4 is for col, add row at front
        let group_pad = Rc::new(add_dummy_back(&instance.group, self.num_vars_feat)); // group is idx by row
        let mean0_pad = Rc::new(add_dummy_front(&instance.mean0, self.num_vars_smpl)); // mean0 is idx by col
        let mean1_pad = Rc::new(add_dummy_front(&instance.mean1, self.num_vars_smpl)); // mean1 is idx by col
        let mut poly_big_disparity =
            ListOfProductsOfPolynomials::<EF>::new(self.num_vars_smpl + self.num_vars_feat);
        poly_big_disparity.add_product(
            [
                eq_r3_pad.clone(),
                eq_r4_pad.clone(),
                instance.sign_deviation.clone(),
                instance.data.clone(),
            ],
            EF::one(),
        );
        poly_big_disparity.add_product(
            [
                eq_r3_pad.clone(),
                eq_r4_pad.clone(),
                instance.sign_deviation.clone(),
                group_pad.clone(),
                mean1_pad.clone(),
            ],
            -EF::one(),
        );
        poly_big_disparity.add_product(
            [
                eq_r3_pad.clone(),
                eq_r4_pad.clone(),
                instance.sign_deviation.clone(),
                mean0_pad.clone(),
            ],
            -EF::one(),
        );
        poly_big_disparity.add_product(
            [
                eq_r3_pad.clone(),
                eq_r4_pad.clone(),
                instance.sign_deviation.clone(),
                group_pad.clone(),
                mean0_pad.clone(),
            ],
            EF::one(),
        );
        poly_big_disparity.add_product(
            [
                eq_r3_pad.clone(),
                eq_r4_pad.clone(),
                instance.deviation.clone(),
            ],
            -EF::one(),
        );
        poly_big_disparity.add_product(
            [
                eq_r3_pad.clone(),
                eq_r4_pad.clone(),
                instance.sign_deviation.clone(),
                instance.sign_deviation.clone(),
            ],
            l[2],
        );
        let (proof_big_disparity, state_big_disparity) =
            MLSumcheck::prove(trans, &poly_big_disparity)
                .expect("fail to prove the sumcheck protocol");
        let sumcheck_point_big_disparity = state_big_disparity.randomness;

        self.proof_big_disparity = SumcheckProof {
            proof: proof_big_disparity,
            info: poly_big_disparity.info(),
            claimed_sum: l[2],
        };

        // big_disparity(s), sign_big(s), data(s), group(s_i), mean0(s_j), mean1(s_j)
        oracle_table.add_point(&instance.deviation.name, &sumcheck_point_big_disparity);
        oracle_table.add_point(&instance.sign_deviation.name, &sumcheck_point_big_disparity);
        oracle_table.add_point(&instance.data.name, &sumcheck_point_big_disparity);
        oracle_table.add_point(
            &instance.group.name,
            &sumcheck_point_big_disparity[0..self.num_vars_smpl],
        );
        oracle_table.add_point(
            &instance.mean0.name,
            &sumcheck_point_big_disparity[self.num_vars_smpl..],
        );
        oracle_table.add_point(
            &instance.mean1.name,
            &sumcheck_point_big_disparity[self.num_vars_smpl..],
        );
    }

    pub fn verify(
        &self,
        oracle_table: &OracleTable<F, EF, H, C, S>,
        trans: &mut Transcript<EF>,
    ) -> bool {
        let r0 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat); // for mean s=0
                                                                                                   // let r1 =
                                                                                                   //     trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat); // for mean s=1
        let r2 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat); // for delta_h_vec check
        let r3 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_smpl); // for big_disparity check
        let r4 =
            trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", self.num_vars_feat); // for big_disparity check
        let l = trans.get_vec_challenge(b"linear point in schwartz-zippel lemma", 3); // for delta_h_vec check

        // \sum_h data(h, r0)(1 - group(h)) = n0 * mean0(r0) + err0(r0)
        // \sum_h data(h, r1) group(h) = n1 * mean1(r1) + err1(r1)
        // batch =>
        // \sum_h data(h, r0) - data(h, r0) group(h) + l0 data(h, r1) group(h) = n0 * mean0(r0) + err0(r0) + l0 * n1 * mean1(r1) + l0 * err1(r1)

        // data(s, r0), data(s, r1), group(s), mean0(r0), err0(r0), mean1(r1), err1(r1)
        let sumcheck_oracle_mean = MLSumcheck::verify(trans, &self.proof_mean)
            .expect("fail to verify the sumcheck protocol");
        // let data_s_r0_eval = oracle_table.get_eval(
        //     &self.data,
        //     &concat_slices(&[&sumcheck_oracle_mean.sumcheck_point, &r0]),
        // )[0];
        let data_s_r0_eval = oracle_table.get_eval(
            &self.data,
            &concat_slices(&[&sumcheck_oracle_mean.sumcheck_point, &r0]),
        )[0];
        let group_s_eval =
            oracle_table.get_eval(&self.group, &sumcheck_oracle_mean.sumcheck_point)[0];
        let mean0_r0_eval = oracle_table.get_eval(&self.mean0, &r0)[0];
        let mean1_r0_eval = oracle_table.get_eval(&self.mean1, &r0)[0];
        let err0_r0_eval = oracle_table.get_eval(&self.err0, &r0)[0];
        let err1_r0_eval = oracle_table.get_eval(&self.err1, &r0)[0];

        // sumcheck_eval = data(s, r0) - data(s, r0) group(s) + l0 data(s, r1) group(s)
        // sumcheck_sum = n0 * mean0(r0) + err0(r0) + l0 * n1 * mean1(r1) + l0 * err1(r1)
        let mut check_mean = sumcheck_oracle_mean.oracle_eval
            == data_s_r0_eval * (EF::one() - group_s_eval) + l[0] * data_s_r0_eval * group_s_eval;
        check_mean &= self.proof_mean.claimed_sum
            == self.num_smpl_0 * mean0_r0_eval
                + err0_r0_eval
                + l[0] * (self.num_smpl_1 * mean1_r0_eval + err1_r0_eval);

        // \sum_h eq(h,r) * disparity(h) - eq(h,r) * sign_d(h) * mean0(h) + eq(h,r) * sign_d(h) * mean1(h) + l1 * eq(h,r) * sign_d(h) * sign_d(h) = l1
        let sumcheck_oracle_disparity = MLSumcheck::verify(trans, &self.proof_disparity)
            .expect("fail to verify the sumcheck protocol");
        let eq_r2_eval = eval_identity_poly(&r2, &sumcheck_oracle_disparity.sumcheck_point);
        let disparity_eval =
            oracle_table.get_eval(&self.disparity, &sumcheck_oracle_disparity.sumcheck_point)[0];
        let sign_d_eval = oracle_table.get_eval(
            &self.sign_disparity,
            &sumcheck_oracle_disparity.sumcheck_point,
        )[0];
        let mean_0_eval =
            oracle_table.get_eval(&self.mean0, &sumcheck_oracle_disparity.sumcheck_point)[0];
        let mean_1_eval =
            oracle_table.get_eval(&self.mean1, &sumcheck_oracle_disparity.sumcheck_point)[0];

        // sumcheck_eval = eq(h,r) * disparity(h) - eq(h,r) * sign_d(h) * mean0(h) + eq(h,r) * sign_d(h) * mean1(h) + l1 * eq(h,r) * sign_d(h) * sign_d(h) = l1
        let mut check_disparity = sumcheck_oracle_disparity.oracle_eval
            == eq_r2_eval * disparity_eval - eq_r2_eval * sign_d_eval * mean_0_eval
                + eq_r2_eval * sign_d_eval * mean_1_eval
                + l[1] * eq_r2_eval * sign_d_eval * sign_d_eval;
        check_disparity &= self.proof_disparity.claimed_sum == l[1];

        // prepare eq(r3,i) eq(r4,j) sign_big(s) big_disparity(s) group(i) mean_s1(j) mean_s0(j) data(i,j)
        let sumcheck_oracle_big_disparity = MLSumcheck::verify(trans, &self.proof_big_disparity)
            .expect("fail to verify the sumcheck protocol");
        let eq_r3_eval = eval_identity_poly(
            &r3,
            &sumcheck_oracle_big_disparity.sumcheck_point[..self.num_vars_smpl],
        );
        let eq_r4_eval = eval_identity_poly(
            &r4,
            &sumcheck_oracle_big_disparity.sumcheck_point[self.num_vars_smpl..],
        );
        let sign_big_s_eval = oracle_table.get_eval(
            &self.sign_deviation,
            &sumcheck_oracle_big_disparity.sumcheck_point,
        )[0];
        let big_disparity_s_eval = oracle_table.get_eval(
            &self.deviation,
            &sumcheck_oracle_big_disparity.sumcheck_point,
        )[0];
        // s = i * num_vars_feat + j
        let group_i_eval = oracle_table.get_eval(
            &self.group,
            &sumcheck_oracle_big_disparity.sumcheck_point[..self.num_vars_smpl],
        )[0];
        let mean_s1_j_eval = oracle_table.get_eval(
            &self.mean1,
            &sumcheck_oracle_big_disparity.sumcheck_point[self.num_vars_smpl..],
        )[0];
        let mean_s0_j_eval = oracle_table.get_eval(
            &self.mean0,
            &sumcheck_oracle_big_disparity.sumcheck_point[self.num_vars_smpl..],
        )[0];
        let data_s_eval =
            oracle_table.get_eval(&self.data, &sumcheck_oracle_big_disparity.sumcheck_point)[0];

        // \sum_iN \sum_jF eq(r3,i)*eq(r4,j) * (sign_big(i,j) * (X(i,j) - group(i)*mean_s1(j) - (1-group(i))*mean_s0(j)) - big_disparity(i,j)) + l2 * eq(r3,i)*eq(r4,j) * sign_big(i,j) * sign_big(i,j) = l2
        let mut check_big_disparity = sumcheck_oracle_big_disparity.oracle_eval
            == eq_r3_eval
                * eq_r4_eval
                * (sign_big_s_eval
                    * (data_s_eval
                        - group_i_eval * mean_s1_j_eval
                        - (EF::one() - group_i_eval) * mean_s0_j_eval)
                    - big_disparity_s_eval)
                + l[2] * eq_r3_eval * eq_r4_eval * sign_big_s_eval * sign_big_s_eval;
        check_big_disparity &= self.proof_big_disparity.claimed_sum == l[2];

        check_mean && check_disparity && check_big_disparity
    }
}

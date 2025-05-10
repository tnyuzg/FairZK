use super::{InfairenceInstance, LookupInstance, MetaDisparityInstance, SpectralNormInstance};
use crate::piop::{BoundInstance, L2NormInstance};
use crate::utils::{absolute_range_tables, u64_to_field};
use crate::{pcs::OracleTable, utils::is_valid};
use algebra::PrimeField;
use pcs::{
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
};

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};

use serde::{Deserialize, Serialize};
use std::{rc::Rc, vec};

// x0 w0 y0 | r0 x1 w1 y1 | r1 x2 w2 y2 | r2 x3 w3 y3
pub struct TotalInstance<F: Field + Serialize + for<'de> Deserialize<'de>> {
    pub num_layers: usize,
    pub num_vars_feats: Vec<usize>,
    pub i: usize,
    pub f: usize,

    pub w: Vec<Rc<DenseMultilinearExtension<F>>>,

    pub meta_disparity_instance: MetaDisparityInstance<F>,
    pub infairence_instance: InfairenceInstance<F>,
    pub spectral_norm_instances: Vec<SpectralNormInstance<F>>,
    pub l2_norms: Vec<L2NormInstance<F>>,
    pub bound: BoundInstance<F>,
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> TotalInstance<F> {
    #[inline]
    pub fn new(
        num_layers: usize,
        num_vars_data: usize,
        num_vars_feats: &[usize],
        i_bit: usize,
        f_bit: usize,
        w: Vec<Rc<DenseMultilinearExtension<F>>>,
        data: Rc<DenseMultilinearExtension<F>>,
        group: Rc<DenseMultilinearExtension<F>>,
    ) -> Self {
        assert!(num_layers + 1 == num_vars_feats.len());
        w.iter()
            .flat_map(|w| w.iter())
            .all(|x| is_valid(*x, i_bit, f_bit));
        data.iter().all(|x| is_valid(*x, i_bit, f_bit));
        group.iter().all(|x| is_valid(*x, i_bit, f_bit));

        let meta_disparity_instance =
            MetaDisparityInstance::new(num_vars_feats[0], num_vars_data, i_bit, f_bit, data, group);

        let infairence_instance = InfairenceInstance::new_overflow(
            num_layers,
            &num_vars_feats,
            i_bit,
            f_bit,
            &meta_disparity_instance.max_deviation,
            &w,
        );

        let mut spectral_norm_instances = Vec::with_capacity(num_layers);
        for i in 0..num_layers - 1 {
            let num_vars_row = num_vars_feats[i];
            let num_vars_col = num_vars_feats[i + 1];
            let sp_norm_instance =
                SpectralNormInstance::new(num_vars_row, num_vars_col, i_bit, f_bit, &w[i]);
            spectral_norm_instances.push(sp_norm_instance);
        }

        let mut l2_norms: Vec<L2NormInstance<F>> = Vec::new();

        // l2 norm from delta_h_vec to delta_0
        l2_norms.push(L2NormInstance::new_essential(
            meta_disparity_instance.disparity.num_vars,
            &meta_disparity_instance.disparity,
        ));

        // l2 norm from Delta_z_vecs to Delta_z's
        for i in 0..num_layers {
            l2_norms.push(L2NormInstance::new_essential(
                infairence_instance.x[i].num_vars,
                &infairence_instance.x[i],
            ));
        }

        // the last layer needs a different l2 norm
        l2_norms.push(L2NormInstance::new_essential(
            num_vars_feats[num_layers - 1],
            &w[num_layers - 1],
        ));

        // get eigen values from spectral norm instances
        let mut eigen_values = Vec::with_capacity(num_layers);
        for i in 0..num_layers - 1 {
            eigen_values.push(u64_to_field(&[spectral_norm_instances[i].spectral_norm as u64])[0]);
        }
        eigen_values.push(l2_norms.last().unwrap().value * l2_norms.last().unwrap().value);

        let delta_0 = l2_norms[0].value;

        // get delta_z_l2_norms from infairence instance
        let mut delta_z_l2_norms = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            delta_z_l2_norms.push(l2_norms[i + 1].value);
        }

        // pad the layers to the next power of two
        let num_vars_layer = num_layers.next_power_of_two().ilog2() as usize;
        if eigen_values.len() < num_layers.next_power_of_two() {
            eigen_values.resize(num_layers.next_power_of_two(), F::zero());
        }
        if delta_z_l2_norms.len() < num_layers.next_power_of_two() {
            delta_z_l2_norms.resize(num_layers.next_power_of_two(), F::zero());
        }
        let bound = BoundInstance::new_essential(
            num_vars_layer,
            f_bit,
            &Rc::new(DenseMultilinearExtension::from_named_vec(
                &"eigen_values".to_string(),
                num_vars_layer,
                eigen_values,
            )),
            &Rc::new(DenseMultilinearExtension::from_named_vec(
                &"delta_z_l2_norms".to_string(),
                num_vars_layer,
                delta_z_l2_norms,
            )),
            delta_0,
            u64_to_field(&[1 << 2 * f_bit as u64])[0],
            u64_to_field(&[1])[0],
        );

        Self {
            num_layers,
            num_vars_feats: num_vars_feats.to_vec(),
            i: i_bit,
            f: f_bit,
            w,
            meta_disparity_instance,
            infairence_instance,
            spectral_norm_instances,
            l2_norms,
            bound,
        }
    }
}

impl<F: PrimeField + Serialize + for<'de> Deserialize<'de>> TotalInstance<F> {
    #[inline]
    pub fn lookups<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let mut lookups = Vec::new();

        lookups.extend(self.infairence_instance.lookups_instances());
        self.spectral_norm_instances
            .iter()
            .for_each(|sn| lookups.extend(sn.lookup_instances()));
        lookups
    }

    #[inline]
    pub fn rangecheck_w<EF>(&self) -> Vec<LookupInstance<F, F, EF>>
    where
        F: Field + Serialize + for<'de> Deserialize<'de>,
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let mut lookups = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let num_vars = self.w[i].num_vars;
            let range_i_f = absolute_range_tables(self.w[i].num_vars, 1 << (self.i + self.f));
            let rangecheck_w_i = LookupInstance::new(
                &vec![self.w[i].clone()],
                &range_i_f,
                format!(
                    "{} rangecheck i {} f {} num_vars {} ",
                    self.w[i].name, num_vars, self.i, self.f
                ),
                format!("num_vars {} rangecheck i {} f {}", num_vars, self.i, self.f),
                1,
            );
            lookups.push(rangecheck_w_i);
        }
        lookups
    }
}

impl<F: Field + Serialize + for<'de> Deserialize<'de>> TotalInstance<F> {
    #[inline]
    pub fn construct_oracles<EF, H, C, S>(&self, table: &mut OracleTable<F, EF, H, C, S>)
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
        H: Hash + Sync + Send,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        // self.meta_disparity_instance.construct_oracles(table);
        self.infairence_instance.construct_oracles(table);
        self.spectral_norm_instances
            .iter()
            .for_each(|x| x.construct_oracles(table));
        self.bound.construct_oracles(table);
        self.l2_norms
            .iter()
            .for_each(|l2| l2.construct_oracles(table));
    }

    #[inline]
    pub fn from_instances(
        num_layers: usize,
        num_vars_feat: Vec<usize>,
        i: usize,
        f: usize,
        w: Vec<Rc<DenseMultilinearExtension<F>>>,
        spectral_norm_instances: Vec<SpectralNormInstance<F>>,
        infairence_instance: InfairenceInstance<F>,
        meta_disparity_instance: MetaDisparityInstance<F>,
        bound: BoundInstance<F>,
        l2_norms: Vec<L2NormInstance<F>>,
    ) -> Self {
        TotalInstance {
            num_layers,
            num_vars_feats: num_vars_feat,
            i,
            f,
            w,
            spectral_norm_instances,
            infairence_instance,
            meta_disparity_instance,
            bound,
            l2_norms,
        }
    }

    #[inline]
    pub fn to_ef<EF>(&self) -> TotalInstance<EF>
    where
        EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
    {
        let w = self.w.iter().map(|w| Rc::new(w.to_ef())).collect();
        let spectral_norm_instances = self
            .spectral_norm_instances
            .iter()
            .map(|sn| sn.to_ef())
            .collect();
        let infairence_instance = self.infairence_instance.to_ef();
        let meta_disparity_instance = self.meta_disparity_instance.to_ef();
        let bound = self.bound.to_ef();
        let l2_norms = self.l2_norms.iter().map(|l2| l2.to_ef()).collect();

        TotalInstance::from_instances(
            self.num_layers,
            self.num_vars_feats.clone(),
            self.i,
            self.f,
            w,
            spectral_norm_instances,
            infairence_instance,
            meta_disparity_instance,
            bound,
            l2_norms,
        )
    }
}

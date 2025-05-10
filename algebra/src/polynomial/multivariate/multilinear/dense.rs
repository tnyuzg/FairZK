// It is derived from https://github.com/arkworks-rs/sumcheck.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Index, Neg, Sub, SubAssign};
use std::slice::{Iter, IterMut};
use std::vec;

use num_traits::Zero;
use rand_distr::Distribution;

use crate::{AbstractExtensionField, Field, FieldUniformSampler};

use super::MultilinearExtension;

/// Stores a multilinear polynomial in dense evaluation form.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct DenseMultilinearExtension<F: Field> {
    /// name
    pub name: String,
    /// Number of variables
    pub num_vars: usize,
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<F>,
}

impl<F: Field> DenseMultilinearExtension<F> {
    /// clear
    #[inline]
    pub fn clear(&mut self) {
        self.evaluations = Vec::new();
    }

    /// name a dense multilinear extension
    #[inline]
    pub fn name(&mut self, name: &String) {
        self.name = name.clone();
    }

    /// Construct an empty instance
    #[inline]
    pub fn new(num_vars: usize) -> Self {
        Self::from_vec(num_vars, vec![F::zero(); 1 << num_vars])
    }

    /// new empty
    #[inline]
    pub fn new_empty(name: &String, num_vars: usize) -> Self {
        Self {
            name: name.clone(),
            num_vars,
            evaluations: Vec::new(),
        }
    }

    /// Construct a new polynomial from a list of evaluations where the index
    /// represents a point in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    #[inline]
    pub fn from_slice(num_vars: usize, evaluations: &[F]) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );
        Self::from_vec(num_vars, evaluations.to_vec())
    }

    /// Construct a new polynomial from a list of evaluations where the index
    /// represents a point in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    #[inline]
    pub fn from_vec(num_vars: usize, evaluations: Vec<F>) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            name: "".to_string(),
            num_vars,
            evaluations,
        }
    }

    /// named
    #[inline]
    pub fn from_named_slice(name: &String, num_vars: usize, evaluations: &[F]) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            name: name.to_string(),
            num_vars,
            evaluations: evaluations.to_vec(),
        }
    }

    /// named   
    #[inline]
    pub fn from_named_vec(name: &String, num_vars: usize, evaluations: Vec<F>) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            name: name.to_string(),
            num_vars,
            evaluations: evaluations,
        }
    }

    /// Returns an iterator that iterates over the evaluations over {0,1}^`num_vars`
    #[inline]
    pub fn iter(&self) -> Iter<'_, F> {
        self.evaluations.iter()
    }

    /// Returns a mutable iterator that iterates over the evaluations over {0,1}^`num_vars`
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, F> {
        self.evaluations.iter_mut()
    }

    /// Split the mle into two mles with one less variable, eliminating the far right variable
    /// original evaluations: f(x, b) for x \in \{0, 1\}^{k-1} and b\{0, 1\}
    /// resulting two mles: f0(x) = f(x, 0) for x \in \{0, 1\}^{k-1} and f1(x) = f(x, 1) for x \in \{0, 1\}^{k-1}
    pub fn split_halves(&self) -> (Self, Self) {
        let left = Self::from_slice(
            self.num_vars - 1,
            &self.evaluations[0..1 << (self.num_vars - 1)],
        );
        let right = Self::from_slice(
            self.num_vars - 1,
            &self.evaluations[1 << (self.num_vars - 1)..],
        );
        (left, right)
    }

    /// Evaluate a point in the extension field.
    #[inline]
    pub fn evaluate_ext<EF>(&self, ext_point: &[EF]) -> EF
    where
        EF: AbstractExtensionField<F>,
    {
        let mut poly: Vec<_> = self
            .evaluations
            .iter()
            .map(|&eval| EF::from_base(eval))
            .collect();
        let nv = self.num_vars;
        let dim = ext_point.len();
        // evaluate nv variable of partial point from left to right
        // with dim rounds and \sum_{i=1}^{dim} 2^(nv - i)
        // (If dim = nv, then the complexity is 2^{nv}.)
        for i in 1..dim + 1 {
            // fix a single variable to evaluate (1 << (nv - i)) evaluations from the last round
            // with complexity of 2^(1 << (nv - i)) field multiplications
            let r = ext_point[i - 1];
            for b in 0..(1 << (nv - i)) {
                let left = poly[b << 1];
                let right = poly[(b << 1) + 1];
                poly[b] = r * (right - left) + left;
            }
        }
        poly.truncate(1 << (nv - dim));
        poly[0]
    }

    /// Evaluate a point in the extension field.
    #[inline]
    pub fn evaluate_ext_opt<EF>(&self, eq_at_r: &DenseMultilinearExtension<EF>) -> EF
    where
        EF: AbstractExtensionField<F>,
    {
        eq_at_r
            .iter()
            .zip(self.iter())
            .fold(EF::zero(), |acc, (c, val)| acc + *c * *val)
    }

    /// Convert to EF version
    #[inline]
    pub fn to_ef<EF: AbstractExtensionField<F>>(&self) -> DenseMultilinearExtension<EF> {
        DenseMultilinearExtension::<EF> {
            name: self.name.clone(),
            num_vars: self.num_vars,
            evaluations: self.evaluations.iter().map(|x| EF::from_base(*x)).collect(),
        }
    }
}

impl<F: Field> MultilinearExtension<F> for DenseMultilinearExtension<F> {
    type Point = [F];

    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn evaluate(&self, point: &Self::Point) -> F {
        assert_eq!(point.len(), self.num_vars, "The point size is invalid.");
        self.fix_variables_front(point)[0]
    }

    #[inline]
    fn random<R>(num_vars: usize, rng: &mut R) -> Self
    where
        R: rand::Rng + rand::CryptoRng,
    {
        Self {
            name: "".to_string(),
            num_vars,
            evaluations: FieldUniformSampler::new()
                .sample_iter(rng)
                .take(1 << num_vars)
                .collect(),
        }
    }

    fn fix_variables_front(&self, partial_point: &[F]) -> Self {
        assert!(
            partial_point.len() <= self.num_vars,
            "invalid size of partial point"
        );
        if partial_point.len() == 0 {
            return self.clone();
        };

        let mut poly = self.evaluations.to_vec();
        let nv = self.num_vars;
        let dim = partial_point.len();
        // evaluate nv variable of partial point from left to right
        // with dim rounds and \sum_{i=1}^{dim} 2^(nv - i)
        // (If dim = nv, then the complexity is 2^{nv}.)
        for i in 1..dim + 1 {
            // fix a single variable to evaluate (1 << (nv - i)) evaluations from the last round
            // with complexity of 2^(1 << (nv - i)) field multiplications
            let r = partial_point[i - 1];
            for b in 0..(1 << (nv - i)) {
                let left = poly[b << 1];
                let right = poly[(b << 1) + 1];
                poly[b] = left + r * (right - left);
            }
        }
        poly.truncate(1 << (nv - dim));
        Self::from_vec(nv - dim, poly)
    }

    fn fix_variables_back(&self, partial_point: &[F]) -> Self {
        assert!(
            partial_point.len() <= self.num_vars,
            "invalid size of partial point"
        );
        let mut poly = self.evaluations.to_vec();
        let nv = self.num_vars;
        let dim = partial_point.len();
        for i in 1..dim + 1 {
            let r = partial_point[dim - i];
            // fix the (i-1)-th variable as r
            for b in 0..(1 << (nv - i)) {
                let left = poly[b];
                let right = poly[b + (1 << (nv - i))];
                poly[b] = left + r * (right - left);
            }
        }
        poly.truncate(1 << (nv - dim));
        Self::from_vec(nv - dim, poly)
    }

    #[inline]
    fn to_evaluations(&self) -> Vec<F> {
        self.evaluations.to_vec()
    }
}

impl<F: Field> Index<usize> for DenseMultilinearExtension<F> {
    type Output = F;

    /// Returns the evaluation of the polynomial at a point represented by index.
    ///
    /// Index represents a vector in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    ///
    /// For dense multilinear polynomial, `index` takes constant time.
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<F: Field> Debug for DenseMultilinearExtension<F> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        write!(f, "DenseML(nv = {}, evaluations = [", self.num_vars)?;
        for i in 0..4.min(self.evaluations.len()) {
            write!(f, "{:?}", self.evaluations[i])?;
        }
        if self.evaluations.len() < 4 {
            write!(f, "])")?;
        } else {
            write!(f, "...])")?;
        }
        Ok(())
    }
}

impl<F: Field> Zero for DenseMultilinearExtension<F> {
    #[inline]
    fn zero() -> Self {
        Self {
            name: "".to_string(),
            num_vars: 0,
            evaluations: vec![F::zero()],
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.num_vars == 0 && self.evaluations[0].is_zero()
    }
}

impl<F: Field> Add for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;
    #[inline]
    fn add(mut self, rhs: DenseMultilinearExtension<F>) -> Self {
        self.add_assign(rhs);
        self
    }
}

impl<'a, F: Field> Add<&'a DenseMultilinearExtension<F>> for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;
    #[inline]
    fn add(mut self, rhs: &'a DenseMultilinearExtension<F>) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<'a, 'b, F: Field> Add<&'a DenseMultilinearExtension<F>> for &'b DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;

    #[inline]
    fn add(self, rhs: &'a DenseMultilinearExtension<F>) -> Self::Output {
        // handle constant zero case
        if rhs.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return rhs.clone();
        }
        assert_eq!(self.num_vars, rhs.num_vars);
        let result: Vec<F> = self.iter().zip(rhs.iter()).map(|(&a, b)| a + b).collect();
        Self::Output::from_vec(self.num_vars, result)
    }
}

impl<F: Field> AddAssign for DenseMultilinearExtension<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter()).for_each(|(x, y)| *x += y);
    }
}

impl<'a, F: Field> AddAssign<&'a DenseMultilinearExtension<F>> for DenseMultilinearExtension<F> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a DenseMultilinearExtension<F>) {
        self.iter_mut().zip(rhs.iter()).for_each(|(x, y)| *x += y);
    }
}

impl<'a, F: Field> AddAssign<(F, &'a DenseMultilinearExtension<F>)>
    for DenseMultilinearExtension<F>
{
    #[inline]
    fn add_assign(&mut self, (f, rhs): (F, &'a DenseMultilinearExtension<F>)) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(x, y)| *x += f.mul(y));
    }
}

impl<F: Field> Neg for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.evaluations.iter_mut().for_each(|x| *x = -(*x));
        self
    }
}

impl<F: Field> Sub for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self.sub_assign(rhs);
        self
    }
}

impl<'a, F: Field> Sub<&'a DenseMultilinearExtension<F>> for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;

    #[inline]
    fn sub(mut self, rhs: &'a DenseMultilinearExtension<F>) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<'a, 'b, F: Field> Sub<&'a DenseMultilinearExtension<F>> for &'b DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<F>;

    #[inline]
    fn sub(self, rhs: &'a DenseMultilinearExtension<F>) -> Self::Output {
        // handle constant zero case
        if rhs.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return rhs.clone();
        }
        assert_eq!(self.num_vars, rhs.num_vars);
        let result: Vec<F> = self.iter().zip(rhs.iter()).map(|(&a, b)| a - b).collect();
        Self::Output::from_vec(self.num_vars, result)
    }
}

impl<F: Field> SubAssign for DenseMultilinearExtension<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter()).for_each(|(x, y)| *x -= y);
    }
}

impl<'a, F: Field> SubAssign<&'a DenseMultilinearExtension<F>> for DenseMultilinearExtension<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a DenseMultilinearExtension<F>) {
        self.iter_mut().zip(rhs.iter()).for_each(|(x, y)| *x -= y);
    }
}

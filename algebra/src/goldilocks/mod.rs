mod extension;

pub use extension::GoldilocksExtension;
use serde::{Deserialize, Serialize};

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::{Inv, One, Pow, Zero};

use crate::{
    modulus::{self, GoldilocksModulus},
    reduce::{
        AddReduce, AddReduceAssign, DivReduce, DivReduceAssign, InvReduce, MulReduce,
        MulReduceAssign, NegReduce, PowReduce, SubReduce, SubReduceAssign,
    },
    Field, Packable, PrimeField, TwoAdicField,
};

/// Implementation of Goldilocks field
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct Goldilocks(u64);

impl Goldilocks {
    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        let mut c = self.0;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= modulus::GOLDILOCKS_P {
            c -= modulus::GOLDILOCKS_P;
        }
        c
    }
}

impl Field for Goldilocks {
    type Value = u64;
    type Order = u64;

    const MODULUS_VALUE: Self::Value = modulus::GOLDILOCKS_P;

    #[inline]
    fn neg_one() -> Self {
        Self(modulus::GOLDILOCKS_P - 1)
    }

    #[inline]
    fn new(value: Self::Value) -> Self {
        Self(value)
    }
}

impl Display for Goldilocks {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add<Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add_reduce(rhs.0, GoldilocksModulus))
    }
}

impl Mul<Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul_reduce(rhs.0, GoldilocksModulus))
    }
}

impl Sub<Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub_reduce(rhs.0, GoldilocksModulus))
    }
}

impl Div<Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div_reduce(rhs.0, GoldilocksModulus))
    }
}

impl AddAssign<Self> for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl SubAssign<Self> for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl MulAssign<Self> for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl DivAssign<Self> for Goldilocks {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0.div_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl Add<&Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        Self(self.0.add_reduce(rhs.0, GoldilocksModulus))
    }
}

impl Sub<&Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        Self(self.0.sub_reduce(rhs.0, GoldilocksModulus))
    }
}

impl Mul<&Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        Self(self.0.mul_reduce(rhs.0, GoldilocksModulus))
    }
}

impl Div<&Self> for Goldilocks {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        Self(self.0.div_reduce(rhs.0, GoldilocksModulus))
    }
}

impl AddAssign<&Self> for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.0.add_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl SubAssign<&Self> for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.0.sub_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl MulAssign<&Self> for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        self.0.mul_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl DivAssign<&Self> for Goldilocks {
    #[inline]
    fn div_assign(&mut self, rhs: &Self) {
        self.0.div_reduce_assign(rhs.0, GoldilocksModulus);
    }
}

impl PartialEq for Goldilocks {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u64() == other.as_canonical_u64()
    }
}

impl Eq for Goldilocks {}

impl PartialOrd for Goldilocks {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Goldilocks {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_canonical_u64().cmp(&other.as_canonical_u64())
    }
}

impl Neg for Goldilocks {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(self.0.neg_reduce(GoldilocksModulus))
    }
}

impl Inv for Goldilocks {
    type Output = Self;
    #[inline]
    fn inv(self) -> Self::Output {
        Self(self.0.inv_reduce(GoldilocksModulus))
    }
}

impl Pow<u64> for Goldilocks {
    type Output = Self;
    #[inline]
    fn pow(self, rhs: u64) -> Self::Output {
        Self(self.0.pow_reduce(rhs, GoldilocksModulus))
    }
}

impl Zero for Goldilocks {
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn set_zero(&mut self) {
        *self = Self(0);
    }

    #[inline]
    fn zero() -> Self {
        Self(0)
    }
}

impl One for Goldilocks {
    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self == Self(1)
    }

    #[inline]
    fn set_one(&mut self) {
        *self = Self(1);
    }

    #[inline]
    fn one() -> Self {
        Self(1)
    }
}

impl PrimeField for Goldilocks {
    #[inline]
    fn is_prime_field() -> bool {
        true
    }

    #[inline]
    fn value(&self) -> <Self as Field>::Value {
        self.0
    }
}

impl Packable for Goldilocks {}

impl TwoAdicField for Goldilocks {
    const TWO_ADICITY: usize = 32;

    fn two_adic_generator(bits: usize) -> Self {
        // TODO: Consider a `match` which may speed this up.
        assert!(bits <= Self::TWO_ADICITY);
        let base = Self(1_753_635_133_440_165_772); // generates the whole 2^TWO_ADICITY group
        exp_power_of_2(base, Self::TWO_ADICITY - bits)
    }
}

#[must_use]
fn exp_power_of_2<F: Field>(x: F, power_log: usize) -> F {
    let mut res = x;
    for _ in 0..power_log {
        res = res * res;
    }
    res
}

impl From<usize> for Goldilocks {
    #[inline]
    fn from(value: usize) -> Self {
        let modulus = Goldilocks::MODULUS_VALUE as usize;
        if value < modulus {
            Self(value as u64)
        } else {
            Self((value - modulus) as u64)
        }
    }
}

//! This place defines some concrete implement of field of the algebra.

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::u64;

use num_traits::{Inv, One, Pow, PrimInt, Zero};
use rand::{CryptoRng, Rng};

use crate::random::UniformBase;
use crate::{AsFrom, AsInto, Widening, WrappingOps};
use std::hash::Hash;

mod prime_fields;
pub use prime_fields::PrimeField;

/// A trait defining the algebraic structure of a mathematical field.
///
/// Fields are algebraic structures with two operations: addition and multiplication,
/// where every nonzero element has a multiplicative inverse. In a field, division
/// by any non-zero element is possible and every element except zero has an inverse.
///
/// The [`Field`] trait extends various Rust standard library traits to ensure field elements
/// can be copied, cloned, debugged, displayed, compared, and have a sense of 'zero' and 'one'.
/// Additionally, it supports standard arithmetic operations like addition, subtraction,
/// multiplication, division, and exponentiation, as well as assignment versions of these operations.
///
/// Types implementing [`Field`] also provide implementations for scalar multiplication,
/// negation, doubling, and squaring operations, both as returning new instances and
/// mutating the current instance in place.
///
/// Implementing this trait enables types to be used within mathematical constructs and
/// algorithms that require field properties, such as many cryptographic systems, coding theory,
/// and computational number theory.
pub trait Field:
    Sized
    + Copy
    + Send
    + Sync
    + 'static
    + Debug
    + Display
    + Default
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + Zero
    + One
    + Hash
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + Neg<Output = Self>
    + Inv<Output = Self>
    + Pow<Self::Order, Output = Self>
{
    /// The inner type of this field.
    type Value: Debug
        + Send
        + Sync
        + PrimInt
        + Widening
        + WrappingOps
        + Into<u64>
        + AsInto<f64>
        + AsFrom<f64>
        + AsFrom<u64>
        + AsInto<u64>
        + UniformBase;

    /// The type of the field's order.
    type Order: Copy;

    /// q
    const MODULUS_VALUE: Self::Value;

    /// -1
    fn neg_one() -> Self;

    /// convert values into field element
    fn new(value: Self::Value) -> Self;

    /// generate a random element.
    fn random<R: CryptoRng + Rng>(rng: &mut R) -> Self {
        let range = <Self::Value as UniformBase>::Sample::as_from(Self::MODULUS_VALUE);
        let thresh = range.wrapping_neg() % range;

        let hi = loop {
            let (lo, hi) = <Self::Value as UniformBase>::gen_sample(rng).widen_mul(range);
            if lo >= thresh {
                break hi;
            }
        };

        Self::new(hi.as_into())
    }
}

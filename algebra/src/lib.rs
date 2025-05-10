#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]

//! Define arithmetic operations.
mod baby_bear;

mod goldilocks;

mod error;
mod extension;
mod field;
mod polynomial;
mod primitive;
mod random;

pub mod modulus;
pub mod reduce;
pub mod utils;

pub use baby_bear::{BabyBear, BabyBearExtension};

pub use goldilocks::{Goldilocks, GoldilocksExtension};

pub use error::AlgebraError;
pub use extension::*;
pub use field::{Field, PrimeField};
pub use polynomial::{
    multivariate::{
        DenseMultilinearExtension, ListOfProductsOfPolynomials, MultilinearExtension,
        PolynomialInfo, SparsePolynomial,
    },
    univariate::Polynomial,
};
pub use primitive::{div_ceil, AsFrom, AsInto, Bits, Widening, WrappingOps};
pub use random::{
    FieldBinarySampler, FieldDiscreteGaussianSampler, FieldTernarySampler, FieldUniformSampler,
};

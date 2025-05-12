//! This module implements some functions and methods for
//! modular arithmetic.

mod baby_bear;
mod barrett;
mod goldilocks;

pub use baby_bear::{
    from_monty, to_monty, BabyBearModulus, MONTY_NEG_ONE, MONTY_ONE, MONTY_TWO, MONTY_ZERO,
    P as BABY_BEAR_P,
};
pub use barrett::BarrettModulus;
pub use goldilocks::{to_canonical_u64, GoldilocksModulus, P as GOLDILOCKS_P};
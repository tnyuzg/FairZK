//! PIOP for various building blocks

// #[macro_export]
// macro_rules! timing {
//     ($label:expr, $func:expr) => {{
//         let start = std::time::Instant::now();
//         let result = $func;
//         let elapsed = start.elapsed().as_millis();
//         println!("{}: {} ms", $label, elapsed);
//         result
//     }};
// }

// #[macro_export]
// macro_rules! time {
//     ($label:expr, $func:expr) => {{
//         let start = std::time::Instant::now();
//         let result = $func;
//         let elapsed = start.elapsed().as_millis();
//         println!("{}: {} ms", $label, elapsed);
//         (result, elapsed)
//     }};
// }

// #[macro_export]
// macro_rules! field_vec {
//     ($t:ty; $elem:expr; $n:expr)=>{
//         vec![<$t>::new($elem);$n]
//     };
//     ($t:ty; $($x:expr),+ $(,)?) => {
//         vec![$(<$t>::new($x)),+]
//     }
// }

pub mod lookup;

pub mod vector_lookup;

pub mod infairence;

pub mod spectral_norm;

pub mod meta_disparity;

pub mod total;

pub mod l2_norm;

pub mod bound;

// pub mod infinite_norm;
// pub mod infinite_norm_snark;

pub use lookup::{LookupIOP, LookupInstance};

pub mod inference;

pub use infairence::{InfairenceIOP, InfairenceInstance};

pub use spectral_norm::{SpectralNormIOP, SpectralNormInstance};

pub use meta_disparity::{MetaDisparityIOP, MetaDisparityInstance};

pub use l2_norm::{L2NormIOP, L2NormInstance};

pub use bound::{BoundIOP, BoundInstance};

pub use vector_lookup::{VectorLookupIOP, VectorLookupInstance};

// pub use infinite_norm::{
//     InfiniteNormEval, InfiniteNormIOP, InfiniteNormInfo, InfiniteNormInstance,
// };
// pub use infinite_norm_snark::InfiniteNormSnark;

pub use inference::{InferenceIOP, InferenceInstance};

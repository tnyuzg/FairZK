#[macro_export]
macro_rules! timing {
    ($label:expr, $func:expr) => {{
        let start = std::time::Instant::now();
        let result = $func;
        let elapsed = start.elapsed().as_millis();
        println!("{}: {} ms", $label, elapsed);
        result
    }};
}

#[macro_export]
macro_rules! time {
    ($label:expr, $func:expr) => {{
        let start = std::time::Instant::now();
        let result = $func;
        let elapsed = start.elapsed().as_millis();
        println!("{}: {} ms", $label, elapsed);
        (result, elapsed)
    }};
}

#[macro_export]
macro_rules! field_vec {
    ($t:ty; $elem:expr; $n:expr)=>{
        vec![<$t>::new($elem);$n]
    };
    ($t:ty; $($x:expr),+ $(,)?) => {
        vec![$(<$t>::new($x)),+]
    }
}

pub mod lookup_snark;

pub mod infairence_snark;

pub mod spectral_norm_snark;

pub mod meta_disparity_snark;

pub mod total_snark;

pub mod inference_snark;
pub mod l2_norm_snark;


pub use infairence_snark::InfairenceSnark;
pub use lookup_snark::LookupSnark;

pub mod vector_lookup_snark;
pub use vector_lookup_snark::VectorLookupSnark;

pub use l2_norm_snark::L2NormSnark;
pub use meta_disparity_snark::MetaDisparitySnark;
pub use spectral_norm_snark::SpectralNormSnark;

pub use inference_snark::InferenceSnark;

// // Requires:
// // - AVX2
#[cfg(target_feature = "avx2")]
pub(crate) mod goldilocks_avx2;
#[cfg(target_feature = "avx2")]
pub(crate) mod poseidon_goldilocks_avx2;

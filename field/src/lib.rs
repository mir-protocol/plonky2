#![allow(incomplete_features)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![cfg_attr(nightly, feature(specialization))]
#![cfg_attr(all(nightly, target_arch = "x86_64"), feature(stdarch_x86_avx512))]
#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub(crate) mod arch;

pub mod batch_util;
pub mod cosets;
pub mod extension;
pub mod fft;
pub mod goldilocks_extensions;
pub mod goldilocks_field;
pub mod interpolation;
pub mod ops;
pub mod packable;
pub mod packed;
pub mod polynomial;
pub mod secp256k1_base;
pub mod secp256k1_scalar;
pub mod types;
pub mod zero_poly_coset;

#[cfg(test)]
mod field_testing;

#[cfg(test)]
mod prime_field_testing;

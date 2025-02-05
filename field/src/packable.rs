use crate::packed::PackedField;
use crate::types::Field;

/// Points us to the default packing for a particular field. There may me multiple choices of
/// PackedField for a particular Field (e.g. every Field is also a PackedField), but this is the
/// recommended one. The recommended packing varies by target_arch and target_feature.
pub trait Packable: Field {
    type Packing: PackedField<Scalar = Self>;
}

#[cfg(nightly)]
impl<F: Field> Packable for F {
    default type Packing = Self;
}

#[cfg(not(nightly))]
impl<F: Field> Packable for F {
    type Packing = Self;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    ))
))]
impl Packable for crate::goldilocks_field::GoldilocksField {
    type Packing = crate::arch::x86_64::avx2_goldilocks_field::Avx2GoldilocksField;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vl"
))]
impl Packable for crate::goldilocks_field::GoldilocksField {
    type Packing = crate::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField;
}

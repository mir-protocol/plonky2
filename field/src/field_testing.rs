use crate::extension::{Extendable, Frobenius};
use crate::ops::Square;
use crate::types::{Field, Sample};

#[macro_export]
macro_rules! test_field_arithmetic {
    ($field:ty) => {
        mod field_arithmetic {
            use alloc::vec::Vec;

            use num::bigint::BigUint;
            use rand::rngs::OsRng;
            use rand::{Rng, RngCore};
            use $crate::types::{Field, Sample};

            #[test]
            fn modular_reduction() {
                let mut rng = OsRng;
                for _ in 0..10 {
                    let x_lo = rng.next_u64();
                    let x_hi = rng.next_u32();
                    let x = (x_lo as u128) + ((x_hi as u128) << 64);
                    let a = <$field>::from_noncanonical_u128(x);
                    let b = <$field>::from_noncanonical_u96((x_lo, x_hi));
                    assert_eq!(a, b);
                }
            }

            #[test]
            fn batch_inversion() {
                for n in 0..20 {
                    let xs = (1..=n as u64)
                        .map(|i| <$field>::from_canonical_u64(i))
                        .collect::<Vec<_>>();
                    let invs = <$field>::batch_multiplicative_inverse(&xs);
                    assert_eq!(invs.len(), n);
                    for (x, inv) in xs.into_iter().zip(invs) {
                        assert_eq!(x * inv, <$field>::ONE);
                    }
                }
            }

            #[test]
            fn primitive_root_order() {
                let max_power = 8.min(<$field>::TWO_ADICITY);
                for n_power in 0..max_power {
                    let root = <$field>::primitive_root_of_unity(n_power);
                    let order = <$field>::generator_order(root);
                    assert_eq!(order, 1 << n_power, "2^{}'th primitive root", n_power);
                }
            }

            #[test]
            fn negation() {
                type F = $field;

                for x in [F::ZERO, F::ONE, F::TWO, F::NEG_ONE] {
                    assert_eq!(x + -x, F::ZERO);
                }
            }

            #[test]
            #[cfg(not(feature = "no_random"))]
            fn exponentiation() {
                type F = $field;

                assert_eq!(F::ZERO.exp_u64(0), <F>::ONE);
                assert_eq!(F::ONE.exp_u64(0), <F>::ONE);
                assert_eq!(F::TWO.exp_u64(0), <F>::ONE);

                assert_eq!(F::ZERO.exp_u64(1), <F>::ZERO);
                assert_eq!(F::ONE.exp_u64(1), <F>::ONE);
                assert_eq!(F::TWO.exp_u64(1), <F>::TWO);

                assert_eq!(F::ZERO.kth_root_u64(1), <F>::ZERO);
                assert_eq!(F::ONE.kth_root_u64(1), <F>::ONE);
                assert_eq!(F::TWO.kth_root_u64(1), <F>::TWO);

                for power in 1..10 {
                    if F::is_monomial_permutation_u64(power) {
                        let x = F::rand();
                        assert_eq!(x.exp_u64(power).kth_root_u64(power), x);
                    }
                }
            }

            #[test]
            #[cfg(not(feature = "no_random"))]
            fn exponentiation_large() {
                type F = $field;

                let mut rng = OsRng;

                let base = F::rand();
                let pow = BigUint::from(rng.gen::<u64>());
                let cycles = rng.gen::<u32>();
                let mul_group_order = F::order() - 1u32;
                let big_pow = &pow + &mul_group_order * cycles;
                let big_pow_wrong = &pow + &mul_group_order * cycles + 1u32;

                assert_eq!(base.exp_biguint(&pow), base.exp_biguint(&big_pow));
                assert_ne!(base.exp_biguint(&pow), base.exp_biguint(&big_pow_wrong));
            }

            #[test]
            #[cfg(not(feature = "no_random"))]
            fn inverses() {
                type F = $field;

                let x = F::rand();
                let x1 = x.inverse();
                let x2 = x1.inverse();
                let x3 = x2.inverse();

                assert_eq!(x, x2);
                assert_eq!(x1, x3);
            }
        }
    };
}

#[allow(clippy::eq_op)]
#[cfg(not(feature = "no_random"))]
pub(crate) fn test_add_neg_sub_mul<BF: Extendable<D>, const D: usize>() {
    let x = BF::Extension::rand();
    let y = BF::Extension::rand();
    let z = BF::Extension::rand();
    assert_eq!(x + (-x), BF::Extension::ZERO);
    assert_eq!(-x, BF::Extension::ZERO - x);
    assert_eq!(x + x, x * BF::Extension::TWO);
    assert_eq!(x * (-x), -x.square());
    assert_eq!(x + y, y + x);
    assert_eq!(x * y, y * x);
    assert_eq!(x * (y * z), (x * y) * z);
    assert_eq!(x - (y + z), (x - y) - z);
    assert_eq!((x + y) - z, x + (y - z));
    assert_eq!(x * (y + z), x * y + x * z);
}

#[cfg(not(feature = "no_random"))]
pub(crate) fn test_inv_div<BF: Extendable<D>, const D: usize>() {
    let x = BF::Extension::rand();
    let y = BF::Extension::rand();
    let z = BF::Extension::rand();
    assert_eq!(x * x.inverse(), BF::Extension::ONE);
    assert_eq!(x.inverse() * x, BF::Extension::ONE);
    assert_eq!(x.square().inverse(), x.inverse().square());
    assert_eq!((x / y) * y, x);
    assert_eq!(x / (y * z), (x / y) / z);
    assert_eq!((x * y) / z, x * (y / z));
}

#[cfg(not(feature = "no_random"))]
pub(crate) fn test_frobenius<BF: Extendable<D>, const D: usize>() {
    let x = BF::Extension::rand();
    assert_eq!(x.exp_biguint(&BF::order()), x.frobenius());
    for count in 2..D {
        assert_eq!(
            x.repeated_frobenius(count),
            (0..count).fold(x, |acc, _| acc.frobenius())
        );
    }
}

#[cfg(not(feature = "no_random"))]
pub(crate) fn test_field_order<BF: Extendable<D>, const D: usize>() {
    let x = BF::Extension::rand();
    assert_eq!(
        x.exp_biguint(&(BF::Extension::order() - 1u8)),
        BF::Extension::ONE
    );
}

pub(crate) fn test_power_of_two_gen<BF: Extendable<D>, const D: usize>() {
    assert_eq!(
        BF::Extension::MULTIPLICATIVE_GROUP_GENERATOR
            .exp_biguint(&(BF::Extension::order() >> BF::Extension::TWO_ADICITY)),
        BF::Extension::POWER_OF_TWO_GENERATOR,
    );
    assert_eq!(
        BF::Extension::POWER_OF_TWO_GENERATOR
            .exp_u64(1 << (BF::Extension::TWO_ADICITY - BF::TWO_ADICITY)),
        BF::POWER_OF_TWO_GENERATOR.into()
    );
}

#[macro_export]
macro_rules! test_field_extension {
    ($field:ty, $d:expr) => {
        mod field_extension {
            #[test]
            #[cfg(not(feature = "no_random"))]
            fn test_add_neg_sub_mul() {
                $crate::field_testing::test_add_neg_sub_mul::<$field, $d>();
            }
            #[test]
            #[cfg(not(feature = "no_random"))]
            fn test_inv_div() {
                $crate::field_testing::test_inv_div::<$field, $d>();
            }
            #[test]
            #[cfg(not(feature = "no_random"))]
            fn test_frobenius() {
                $crate::field_testing::test_frobenius::<$field, $d>();
            }
            #[test]
            #[cfg(not(feature = "no_random"))]
            fn test_field_order() {
                $crate::field_testing::test_field_order::<$field, $d>();
            }
            #[test]
            fn test_power_of_two_gen() {
                $crate::field_testing::test_power_of_two_gen::<$field, $d>();
            }
        }
    };
}

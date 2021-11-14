use std::convert::TryInto;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::bigint::BigUint;
use num::{Integer, One, Zero};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::field::extension_field::Frobenius;
use crate::hash::gmimc::GMiMC;
use crate::hash::poseidon::Poseidon;
use crate::util::bits_u64;

/// A prime order field with the features we need to use it as a base field in our argument system.
pub trait RichField: PrimeField + GMiMC<12> + Poseidon<12> {}

/// A finite field.
pub trait Field:
    'static
    + Copy
    + Eq
    + Hash
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sum
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Product
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Debug
    + Default
    + Display
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
{
    type PrimeField: PrimeField;

    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;

    const CHARACTERISTIC: u64;

    /// The 2-adicity of this field's multiplicative group.
    const TWO_ADICITY: usize;

    /// Generator of the entire multiplicative group, i.e. all non-zero elements.
    const MULTIPLICATIVE_GROUP_GENERATOR: Self;
    /// Generator of a multiplicative subgroup of order `2^TWO_ADICITY`.
    const POWER_OF_TWO_GENERATOR: Self;

    fn order() -> BigUint;

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    #[inline]
    fn is_nonzero(&self) -> bool {
        *self != Self::ZERO
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    #[inline]
    fn double(&self) -> Self {
        *self + *self
    }

    #[inline]
    fn square(&self) -> Self {
        *self * *self
    }

    #[inline]
    fn cube(&self) -> Self {
        self.square() * *self
    }

    /// Compute the multiplicative inverse of this field element.
    fn try_inverse(&self) -> Option<Self>;

    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    fn batch_multiplicative_inverse(x: &[Self]) -> Vec<Self> {
        // This is Montgomery's trick. At a high level, we invert the product of the given field
        // elements, then derive the individual inverses from that via multiplication.

        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![x[0].inverse()];
        }

        // Fill buf with cumulative product of x.
        let mut buf = Vec::with_capacity(n);
        let mut cumul_prod = x[0];
        buf.push(cumul_prod);
        for i in 1..n {
            cumul_prod *= x[i];
            buf.push(cumul_prod);
        }

        // At this stage buf contains the the cumulative product of x. We reuse the buffer for
        // efficiency. At the end of the loop, it is filled with inverses of x.
        let mut a_inv = cumul_prod.inverse();
        buf[n - 1] = buf[n - 2] * a_inv;
        for i in (1..n - 1).rev() {
            a_inv = x[i + 1] * a_inv;
            // buf[i - 1] has not been written to by this loop, so it equals x[0] * ... x[n - 1].
            buf[i] = buf[i - 1] * a_inv;
            // buf[i] now holds the inverse of x[i].
        }
        buf[0] = x[1] * a_inv;
        buf
    }

    /// Compute the inverse of 2^exp in this field.
    #[inline]
    fn inverse_2exp(exp: usize) -> Self {
        // The inverse of 2^exp is p-(p-1)/2^exp when char(F) = p and
        // exp is at most the t=TWO_ADICITY of the prime field. When
        // exp exceeds t, we repeatedly multiply by 2^-t and reduce
        // exp until it's in the right range.

        let p = Self::CHARACTERISTIC;

        // NB: The only reason this is split into two cases is to save
        // the multiplication (and possible calculation of
        // inverse_2_pow_adicity) in the usual case that exp <=
        // TWO_ADICITY. Can remove the branch and simplify if that
        // saving isn't worth it.

        if exp > Self::PrimeField::TWO_ADICITY {
            // NB: This should be a compile-time constant
            let inverse_2_pow_adicity: Self =
                Self::from_canonical_u64(p - ((p - 1) >> Self::PrimeField::TWO_ADICITY));

            let mut res = inverse_2_pow_adicity;
            let mut e = exp - Self::PrimeField::TWO_ADICITY;

            while e > Self::PrimeField::TWO_ADICITY {
                res *= inverse_2_pow_adicity;
                e -= Self::PrimeField::TWO_ADICITY;
            }
            res * Self::from_canonical_u64(p - ((p - 1) >> e))
        } else {
            Self::from_canonical_u64(p - ((p - 1) >> exp))
        }
    }

    fn primitive_root_of_unity(n_log: usize) -> Self {
        assert!(n_log <= Self::TWO_ADICITY);
        let base = Self::POWER_OF_TWO_GENERATOR;
        base.exp_power_of_2(Self::TWO_ADICITY - n_log)
    }

    /// Computes a multiplicative subgroup whose order is known in advance.
    fn cyclic_subgroup_known_order(generator: Self, order: usize) -> Vec<Self> {
        generator.powers().take(order).collect()
    }

    /// Computes the subgroup generated by the root of unity of a given order generated by `Self::primitive_root_of_unity`.
    fn two_adic_subgroup(n_log: usize) -> Vec<Self> {
        let generator = Self::primitive_root_of_unity(n_log);
        generator.powers().take(1 << n_log).collect()
    }

    fn cyclic_subgroup_unknown_order(generator: Self) -> Vec<Self> {
        let mut subgroup = Vec::new();
        for power in generator.powers() {
            if power.is_one() && !subgroup.is_empty() {
                break;
            }
            subgroup.push(power);
        }
        subgroup
    }

    fn generator_order(generator: Self) -> usize {
        generator.powers().skip(1).position(|y| y.is_one()).unwrap() + 1
    }

    /// Computes a coset of a multiplicative subgroup whose order is known in advance.
    fn cyclic_subgroup_coset_known_order(generator: Self, shift: Self, order: usize) -> Vec<Self> {
        let subgroup = Self::cyclic_subgroup_known_order(generator, order);
        subgroup.into_iter().map(|x| x * shift).collect()
    }

    // TODO: move these to a new `PrimeField` trait (for all prime fields, not just 64-bit ones)
    fn from_biguint(n: BigUint) -> Self;

    fn to_biguint(&self) -> BigUint;

    fn from_canonical_u64(n: u64) -> Self;

    fn from_canonical_u32(n: u32) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    fn from_bool(b: bool) -> Self {
        Self::from_canonical_u64(b as u64)
    }

    /// Returns `n % Self::CHARACTERISTIC`.
    fn from_noncanonical_u128(n: u128) -> Self;

    /// Returns `n % Self::CHARACTERISTIC`. May be cheaper than from_noncanonical_u128 when we know
    /// that n < 2 ** 96.
    #[inline]
    fn from_noncanonical_u96((n_lo, n_hi): (u64, u32)) -> Self {
        // Default implementation.
        let n: u128 = ((n_hi as u128) << 64) + (n_lo as u128);
        Self::from_noncanonical_u128(n)
    }

    fn rand_from_rng<R: Rng>(rng: &mut R) -> Self;

    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    fn exp_u64(&self, power: u64) -> Self {
        let mut current = *self;
        let mut product = Self::ONE;

        for j in 0..bits_u64(power) {
            if (power >> j & 1) != 0 {
                product *= current;
            }
            current = current.square();
        }
        product
    }

    fn exp_biguint(&self, power: &BigUint) -> Self {
        let mut result = Self::ONE;
        for &digit in power.to_u64_digits().iter().rev() {
            result = result.exp_power_of_2(64);
            result *= self.exp_u64(digit);
        }
        result
    }

    /// Returns whether `x^power` is a permutation of this field.
    fn is_monomial_permutation_u64(power: u64) -> bool {
        match power {
            0 => false,
            1 => true,
            _ => (Self::order() - 1u32).gcd(&BigUint::from(power)).is_one(),
        }
    }

    fn kth_root_u64(&self, k: u64) -> Self {
        let p = Self::order().clone();
        let p_minus_1 = &p - 1u32;
        debug_assert!(
            Self::is_monomial_permutation_u64(k),
            "Not a permutation of this field"
        );

        // By Fermat's little theorem, x^p = x and x^(p - 1) = 1, so x^(p + n(p - 1)) = x for any n.
        // Our assumption that the k'th root operation is a permutation implies gcd(p - 1, k) = 1,
        // so there exists some n such that p + n(p - 1) is a multiple of k. Once we find such an n,
        // we can rewrite the above as
        //    x^((p + n(p - 1))/k)^k = x,
        // implying that x^((p + n(p - 1))/k) is a k'th root of x.
        for n in 0..k {
            let numerator = &p + &p_minus_1 * n;
            if (&numerator % k).is_zero() {
                let power = (numerator / k) % p_minus_1;
                return self.exp_biguint(&power);
            }
        }
        panic!(
            "x^{} and x^(1/{}) are not permutations of this field, or we have a bug!",
            k, k
        );
    }

    fn cube_root(&self) -> Self {
        self.kth_root_u64(3)
    }

    fn powers(&self) -> Powers<Self> {
        Powers {
            base: *self,
            current: Self::ONE,
        }
    }

    fn rand() -> Self {
        Self::rand_from_rng(&mut rand::thread_rng())
    }

    fn rand_arr<const N: usize>() -> [Self; N] {
        Self::rand_vec(N).try_into().unwrap()
    }

    fn rand_vec(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::rand()).collect()
    }

    /// Representative `g` of the coset used in FRI, so that LDEs in FRI are done over `gH`.
    fn coset_shift() -> Self {
        Self::MULTIPLICATIVE_GROUP_GENERATOR
    }

    /// Equivalent to *self + x * y, but may be cheaper.
    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        // Default implementation.
        *self + x * y
    }
}

/// A finite field of prime order less than 2^64.
pub trait PrimeField: Field {
    const ORDER: u64;

    /// The number of bits required to encode any field element.
    fn bits() -> usize {
        bits_u64(Self::NEG_ONE.to_canonical_u64())
    }

    fn to_canonical_u64(&self) -> u64;

    fn to_noncanonical_u64(&self) -> u64;

    fn from_noncanonical_u64(n: u64) -> Self;

    #[inline]
    fn add_one(&self) -> Self {
        unsafe { self.add_canonical_u64(1) }
    }

    #[inline]
    fn sub_one(&self) -> Self {
        unsafe { self.sub_canonical_u64(1) }
    }

    /// Equivalent to *self + Self::from_canonical_u64(rhs), but may be cheaper. The caller must
    /// ensure that 0 <= rhs < Self::ORDER. The function may return incorrect results if this
    /// precondition is not met. It is marked unsafe for this reason.
    #[inline]
    unsafe fn add_canonical_u64(&self, rhs: u64) -> Self {
        // Default implementation.
        *self + Self::from_canonical_u64(rhs)
    }

    /// Equivalent to *self - Self::from_canonical_u64(rhs), but may be cheaper. The caller must
    /// ensure that 0 <= rhs < Self::ORDER. The function may return incorrect results if this
    /// precondition is not met. It is marked unsafe for this reason.
    #[inline]
    unsafe fn sub_canonical_u64(&self, rhs: u64) -> Self {
        // Default implementation.
        *self - Self::from_canonical_u64(rhs)
    }
}

/// An iterator over the powers of a certain base element `b`: `b^0, b^1, b^2, ...`.
#[derive(Clone)]
pub struct Powers<F: Field> {
    base: F,
    current: F,
}

impl<F: Field> Iterator for Powers<F> {
    type Item = F;

    fn next(&mut self) -> Option<F> {
        let result = self.current;
        self.current *= self.base;
        Some(result)
    }
}

impl<F: Field> Powers<F> {
    /// Apply the Frobenius automorphism `k` times.
    pub fn repeated_frobenius<const D: usize>(self, k: usize) -> Self
    where
        F: Frobenius<D>,
    {
        let Self { base, current } = self;
        Self {
            base: base.repeated_frobenius(k),
            current: current.repeated_frobenius(k),
        }
    }
}

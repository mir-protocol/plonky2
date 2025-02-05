use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::bigint::BigUint;
use num::{Integer, One, ToPrimitive, Zero};
use plonky2_util::bits_u64;
#[cfg(feature = "os_rng")]
use rand::rngs::OsRng;
#[cfg(not(feature = "os_rng"))]
use rand::SeedableRng;
#[cfg(not(feature = "os_rng"))]
use rand_chacha::ChaCha8Rng;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::extension::Frobenius;
use crate::ops::Square;

/// Sampling
pub trait Sample: Sized {
    /// Samples a single value using `rng`.
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized;

    #[cfg(feature = "os_rng")]
    /// Samples a single value using the [`OsRng`].
    #[inline]
    fn rand() -> Self {
        Self::sample(&mut OsRng)
    }

    #[cfg(not(feature = "os_rng"))]
    /// Samples a single value using seeded [`ChaCha8Rng`].
    #[inline]
    fn rand() -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
        Self::sample(&mut rng)
    }

    /// Samples a [`Vec`] of values of length `n` using [`OsRng`].
    #[inline]
    fn rand_vec(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::rand()).collect()
    }

    /// Samples an array of values of length `N` using [`OsRng`].
    #[inline]
    fn rand_array<const N: usize>() -> [Self; N] {
        Self::rand_vec(N)
            .try_into()
            .ok()
            .expect("This conversion can never fail.")
    }
}

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
    + Square
    + Product
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Debug
    + Default
    + Display
    + Sample
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;

    /// The 2-adicity of this field's multiplicative group.
    const TWO_ADICITY: usize;

    /// The field's characteristic and it's 2-adicity.
    /// Set to `None` when the characteristic doesn't fit in a u64.
    const CHARACTERISTIC_TWO_ADICITY: usize;

    /// Generator of the entire multiplicative group, i.e. all non-zero elements.
    const MULTIPLICATIVE_GROUP_GENERATOR: Self;
    /// Generator of a multiplicative subgroup of order `2^TWO_ADICITY`.
    const POWER_OF_TWO_GENERATOR: Self;

    /// The bit length of the field order.
    const BITS: usize;

    fn order() -> BigUint;
    fn characteristic() -> BigUint;

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
    fn cube(&self) -> Self {
        self.square() * *self
    }

    fn triple(&self) -> Self {
        *self * (Self::ONE + Self::TWO)
    }

    /// Compute the multiplicative inverse of this field element.
    fn try_inverse(&self) -> Option<Self>;

    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    fn batch_multiplicative_inverse(x: &[Self]) -> Vec<Self> {
        // This is Montgomery's trick. At a high level, we invert the product of the given field
        // elements, then derive the individual inverses from that via multiplication.

        // The usual Montgomery trick involves calculating an array of cumulative products,
        // resulting in a long dependency chain. To increase instruction-level parallelism, we
        // compute WIDTH separate cumulative product arrays that only meet at the end.

        // Higher WIDTH increases instruction-level parallelism, but too high a value will cause us
        // to run out of registers.
        const WIDTH: usize = 4;
        // JN note: WIDTH is 4. The code is specialized to this value and will need
        // modification if it is changed. I tried to make it more generic, but Rust's const
        // generics are not yet good enough.

        // Handle special cases. Paradoxically, below is repetitive but concise.
        // The branches should be very predictable.
        let n = x.len();
        if n == 0 {
            return Vec::new();
        } else if n == 1 {
            return vec![x[0].inverse()];
        } else if n == 2 {
            let x01 = x[0] * x[1];
            let x01inv = x01.inverse();
            return vec![x01inv * x[1], x01inv * x[0]];
        } else if n == 3 {
            let x01 = x[0] * x[1];
            let x012 = x01 * x[2];
            let x012inv = x012.inverse();
            let x01inv = x012inv * x[2];
            return vec![x01inv * x[1], x01inv * x[0], x012inv * x01];
        }
        debug_assert!(n >= WIDTH);

        // Buf is reused for a few things to save allocations.
        // Fill buf with cumulative product of x, only taking every 4th value. Concretely, buf will
        // be [
        //   x[0], x[1], x[2], x[3],
        //   x[0] * x[4], x[1] * x[5], x[2] * x[6], x[3] * x[7],
        //   x[0] * x[4] * x[8], x[1] * x[5] * x[9], x[2] * x[6] * x[10], x[3] * x[7] * x[11],
        //   ...
        // ].
        // If n is not a multiple of WIDTH, the result is truncated from the end. For example,
        // for n == 5, we get [x[0], x[1], x[2], x[3], x[0] * x[4]].
        let mut buf: Vec<Self> = Vec::with_capacity(n);
        // cumul_prod holds the last WIDTH elements of buf. This is redundant, but it's how we
        // convince LLVM to keep the values in the registers.
        let mut cumul_prod: [Self; WIDTH] = x[..WIDTH].try_into().unwrap();
        buf.extend(cumul_prod);
        for (i, &xi) in x[WIDTH..].iter().enumerate() {
            cumul_prod[i % WIDTH] *= xi;
            buf.push(cumul_prod[i % WIDTH]);
        }
        debug_assert_eq!(buf.len(), n);

        let mut a_inv = {
            // This is where the four dependency chains meet.
            // Take the last four elements of buf and invert them all.
            let c01 = cumul_prod[0] * cumul_prod[1];
            let c23 = cumul_prod[2] * cumul_prod[3];
            let c0123 = c01 * c23;
            let c0123inv = c0123.inverse();
            let c01inv = c0123inv * c23;
            let c23inv = c0123inv * c01;
            [
                c01inv * cumul_prod[1],
                c01inv * cumul_prod[0],
                c23inv * cumul_prod[3],
                c23inv * cumul_prod[2],
            ]
        };

        for i in (WIDTH..n).rev() {
            // buf[i - WIDTH] has not been written to by this loop, so it equals
            // x[i % WIDTH] * x[i % WIDTH + WIDTH] * ... * x[i - WIDTH].
            buf[i] = buf[i - WIDTH] * a_inv[i % WIDTH];
            // buf[i] now holds the inverse of x[i].
            a_inv[i % WIDTH] *= x[i];
        }
        for i in (0..WIDTH).rev() {
            buf[i] = a_inv[i];
        }

        for (&bi, &xi) in buf.iter().zip(x) {
            // Sanity check only.
            debug_assert_eq!(bi * xi, Self::ONE);
        }

        buf
    }

    /// Compute the inverse of 2^exp in this field.
    #[inline]
    fn inverse_2exp(exp: usize) -> Self {
        // Let p = char(F). Since 2^exp is in the prime subfield, i.e. an
        // element of GF_p, its inverse must be as well. Thus we may add
        // multiples of p without changing the result. In particular,
        // 2^-exp = 2^-exp - p 2^-exp
        //        = 2^-exp (1 - p)
        //        = p - (p - 1) / 2^exp

        // If this field's two adicity, t, is at least exp, then 2^exp divides
        // p - 1, so this division can be done with a simple bit shift. If
        // exp > t, we repeatedly multiply by 2^-t and reduce exp until it's in
        // the right range.

        if let Some(p) = Self::characteristic().to_u64() {
            // NB: The only reason this is split into two cases is to save
            // the multiplication (and possible calculation of
            // inverse_2_pow_adicity) in the usual case that exp <=
            // TWO_ADICITY. Can remove the branch and simplify if that
            // saving isn't worth it.

            if exp > Self::CHARACTERISTIC_TWO_ADICITY {
                // NB: This should be a compile-time constant
                let inverse_2_pow_adicity: Self =
                    Self::from_canonical_u64(p - ((p - 1) >> Self::CHARACTERISTIC_TWO_ADICITY));

                let mut res = inverse_2_pow_adicity;
                let mut e = exp - Self::CHARACTERISTIC_TWO_ADICITY;

                while e > Self::CHARACTERISTIC_TWO_ADICITY {
                    res *= inverse_2_pow_adicity;
                    e -= Self::CHARACTERISTIC_TWO_ADICITY;
                }
                res * Self::from_canonical_u64(p - ((p - 1) >> e))
            } else {
                Self::from_canonical_u64(p - ((p - 1) >> exp))
            }
        } else {
            Self::TWO.inverse().exp_u64(exp as u64)
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

    /// Returns `n % Self::characteristic()`.
    fn from_noncanonical_biguint(n: BigUint) -> Self;

    /// Returns `n`. Assumes that `n` is already in canonical form, i.e. `n < Self::order()`.
    // TODO: Should probably be unsafe.
    fn from_canonical_u64(n: u64) -> Self;

    /// Returns `n`. Assumes that `n` is already in canonical form, i.e. `n < Self::order()`.
    // TODO: Should probably be unsafe.
    fn from_canonical_u32(n: u32) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    /// Returns `n`. Assumes that `n` is already in canonical form, i.e. `n < Self::order()`.
    // TODO: Should probably be unsafe.
    fn from_canonical_u16(n: u16) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    /// Returns `n`. Assumes that `n` is already in canonical form, i.e. `n < Self::order()`.
    // TODO: Should probably be unsafe.
    fn from_canonical_u8(n: u8) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    /// Returns `n`. Assumes that `n` is already in canonical form, i.e. `n < Self::order()`.
    // TODO: Should probably be unsafe.
    fn from_canonical_usize(n: usize) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    fn from_bool(b: bool) -> Self {
        Self::from_canonical_u64(b as u64)
    }

    /// Returns `n % Self::characteristic()`.
    fn from_noncanonical_u128(n: u128) -> Self;

    /// Returns `x % Self::CHARACTERISTIC`.
    fn from_noncanonical_u64(n: u64) -> Self;

    /// Returns `n` as an element of this field.
    fn from_noncanonical_i64(n: i64) -> Self;

    /// Returns `n % Self::characteristic()`. May be cheaper than from_noncanonical_u128 when we know
    /// that `n < 2 ** 96`.
    #[inline]
    fn from_noncanonical_u96((n_lo, n_hi): (u64, u32)) -> Self {
        // Default implementation.
        let n: u128 = ((n_hi as u128) << 64) + (n_lo as u128);
        Self::from_noncanonical_u128(n)
    }

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
            if ((power >> j) & 1) != 0 {
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
        let p = Self::order();
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
        self.shifted_powers(Self::ONE)
    }

    fn shifted_powers(&self, start: Self) -> Powers<Self> {
        Powers {
            base: *self,
            current: start,
        }
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

pub trait PrimeField: Field {
    fn to_canonical_biguint(&self) -> BigUint;

    fn is_quadratic_residue(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        // This is based on Euler's criterion.
        let power = Self::NEG_ONE.to_canonical_biguint() / 2u8;
        let exp = self.exp_biguint(&power);
        if exp == Self::ONE {
            return true;
        }
        if exp == Self::NEG_ONE {
            return false;
        }
        panic!("Unreachable")
    }

    fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            Some(*self)
        } else if self.is_quadratic_residue() {
            let t = (Self::order() - BigUint::from(1u32))
                / (BigUint::from(2u32).pow(Self::TWO_ADICITY as u32));
            let mut z = Self::POWER_OF_TWO_GENERATOR;
            let mut w = self.exp_biguint(&((t - BigUint::from(1u32)) / BigUint::from(2u32)));
            let mut x = w * *self;
            let mut b = x * w;

            let mut v = Self::TWO_ADICITY;

            while !b.is_one() {
                let mut k = 0usize;
                let mut b2k = b;
                while !b2k.is_one() {
                    b2k = b2k * b2k;
                    k += 1;
                }
                let j = v - k - 1;
                w = z;
                for _ in 0..j {
                    w = w * w;
                }

                z = w * w;
                b *= z;
                x *= w;
                v = k;
            }
            Some(x)
        } else {
            None
        }
    }
}

/// A finite field of order less than 2^64.
pub trait Field64: Field {
    const ORDER: u64;

    /// Returns `n` as an element of this field. Assumes that `0 <= n < Self::ORDER`.
    // TODO: Move to `Field`.
    // TODO: Should probably be unsafe.
    #[inline]
    fn from_canonical_i64(n: i64) -> Self {
        Self::from_canonical_u64(n as u64)
    }

    #[inline]
    // TODO: Move to `Field`.
    fn add_one(&self) -> Self {
        unsafe { self.add_canonical_u64(1) }
    }

    #[inline]
    // TODO: Move to `Field`.
    fn sub_one(&self) -> Self {
        unsafe { self.sub_canonical_u64(1) }
    }

    /// # Safety
    /// Equivalent to *self + Self::from_canonical_u64(rhs), but may be cheaper. The caller must
    /// ensure that 0 <= rhs < Self::ORDER. The function may return incorrect results if this
    /// precondition is not met. It is marked unsafe for this reason.
    // TODO: Move to `Field`.
    #[inline]
    unsafe fn add_canonical_u64(&self, rhs: u64) -> Self {
        // Default implementation.
        *self + Self::from_canonical_u64(rhs)
    }

    /// # Safety
    /// Equivalent to *self - Self::from_canonical_u64(rhs), but may be cheaper. The caller must
    /// ensure that 0 <= rhs < Self::ORDER. The function may return incorrect results if this
    /// precondition is not met. It is marked unsafe for this reason.
    // TODO: Move to `Field`.
    #[inline]
    unsafe fn sub_canonical_u64(&self, rhs: u64) -> Self {
        // Default implementation.
        *self - Self::from_canonical_u64(rhs)
    }
}

/// A finite field of prime order less than 2^64.
pub trait PrimeField64: PrimeField + Field64 {
    fn to_canonical_u64(&self) -> u64;

    fn to_noncanonical_u64(&self) -> u64;

    #[inline(always)]
    fn to_canonical(&self) -> Self {
        Self::from_canonical_u64(self.to_canonical_u64())
    }
}

/// An iterator over the powers of a certain base element `b`: `b^0, b^1, b^2, ...`.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    fn nth(&mut self, n: usize) -> Option<F> {
        let result = self.current * self.base.exp_u64(n.try_into().unwrap());
        self.current = result * self.base;
        Some(result)
    }

    fn last(self) -> Option<F> {
        panic!("called `Iterator::last()` on an infinite sequence")
    }

    fn count(self) -> usize {
        panic!("called `Iterator::count()` on an infinite sequence")
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

#[cfg(test)]
mod tests {
    use super::Field;
    use crate::goldilocks_field::GoldilocksField;

    #[test]
    fn test_powers_nth() {
        type F = GoldilocksField;

        const N: usize = 10;
        let powers_of_two: Vec<F> = F::TWO.powers().take(N).collect();

        for (n, &expect) in powers_of_two.iter().enumerate() {
            let mut iter = F::TWO.powers();
            assert_eq!(iter.nth(n), Some(expect));

            for &expect_next in &powers_of_two[n + 1..] {
                assert_eq!(iter.next(), Some(expect_next));
            }
        }
    }
}

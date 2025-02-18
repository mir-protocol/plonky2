use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::bigint::BigUint;
use num::traits::Pow;
use serde::{Deserialize, Serialize};

use crate::extension::{Extendable, FieldExtension, Frobenius, OEF};
use crate::ops::Square;
use crate::types::{Field, Sample};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct QuinticExtension<F: Extendable<5>>(pub [F; 5]);

impl<F: Extendable<5>> Default for QuinticExtension<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Extendable<5>> OEF<5> for QuinticExtension<F> {
    const W: F = F::W;
    const DTH_ROOT: F = F::DTH_ROOT;
}

impl<F: Extendable<5>> Frobenius<5> for QuinticExtension<F> {}

impl<F: Extendable<5>> FieldExtension<5> for QuinticExtension<F> {
    type BaseField = F;

    fn to_basefield_array(&self) -> [F; 5] {
        self.0
    }

    fn from_basefield_array(arr: [F; 5]) -> Self {
        Self(arr)
    }

    fn from_basefield(x: F) -> Self {
        x.into()
    }
}

impl<F: Extendable<5>> From<F> for QuinticExtension<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO, F::ZERO, F::ZERO, F::ZERO])
    }
}

impl<F: Extendable<5>> Sample for QuinticExtension<F> {
    #[inline]
    #[cfg(not(feature = "no_random"))]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        Self::from_basefield_array([
            F::sample(rng),
            F::sample(rng),
            F::sample(rng),
            F::sample(rng),
            F::sample(rng),
        ])
    }
}

impl<F: Extendable<5>> Field for QuinticExtension<F> {
    const ZERO: Self = Self([F::ZERO; 5]);
    const ONE: Self = Self([F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const TWO: Self = Self([F::TWO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const NEG_ONE: Self = Self([F::NEG_ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

    // `p^5 - 1 = (p - 1)(p^4 + p^3 + p^2 + p + 1)`. The `p - 1` term
    // has a two-adicity of `F::TWO_ADICITY` and the term `p^4 + p^3 +
    // p^2 + p + 1` is odd since it is the sum of an odd number of odd
    // terms. Hence the two-adicity of `p^5 - 1` is the same as for
    // `p - 1`.
    const TWO_ADICITY: usize = F::TWO_ADICITY;
    const CHARACTERISTIC_TWO_ADICITY: usize = F::CHARACTERISTIC_TWO_ADICITY;

    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Self(F::EXT_MULTIPLICATIVE_GROUP_GENERATOR);
    const POWER_OF_TWO_GENERATOR: Self = Self(F::EXT_POWER_OF_TWO_GENERATOR);

    const BITS: usize = F::BITS * 5;

    fn order() -> BigUint {
        F::order().pow(5u32)
    }
    fn characteristic() -> BigUint {
        F::characteristic()
    }

    // Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Writing 'a' for self:
        let d = self.frobenius(); // d = a^p
        let e = d * d.frobenius(); // e = a^(p + p^2)
        let f = e * e.repeated_frobenius(2); // f = a^(p + p^2 + p^3 + p^4)

        // f contains a^(r-1) and a^r is in the base field.
        debug_assert!(FieldExtension::<5>::is_in_basefield(&(*self * f)));

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product. The equation is
        // extracted from Mul::mul(...) below.
        let Self([a0, a1, a2, a3, a4]) = *self;
        let Self([b0, b1, b2, b3, b4]) = f;
        let g = a0 * b0 + <Self as OEF<5>>::W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1);

        Some(FieldExtension::<5>::scalar_mul(&f, g.inverse()))
    }

    fn from_noncanonical_biguint(n: BigUint) -> Self {
        F::from_noncanonical_biguint(n).into()
    }

    fn from_canonical_u64(n: u64) -> Self {
        F::from_canonical_u64(n).into()
    }

    fn from_noncanonical_u128(n: u128) -> Self {
        F::from_noncanonical_u128(n).into()
    }

    fn from_noncanonical_i64(n: i64) -> Self {
        F::from_noncanonical_i64(n).into()
    }

    fn from_noncanonical_u64(n: u64) -> Self {
        F::from_noncanonical_u64(n).into()
    }
}

impl<F: Extendable<5>> Display for QuinticExtension<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} + {}*a + {}*a^2 + {}*a^3 + {}*a^4",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4]
        )
    }
}

impl<F: Extendable<5>> Debug for QuinticExtension<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: Extendable<5>> Neg for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3], -self.0[4]])
    }
}

impl<F: Extendable<5>> Add for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }
}

impl<F: Extendable<5>> AddAssign for QuinticExtension<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Extendable<5>> Sum for QuinticExtension<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: Extendable<5>> Sub for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
        ])
    }
}

impl<F: Extendable<5>> SubAssign for QuinticExtension<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Extendable<5>> Mul for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    default fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = self;
        let Self([b0, b1, b2, b3, b4]) = rhs;
        let w = <Self as OEF<5>>::W;

        let c0 = a0 * b0 + w * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1);
        let c1 = a0 * b1 + a1 * b0 + w * (a2 * b4 + a3 * b3 + a4 * b2);
        let c2 = a0 * b2 + a1 * b1 + a2 * b0 + w * (a3 * b4 + a4 * b3);
        let c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + w * a4 * b4;
        let c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        Self([c0, c1, c2, c3, c4])
    }
}

impl<F: Extendable<5>> MulAssign for QuinticExtension<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Extendable<5>> Square for QuinticExtension<F> {
    #[inline(always)]
    fn square(&self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = *self;
        let w = <Self as OEF<5>>::W;
        let double_w = <Self as OEF<5>>::W.double();

        let c0 = a0.square() + double_w * (a1 * a4 + a2 * a3);
        let double_a0 = a0.double();
        let c1 = double_a0 * a1 + double_w * a2 * a4 + w * a3 * a3;
        let c2 = double_a0 * a2 + a1 * a1 + double_w * a4 * a3;
        let double_a1 = a1.double();
        let c3 = double_a0 * a3 + double_a1 * a2 + w * a4 * a4;
        let c4 = double_a0 * a4 + double_a1 * a3 + a2 * a2;

        Self([c0, c1, c2, c3, c4])
    }
}

impl<F: Extendable<5>> Product for QuinticExtension<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: Extendable<5>> Div for QuinticExtension<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: Extendable<5>> DivAssign for QuinticExtension<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(test)]
mod tests {
    mod goldilocks {
        use crate::{test_field_arithmetic, test_field_extension};

        test_field_extension!(crate::goldilocks_field::GoldilocksField, 5);
        test_field_arithmetic!(
            crate::extension::quintic::QuinticExtension<
                crate::goldilocks_field::GoldilocksField,
            >
        );
    }
}

use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::extension::OEF;

/// Let `F_D` be the optimal extension field `F[X]/(X^D-W)`. Then `ExtensionAlgebra<F_D>` is the quotient `F_D[X]/(X^D-W)`.
/// It's a `D`-dimensional algebra over `F_D` useful to lift the multiplication over `F_D` to a multiplication over `(F_D)^D`.
#[derive(Copy, Clone)]
pub struct ExtensionAlgebra<F: OEF<D>, const D: usize>(pub [F; D]);

impl<F: OEF<D>, const D: usize> ExtensionAlgebra<F, D> {
    pub const ZERO: Self = Self([F::ZERO; D]);

    pub fn one() -> Self {
        F::ONE.into()
    }

    pub const fn from_basefield_array(arr: [F; D]) -> Self {
        Self(arr)
    }

    pub const fn to_basefield_array(self) -> [F; D] {
        self.0
    }

    pub fn scalar_mul(&self, scalar: F) -> Self {
        let mut res = self.0;
        res.iter_mut().for_each(|x| {
            *x *= scalar;
        });
        Self(res)
    }
}

impl<F: OEF<D>, const D: usize> From<F> for ExtensionAlgebra<F, D> {
    fn from(x: F) -> Self {
        let mut arr = [F::ZERO; D];
        arr[0] = x;
        Self(arr)
    }
}

impl<F: OEF<D>, const D: usize> Display for ExtensionAlgebra<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({})", self.0[0])?;
        for i in 1..D {
            write!(f, " + ({})*b^{i}", self.0[i])?;
        }
        Ok(())
    }
}

impl<F: OEF<D>, const D: usize> Debug for ExtensionAlgebra<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: OEF<D>, const D: usize> Neg for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        let mut arr = self.0;
        arr.iter_mut().for_each(|x| *x = -*x);
        Self(arr)
    }
}

impl<F: OEF<D>, const D: usize> Add for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = self.0;
        arr.iter_mut().zip(&rhs.0).for_each(|(x, &y)| *x += y);
        Self(arr)
    }
}

impl<F: OEF<D>, const D: usize> AddAssign for ExtensionAlgebra<F, D> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: OEF<D>, const D: usize> Sum for ExtensionAlgebra<F, D> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: OEF<D>, const D: usize> Sub for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = self.0;
        arr.iter_mut().zip(&rhs.0).for_each(|(x, &y)| *x -= y);
        Self(arr)
    }
}

impl<F: OEF<D>, const D: usize> SubAssign for ExtensionAlgebra<F, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: OEF<D>, const D: usize> Mul for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = [F::ZERO; D];
        let w = F::from_basefield(F::W);
        for i in 0..D {
            for j in 0..D {
                res[(i + j) % D] += if i + j < D {
                    self.0[i] * rhs.0[j]
                } else {
                    w * self.0[i] * rhs.0[j]
                }
            }
        }
        Self(res)
    }
}

impl<F: OEF<D>, const D: usize> MulAssign for ExtensionAlgebra<F, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: OEF<D>, const D: usize> Product for ExtensionAlgebra<F, D> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

/// A polynomial in coefficient form.
#[derive(Clone, Debug)]
pub struct PolynomialCoeffsAlgebra<F: OEF<D>, const D: usize> {
    pub(crate) coeffs: Vec<ExtensionAlgebra<F, D>>,
}

impl<F: OEF<D>, const D: usize> PolynomialCoeffsAlgebra<F, D> {
    pub fn new(coeffs: Vec<ExtensionAlgebra<F, D>>) -> Self {
        PolynomialCoeffsAlgebra { coeffs }
    }

    pub fn eval(&self, x: ExtensionAlgebra<F, D>) -> ExtensionAlgebra<F, D> {
        self.coeffs
            .iter()
            .rev()
            .fold(ExtensionAlgebra::ZERO, |acc, &c| acc * x + c)
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_with_powers(&self, powers: &[ExtensionAlgebra<F, D>]) -> ExtensionAlgebra<F, D> {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + c * x)
    }

    pub fn eval_base(&self, x: F) -> ExtensionAlgebra<F, D> {
        self.coeffs
            .iter()
            .rev()
            .fold(ExtensionAlgebra::ZERO, |acc, &c| acc.scalar_mul(x) + c)
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_base_with_powers(&self, powers: &[F]) -> ExtensionAlgebra<F, D> {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + x.scalar_mul(c))
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use alloc::vec::Vec;

    use itertools::Itertools;

    use crate::extension::algebra::ExtensionAlgebra;
    use crate::extension::{Extendable, FieldExtension};
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Sample};

    /// Tests that the multiplication on the extension algebra lifts that of the field extension.
    fn test_extension_algebra<F: Extendable<D>, const D: usize>() {
        #[derive(Copy, Clone, Debug)]
        enum ZeroOne {
            Zero,
            One,
        }

        let to_field = |zo: &ZeroOne| match zo {
            ZeroOne::Zero => F::ZERO,
            ZeroOne::One => F::ONE,
        };
        let to_fields = |x: &[ZeroOne], y: &[ZeroOne]| -> (F::Extension, F::Extension) {
            let mut arr0 = [F::ZERO; D];
            let mut arr1 = [F::ZERO; D];
            arr0.copy_from_slice(&x.iter().map(to_field).collect::<Vec<_>>());
            arr1.copy_from_slice(&y.iter().map(to_field).collect::<Vec<_>>());
            (
                <F as Extendable<D>>::Extension::from_basefield_array(arr0),
                <F as Extendable<D>>::Extension::from_basefield_array(arr1),
            )
        };

        // Standard MLE formula.
        let selector = |xs: Vec<ZeroOne>, ts: &[F::Extension]| -> F::Extension {
            (0..2 * D)
                .map(|i| match xs[i] {
                    ZeroOne::Zero => F::Extension::ONE - ts[i],
                    ZeroOne::One => ts[i],
                })
                .product()
        };

        let mul_mle = |ts: Vec<F::Extension>| -> [F::Extension; D] {
            let mut ans = [F::Extension::ZERO; D];
            for xs in (0..2 * D)
                .map(|_| vec![ZeroOne::Zero, ZeroOne::One])
                .multi_cartesian_product()
            {
                let (a, b) = to_fields(&xs[..D], &xs[D..]);
                let c = a * b;
                let res = selector(xs, &ts);
                for i in 0..D {
                    ans[i] += res.scalar_mul(c.to_basefield_array()[i]);
                }
            }
            ans
        };

        let ts = F::Extension::rand_vec(2 * D);
        let mut arr0 = [F::Extension::ZERO; D];
        let mut arr1 = [F::Extension::ZERO; D];
        arr0.copy_from_slice(&ts[..D]);
        arr1.copy_from_slice(&ts[D..]);
        let x = ExtensionAlgebra::from_basefield_array(arr0);
        let y = ExtensionAlgebra::from_basefield_array(arr1);
        let z = x * y;

        assert_eq!(z.0, mul_mle(ts));
    }

    mod base {
        use super::*;

        #[test]
        fn test_algebra() {
            test_extension_algebra::<GoldilocksField, 1>();
        }
    }

    mod quadratic {
        use super::*;

        #[test]
        fn test_algebra() {
            test_extension_algebra::<GoldilocksField, 2>();
        }
    }

    mod quartic {
        use super::*;

        #[test]
        fn test_algebra() {
            test_extension_algebra::<GoldilocksField, 4>();
        }
    }
}

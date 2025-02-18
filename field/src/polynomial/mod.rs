pub(crate) mod division;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use anyhow::{ensure, Result};
use itertools::Itertools;
use plonky2_util::log2_strict;
use serde::{Deserialize, Serialize};

use crate::extension::{Extendable, FieldExtension};
use crate::fft::{fft, fft_with_options, ifft, FftRootTable};
use crate::types::Field;

/// A polynomial in point-value form.
///
/// The points are implicitly `g^i`, where `g` generates the subgroup whose size equals the number
/// of points.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialValues<F: Field> {
    pub values: Vec<F>,
}

impl<F: Field> PolynomialValues<F> {
    pub fn new(values: Vec<F>) -> Self {
        // Check that a subgroup exists of this size, which should be a power of two.
        debug_assert!(log2_strict(values.len()) <= F::TWO_ADICITY);
        PolynomialValues { values }
    }

    pub fn constant(value: F, len: usize) -> Self {
        Self::new(vec![value; len])
    }

    pub fn zero(len: usize) -> Self {
        Self::constant(F::ZERO, len)
    }

    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|x| x.is_zero())
    }

    /// Returns the polynomial whole value is one at the given index, and zero elsewhere.
    pub fn selector(len: usize, index: usize) -> Self {
        let mut result = Self::zero(len);
        result.values[index] = F::ONE;
        result
    }

    /// The number of values stored.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn ifft(self) -> PolynomialCoeffs<F> {
        ifft(self)
    }

    /// Returns the polynomial whose evaluation on the coset `shift*H` is `self`.
    pub fn coset_ifft(self, shift: F) -> PolynomialCoeffs<F> {
        let mut shifted_coeffs = self.ifft();
        shifted_coeffs
            .coeffs
            .iter_mut()
            .zip(shift.inverse().powers())
            .for_each(|(c, r)| {
                *c *= r;
            });
        shifted_coeffs
    }

    pub fn lde_multiple(polys: Vec<Self>, rate_bits: usize) -> Vec<Self> {
        polys.into_iter().map(|p| p.lde(rate_bits)).collect()
    }

    pub fn lde(self, rate_bits: usize) -> Self {
        let coeffs = ifft(self).lde(rate_bits);
        fft_with_options(coeffs, Some(rate_bits), None)
    }

    /// Low-degree extend `Self` (seen as evaluations over the subgroup) onto a coset.
    pub fn lde_onto_coset(self, rate_bits: usize) -> Self {
        let coeffs = ifft(self).lde(rate_bits);
        coeffs.coset_fft_with_options(F::coset_shift(), Some(rate_bits), None)
    }

    pub fn degree(&self) -> usize {
        self.degree_plus_one().saturating_sub(1)
    }

    pub fn degree_plus_one(&self) -> usize {
        self.clone().ifft().degree_plus_one()
    }

    /// Adds `rhs * rhs_weight` to `self`. Assumes `self.len() == rhs.len()`.
    pub fn add_assign_scaled(&mut self, rhs: &Self, rhs_weight: F) {
        self.values
            .iter_mut()
            .zip_eq(&rhs.values)
            .for_each(|(self_v, rhs_v)| *self_v += *rhs_v * rhs_weight)
    }
}

impl<F: Field> From<Vec<F>> for PolynomialValues<F> {
    fn from(values: Vec<F>) -> Self {
        Self::new(values)
    }
}

/// A polynomial in coefficient form.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PolynomialCoeffs<F: Field> {
    pub coeffs: Vec<F>,
}

impl<F: Field> PolynomialCoeffs<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        PolynomialCoeffs { coeffs }
    }

    /// The empty list of coefficients, which is the smallest encoding of the zero polynomial.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn zero(len: usize) -> Self {
        Self::new(vec![F::ZERO; len])
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|x| x.is_zero())
    }

    /// The number of coefficients. This does not filter out any zero coefficients, so it is not
    /// necessarily related to the degree.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn log_len(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn chunks(&self, chunk_size: usize) -> Vec<Self> {
        self.coeffs
            .chunks(chunk_size)
            .map(|chunk| PolynomialCoeffs::new(chunk.to_vec()))
            .collect()
    }

    pub fn eval(&self, x: F) -> F {
        self.coeffs
            .iter()
            .rev()
            .fold(F::ZERO, |acc, &c| acc * x + c)
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_with_powers(&self, powers: &[F]) -> F {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + c * x)
    }

    pub fn eval_base<const D: usize>(&self, x: F::BaseField) -> F
    where
        F: FieldExtension<D>,
    {
        self.coeffs
            .iter()
            .rev()
            .fold(F::ZERO, |acc, &c| acc.scalar_mul(x) + c)
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_base_with_powers<const D: usize>(&self, powers: &[F::BaseField]) -> F
    where
        F: FieldExtension<D>,
    {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + x.scalar_mul(c))
    }

    pub fn lde_multiple(polys: Vec<&Self>, rate_bits: usize) -> Vec<Self> {
        polys.into_iter().map(|p| p.lde(rate_bits)).collect()
    }

    pub fn lde(&self, rate_bits: usize) -> Self {
        self.padded(self.len() << rate_bits)
    }

    pub fn pad(&mut self, new_len: usize) -> Result<()> {
        ensure!(
            new_len >= self.len(),
            "Trying to pad a polynomial of length {} to a length of {}.",
            self.len(),
            new_len
        );
        self.coeffs.resize(new_len, F::ZERO);
        Ok(())
    }

    pub fn padded(&self, new_len: usize) -> Self {
        let mut poly = self.clone();
        poly.pad(new_len).unwrap();
        poly
    }

    /// Removes any leading zero coefficients.
    pub fn trim(&mut self) {
        self.coeffs.truncate(self.degree_plus_one());
    }

    /// Removes some leading zero coefficients, such that a desired length is reached. Fails if a
    /// nonzero coefficient is encountered before then.
    pub fn trim_to_len(&mut self, len: usize) -> Result<()> {
        ensure!(self.len() >= len);
        ensure!(self.coeffs[len..].iter().all(F::is_zero));
        self.coeffs.truncate(len);
        Ok(())
    }

    /// Removes any leading zero coefficients.
    pub fn trimmed(&self) -> Self {
        let coeffs = self.coeffs[..self.degree_plus_one()].to_vec();
        Self { coeffs }
    }

    /// Degree of the polynomial + 1, or 0 for a polynomial with no non-zero coefficients.
    pub fn degree_plus_one(&self) -> usize {
        (0usize..self.len())
            .rev()
            .find(|&i| self.coeffs[i].is_nonzero())
            .map_or(0, |i| i + 1)
    }

    /// Leading coefficient.
    pub fn lead(&self) -> F {
        self.coeffs
            .iter()
            .rev()
            .find(|x| x.is_nonzero())
            .map_or(F::ZERO, |x| *x)
    }

    /// Reverse the order of the coefficients, not taking into account the leading zero coefficients.
    pub(crate) fn rev(&self) -> Self {
        Self::new(self.trimmed().coeffs.into_iter().rev().collect())
    }

    pub fn fft(self) -> PolynomialValues<F> {
        fft(self)
    }

    pub fn fft_with_options(
        self,
        zero_factor: Option<usize>,
        root_table: Option<&FftRootTable<F>>,
    ) -> PolynomialValues<F> {
        fft_with_options(self, zero_factor, root_table)
    }

    /// Returns the evaluation of the polynomial on the coset `shift*H`.
    pub fn coset_fft(&self, shift: F) -> PolynomialValues<F> {
        self.coset_fft_with_options(shift, None, None)
    }

    /// Returns the evaluation of the polynomial on the coset `shift*H`.
    pub fn coset_fft_with_options(
        &self,
        shift: F,
        zero_factor: Option<usize>,
        root_table: Option<&FftRootTable<F>>,
    ) -> PolynomialValues<F> {
        let modified_poly: Self = shift
            .powers()
            .zip(&self.coeffs)
            .map(|(r, &c)| r * c)
            .collect::<Vec<_>>()
            .into();
        modified_poly.fft_with_options(zero_factor, root_table)
    }

    pub fn to_extension<const D: usize>(&self) -> PolynomialCoeffs<F::Extension>
    where
        F: Extendable<D>,
    {
        PolynomialCoeffs::new(self.coeffs.iter().map(|&c| c.into()).collect())
    }

    pub fn mul_extension<const D: usize>(&self, rhs: F::Extension) -> PolynomialCoeffs<F::Extension>
    where
        F: Extendable<D>,
    {
        PolynomialCoeffs::new(self.coeffs.iter().map(|&c| rhs.scalar_mul(c)).collect())
    }
}

impl<F: Field> PartialEq for PolynomialCoeffs<F> {
    fn eq(&self, other: &Self) -> bool {
        let max_terms = self.coeffs.len().max(other.coeffs.len());
        for i in 0..max_terms {
            let self_i = self.coeffs.get(i).cloned().unwrap_or(F::ZERO);
            let other_i = other.coeffs.get(i).cloned().unwrap_or(F::ZERO);
            if self_i != other_i {
                return false;
            }
        }
        true
    }
}

impl<F: Field> Eq for PolynomialCoeffs<F> {}

impl<F: Field> From<Vec<F>> for PolynomialCoeffs<F> {
    fn from(coeffs: Vec<F>) -> Self {
        Self::new(coeffs)
    }
}

impl<F: Field> Add for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    fn add(self, rhs: Self) -> Self::Output {
        let len = max(self.len(), rhs.len());
        let a = self.padded(len).coeffs;
        let b = rhs.padded(len).coeffs;
        let coeffs = a.into_iter().zip(b).map(|(x, y)| x + y).collect();
        PolynomialCoeffs::new(coeffs)
    }
}

impl<F: Field> Sum for PolynomialCoeffs<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::empty(), |acc, p| &acc + &p)
    }
}

impl<F: Field> Sub for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        let len = max(self.len(), rhs.len());
        let mut coeffs = self.padded(len).coeffs;
        for (i, &c) in rhs.coeffs.iter().enumerate() {
            coeffs[i] -= c;
        }
        PolynomialCoeffs::new(coeffs)
    }
}

impl<F: Field> AddAssign for PolynomialCoeffs<F> {
    fn add_assign(&mut self, rhs: Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, r) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *l += r;
        }
    }
}

impl<F: Field> AddAssign<&Self> for PolynomialCoeffs<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, &r) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *l += r;
        }
    }
}

impl<F: Field> SubAssign for PolynomialCoeffs<F> {
    fn sub_assign(&mut self, rhs: Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, r) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *l -= r;
        }
    }
}

impl<F: Field> SubAssign<&Self> for PolynomialCoeffs<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, &r) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *l -= r;
        }
    }
}

impl<F: Field> Mul<F> for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    fn mul(self, rhs: F) -> Self::Output {
        let coeffs = self.coeffs.iter().map(|&x| rhs * x).collect();
        PolynomialCoeffs::new(coeffs)
    }
}

impl<F: Field> MulAssign<F> for PolynomialCoeffs<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.coeffs.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<F: Field> Mul for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        let new_len = (self.len() + rhs.len()).next_power_of_two();
        let a = self.padded(new_len);
        let b = rhs.padded(new_len);
        let a_evals = a.fft();
        let b_evals = b.fft();

        let mul_evals: Vec<F> = a_evals
            .values
            .into_iter()
            .zip(b_evals.values)
            .map(|(pa, pb)| pa * pb)
            .collect();
        ifft(mul_evals.into())
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use std::time::Instant;

    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Sample;

    #[test]
    fn test_trimmed() {
        type F = GoldilocksField;

        assert_eq!(
            PolynomialCoeffs::<F> { coeffs: vec![] }.trimmed(),
            PolynomialCoeffs::<F> { coeffs: vec![] }
        );
        assert_eq!(
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ZERO]
            }
            .trimmed(),
            PolynomialCoeffs::<F> { coeffs: vec![] }
        );
        assert_eq!(
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ONE, F::TWO, F::ZERO, F::ZERO]
            }
            .trimmed(),
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ONE, F::TWO]
            }
        );
    }

    #[test]
    fn test_coset_fft() {
        type F = GoldilocksField;

        let k = 8;
        let n = 1 << k;
        let poly = PolynomialCoeffs::new(F::rand_vec(n));
        let shift = F::rand();
        let coset_evals = poly.coset_fft(shift).values;

        let generator = F::primitive_root_of_unity(k);
        let naive_coset_evals = F::cyclic_subgroup_coset_known_order(generator, shift, n)
            .into_iter()
            .map(|x| poly.eval(x))
            .collect::<Vec<_>>();
        assert_eq!(coset_evals, naive_coset_evals);

        let ifft_coeffs = PolynomialValues::new(coset_evals).coset_ifft(shift);
        assert_eq!(poly, ifft_coeffs);
    }

    #[test]
    fn test_coset_ifft() {
        type F = GoldilocksField;

        let k = 8;
        let n = 1 << k;
        let evals = PolynomialValues::new(F::rand_vec(n));
        let shift = F::rand();
        let coeffs = evals.clone().coset_ifft(shift);

        let generator = F::primitive_root_of_unity(k);
        let naive_coset_evals = F::cyclic_subgroup_coset_known_order(generator, shift, n)
            .into_iter()
            .map(|x| coeffs.eval(x))
            .collect::<Vec<_>>();
        assert_eq!(evals, naive_coset_evals.into());

        let fft_evals = coeffs.coset_fft(shift);
        assert_eq!(evals, fft_evals);
    }

    #[test]
    fn test_polynomial_multiplication() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000));
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg));
        let m1 = &a * &b;
        let m2 = &a * &b;
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(m1.eval(x), a.eval(x) * b.eval(x));
            assert_eq!(m2.eval(x), a.eval(x) * b.eval(x));
        }
    }

    #[test]
    fn test_inv_mod_xn() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let a_deg = rng.gen_range(0..1_000);
        let n = rng.gen_range(1..1_000);
        let mut a = PolynomialCoeffs::new(F::rand_vec(a_deg + 1));
        if a.coeffs[0].is_zero() {
            a.coeffs[0] = F::ONE; // First coefficient needs to be nonzero.
        }
        let b = a.inv_mod_xn(n);
        let mut m = &a * &b;
        m.coeffs.truncate(n);
        m.trim();
        assert_eq!(
            m,
            PolynomialCoeffs::new(vec![F::ONE]),
            "a: {:#?}, b:{:#?}, n:{:#?}, m:{:#?}",
            a,
            b,
            n,
            m
        );
    }

    #[test]
    fn test_polynomial_long_division() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000));
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg));
        let (q, r) = a.div_rem_long_division(&b);
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x));
        }
    }

    #[test]
    fn test_polynomial_division() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000));
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg));
        let (q, r) = a.div_rem(&b);
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x));
        }
    }

    #[test]
    fn test_polynomial_division_by_constant() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let a_deg = rng.gen_range(1..10_000);
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::from(vec![F::rand()]);
        let (q, r) = a.div_rem(&b);
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x));
        }
    }

    // Test to see which polynomial division method is faster for divisions of the type
    // `(X^n - 1)/(X - a)
    #[test]
    fn test_division_linear() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let l = 14;
        let n = 1 << l;
        let g = F::primitive_root_of_unity(l);
        let xn_minus_one = {
            let mut xn_min_one_vec = vec![F::ZERO; n + 1];
            xn_min_one_vec[n] = F::ONE;
            xn_min_one_vec[0] = F::NEG_ONE;
            PolynomialCoeffs::new(xn_min_one_vec)
        };

        let a = g.exp_u64(rng.gen_range(0..(n as u64)));
        let denom = PolynomialCoeffs::new(vec![-a, F::ONE]);
        let now = Instant::now();
        xn_minus_one.div_rem(&denom);
        println!("Division time: {:?}", now.elapsed());
        let now = Instant::now();
        xn_minus_one.div_rem_long_division(&denom);
        println!("Division time: {:?}", now.elapsed());
    }

    #[test]
    fn eq() {
        type F = GoldilocksField;
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO, F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ONE]),
            PolynomialCoeffs::new(vec![F::ONE, F::ZERO])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![F::ONE])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO, F::ONE])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ONE, F::ZERO])
        );
    }
}

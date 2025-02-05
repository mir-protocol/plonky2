#[cfg(nightly)]
use core::ops::Mul;

use crate::extension::quadratic::QuadraticExtension;
use crate::extension::quartic::QuarticExtension;
use crate::extension::quintic::QuinticExtension;
use crate::extension::{Extendable, Frobenius};
use crate::goldilocks_field::GoldilocksField;
use crate::types::Field;

impl Frobenius<1> for GoldilocksField {}

impl Extendable<2> for GoldilocksField {
    type Extension = QuadraticExtension<Self>;

    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    const W: Self = Self(7);

    // DTH_ROOT = W^((ORDER - 1)/2)
    const DTH_ROOT: Self = Self(18446744069414584320);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 2] = [Self(0), Self(11713931119993638672)];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 2] = [Self(0), Self(7226896044987257365)];
}

#[cfg(nightly)]
impl Mul for QuadraticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1]) = self;
        let Self([b0, b1]) = rhs;
        let c = self::specialized::ext2_mul([a0.0, a1.0], [b0.0, b1.0]);
        Self(c)
    }
}

impl Extendable<4> for GoldilocksField {
    type Extension = QuarticExtension<Self>;

    const W: Self = Self(7);

    // DTH_ROOT = W^((ORDER - 1)/4)
    const DTH_ROOT: Self = Self(281474976710656);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 4] =
        [Self(0), Self(8295451483910296135), Self(0), Self(0)];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 4] =
        [Self(0), Self(0), Self(0), Self(17216955519093520442)];
}

#[cfg(nightly)]
impl Mul for QuarticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = rhs;
        let c = self::specialized::ext4_mul([a0.0, a1.0, a2.0, a3.0], [b0.0, b1.0, b2.0, b3.0]);
        Self(c)
    }
}

impl Extendable<5> for GoldilocksField {
    type Extension = QuinticExtension<Self>;

    const W: Self = Self(3);

    // DTH_ROOT = W^((ORDER - 1)/5)
    const DTH_ROOT: Self = Self(1041288259238279555);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 5] = [
        Self(4624713872807171977),
        Self(381988216716071028),
        Self(14499722700050429911),
        Self(4870631734967222356),
        Self(4518902370426242880),
    ];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 5] = [
        Self::POWER_OF_TWO_GENERATOR,
        Self(0),
        Self(0),
        Self(0),
        Self(0),
    ];
}

#[cfg(nightly)]
impl Mul for QuinticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = self;
        let Self([b0, b1, b2, b3, b4]) = rhs;
        let c = self::specialized::ext5_mul(
            [a0.0, a1.0, a2.0, a3.0, a4.0],
            [b0.0, b1.0, b2.0, b3.0, b4.0],
        );
        Self(c)
    }
}

#[cfg(nightly)]
mod specialized {
    use static_assertions::const_assert;

    use super::*;
    use crate::goldilocks_field::reduce160;

    /*
     * The functions extD_add_prods[0-4] are helper functions for
     * computing products for extensions of degree D over the Goldilocks
     * field. They are faster than the generic method because all
     * reductions are delayed until the end which means only one per
     * result coefficient is necessary.
     */

    /// Return `a`, `b` such that `a + b*2^128 = 3*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
    #[inline(always)]
    const fn u160_times_3(x: u128, y: u32) -> (u128, u32) {
        let (s, cy) = x.overflowing_add(x << 1);
        (s, 3 * y + (x >> 127) as u32 + cy as u32)
    }

    /// Return `a`, `b` such that `a + b*2^128 = 7*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
    #[inline(always)]
    const fn u160_times_7(x: u128, y: u32) -> (u128, u32) {
        let (d, br) = (x << 3).overflowing_sub(x);
        // NB: subtracting the borrow can't underflow
        (d, 7 * y + (x >> (128 - 3)) as u32 - br as u32)
    }

    /*
     * Quadratic multiplication and squaring
     */

    #[inline(always)]
    fn ext2_add_prods0(a: &[u64; 2], b: &[u64; 2]) -> GoldilocksField {
        // Computes a0 * b0 + W * a1 * b1;
        let [a0, a1] = *a;
        let [b0, b1] = *b;

        let cy;

        // W * a1 * b1
        let (mut cumul_lo, mut cumul_hi) = u160_times_7((a1 as u128) * (b1 as u128), 0u32);

        // a0 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext2_add_prods1(a: &[u64; 2], b: &[u64; 2]) -> GoldilocksField {
        // Computes a0 * b1 + a1 * b0;
        let [a0, a1] = *a;
        let [b0, b1] = *b;

        let cy;

        // a0 * b1
        let mut cumul_lo = (a0 as u128) * (b1 as u128);

        // a1 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
        let cumul_hi = cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    /// Multiply a and b considered as elements of GF(p^2).
    #[inline(always)]
    pub(crate) fn ext2_mul(a: [u64; 2], b: [u64; 2]) -> [GoldilocksField; 2] {
        // The code in ext2_add_prods[01] assumes the quadratic extension
        // generator is 7.
        const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);

        let c0 = ext2_add_prods0(&a, &b);
        let c1 = ext2_add_prods1(&a, &b);
        [c0, c1]
    }

    /*
     * Quartic multiplication and squaring
     */

    #[inline(always)]
    fn ext4_add_prods0(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
        // Computes c0 = a0 * b0 + W * (a1 * b3 + a2 * b2 + a3 * b1)

        let [a0, a1, a2, a3] = *a;
        let [b0, b1, b2, b3] = *b;

        let mut cy;

        // a1 * b3
        let mut cumul_lo = (a1 as u128) * (b3 as u128);

        // a2 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b2 as u128));
        let mut cumul_hi = cy as u32;

        // a3 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // * W
        (cumul_lo, cumul_hi) = u160_times_7(cumul_lo, cumul_hi);

        // a0 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext4_add_prods1(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
        // Computes c1 = a0 * b1 + a1 * b0 + W * (a2 * b3 + a3 * b2);

        let [a0, a1, a2, a3] = *a;
        let [b0, b1, b2, b3] = *b;

        let mut cy;

        // a2 * b3
        let mut cumul_lo = (a2 as u128) * (b3 as u128);

        // a3 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b2 as u128));
        let mut cumul_hi = cy as u32;

        // * W
        (cumul_lo, cumul_hi) = u160_times_7(cumul_lo, cumul_hi);

        // a0 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a1 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext4_add_prods2(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
        // Computes c2 = a0 * b2 + a1 * b1 + a2 * b0 + W * a3 * b3;

        let [a0, a1, a2, a3] = *a;
        let [b0, b1, b2, b3] = *b;

        let mut cy;

        // W * a3 * b3
        let (mut cumul_lo, mut cumul_hi) = u160_times_7((a3 as u128) * (b3 as u128), 0u32);

        // a0 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b2 as u128));
        cumul_hi += cy as u32;

        // a1 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a2 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext4_add_prods3(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
        // Computes c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;

        let [a0, a1, a2, a3] = *a;
        let [b0, b1, b2, b3] = *b;

        let mut cy;

        // a0 * b3
        let mut cumul_lo = (a0 as u128) * (b3 as u128);

        // a1 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b2 as u128));
        let mut cumul_hi = cy as u32;

        // a2 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a3 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    /// Multiply a and b considered as elements of GF(p^4).
    #[inline(always)]
    pub(crate) fn ext4_mul(a: [u64; 4], b: [u64; 4]) -> [GoldilocksField; 4] {
        // The code in ext4_add_prods[0-3] assumes the quartic extension
        // generator is 7.
        const_assert!(<GoldilocksField as Extendable<4>>::W.0 == 7u64);

        let c0 = ext4_add_prods0(&a, &b);
        let c1 = ext4_add_prods1(&a, &b);
        let c2 = ext4_add_prods2(&a, &b);
        let c3 = ext4_add_prods3(&a, &b);
        [c0, c1, c2, c3]
    }

    /*
     * Quintic multiplication and squaring
     */

    #[inline(always)]
    fn ext5_add_prods0(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
        // Computes c0 = a0 * b0 + W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1)

        let [a0, a1, a2, a3, a4] = *a;
        let [b0, b1, b2, b3, b4] = *b;

        let mut cy;

        // a1 * b4
        let mut cumul_lo = (a1 as u128) * (b4 as u128);

        // a2 * b3
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b3 as u128));
        let mut cumul_hi = cy as u32;

        // a3 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b2 as u128));
        cumul_hi += cy as u32;

        // a4 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // * W
        (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

        // a0 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext5_add_prods1(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
        // Computes c1 = a0 * b1 + a1 * b0 + W * (a2 * b4 + a3 * b3 + a4 * b2);

        let [a0, a1, a2, a3, a4] = *a;
        let [b0, b1, b2, b3, b4] = *b;

        let mut cy;

        // a2 * b4
        let mut cumul_lo = (a2 as u128) * (b4 as u128);

        // a3 * b3
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b3 as u128));
        let mut cumul_hi = cy as u32;

        // a4 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b2 as u128));
        cumul_hi += cy as u32;

        // * W
        (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

        // a0 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a1 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext5_add_prods2(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
        // Computes c2 = a0 * b2 + a1 * b1 + a2 * b0 + W * (a3 * b4 + a4 * b3);

        let [a0, a1, a2, a3, a4] = *a;
        let [b0, b1, b2, b3, b4] = *b;

        let mut cy;

        // a3 * b4
        let mut cumul_lo = (a3 as u128) * (b4 as u128);

        // a4 * b3
        (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b3 as u128));
        let mut cumul_hi = cy as u32;

        // * W
        (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

        // a0 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b2 as u128));
        cumul_hi += cy as u32;

        // a1 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a2 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext5_add_prods3(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
        // Computes c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + W * a4 * b4;

        let [a0, a1, a2, a3, a4] = *a;
        let [b0, b1, b2, b3, b4] = *b;

        let mut cy;

        // W * a4 * b4
        let (mut cumul_lo, mut cumul_hi) = u160_times_3((a4 as u128) * (b4 as u128), 0u32);

        // a0 * b3
        (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b3 as u128));
        cumul_hi += cy as u32;

        // a1 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b2 as u128));
        cumul_hi += cy as u32;

        // a2 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a3 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    #[inline(always)]
    fn ext5_add_prods4(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
        // Computes c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        let [a0, a1, a2, a3, a4] = *a;
        let [b0, b1, b2, b3, b4] = *b;

        let mut cy;

        // a0 * b4
        let mut cumul_lo = (a0 as u128) * (b4 as u128);

        // a1 * b3
        (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b3 as u128));
        let mut cumul_hi = cy as u32;

        // a2 * b2
        (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b2 as u128));
        cumul_hi += cy as u32;

        // a3 * b1
        (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b1 as u128));
        cumul_hi += cy as u32;

        // a4 * b0
        (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b0 as u128));
        cumul_hi += cy as u32;

        unsafe { reduce160(cumul_lo, cumul_hi) }
    }

    /// Multiply a and b considered as elements of GF(p^5).
    #[inline(always)]
    pub(crate) fn ext5_mul(a: [u64; 5], b: [u64; 5]) -> [GoldilocksField; 5] {
        // The code in ext5_add_prods[0-4] assumes the quintic extension
        // generator is 3.
        const_assert!(<GoldilocksField as Extendable<5>>::W.0 == 3u64);

        let c0 = ext5_add_prods0(&a, &b);
        let c1 = ext5_add_prods1(&a, &b);
        let c2 = ext5_add_prods2(&a, &b);
        let c3 = ext5_add_prods3(&a, &b);
        let c4 = ext5_add_prods4(&a, &b);
        [c0, c1, c2, c3, c4]
    }
}

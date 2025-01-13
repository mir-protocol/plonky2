use core::arch::x86_64::*;

const MSB_1: i64 = 0x8000000000000000u64 as i64;
const P_N_1: i64 = 0xFFFFFFFF;

#[inline(always)]
pub fn shift_avx(a: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        _mm256_xor_si256(*a, msb)
    }
}

#[inline(always)]
pub fn add_avx_a_sc(a_sc: &__m256i, b: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_add_epi64(*a_sc, *b);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let mask_ = _mm256_cmpgt_epi64(*a_sc, c0_s);
        let corr_ = _mm256_and_si256(mask_, p_n);
        let c_s = _mm256_add_epi64(c0_s, corr_);
        shift_avx(&c_s)
    }
}

#[inline(always)]
pub fn add_avx(a: &__m256i, b: &__m256i) -> __m256i {
    let a_sc = shift_avx(a);
    add_avx_a_sc(&a_sc, b)
}

#[inline(always)]
pub fn add_avx_s_b_small(a_s: &__m256i, b_small: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_add_epi64(*a_s, *b_small);
        let mask_ = _mm256_cmpgt_epi32(*a_s, c0_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_add_epi64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn sub_avx_s_b_small(a_s: &__m256i, b: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_sub_epi64(*a_s, *b);
        let mask_ = _mm256_cmpgt_epi32(c0_s, *a_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_sub_epi64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn reduce_avx_128_64(c_h: &__m256i, c_l: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let c_hh = _mm256_srli_epi64(*c_h, 32);
        let c_ls = _mm256_xor_si256(*c_l, msb);
        let c1_s = sub_avx_s_b_small(&c_ls, &c_hh);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let c2 = _mm256_mul_epu32(*c_h, p_n);
        let c_s = add_avx_s_b_small(&c1_s, &c2);
        _mm256_xor_si256(c_s, msb)
    }
}

// Here we suppose c_h < 2^32
#[inline(always)]
pub fn reduce_avx_96_64(c_h: &__m256i, c_l: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let c_ls = _mm256_xor_si256(*c_l, msb);
        let c2 = _mm256_mul_epu32(*c_h, p_n);
        let c_s = add_avx_s_b_small(&c_ls, &c2);
        _mm256_xor_si256(c_s, msb)
    }
}

#[inline(always)]
pub fn mult_avx_128(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    unsafe {
        let a_h = _mm256_srli_epi64(*a, 32);
        let b_h = _mm256_srli_epi64(*b, 32);
        let c_hh = _mm256_mul_epu32(a_h, b_h);
        let c_hl = _mm256_mul_epu32(a_h, *b);
        let c_lh = _mm256_mul_epu32(*a, b_h);
        let c_ll = _mm256_mul_epu32(*a, *b);
        let c_ll_h = _mm256_srli_epi64(c_ll, 32);
        let r0 = _mm256_add_epi64(c_hl, c_ll_h);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let r0_l = _mm256_and_si256(r0, p_n);
        let r0_h = _mm256_srli_epi64(r0, 32);
        let r1 = _mm256_add_epi64(c_lh, r0_l);
        let r1_l = _mm256_slli_epi64(r1, 32);
        let c_l = _mm256_blend_epi32(c_ll, r1_l, 0xaa);
        let r2 = _mm256_add_epi64(c_hh, r0_h);
        let r1_h = _mm256_srli_epi64(r1, 32);
        let c_h = _mm256_add_epi64(r2, r1_h);
        (c_h, c_l)
    }
}

#[inline(always)]
pub fn mult_avx(a: &__m256i, b: &__m256i) -> __m256i {
    let (c_h, c_l) = mult_avx_128(a, b);
    reduce_avx_128_64(&c_h, &c_l)
}

// Multiply two 64bit numbers with the assumption that the product does not averflow.
#[inline]
pub unsafe fn mul64_no_overflow(a: &__m256i, b: &__m256i) -> __m256i {
    let r = _mm256_mul_epu32(*a, *b);
    let ah = _mm256_srli_epi64(*a, 32);
    let bh = _mm256_srli_epi64(*b, 32);
    let r1 = _mm256_mul_epu32(*a, bh);
    let r1 = _mm256_slli_epi64(r1, 32);
    let r = _mm256_add_epi64(r, r1);
    let r1 = _mm256_mul_epu32(ah, *b);
    let r1 = _mm256_slli_epi64(r1, 32);
    let r = _mm256_add_epi64(r, r1);
    r
}

#[inline]
pub unsafe fn add64_no_carry(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    /*
     * a and b are signed 4 x i64. Suppose a and b represent only one i64, then:
     * - (test 1): if a < 2^63 and b < 2^63 (this means a >= 0 and b >= 0) => sum does not overflow => cout = 0
     * - if a >= 2^63 and b >= 2^63 => sum overflows so sum = a + b and cout = 1
     * - (test 2): if (a < 2^63 and b >= 2^63) or (a >= 2^63 and b < 2^63)
     *   - (test 3): if a + b < 2^64 (this means a + b is negative in signed representation) => no overflow so cout = 0
     *   - (test 3): if a + b >= 2^64 (this means a + b becomes positive in signed representation, that is, a + b >= 0) => there is overflow so cout = 1
     */
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let r = _mm256_add_epi64(*a, *b);
    let ma = _mm256_cmpgt_epi64(zeros, *a);
    let mb = _mm256_cmpgt_epi64(zeros, *b);
    let m1 = _mm256_and_si256(ma, mb); // test 1
    let m2 = _mm256_xor_si256(ma, mb); // test 2
    let m23 = _mm256_cmpgt_epi64(zeros, r); // test 3
    let m2 = _mm256_andnot_si256(m23, m2);
    let m = _mm256_or_si256(m1, m2);
    let co = _mm256_and_si256(m, ones);
    (r, co)
}

#[inline(always)]
pub fn sqr_avx_128(a: &__m256i) -> (__m256i, __m256i) {
    unsafe {
        let a_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*a)));
        let c_ll = _mm256_mul_epu32(*a, *a);
        let c_lh = _mm256_mul_epu32(*a, a_h);
        let c_hh = _mm256_mul_epu32(a_h, a_h);
        let c_ll_hi = _mm256_srli_epi64(c_ll, 33);
        let t0 = _mm256_add_epi64(c_lh, c_ll_hi);
        let t0_hi = _mm256_srli_epi64(t0, 31);
        let res_hi = _mm256_add_epi64(c_hh, t0_hi);
        let c_lh_lo = _mm256_slli_epi64(c_lh, 33);
        let res_lo = _mm256_add_epi64(c_ll, c_lh_lo);
        (res_hi, res_lo)
    }
}

#[inline(always)]
pub fn sqr_avx(a: &__m256i) -> __m256i {
    let (c_h, c_l) = sqr_avx_128(a);
    reduce_avx_128_64(&c_h, &c_l)
}

#[inline(always)]
pub fn sbox_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i) {
    // x^2
    let p10 = sqr_avx(s0);
    let p11 = sqr_avx(s1);
    let p12 = sqr_avx(s2);
    // x^3
    let p30 = mult_avx(&p10, s0);
    let p31 = mult_avx(&p11, s1);
    let p32 = mult_avx(&p12, s2);
    // x^4 = (x^2)^2
    let p40 = sqr_avx(&p10);
    let p41 = sqr_avx(&p11);
    let p42 = sqr_avx(&p12);
    // x^7
    *s0 = mult_avx(&p40, &p30);
    *s1 = mult_avx(&p41, &p31);
    *s2 = mult_avx(&p42, &p32);
}

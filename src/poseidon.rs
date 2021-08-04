//! Implementation of the Poseidon hash function, as described in
//! https://eprint.iacr.org/2019/458.pdf

use unroll::unroll_for_loops;
use crate::poseidon_constants::{
    WIDTH,
    HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS,
    MDS_MATRIX_EXPS, ALL_ROUND_CONSTANTS,
    FAST_PARTIAL_ROUND_CONSTANTS,
    FAST_PARTIAL_ROUND_VS, FAST_PARTIAL_ROUND_W_HATS,
    FAST_PARTIAL_ROUND_INITAL_MATRIX};

use crate::field::field::Field;

#[inline]
#[unroll_for_loops]
fn constant_layer<F: Field>(state: &mut [F; WIDTH], round_ctr: usize) {
    for i in 0..WIDTH {
        state[i] += F::from_canonical_u64(ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr]);
    }
}

#[inline]
#[unroll_for_loops]
fn sbox_layer<F: Field>(state: &mut [F; WIDTH]) {
    for i in 0..WIDTH {
        state[i] = state[i].cube();
    }
}

#[inline]
#[unroll_for_loops]
fn mds_row_shf(r: usize, v: &[u64; WIDTH]) -> u128 {
    debug_assert!(r < WIDTH);
    // TODO: Double-check that the calculations associated with the
    // zeros in this const array are not removed by the compiler; they
    // weren't removed when I used MDS_MATRIX_EXPS[(i + r) % WIDTH],
    // but they seem to be when using MDS_MATRIX_EXPS[i].

    let mut res = 0u128;
    for i in 0..WIDTH {
        res += (v[(i + WIDTH - r) % WIDTH] as u128) << MDS_MATRIX_EXPS[i];
    }
    res
}

#[inline]
#[unroll_for_loops]
fn mds_layer<F: Field>(state_: &[F; WIDTH]) -> [F; WIDTH] {
    let mut result = [F::ZERO; WIDTH];

    // TODO: Need a better way to do this; we only want the raw u64 anyway.
    let mut state = [0u64; WIDTH];
    for r in 0..WIDTH {
        state[r] = state_[r].to_canonical_u64();
    }

    for r in 0..WIDTH {
        result[r] = F::from_canonical_u128(mds_row_shf(r, &state));
    }
    result
}

#[inline]
#[unroll_for_loops]
fn mds_partial_layer_init<F: Field>(state: &[F; WIDTH]) -> [F; WIDTH] {
    let mut result = [F::ZERO; WIDTH];

    // Initial matrix has first row/column = [1, 0, ..., 0];

    // r = 0
    result[0] = state[0];

    for r in 1..WIDTH {
        for c in 1..WIDTH {
            let t = F::from_canonical_u64(FAST_PARTIAL_ROUND_INITAL_MATRIX[r - 1][c - 1]);
            result[r] += t * state[c];
        }
    }
    result
}

/// Computes s*A where s is the state row vector and A is the matrix
///
///    [ M_00  | v  ]
///    [ ------+--- ]
///    [ w_hat | Id ]
///
/// M_00 is a scalar, v is 1x(t-1), w_hat is (t-1)x1 and Id is the
/// (t-1)x(t-1) identity matrix.
#[inline]
#[unroll_for_loops]
fn mds_partial_layer_fast<F: Field>(state: &[F; WIDTH], r: usize) -> [F; WIDTH] {
    // Set d = [M_00 | w^] dot [state]
    const MDS_TOP_LEFT: u64 = 1u64 << MDS_MATRIX_EXPS[0];
    let mut d = F::from_canonical_u64(MDS_TOP_LEFT) * state[0];
    for i in 1..WIDTH {
        let t = F::from_canonical_u64(FAST_PARTIAL_ROUND_W_HATS[r][i - 1]);
        d += t * state[i]
    }

    // result = [d] concat [state[0] * v + state[shift up by 1]]
    let mut result = [F::ZERO; WIDTH];
    result[0] = d;
    for i in 1..WIDTH {
        let t = F::from_canonical_u64(FAST_PARTIAL_ROUND_VS[r][i - 1]);
        result[i] = state[0] * t + state[i];
    }
    result
}

#[inline]
#[unroll_for_loops]
fn full_rounds<F: Field>(state: &mut [F; WIDTH], round_ctr: &mut usize) {
    for _ in 0..HALF_N_FULL_ROUNDS {
        constant_layer(state, *round_ctr);
        sbox_layer(state);
        *state = mds_layer(state);
        *round_ctr += 1;
    }
}


#[inline]
#[unroll_for_loops]
fn partial_rounds<F: Field>(state: &mut [F; WIDTH], round_ctr: &mut usize) {
    for _ in 0..N_PARTIAL_ROUNDS {
        constant_layer(state, *round_ctr);
        state[0] = state[0].cube();
        *state = mds_layer(state);
        *round_ctr += 1;
    }
}

#[inline]
#[unroll_for_loops]
fn partial_rounds_fast<F: Field>(state: &mut [F; WIDTH], round_ctr: &mut usize) {
    constant_layer(state, *round_ctr);

    *state = mds_partial_layer_init(state);
    // One less than N_PARTIAL_ROUNDS because we do the last one
    // separately at the end.
    for i in 0..(N_PARTIAL_ROUNDS - 1) {
        state[0] = state[0].cube();
        state[0] += F::from_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
        *state = mds_partial_layer_fast(state, i);
    }
    state[0] = state[0].cube();
    *state = mds_partial_layer_fast(state, N_PARTIAL_ROUNDS - 1);
    *round_ctr += N_PARTIAL_ROUNDS;
}

#[unroll_for_loops]
pub fn poseidon_fast<F: Field>(input: [F; WIDTH]) -> [F; WIDTH] {
    let mut state = input;
    let mut round_ctr = 0;

    full_rounds(&mut state, &mut round_ctr);
    partial_rounds_fast(&mut state, &mut round_ctr);
    full_rounds(&mut state, &mut round_ctr);

    state
}

#[unroll_for_loops]
pub fn poseidon<F: Field>(input: [F; WIDTH]) -> [F; WIDTH] {
    let mut state = input;
    let mut round_ctr = 0;

    full_rounds(&mut state, &mut round_ctr);
    partial_rounds(&mut state, &mut round_ctr);
    full_rounds(&mut state, &mut round_ctr);

    state
}

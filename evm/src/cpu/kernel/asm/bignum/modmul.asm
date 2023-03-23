// Arithmetic on little-endian integers represented with 128-bit limbs.
// All integers must be under a given length bound, and are padded with leading zeroes.

// Stores a * b % m in output_loc, leaving a, b, and m unchanged.
// a, b, and m must have the same length.
// Both output_loc and scratch_1 must have size length.
// Both scratch_2 and scratch_3 have size 2 * length and be initialized with zeroes.
global modmul_bignum:
    // stack: len, a_loc, b_loc, m_loc, out_loc, s1 (=scratch_1), s2, s3, retdest
    DUP1
    ISZERO
    %jumpi(len_zero)
    
    // The prover provides x := (a * b) % m, which we store in output_loc.
    
    PUSH 0
    // stack: i=0, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
modmul_remainder_loop:
    // stack: i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    PROVER_INPUT(bignum_modmul)
    // stack: PI, i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    DUP7
    DUP3
    ADD
    // stack: out_loc[i], PI, i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %mstore_kernel_general
    // stack: i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %increment
    DUP2
    DUP2
    // stack: i+1, len, i+1, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    SUB // functions as NEQ
    // stack: i+1!=len, i+1, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %jumpi(modmul_remainder_loop)
// end of modmul_remainder_loop
    // stack: i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    POP

    // The prover provides k := (a * b) / m, which we store in scratch_1.

    // stack: len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    PUSH 0
    // stack: i=0, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
modmul_quotient_loop:
    // stack: i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    PROVER_INPUT(bignum_modmul)
    // stack: PI, i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    DUP8
    DUP3
    ADD
    // stack: s1[i], PI, i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %mstore_kernel_general
    // stack: i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %increment
    DUP2
    DUP2
    // stack: i+1, len, i+1, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    SUB // functions as NEQ
    // stack: i+1!=len, i+1, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %jumpi(modmul_quotient_loop)
// end of modmul_quotient_loop
    // stack: i, len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    POP
    // stack: len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest

    // Verification step 1: calculate x + k * m.

    // Store k * m in scratch_2.
    PUSH modmul_return_1
    %stack (return, len, a, b, m, out, s1, s2) -> (len, s1, m, s2, return, len, a, b, out, s2)
    // stack: len, s1, m_loc, s2, modmul_return_1, len, a_loc, b_loc, out_loc, s2, s3, retdest
    %jump(mul_bignum)
modmul_return_1:
    // stack: len, a_loc, b_loc, out_loc, s2, s3, retdest

    // Add x into k * m (in scratch_2).
    PUSH modmul_return_2
    %stack (return, len, a, b, out, s2) -> (len, s2, out, return, len, a, b, s2)
    // stack: len, s2, out_loc, modmul_return_2, len, a_loc, b_loc, s2, s3, retdest
    %jump(add_bignum)
modmul_return_2:
    // stack: carry, len, a_loc, b_loc, s2, s3, retdest
    ISZERO
    %jumpi(no_carry)

    // stack: len, a_loc, b_loc, s2, s3, retdest
    DUP4
    DUP2
    ADD
    // stack: cur_loc=s2 + len, len, a_loc, b_loc, s2, s3, retdest
increment_loop:
    // stack: cur_loc, len, a_loc, b_loc, s2, s3, retdest
    DUP1
    %mload_kernel_general
    // stack: val, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    %increment
    DUP1
    // stack: val+1, val+1, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    %eq_const(@BIGNUM_LIMB_BASE)
    DUP1
    ISZERO
    // stack: val+1!=limb_base, val+1==limb_base, val+1, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    SWAP1
    SWAP2
    // stack: val+1, val+1!=limb_base, val+1==limb_base, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    MUL
    // stack: to_write=(val+1)*(val+1!=limb_base), continue=val+1==limb_base, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    DUP3
    // stack: cur_loc, to_write, continue, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    %mstore_kernel_general
    // stack: continue, cur_loc, len, a_loc, b_loc, s2, s3, retdest
    SWAP1
    %increment
    DUP1
    DUP8
    // stack: s3, cur_loc + 1, cur_loc + 1, continue, len, a_loc, b_loc, s2, s3, retdest
    EQ
    ISZERO
    // stack: cur_loc + 1 != s3, cur_loc + 1, continue, len, a_loc, b_loc, s2, s3, retdest
    SWAP1
    SWAP2
    // stack: continue, cur_loc + 1 != s3, cur_loc + 1, len, a_loc, b_loc, s2, s3, retdest
    MUL
    // stack: new_continue=continue*(cur_loc + 1 != s3), cur_loc + 1, len, a_loc, b_loc, s2, s3, retdest
    %jumpi(increment_loop)
    // stack: cur_loc + 1, len, a_loc, b_loc, s2, s3, retdest
    POP
no_carry:
    // stack: len, a_loc, b_loc, s2, s3, retdest

    // Calculate a * b.

    // Store a * b in scratch_3.
    PUSH modmul_return_3
    %stack (return, len, a, b, s2, s3) -> (len, a, b, s3, return, len, s2, s3)
    // stack: len, a_loc, b_loc, s3, modmul_return_3, len, s2, s3, retdest
    %jump(mul_bignum)
modmul_return_3:
    // stack: len, s2, s3, retdest

    // Check that x + k * m = a * b.
    // Walk through scratch_2 and scratch_3, checking that they are equal.
    // stack: n=len, i=s2, j=s3, retdest
modmul_check_loop:
    // stack: n, i, j, retdest
    %stack (l, idx: 2) -> (idx, l, idx)
    // stack: i, j, n, i, j, retdest
    %mload_kernel_general
    SWAP1
    %mload_kernel_general
    SWAP1
    // stack: mem[i], mem[j], n, i, j, retdest
    %assert_eq
    // stack: n, i, j, retdest
    %decrement
    SWAP1
    %increment
    SWAP2
    %increment
    SWAP2
    SWAP1
    // stack: n-1, i+1, j+1, retdest
    DUP1
    // stack: n-1, n-1, i+1, j+1, retdest
    %jumpi(modmul_check_loop)
// end of modmul_check_loop
    // stack: n-1, i+1, j+1, retdest
    %pop3
    // stack: retdest
    JUMP
len_zero:
    // stack: len, a_loc, b_loc, m_loc, out_loc, s1, s2, s3, retdest
    %pop8
    // stack: retdest
    JUMP
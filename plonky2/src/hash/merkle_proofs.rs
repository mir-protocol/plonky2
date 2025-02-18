#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::ops::RangeInclusive;

use anyhow::{ensure, Result};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::VerifierCircuitTarget;
use crate::plonk::config::{AlgebraicHasher, GenericHashOut, Hasher};

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
pub struct MerkleProof<F: RichField, H: Hasher<F>> {
    /// The Merkle digest of each sibling subtree, staying from the bottommost layer.
    pub siblings: Vec<H::Hash>,
}

impl<F: RichField, H: Hasher<F>> MerkleProof<F, H> {
    pub fn len(&self) -> usize {
        self.siblings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleProofTarget {
    /// The Merkle digest of each sibling subtree, staying from the bottommost layer.
    pub siblings: Vec<HashOutTarget>,
}

/// Verifies that the given leaf data is present at the given index in the Merkle tree with the
/// given root.
pub fn verify_merkle_proof<F: RichField, H: Hasher<F>>(
    leaf_data: Vec<F>,
    leaf_index: usize,
    merkle_root: H::Hash,
    proof: &MerkleProof<F, H>,
) -> Result<()> {
    let merkle_cap = MerkleCap(vec![merkle_root]);
    verify_merkle_proof_to_cap(leaf_data, leaf_index, &merkle_cap, proof)
}

/// Verifies that the given leaf data is present at the given index in the Merkle tree with the
/// given cap.
pub fn verify_merkle_proof_to_cap<F: RichField, H: Hasher<F>>(
    leaf_data: Vec<F>,
    leaf_index: usize,
    merkle_cap: &MerkleCap<F, H>,
    proof: &MerkleProof<F, H>,
) -> Result<()> {
    verify_batch_merkle_proof_to_cap(
        &[leaf_data.clone()],
        &[proof.siblings.len()],
        leaf_index,
        merkle_cap,
        proof,
    )
}

/// Verifies that the given leaf data is present at the given index in the Field Merkle tree with the
/// given cap.
pub fn verify_batch_merkle_proof_to_cap<F: RichField, H: Hasher<F>>(
    leaf_data: &[Vec<F>],
    leaf_heights: &[usize],
    mut leaf_index: usize,
    merkle_cap: &MerkleCap<F, H>,
    proof: &MerkleProof<F, H>,
) -> Result<()> {
    assert_eq!(leaf_data.len(), leaf_heights.len());
    let mut current_digest = H::hash_or_noop(&leaf_data[0]);
    let mut current_height = leaf_heights[0];
    let mut leaf_data_index = 1;
    for &sibling_digest in &proof.siblings {
        let bit = leaf_index & 1;
        leaf_index >>= 1;
        current_digest = if bit == 1 {
            H::two_to_one(sibling_digest, current_digest)
        } else {
            H::two_to_one(current_digest, sibling_digest)
        };
        current_height -= 1;

        if leaf_data_index < leaf_heights.len() && current_height == leaf_heights[leaf_data_index] {
            let mut new_leaves = current_digest.to_vec();
            new_leaves.extend_from_slice(&leaf_data[leaf_data_index]);
            current_digest = H::hash_or_noop(&new_leaves);
            leaf_data_index += 1;
        }
    }
    assert_eq!(leaf_data_index, leaf_data.len());
    ensure!(
        current_digest == merkle_cap.0[leaf_index],
        "Invalid Merkle proof."
    );

    Ok(())
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Verifies that the given leaf data is present at the given index in the Merkle tree with the
    /// given root. The index is given by its little-endian bits.
    pub fn verify_merkle_proof<H: AlgebraicHasher<F>>(
        &mut self,
        leaf_data: Vec<Target>,
        leaf_index_bits: &[BoolTarget],
        merkle_root: HashOutTarget,
        proof: &MerkleProofTarget,
    ) {
        let merkle_cap = MerkleCapTarget(vec![merkle_root]);
        self.verify_merkle_proof_to_cap::<H>(leaf_data, leaf_index_bits, &merkle_cap, proof);
    }

    /// Verifies that the given leaf data is present at the given index in the Merkle tree with the
    /// given cap. The index is given by its little-endian bits.
    pub fn verify_merkle_proof_to_cap<H: AlgebraicHasher<F>>(
        &mut self,
        leaf_data: Vec<Target>,
        leaf_index_bits: &[BoolTarget],
        merkle_cap: &MerkleCapTarget,
        proof: &MerkleProofTarget,
    ) {
        let cap_index = self.le_sum(leaf_index_bits[proof.siblings.len()..].iter().copied());
        self.verify_merkle_proof_to_cap_with_cap_index::<H>(
            leaf_data,
            leaf_index_bits,
            cap_index,
            merkle_cap,
            proof,
        );
    }

    /// Same as `verify_merkle_proof_to_cap`, except with the final "cap index" as separate parameter,
    /// rather than being contained in `leaf_index_bits`.
    pub(crate) fn verify_merkle_proof_to_cap_with_cap_index<H: AlgebraicHasher<F>>(
        &mut self,
        leaf_data: Vec<Target>,
        leaf_index_bits: &[BoolTarget],
        cap_index: Target,
        merkle_cap: &MerkleCapTarget,
        proof: &MerkleProofTarget,
    ) {
        debug_assert!(H::AlgebraicPermutation::RATE >= NUM_HASH_OUT_ELTS);

        let zero = self.zero();
        let mut state: HashOutTarget = self.hash_or_noop::<H>(leaf_data);
        debug_assert_eq!(state.elements.len(), NUM_HASH_OUT_ELTS);

        for (&bit, &sibling) in leaf_index_bits.iter().zip(&proof.siblings) {
            debug_assert_eq!(sibling.elements.len(), NUM_HASH_OUT_ELTS);

            let mut perm_inputs = H::AlgebraicPermutation::default();
            perm_inputs.set_from_slice(&state.elements, 0);
            perm_inputs.set_from_slice(&sibling.elements, NUM_HASH_OUT_ELTS);
            // Ensure the rest of the state, if any, is zero:
            perm_inputs.set_from_iter(core::iter::repeat(zero), 2 * NUM_HASH_OUT_ELTS);
            let perm_outs = self.permute_swapped::<H>(perm_inputs, bit);
            let hash_outs = perm_outs.squeeze()[0..NUM_HASH_OUT_ELTS]
                .try_into()
                .unwrap();
            state = HashOutTarget {
                elements: hash_outs,
            };
        }

        for i in 0..NUM_HASH_OUT_ELTS {
            let result = self.random_access(
                cap_index,
                merkle_cap.0.iter().map(|h| h.elements[i]).collect(),
            );
            self.connect(result, state.elements[i]);
        }
    }

    /// Same as `verify_merkle_proof_to_cap`, except with the final "cap index" as separate parameter,
    /// rather than being contained in `leaf_index_bits`.
    pub(crate) fn verify_merkle_proof_to_cap_with_cap_indices<H: AlgebraicHasher<F>>(
        &mut self,
        condition: Target,
        leaf_data: Vec<Target>,
        leaf_index_bits: &[BoolTarget],
        log_n_range: RangeInclusive<usize>,
        n_index: Target,
        cap_index: Target,
        merkle_cap: &MerkleCapTarget,
        proof: &MerkleProofTarget,
    ) {
        debug_assert!(H::AlgebraicPermutation::RATE >= NUM_HASH_OUT_ELTS);

        let zero = self.zero();
        let mut state: HashOutTarget = self.hash_or_noop::<H>(leaf_data);
        debug_assert_eq!(state.elements.len(), NUM_HASH_OUT_ELTS);

        let num_log_n = log_n_range.clone().count();
        let mut final_states = vec![state; num_log_n];

        for (&bit, &sibling) in leaf_index_bits.iter().zip(&proof.siblings) {
            debug_assert_eq!(sibling.elements.len(), NUM_HASH_OUT_ELTS);

            let mut perm_inputs = H::AlgebraicPermutation::default();
            perm_inputs.set_from_slice(&state.elements, 0);
            perm_inputs.set_from_slice(&sibling.elements, NUM_HASH_OUT_ELTS);
            // Ensure the rest of the state, if any, is zero:
            perm_inputs.set_from_iter(core::iter::repeat(zero), 2 * NUM_HASH_OUT_ELTS);
            let perm_outs = self.permute_swapped::<H>(perm_inputs, bit);
            let hash_outs = perm_outs.squeeze()[0..NUM_HASH_OUT_ELTS]
                .try_into()
                .unwrap();
            state = HashOutTarget {
                elements: hash_outs,
            };
            // Store state at specific indices
            for n in 0..num_log_n - 1 {
                final_states[n] = final_states[n + 1];
            }
            final_states[num_log_n - 1] = state;
        }

        for i in 0..NUM_HASH_OUT_ELTS {
            let result = self.random_access(
                cap_index,
                merkle_cap.0.iter().map(|h| h.elements[i]).collect(),
            );
            let state = self.random_access(
                n_index,
                final_states.iter().map(|s| s.elements[i]).collect(),
            );
            self.conditional_assert_eq(condition, result, state);
        }
    }

    /// Same as `verify_batch_merkle_proof_to_cap`, except with the final "cap index" as separate parameter,
    /// rather than being contained in `leaf_index_bits`.
    pub(crate) fn verify_batch_merkle_proof_to_cap_with_cap_index<H: AlgebraicHasher<F>>(
        &mut self,
        leaf_data: &[Vec<Target>],
        leaf_heights: &[usize],
        leaf_index_bits: &[BoolTarget],
        cap_index: Target,
        merkle_cap: &MerkleCapTarget,
        proof: &MerkleProofTarget,
    ) {
        debug_assert!(H::AlgebraicPermutation::RATE >= NUM_HASH_OUT_ELTS);

        let zero = self.zero();
        let mut state: HashOutTarget = self.hash_or_noop::<H>(leaf_data[0].clone());
        debug_assert_eq!(state.elements.len(), NUM_HASH_OUT_ELTS);

        let mut current_height = leaf_heights[0];
        let mut leaf_data_index = 1;
        for (&bit, &sibling) in leaf_index_bits.iter().zip(&proof.siblings) {
            debug_assert_eq!(sibling.elements.len(), NUM_HASH_OUT_ELTS);

            let mut perm_inputs = H::AlgebraicPermutation::default();
            perm_inputs.set_from_slice(&state.elements, 0);
            perm_inputs.set_from_slice(&sibling.elements, NUM_HASH_OUT_ELTS);
            // Ensure the rest of the state, if any, is zero:
            perm_inputs.set_from_iter(core::iter::repeat(zero), 2 * NUM_HASH_OUT_ELTS);
            let perm_outs = self.permute_swapped::<H>(perm_inputs, bit);
            let hash_outs = perm_outs.squeeze()[0..NUM_HASH_OUT_ELTS]
                .try_into()
                .unwrap();
            state = HashOutTarget {
                elements: hash_outs,
            };
            current_height -= 1;

            if leaf_data_index < leaf_heights.len()
                && current_height == leaf_heights[leaf_data_index]
            {
                let mut new_leaves = state.elements.to_vec();
                new_leaves.extend_from_slice(&leaf_data[leaf_data_index]);
                state = self.hash_or_noop::<H>(new_leaves);

                leaf_data_index += 1;
            }
        }

        for i in 0..NUM_HASH_OUT_ELTS {
            let result = self.random_access(
                cap_index,
                merkle_cap.0.iter().map(|h| h.elements[i]).collect(),
            );
            self.connect(result, state.elements[i]);
        }
    }

    pub fn connect_hashes(&mut self, x: HashOutTarget, y: HashOutTarget) {
        for i in 0..NUM_HASH_OUT_ELTS {
            self.connect(x.elements[i], y.elements[i]);
        }
    }

    pub fn connect_merkle_caps(&mut self, x: &MerkleCapTarget, y: &MerkleCapTarget) {
        for (h0, h1) in x.0.iter().zip_eq(&y.0) {
            self.connect_hashes(*h0, *h1);
        }
    }

    pub fn connect_verifier_data(&mut self, x: &VerifierCircuitTarget, y: &VerifierCircuitTarget) {
        self.connect_merkle_caps(&x.constants_sigmas_cap, &y.constants_sigmas_cap);
        self.connect_hashes(x.circuit_digest, y.circuit_digest);
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::field::types::Field;
    use crate::hash::merkle_tree::MerkleTree;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    fn random_data<F: Field>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    #[test]
    fn test_recursive_merkle_proof() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_config();
        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let log_n = 8;
        let n = 1 << log_n;
        let cap_height = 1;
        let leaves = random_data::<F>(n, 7);
        let tree = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
        let i: usize = OsRng.gen_range(0..n);
        let proof = tree.prove(i);

        let proof_t = MerkleProofTarget {
            siblings: builder.add_virtual_hashes(proof.siblings.len()),
        };
        for i in 0..proof.siblings.len() {
            pw.set_hash_target(proof_t.siblings[i], proof.siblings[i])?;
        }

        let cap_t = builder.add_virtual_cap(cap_height);
        pw.set_cap_target(&cap_t, &tree.cap)?;

        let i_c = builder.constant(F::from_canonical_usize(i));
        let i_bits = builder.split_le(i_c, log_n);

        let data = builder.add_virtual_targets(tree.leaves[i].len());
        for j in 0..data.len() {
            pw.set_target(data[j], tree.leaves[i][j])?;
        }

        builder.verify_merkle_proof_to_cap::<<C as GenericConfig<D>>::InnerHasher>(
            data, &i_bits, &cap_t, &proof_t,
        );

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }
}

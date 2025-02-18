#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hashbrown::HashMap;
use num::Integer;

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::Hasher;

/// Compress multiple Merkle proofs on the same tree by removing redundancy in the Merkle paths.
pub(crate) fn compress_merkle_proofs<F: RichField, H: Hasher<F>>(
    cap_height: usize,
    indices: &[usize],
    proofs: &[MerkleProof<F, H>],
) -> Vec<MerkleProof<F, H>> {
    assert!(!proofs.is_empty());
    let height = cap_height + proofs[0].siblings.len();
    let num_leaves = 1 << height;
    let mut compressed_proofs = Vec::with_capacity(proofs.len());
    // Holds the known nodes in the tree at a given time. The root is at index 1.
    // Valid indices are 1 through n, and each element at index `i` has
    // children at indices `2i` and `2i +1` its parent at index `floor(i ∕ 2)`.
    let mut known = vec![false; 2 * num_leaves];
    for &i in indices {
        // The path from a leaf to the cap is known.
        for j in 0..(height - cap_height) {
            known[(i + num_leaves) >> j] = true;
        }
    }
    // For each proof collect all the unknown proof elements.
    for (&i, p) in indices.iter().zip(proofs) {
        let mut compressed_proof = MerkleProof {
            siblings: Vec::new(),
        };
        let mut index = i + num_leaves;
        for &sibling in &p.siblings {
            let sibling_index = index ^ 1;
            if !known[sibling_index] {
                // If the sibling is not yet known, add it to the proof and set it to known.
                compressed_proof.siblings.push(sibling);
                known[sibling_index] = true;
            }
            // Go up the tree and set the parent to known.
            index >>= 1;
            known[index] = true;
        }
        compressed_proofs.push(compressed_proof);
    }

    compressed_proofs
}

/// Decompress compressed Merkle proofs.
/// Note: The data and indices must be in the same order as in `compress_merkle_proofs`.
pub(crate) fn decompress_merkle_proofs<F: RichField, H: Hasher<F>>(
    leaves_data: &[Vec<F>],
    leaves_indices: &[usize],
    compressed_proofs: &[MerkleProof<F, H>],
    height: usize,
    cap_height: usize,
) -> Vec<MerkleProof<F, H>> {
    let num_leaves = 1 << height;
    let compressed_proofs = compressed_proofs.to_vec();
    let mut decompressed_proofs = Vec::with_capacity(compressed_proofs.len());
    // Holds the already seen nodes in the tree along with their value.
    let mut seen = HashMap::new();

    for (&i, v) in leaves_indices.iter().zip(leaves_data) {
        // Observe the leaves.
        seen.insert(i + num_leaves, H::hash_or_noop(v));
    }

    // Iterators over the siblings.
    let mut siblings = compressed_proofs
        .iter()
        .map(|p| p.siblings.iter())
        .collect::<Vec<_>>();
    // Fill the `seen` map from the bottom of the tree to the cap.
    for layer_height in 0..height - cap_height {
        for (&i, p) in leaves_indices.iter().zip(siblings.iter_mut()) {
            let index = (i + num_leaves) >> layer_height;
            let current_hash = seen[&index];
            let sibling_index = index ^ 1;
            let sibling_hash = *seen
                .entry(sibling_index)
                .or_insert_with(|| *p.next().unwrap());
            let parent_hash = if index.is_even() {
                H::two_to_one(current_hash, sibling_hash)
            } else {
                H::two_to_one(sibling_hash, current_hash)
            };
            seen.insert(index >> 1, parent_hash);
        }
    }
    // For every index, go up the tree by querying `seen` to get node values.
    for &i in leaves_indices {
        let mut decompressed_proof = MerkleProof {
            siblings: Vec::new(),
        };
        let mut index = i + num_leaves;
        for _ in 0..height - cap_height {
            let sibling_index = index ^ 1;
            let h = seen[&sibling_index];
            decompressed_proof.siblings.push(h);
            index >>= 1;
        }

        decompressed_proofs.push(decompressed_proof);
    }

    decompressed_proofs
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::field::types::Sample;
    use crate::hash::merkle_tree::MerkleTree;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn test_path_compression() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let h = 10;
        let cap_height = 3;
        let vs = (0..1 << h).map(|_| vec![F::rand()]).collect::<Vec<_>>();
        let mt = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(vs.clone(), cap_height);

        let mut rng = OsRng;
        let k = rng.gen_range(1..=1 << h);
        let indices = (0..k).map(|_| rng.gen_range(0..1 << h)).collect::<Vec<_>>();
        let proofs = indices.iter().map(|&i| mt.prove(i)).collect::<Vec<_>>();

        let compressed_proofs = compress_merkle_proofs(cap_height, &indices, &proofs);
        let decompressed_proofs = decompress_merkle_proofs(
            &indices.iter().map(|&i| vs[i].clone()).collect::<Vec<_>>(),
            &indices,
            &compressed_proofs,
            h,
            cap_height,
        );

        assert_eq!(proofs, decompressed_proofs);

        #[cfg(feature = "std")]
        {
            let compressed_proof_bytes = serde_cbor::to_vec(&compressed_proofs).unwrap();
            println!(
                "Compressed proof length: {} bytes",
                compressed_proof_bytes.len()
            );
            let proof_bytes = serde_cbor::to_vec(&proofs).unwrap();
            println!("Proof length: {} bytes", proof_bytes.len());
        }
    }
}

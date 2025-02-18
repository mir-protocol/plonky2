#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use itertools::Itertools;

use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::merkle_proofs::MerkleProof;
use crate::hash::merkle_tree::{
    capacity_up_to_mut, fill_digests_buf, merkle_tree_prove, MerkleCap,
};
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchMerkleTree<F: RichField, H: Hasher<F>> {
    /// The data stored in the Merkle tree leaves.
    pub leaves: Vec<Vec<Vec<F>>>,

    /// Merkle tree node hashes, analogous to `digests` in `MerkleTree`.
    pub digests: Vec<H::Hash>,

    /// Represents the roots of the Merkle tree. This allows for using any layer as the root of the tree.
    pub cap: MerkleCap<F, H>,

    /// Represents the heights at which leaves reside within the tree.
    pub leaf_heights: Vec<usize>,
}

impl<F: RichField, H: Hasher<F>> BatchMerkleTree<F, H> {
    /// Each element in the `leaves` vector represents a matrix (a vector of vectors).
    /// The height of each matrix should be a power of two.
    /// The `leaves` vector should be sorted by matrix height, from tallest to shortest, with no duplicate heights.
    pub fn new(mut leaves: Vec<Vec<Vec<F>>>, cap_height: usize) -> Self {
        assert!(!leaves.is_empty());
        assert!(leaves.iter().all(|leaf| leaf.len().is_power_of_two()));
        assert!(leaves
            .windows(2)
            .all(|pair| { pair[0].len() > pair[1].len() }));

        let last_leaves_cap_height = log2_strict(leaves.last().unwrap().len());
        assert!(
            cap_height <= last_leaves_cap_height,
            "cap_height={} should be at most last_leaves_cap_height={}",
            cap_height,
            last_leaves_cap_height
        );

        let mut leaf_heights = Vec::with_capacity(leaves.len());

        let leaves_len = leaves[0].len();
        let num_digests = 2 * (leaves_len - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);
        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let mut digests_buf_pos = 0;

        let mut cap = vec![];
        let dummy_leaves = vec![vec![F::ZERO]; 1 << cap_height];
        leaves.push(dummy_leaves);
        for window in leaves.windows(2) {
            let cur = &window[0];
            let next = &window[1];

            let cur_leaf_len = cur.len();
            let next_cap_len = next.len();
            let next_cap_height = log2_strict(next_cap_len);

            leaf_heights.push(log2_strict(cur_leaf_len));

            let num_tmp_digests = 2 * (cur_leaf_len - next_cap_len);

            if cur_leaf_len == leaves_len {
                // The bottom leaf layer
                cap = Vec::with_capacity(next_cap_len);
                let tmp_cap_buf = capacity_up_to_mut(&mut cap, next_cap_len);
                fill_digests_buf::<F, H>(
                    &mut digests_buf[digests_buf_pos..(digests_buf_pos + num_tmp_digests)],
                    tmp_cap_buf,
                    &cur[..],
                    next_cap_height,
                );
            } else {
                // The rest leaf layers
                let new_leaves: Vec<Vec<F>> = cap
                    .iter()
                    .enumerate()
                    .map(|(i, cap_hash)| {
                        let mut new_hash = Vec::with_capacity(NUM_HASH_OUT_ELTS + cur[i].len());
                        new_hash.extend(&cap_hash.to_vec());
                        new_hash.extend(&cur[i]);
                        new_hash
                    })
                    .collect();
                cap.clear();
                cap.reserve_exact(next_cap_len);
                let tmp_cap_buf = capacity_up_to_mut(&mut cap, next_cap_len);
                fill_digests_buf::<F, H>(
                    &mut digests_buf[digests_buf_pos..(digests_buf_pos + num_tmp_digests)],
                    tmp_cap_buf,
                    &new_leaves[..],
                    next_cap_height,
                );
            }

            unsafe {
                // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
                // `num_digests` and `len_cap`, resp.
                cap.set_len(next_cap_len);
            }

            digests_buf_pos += num_tmp_digests;
        }

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
        }

        // remove dummy leaves
        leaves.pop();

        Self {
            leaves,
            digests,
            cap: MerkleCap(cap),
            leaf_heights,
        }
    }

    /// Create a Merkle proof from a leaf index.
    pub fn open_batch(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let mut digests_buf_pos = 0;
        let initial_leaf_height = log2_strict(self.leaves[0].len());
        let mut siblings = vec![];
        let mut cap_heights = self.leaf_heights.clone();
        cap_heights.push(log2_strict(self.cap.len()));
        for window in cap_heights.windows(2) {
            let cur_cap_height = window[0];
            let next_cap_height = window[1];
            let num_digests: usize = 2 * ((1 << cur_cap_height) - (1 << next_cap_height));
            siblings.extend::<Vec<_>>(merkle_tree_prove::<F, H>(
                leaf_index >> (initial_leaf_height - cur_cap_height),
                1 << cur_cap_height,
                next_cap_height,
                &self.digests[digests_buf_pos..digests_buf_pos + num_digests],
            ));
            digests_buf_pos += num_digests;
        }

        MerkleProof { siblings }
    }

    pub fn values(&self, leaf_index: usize) -> Vec<Vec<F>> {
        let leaves_cap_height = log2_strict(self.leaves[0].len());
        self.leaves
            .iter()
            .zip(&self.leaf_heights)
            .map(|(leaves, cap_height)| {
                leaves[leaf_index >> (leaves_cap_height - cap_height)].clone()
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use anyhow::Result;
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::Field;

    use super::*;
    use crate::hash::merkle_proofs::verify_batch_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type H = <C as GenericConfig<D>>::Hasher;

    #[test]
    fn commit_single() -> Result<()> {
        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        //   0 0
        // ]
        let mat_1 = vec![
            vec![F::ZERO, F::ONE],
            vec![F::TWO, F::ONE],
            vec![F::TWO, F::TWO],
            vec![F::ZERO, F::ZERO],
        ];
        let fmt: BatchMerkleTree<GoldilocksField, H> = BatchMerkleTree::new(vec![mat_1], 0);

        let mat_1_leaf_hashes = [
            H::hash_or_noop(&[F::ZERO, F::ONE]),
            H::hash_or_noop(&[F::TWO, F::ONE]),
            H::hash_or_noop(&[F::TWO, F::TWO]),
            H::hash_or_noop(&[F::ZERO, F::ZERO]),
        ];
        assert_eq!(mat_1_leaf_hashes[0..2], fmt.digests[0..2]);
        assert_eq!(mat_1_leaf_hashes[2..4], fmt.digests[4..6]);

        let layer_1 = [
            H::two_to_one(mat_1_leaf_hashes[0], mat_1_leaf_hashes[1]),
            H::two_to_one(mat_1_leaf_hashes[2], mat_1_leaf_hashes[3]),
        ];
        assert_eq!(layer_1, fmt.digests[2..4]);

        let root = H::two_to_one(layer_1[0], layer_1[1]);
        assert_eq!(fmt.cap.flatten(), root.to_vec());

        let proof = fmt.open_batch(2);
        assert_eq!(proof.siblings, [mat_1_leaf_hashes[3], layer_1[0]]);

        let opened_values = fmt.values(2);
        assert_eq!(opened_values, [vec![F::TWO, F::TWO]]);

        verify_batch_merkle_proof_to_cap(&opened_values, &fmt.leaf_heights, 2, &fmt.cap, &proof)?;
        Ok(())
    }

    #[test]
    fn commit_mixed() -> Result<()> {
        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        //   0 0
        // ]
        let mat_1 = vec![
            vec![F::ZERO, F::ONE],
            vec![F::TWO, F::ONE],
            vec![F::TWO, F::TWO],
            vec![F::ZERO, F::ZERO],
        ];

        // mat_2 = [
        //   1 2 1
        //   0 2 2
        // ]
        let mat_2 = vec![vec![F::ONE, F::TWO, F::ONE], vec![F::ZERO, F::TWO, F::TWO]];

        let fmt: BatchMerkleTree<GoldilocksField, H> =
            BatchMerkleTree::new(vec![mat_1, mat_2.clone()], 0);

        let mat_1_leaf_hashes = [
            H::hash_or_noop(&[F::ZERO, F::ONE]),
            H::hash_or_noop(&[F::TWO, F::ONE]),
            H::hash_or_noop(&[F::TWO, F::TWO]),
            H::hash_or_noop(&[F::ZERO, F::ZERO]),
        ];
        assert_eq!(mat_1_leaf_hashes, fmt.digests[0..4]);

        let hidden_layer = [
            H::two_to_one(mat_1_leaf_hashes[0], mat_1_leaf_hashes[1]).to_vec(),
            H::two_to_one(mat_1_leaf_hashes[2], mat_1_leaf_hashes[3]).to_vec(),
        ];
        let new_leaves = hidden_layer
            .iter()
            .zip(mat_2.iter())
            .map(|(row1, row2)| {
                let mut new_row = row1.clone();
                new_row.extend_from_slice(row2);
                new_row
            })
            .collect::<Vec<Vec<F>>>();
        let layer_1 = [
            H::hash_or_noop(&new_leaves[0]),
            H::hash_or_noop(&new_leaves[1]),
        ];
        assert_eq!(layer_1, fmt.digests[4..]);

        let root = H::two_to_one(layer_1[0], layer_1[1]);
        assert_eq!(fmt.cap.flatten(), root.to_vec());

        let proof = fmt.open_batch(1);
        assert_eq!(proof.siblings, [mat_1_leaf_hashes[0], layer_1[1]]);

        let opened_values = fmt.values(1);
        assert_eq!(
            opened_values,
            [vec![F::TWO, F::ONE], vec![F::ONE, F::TWO, F::ONE]]
        );

        verify_batch_merkle_proof_to_cap(&opened_values, &fmt.leaf_heights, 1, &fmt.cap, &proof)?;
        Ok(())
    }

    #[test]
    #[cfg(not(feature = "no_random"))]
    fn test_batch_merkle_trees() -> Result<()> {
        let leaves_1 = crate::hash::merkle_tree::tests::random_data::<F>(1024, 7);
        let leaves_2 = crate::hash::merkle_tree::tests::random_data::<F>(64, 3);
        let leaves_3 = crate::hash::merkle_tree::tests::random_data::<F>(32, 100);

        let fmt: BatchMerkleTree<GoldilocksField, H> =
            BatchMerkleTree::new(vec![leaves_1, leaves_2, leaves_3], 3);
        for index in [0, 1023, 512, 255] {
            let proof = fmt.open_batch(index);
            let opened_values = fmt.values(index);
            verify_batch_merkle_proof_to_cap(
                &opened_values,
                &fmt.leaf_heights,
                index,
                &fmt.cap,
                &proof,
            )?;
        }

        Ok(())
    }

    #[test]
    #[cfg(not(feature = "no_random"))]
    fn test_batch_merkle_trees_cap_at_leaves_height() -> Result<()> {
        let leaves_1 = crate::hash::merkle_tree::tests::random_data::<F>(16, 7);

        let fmt: BatchMerkleTree<GoldilocksField, H> = BatchMerkleTree::new(vec![leaves_1], 4);
        for index in 0..16 {
            let proof = fmt.open_batch(index);
            let opened_values = fmt.values(index);
            verify_batch_merkle_proof_to_cap(
                &opened_values,
                &fmt.leaf_heights,
                index,
                &fmt.cap,
                &proof,
            )?;
        }

        Ok(())
    }
}

mod allocator;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, BatchSize};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::keccak::KeccakHash;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::Hasher;
use tynm::type_name;
use poseidon2_plonky2::poseidon2_hash::Poseidon2Hash;

const ELEMS_PER_LEAF: usize = 135;

pub(crate) fn bench_merkle_tree<F: RichField, H: Hasher<F>>(c: &mut Criterion) {
    let mut group = c.benchmark_group(&format!(
        "merkle-tree<{}, {}>",
        type_name::<F>(),
        type_name::<H>()
    ));
    group.sample_size(30);

    for size_log in [13, 14, 15] {
        let size = 1 << size_log;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter_batched(
                || vec![F::rand_vec(ELEMS_PER_LEAF); size],
                |leaves| MerkleTree::<F, H>::new(leaves, 0),
                BatchSize::SmallInput,
            );
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_merkle_tree::<GoldilocksField, PoseidonHash>(c);
    bench_merkle_tree::<GoldilocksField, Poseidon2Hash>(c);
    bench_merkle_tree::<GoldilocksField, KeccakHash<25>>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
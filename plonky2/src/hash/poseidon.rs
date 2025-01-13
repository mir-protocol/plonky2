//! Implementation of the Poseidon hash function, as described in
//! <https://eprint.iacr.org/2019/458.pdf>

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use plonky2_field::packed::PackedField;
use unroll::unroll_for_loops;

#[cfg(target_feature = "avx2")]
use super::arch::x86_64::poseidon_goldilocks_avx2::poseidon_avx;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::{Field, PrimeField64};
use crate::gates::gate::Gate;
use crate::gates::poseidon::PoseidonGate;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

// The number of full rounds and partial rounds is given by the
// calc_round_numbers.py script. They happen to be the same for both
// width 8 and width 12 with s-box x^7.
//
// NB: Changing any of these values will require regenerating all of
// the precomputed constant arrays in this file.
pub const HALF_N_FULL_ROUNDS: usize = 4;
pub(crate) const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const N_ROUNDS: usize = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;
const MAX_WIDTH: usize = 12; // we only have width 8 and 12, and 12 is bigger. :)

#[inline(always)]
pub(crate) const fn add_u160_u128((x_lo, x_hi): (u128, u32), y: u128) -> (u128, u32) {
    let (res_lo, over) = x_lo.overflowing_add(y);
    let res_hi = x_hi + (over as u32);
    (res_lo, res_hi)
}

#[inline(always)]
pub(crate) fn reduce_u160<F: PrimeField64>((n_lo, n_hi): (u128, u32)) -> F {
    let n_lo_hi = (n_lo >> 64) as u64;
    let n_lo_lo = n_lo as u64;
    let reduced_hi: u64 = F::from_noncanonical_u96((n_lo_hi, n_hi)).to_noncanonical_u64();
    let reduced128: u128 = ((reduced_hi as u128) << 64) + (n_lo_lo as u128);
    F::from_noncanonical_u128(reduced128)
}

/// Note that these work for the Goldilocks field, but not necessarily others. See
/// `generate_constants` about how these were generated. We include enough for a width of 12;
/// smaller widths just use a subset.
#[rustfmt::skip]
pub const ALL_ROUND_CONSTANTS: [u64; MAX_WIDTH * N_ROUNDS]  = [
    // WARNING: The AVX2 Goldilocks specialization relies on all round constants being in
    // 0..0xfffeeac900011537. If these constants are randomly regenerated, there is a ~.6% chance
    // that this condition will no longer hold.
    //
    // WARNING: If these are changed in any way, then all the
    // implementations of Poseidon must be regenerated. See comments
    // in `poseidon_goldilocks.rs`.
    0xb585f766f2144405, 0x7746a55f43921ad7, 0xb2fb0d31cee799b4, 0x0f6760a4803427d7,
    0xe10d666650f4e012, 0x8cae14cb07d09bf1, 0xd438539c95f63e9f, 0xef781c7ce35b4c3d,
    0xcdc4a239b0c44426, 0x277fa208bf337bff, 0xe17653a29da578a1, 0xc54302f225db2c76,
    0x86287821f722c881, 0x59cd1a8a41c18e55, 0xc3b919ad495dc574, 0xa484c4c5ef6a0781,
    0x308bbd23dc5416cc, 0x6e4a40c18f30c09c, 0x9a2eedb70d8f8cfa, 0xe360c6e0ae486f38,
    0xd5c7718fbfc647fb, 0xc35eae071903ff0b, 0x849c2656969c4be7, 0xc0572c8c08cbbbad,
    0xe9fa634a21de0082, 0xf56f6d48959a600d, 0xf7d713e806391165, 0x8297132b32825daf,
    0xad6805e0e30b2c8a, 0xac51d9f5fcf8535e, 0x502ad7dc18c2ad87, 0x57a1550c110b3041,
    0x66bbd30e6ce0e583, 0x0da2abef589d644e, 0xf061274fdb150d61, 0x28b8ec3ae9c29633,
    0x92a756e67e2b9413, 0x70e741ebfee96586, 0x019d5ee2af82ec1c, 0x6f6f2ed772466352,
    0x7cf416cfe7e14ca1, 0x61df517b86a46439, 0x85dc499b11d77b75, 0x4b959b48b9c10733,
    0xe8be3e5da8043e57, 0xf5c0bc1de6da8699, 0x40b12cbf09ef74bf, 0xa637093ecb2ad631,
    0x3cc3f892184df408, 0x2e479dc157bf31bb, 0x6f49de07a6234346, 0x213ce7bede378d7b,
    0x5b0431345d4dea83, 0xa2de45780344d6a1, 0x7103aaf94a7bf308, 0x5326fc0d97279301,
    0xa9ceb74fec024747, 0x27f8ec88bb21b1a3, 0xfceb4fda1ded0893, 0xfac6ff1346a41675,
    0x7131aa45268d7d8c, 0x9351036095630f9f, 0xad535b24afc26bfb, 0x4627f5c6993e44be,
    0x645cf794b8f1cc58, 0x241c70ed0af61617, 0xacb8e076647905f1, 0x3737e9db4c4f474d,
    0xe7ea5e33e75fffb6, 0x90dee49fc9bfc23a, 0xd1b1edf76bc09c92, 0x0b65481ba645c602,
    0x99ad1aab0814283b, 0x438a7c91d416ca4d, 0xb60de3bcc5ea751c, 0xc99cab6aef6f58bc,
    0x69a5ed92a72ee4ff, 0x5e7b329c1ed4ad71, 0x5fc0ac0800144885, 0x32db829239774eca,
    0x0ade699c5830f310, 0x7cc5583b10415f21, 0x85df9ed2e166d64f, 0x6604df4fee32bcb1,
    0xeb84f608da56ef48, 0xda608834c40e603d, 0x8f97fe408061f183, 0xa93f485c96f37b89,
    0x6704e8ee8f18d563, 0xcee3e9ac1e072119, 0x510d0e65e2b470c1, 0xf6323f486b9038f0,
    0x0b508cdeffa5ceef, 0xf2417089e4fb3cbd, 0x60e75c2890d15730, 0xa6217d8bf660f29c,
    0x7159cd30c3ac118e, 0x839b4e8fafead540, 0x0d3f3e5e82920adc, 0x8f7d83bddee7bba8,
    0x780f2243ea071d06, 0xeb915845f3de1634, 0xd19e120d26b6f386, 0x016ee53a7e5fecc6,
    0xcb5fd54e7933e477, 0xacb8417879fd449f, 0x9c22190be7f74732, 0x5d693c1ba3ba3621,
    0xdcef0797c2b69ec7, 0x3d639263da827b13, 0xe273fd971bc8d0e7, 0x418f02702d227ed5,
    0x8c25fda3b503038c, 0x2cbaed4daec8c07c, 0x5f58e6afcdd6ddc2, 0x284650ac5e1b0eba,
    0x635b337ee819dab5, 0x9f9a036ed4f2d49f, 0xb93e260cae5c170e, 0xb0a7eae879ddb76d,
    0xd0762cbc8ca6570c, 0x34c6efb812b04bf5, 0x40bf0ab5fa14c112, 0xb6b570fc7c5740d3,
    0x5a27b9002de33454, 0xb1a5b165b6d2b2d2, 0x8722e0ace9d1be22, 0x788ee3b37e5680fb,
    0x14a726661551e284, 0x98b7672f9ef3b419, 0xbb93ae776bb30e3a, 0x28fd3b046380f850,
    0x30a4680593258387, 0x337dc00c61bd9ce1, 0xd5eca244c7a4ff1d, 0x7762638264d279bd,
    0xc1e434bedeefd767, 0x0299351a53b8ec22, 0xb2d456e4ad251b80, 0x3e9ed1fda49cea0b,
    0x2972a92ba450bed8, 0x20216dd77be493de, 0xadffe8cf28449ec6, 0x1c4dbb1c4c27d243,
    0x15a16a8a8322d458, 0x388a128b7fd9a609, 0x2300e5d6baedf0fb, 0x2f63aa8647e15104,
    0xf1c36ce86ecec269, 0x27181125183970c9, 0xe584029370dca96d, 0x4d9bbc3e02f1cfb2,
    0xea35bc29692af6f8, 0x18e21b4beabb4137, 0x1e3b9fc625b554f4, 0x25d64362697828fd,
    0x5a3f1bb1c53a9645, 0xdb7f023869fb8d38, 0xb462065911d4e1fc, 0x49c24ae4437d8030,
    0xd793862c112b0566, 0xaadd1106730d8feb, 0xc43b6e0e97b0d568, 0xe29024c18ee6fca2,
    0x5e50c27535b88c66, 0x10383f20a4ff9a87, 0x38e8ee9d71a45af8, 0xdd5118375bf1a9b9,
    0x775005982d74d7f7, 0x86ab99b4dde6c8b0, 0xb1204f603f51c080, 0xef61ac8470250ecf,
    0x1bbcd90f132c603f, 0x0cd1dabd964db557, 0x11a3ae5beb9d1ec9, 0xf755bfeea585d11d,
    0xa3b83250268ea4d7, 0x516306f4927c93af, 0xddb4ac49c9efa1da, 0x64bb6dec369d4418,
    0xf9cc95c22b4c1fcc, 0x08d37f755f4ae9f6, 0xeec49b613478675b, 0xf143933aed25e0b0,
    0xe4c5dd8255dfc622, 0xe7ad7756f193198e, 0x92c2318b87fff9cb, 0x739c25f8fd73596d,
    0x5636cac9f16dfed0, 0xdd8f909a938e0172, 0xc6401fe115063f5b, 0x8ad97b33f1ac1455,
    0x0c49366bb25e8513, 0x0784d3d2f1698309, 0x530fb67ea1809a81, 0x410492299bb01f49,
    0x139542347424b9ac, 0x9cb0bd5ea1a1115e, 0x02e3f615c38f49a1, 0x985d4f4a9c5291ef,
    0x775b9feafdcd26e7, 0x304265a6384f0f2d, 0x593664c39773012c, 0x4f0a2e5fb028f2ce,
    0xdd611f1000c17442, 0xd8185f9adfea4fd0, 0xef87139ca9a3ab1e, 0x3ba71336c34ee133,
    0x7d3a455d56b70238, 0x660d32e130182684, 0x297a863f48cd1f43, 0x90e0a736a751ebb7,
    0x549f80ce550c4fd3, 0x0f73b2922f38bd64, 0x16bf1f73fb7a9c3f, 0x6d1f5a59005bec17,
    0x02ff876fa5ef97c4, 0xc5cb72a2a51159b0, 0x8470f39d2d5c900e, 0x25abb3f1d39fcb76,
    0x23eb8cc9b372442f, 0xd687ba55c64f6364, 0xda8d9e90fd8ff158, 0xe3cbdc7d2fe45ea7,
    0xb9a8c9b3aee52297, 0xc0d28a5c10960bd3, 0x45d7ac9b68f71a34, 0xeeb76e397069e804,
    0x3d06c8bd1514e2d9, 0x9c9c98207cb10767, 0x65700b51aedfb5ef, 0x911f451539869408,
    0x7ae6849fbc3a0ec6, 0x3bb340eba06afe7e, 0xb46e9d8b682ea65e, 0x8dcf22f9a3b34356,
    0x77bdaeda586257a7, 0xf19e400a5104d20d, 0xc368a348e46d950f, 0x9ef1cd60e679f284,
    0xe89cd854d5d01d33, 0x5cd377dc8bb882a2, 0xa7b0fb7883eee860, 0x7684403ec392950d,
    0x5fa3f06f4fed3b52, 0x8df57ac11bc04831, 0x2db01efa1e1e1897, 0x54846de4aadb9ca2,
    0xba6745385893c784, 0x541d496344d2c75b, 0xe909678474e687fe, 0xdfe89923f6c9c2ff,
    0xece5a71e0cfedc75, 0x5ff98fd5d51fe610, 0x83e8941918964615, 0x5922040b47f150c1,
    0xf97d750e3dd94521, 0x5080d4c2b86f56d7, 0xa7de115b56c78d70, 0x6a9242ac87538194,
    0xf7856ef7f9173e44, 0x2265fc92feb0dc09, 0x17dfc8e4f7ba8a57, 0x9001a64209f21db8,
    0x90004c1371b893c5, 0xb932b7cf752e5545, 0xa0b1df81b6fe59fc, 0x8ef1dd26770af2c2,
    0x0541a4f9cfbeed35, 0x9e61106178bfc530, 0xb3767e80935d8af2, 0x0098d5782065af06,
    0x31d191cd5c1466c7, 0x410fefafa319ac9d, 0xbdf8f242e316c4ab, 0x9e8cd55b57637ed0,
    0xde122bebe9a39368, 0x4d001fd58f002526, 0xca6637000eb4a9f8, 0x2f2339d624f91f78,
    0x6d1a7918c80df518, 0xdf9a4939342308e9, 0xebc2151ee6c8398c, 0x03cc2ba8a1116515,
    0xd341d037e840cf83, 0x387cb5d25af4afcc, 0xbba2515f22909e87, 0x7248fe7705f38e47,
    0x4d61e56a525d225a, 0x262e963c8da05d3d, 0x59e89b094d220ec2, 0x055d5b52b78b9c5e,
    0x82b27eb33514ef99, 0xd30094ca96b7ce7b, 0xcf5cb381cd0a1535, 0xfeed4db6919e5a7c,
    0x41703f53753be59f, 0x5eeea940fcde8b6f, 0x4cd1f1b175100206, 0x4a20358574454ec0,
    0x1478d361dbbf9fac, 0x6f02dc07d141875c, 0x296a202ed8e556a2, 0x2afd67999bf32ee5,
    0x7acfd96efa95491d, 0x6798ba0c0abb2c6d, 0x34c6f57b26c92122, 0x5736e1bad206b5de,
    0x20057d2a0056521b, 0x3dea5bd5d0578bd7, 0x16e50d897d4634ac, 0x29bff3ecb9b7a6e3,
    0x475cd3205a3bdcde, 0x18a42105c31b7e88, 0x023e7414af663068, 0x15147108121967d7,
    0xe4a3dff1d7d6fef9, 0x01a8d1a588085737, 0x11b4c74eda62beef, 0xe587cc0d69a73346,
    0x1ff7327017aa2a6e, 0x594e29c42473d06b, 0xf6f31db1899b12d5, 0xc02ac5e47312d3ca,
    0xe70201e960cb78b8, 0x6f90ff3b6a65f108, 0x42747a7245e7fa84, 0xd1f507e43ab749b2,
    0x1c86d265f15750cd, 0x3996ce73dd832c1c, 0x8e7fba02983224bd, 0xba0dec7103255dd4,
    0x9e9cbd781628fc5b, 0xdae8645996edd6a5, 0xdebe0853b1a1d378, 0xa49229d24d014343,
    0x7be5b9ffda905e1c, 0xa3c95eaec244aa30, 0x0230bca8f4df0544, 0x4135c2bebfe148c6,
    0x166fc0cc438a3c72, 0x3762b59a8ae83efa, 0xe8928a4c89114750, 0x2a440b51a4945ee5,
    0x80cefd2b7d99ff83, 0xbb9879c6e61fd62a, 0x6e7c8f1a84265034, 0x164bb2de1bbeddc8,
    0xf3c12fe54d5c653b, 0x40b9e922ed9771e2, 0x551f5b0fbe7b1840, 0x25032aa7c4cb1811,
    0xaaed34074b164346, 0x8ffd96bbf9c9c81d, 0x70fc91eb5937085c, 0x7f795e2a5f915440,
    0x4543d9df5476d3cb, 0xf172d73e004fc90d, 0xdfd1c4febcc81238, 0xbc8dfb627fe558fc,
];

pub trait Poseidon: PrimeField64 {
    // Total number of round constants required: width of the input
    // times number of rounds.
    const N_ROUND_CONSTANTS: usize = SPONGE_WIDTH * N_ROUNDS;

    // The MDS matrix we use is C + D, where C is the circulant matrix whose first
    // row is given by `MDS_MATRIX_CIRC`, and D is the diagonal matrix whose
    // diagonal is given by `MDS_MATRIX_DIAG`.
    const MDS_MATRIX_CIRC: [u64; SPONGE_WIDTH];
    const MDS_MATRIX_DIAG: [u64; SPONGE_WIDTH];

    // Precomputed constants for the fast Poseidon calculation. See
    // the paper.
    const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [u64; SPONGE_WIDTH];
    const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS];
    const FAST_PARTIAL_ROUND_VS: [[u64; SPONGE_WIDTH - 1]; N_PARTIAL_ROUNDS];
    const FAST_PARTIAL_ROUND_W_HATS: [[u64; SPONGE_WIDTH - 1]; N_PARTIAL_ROUNDS];
    const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[u64; SPONGE_WIDTH - 1]; SPONGE_WIDTH - 1];

    #[inline(always)]
    #[unroll_for_loops]
    fn mds_row_shf(r: usize, v: &[u64; SPONGE_WIDTH]) -> u128 {
        debug_assert!(r < SPONGE_WIDTH);
        // The values of `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are
        // known to be small, so we can accumulate all the products for
        // each row and reduce just once at the end (done by the
        // caller).

        // NB: Unrolling this, calculating each term independently, and
        // summing at the end, didn't improve performance for me.
        let mut res = 0u128;

        // This is a hacky way of fully unrolling the loop.
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                res += (v[(i + r) % SPONGE_WIDTH] as u128) * (Self::MDS_MATRIX_CIRC[i] as u128);
            }
        }
        res += (v[r] as u128) * (Self::MDS_MATRIX_DIAG[r] as u128);

        res
    }

    /// Same as `mds_row_shf` for field extensions of `Self`.
    fn mds_row_shf_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        r: usize,
        v: &[F; SPONGE_WIDTH],
    ) -> F {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = F::ZERO;

        for i in 0..SPONGE_WIDTH {
            res += v[(i + r) % SPONGE_WIDTH] * F::from_canonical_u64(Self::MDS_MATRIX_CIRC[i]);
        }
        res += v[r] * F::from_canonical_u64(Self::MDS_MATRIX_DIAG[r]);

        res
    }

    /// Same as `mds_row_shf` for `PackedField`.
    fn mds_row_shf_packed_field<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
    >(
        r: usize,
        v: &[P; SPONGE_WIDTH],
    ) -> P
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = P::ZEROS;

        for i in 0..SPONGE_WIDTH {
            res +=
                v[(i + r) % SPONGE_WIDTH] * P::Scalar::from_canonical_u64(Self::MDS_MATRIX_CIRC[i]);
        }
        res += v[r] * P::Scalar::from_canonical_u64(Self::MDS_MATRIX_DIAG[r]);

        res
    }

    /// Recursive version of `mds_row_shf`.
    fn mds_row_shf_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        r: usize,
        v: &[ExtensionTarget<D>; SPONGE_WIDTH],
    ) -> ExtensionTarget<D>
    where
        Self: RichField + Extendable<D>,
    {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = builder.zero_extension();

        for i in 0..SPONGE_WIDTH {
            let c = Self::from_canonical_u64(<Self as Poseidon>::MDS_MATRIX_CIRC[i]);
            res = builder.mul_const_add_extension(c, v[(i + r) % SPONGE_WIDTH], res);
        }
        {
            let c = Self::from_canonical_u64(<Self as Poseidon>::MDS_MATRIX_DIAG[r]);
            res = builder.mul_const_add_extension(c, v[r], res);
        }

        res
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn mds_layer(state_: &[Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut result = [Self::ZERO; SPONGE_WIDTH];

        let mut state = [0u64; SPONGE_WIDTH];
        for r in 0..SPONGE_WIDTH {
            state[r] = state_[r].to_noncanonical_u64();
        }

        // This is a hacky way of fully unrolling the loop.
        for r in 0..12 {
            if r < SPONGE_WIDTH {
                let sum = Self::mds_row_shf(r, &state);
                let sum_lo = sum as u64;
                let sum_hi = (sum >> 64) as u32;
                result[r] = Self::from_noncanonical_u96((sum_lo, sum_hi));
            }
        }

        result
    }

    /// Same as `mds_layer` for field extensions of `Self`.
    fn mds_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; SPONGE_WIDTH],
    ) -> [F; SPONGE_WIDTH] {
        let mut result = [F::ZERO; SPONGE_WIDTH];

        for r in 0..SPONGE_WIDTH {
            result[r] = Self::mds_row_shf_field(r, state);
        }

        result
    }

    /// Same as `mds_layer` for `PackedField`.
    fn mds_layer_packed_field<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
    >(
        state: &[P; SPONGE_WIDTH],
    ) -> [P; SPONGE_WIDTH]
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut result = [P::ZEROS; SPONGE_WIDTH];

        for r in 0..SPONGE_WIDTH {
            result[r] = Self::mds_row_shf_packed_field(r, state);
        }

        result
    }

    /// Recursive version of `mds_layer`.
    fn mds_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &[ExtensionTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionTarget<D>; SPONGE_WIDTH]
    where
        Self: RichField + Extendable<D>,
    {
        // If we have enough routed wires, we will use PoseidonMdsGate.
        let mds_gate = PoseidonMdsGate::<Self, D>::new();
        if builder.config.num_routed_wires >= mds_gate.num_wires() {
            let index = builder.add_gate(mds_gate, vec![]);
            for i in 0..SPONGE_WIDTH {
                let input_wire = PoseidonMdsGate::<Self, D>::wires_input(i);
                builder.connect_extension(state[i], ExtensionTarget::from_range(index, input_wire));
            }
            (0..SPONGE_WIDTH)
                .map(|i| {
                    let output_wire = PoseidonMdsGate::<Self, D>::wires_output(i);
                    ExtensionTarget::from_range(index, output_wire)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        } else {
            let mut result = [builder.zero_extension(); SPONGE_WIDTH];

            for r in 0..SPONGE_WIDTH {
                result[r] = Self::mds_row_shf_circuit(builder, r, state);
            }

            result
        }
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn partial_first_constant_layer<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; SPONGE_WIDTH],
    ) {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(Self::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
            }
        }
    }

    /// Same as `partial_first_constant_layer` for `PackedField`.
    #[inline(always)]
    #[unroll_for_loops]
    fn partial_first_constant_layer_packed_field<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
    >(
        state: &mut [P; SPONGE_WIDTH],
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                state[i] +=
                    P::Scalar::from_canonical_u64(Self::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
            }
        }
    }

    /// Recursive version of `partial_first_constant_layer`.
    fn partial_first_constant_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..SPONGE_WIDTH {
            let c = <Self as Poseidon>::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i];
            let c = Self::Extension::from_canonical_u64(c);
            let c = builder.constant_extension(c);
            state[i] = builder.add_extension(state[i], c);
        }
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn mds_partial_layer_init<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; SPONGE_WIDTH],
    ) -> [F; SPONGE_WIDTH] {
        let mut result = [F::ZERO; SPONGE_WIDTH];

        // Initial matrix has first row/column = [1, 0, ..., 0];

        // c = 0
        result[0] = state[0];

        for r in 1..12 {
            if r < SPONGE_WIDTH {
                for c in 1..12 {
                    if c < SPONGE_WIDTH {
                        // NB: FAST_PARTIAL_ROUND_INITIAL_MATRIX is stored in
                        // row-major order so that this dot product is cache
                        // friendly.
                        let t = F::from_canonical_u64(
                            Self::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1],
                        );
                        result[c] += state[r] * t;
                    }
                }
            }
        }
        result
    }

    /// Same as `mds_partial_layer_init` for `PackedField`.
    #[inline(always)]
    #[unroll_for_loops]
    fn mds_partial_layer_init_packed_field<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
    >(
        state: &[P; SPONGE_WIDTH],
    ) -> [P; SPONGE_WIDTH]
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut result = [P::ZEROS; SPONGE_WIDTH];

        // Initial matrix has first row/column = [1, 0, ..., 0];

        // c = 0
        result[0] = state[0];

        for r in 1..12 {
            if r < SPONGE_WIDTH {
                for c in 1..12 {
                    if c < SPONGE_WIDTH {
                        // NB: FAST_PARTIAL_ROUND_INITIAL_MATRIX is stored in
                        // row-major order so that this dot product is cache
                        // friendly.
                        let t = P::Scalar::from_canonical_u64(
                            Self::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1],
                        );
                        result[c] += state[r] * t;
                    }
                }
            }
        }
        result
    }
    /// Recursive version of `mds_partial_layer_init`.
    fn mds_partial_layer_init_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &[ExtensionTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionTarget<D>; SPONGE_WIDTH]
    where
        Self: RichField + Extendable<D>,
    {
        let mut result = [builder.zero_extension(); SPONGE_WIDTH];

        result[0] = state[0];

        for r in 1..SPONGE_WIDTH {
            for c in 1..SPONGE_WIDTH {
                let t = <Self as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1];
                let t = Self::Extension::from_canonical_u64(t);
                let t = builder.constant_extension(t);
                result[c] = builder.mul_add_extension(t, state[r], result[c]);
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
    #[inline(always)]
    #[unroll_for_loops]
    fn mds_partial_layer_fast(state: &[Self; SPONGE_WIDTH], r: usize) -> [Self; SPONGE_WIDTH] {
        // Set d = [M_00 | w^] dot [state]

        let mut d_sum = (0u128, 0u32); // u160 accumulator
        for i in 1..12 {
            if i < SPONGE_WIDTH {
                let t = Self::FAST_PARTIAL_ROUND_W_HATS[r][i - 1] as u128;
                let si = state[i].to_noncanonical_u64() as u128;
                d_sum = add_u160_u128(d_sum, si * t);
            }
        }
        let s0 = state[0].to_noncanonical_u64() as u128;
        let mds0to0 = (Self::MDS_MATRIX_CIRC[0] + Self::MDS_MATRIX_DIAG[0]) as u128;
        d_sum = add_u160_u128(d_sum, s0 * mds0to0);
        let d = reduce_u160::<Self>(d_sum);

        // result = [d] concat [state[0] * v + state[shift up by 1]]
        let mut result = [Self::ZERO; SPONGE_WIDTH];
        result[0] = d;
        for i in 1..12 {
            if i < SPONGE_WIDTH {
                let t = Self::from_canonical_u64(Self::FAST_PARTIAL_ROUND_VS[r][i - 1]);
                result[i] = state[i].multiply_accumulate(state[0], t);
            }
        }
        result
    }

    /// Same as `mds_partial_layer_fast` for field extensions of `Self`.
    fn mds_partial_layer_fast_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; SPONGE_WIDTH],
        r: usize,
    ) -> [F; SPONGE_WIDTH] {
        let s0 = state[0];
        let mds0to0 = Self::MDS_MATRIX_CIRC[0] + Self::MDS_MATRIX_DIAG[0];
        let mut d = s0 * F::from_canonical_u64(mds0to0);
        for i in 1..SPONGE_WIDTH {
            let t = F::from_canonical_u64(Self::FAST_PARTIAL_ROUND_W_HATS[r][i - 1]);
            d += state[i] * t;
        }

        // result = [d] concat [state[0] * v + state[shift up by 1]]
        let mut result = [F::ZERO; SPONGE_WIDTH];
        result[0] = d;
        for i in 1..SPONGE_WIDTH {
            let t = F::from_canonical_u64(Self::FAST_PARTIAL_ROUND_VS[r][i - 1]);
            result[i] = state[0] * t + state[i];
        }
        result
    }

    /// Same as `mds_partial_layer_fast` for `PackedField.
    fn mds_partial_layer_fast_packed_field<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
    >(
        state: &[P; SPONGE_WIDTH],
        r: usize,
    ) -> [P; SPONGE_WIDTH]
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let s0 = state[0];
        let mds0to0 = Self::MDS_MATRIX_CIRC[0] + Self::MDS_MATRIX_DIAG[0];
        let mut d = s0 * P::Scalar::from_canonical_u64(mds0to0);
        for i in 1..SPONGE_WIDTH {
            let t = P::Scalar::from_canonical_u64(Self::FAST_PARTIAL_ROUND_W_HATS[r][i - 1]);
            d += state[i] * t;
        }

        // result = [d] concat [state[0] * v + state[shift up by 1]]
        let mut result = [P::ZEROS; SPONGE_WIDTH];
        result[0] = d;
        for i in 1..SPONGE_WIDTH {
            let t = P::Scalar::from_canonical_u64(Self::FAST_PARTIAL_ROUND_VS[r][i - 1]);
            result[i] = state[0] * t + state[i];
        }
        result
    }

    /// Recursive version of `mds_partial_layer_fast`.
    fn mds_partial_layer_fast_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &[ExtensionTarget<D>; SPONGE_WIDTH],
        r: usize,
    ) -> [ExtensionTarget<D>; SPONGE_WIDTH]
    where
        Self: RichField + Extendable<D>,
    {
        let s0 = state[0];
        let mds0to0 = Self::MDS_MATRIX_CIRC[0] + Self::MDS_MATRIX_DIAG[0];
        let mut d = builder.mul_const_extension(Self::from_canonical_u64(mds0to0), s0);
        for i in 1..SPONGE_WIDTH {
            let t = <Self as Poseidon>::FAST_PARTIAL_ROUND_W_HATS[r][i - 1];
            let t = Self::Extension::from_canonical_u64(t);
            let t = builder.constant_extension(t);
            d = builder.mul_add_extension(t, state[i], d);
        }

        let mut result = [builder.zero_extension(); SPONGE_WIDTH];
        result[0] = d;
        for i in 1..SPONGE_WIDTH {
            let t = <Self as Poseidon>::FAST_PARTIAL_ROUND_VS[r][i - 1];
            let t = Self::Extension::from_canonical_u64(t);
            let t = builder.constant_extension(t);
            result[i] = builder.mul_add_extension(t, state[0], state[i]);
        }
        result
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn constant_layer(state: &mut [Self; SPONGE_WIDTH], round_ctr: usize) {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                let round_constant = ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr];
                unsafe {
                    state[i] = state[i].add_canonical_u64(round_constant);
                }
            }
        }
    }

    /// Same as `constant_layer` for field extensions of `Self`.
    fn constant_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; SPONGE_WIDTH],
        round_ctr: usize,
    ) {
        for i in 0..SPONGE_WIDTH {
            state[i] += F::from_canonical_u64(ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr]);
        }
    }

    /// Same as `constant_layer` for PackedFields.
    fn constant_layer_packed_field<
        F: RichField + Extendable<D>,
        const D: usize,
        FE,
        P,
        const D2: usize,
    >(
        state: &mut [P; SPONGE_WIDTH],
        round_ctr: usize,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        for i in 0..SPONGE_WIDTH {
            state[i] +=
                P::Scalar::from_canonical_u64(ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr]);
        }
    }

    /// Recursive version of `constant_layer`.
    fn constant_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
        round_ctr: usize,
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..SPONGE_WIDTH {
            let c = ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr];
            let c = Self::Extension::from_canonical_u64(c);
            let c = builder.constant_extension(c);
            state[i] = builder.add_extension(state[i], c);
        }
    }

    #[inline(always)]
    fn sbox_monomial<F: FieldExtension<D, BaseField = Self>, const D: usize>(x: F) -> F {
        // x |--> x^7
        let x2 = x.square();
        let x4 = x2.square();
        let x3 = x * x2;
        x3 * x4
    }

    /// Recursive version of `sbox_monomial`.
    fn sbox_monomial_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        x: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        Self: RichField + Extendable<D>,
    {
        // x |--> x^7
        builder.exp_u64_extension(x, 7)
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn sbox_layer(state: &mut [Self; SPONGE_WIDTH]) {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                state[i] = Self::sbox_monomial(state[i]);
            }
        }
    }

    /// Same as `sbox_layer` for field extensions of `Self`.
    fn sbox_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; SPONGE_WIDTH],
    ) {
        for i in 0..SPONGE_WIDTH {
            state[i] = Self::sbox_monomial(state[i]);
        }
    }

    /// Recursive version of `sbox_layer`.
    fn sbox_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..SPONGE_WIDTH {
            state[i] = <Self as Poseidon>::sbox_monomial_circuit(builder, state[i]);
        }
    }

    #[inline]
    fn full_rounds(state: &mut [Self; SPONGE_WIDTH], round_ctr: &mut usize) {
        for _ in 0..HALF_N_FULL_ROUNDS {
            Self::constant_layer(state, *round_ctr);
            Self::sbox_layer(state);
            *state = Self::mds_layer(state);
            *round_ctr += 1;
        }
    }

    #[inline]
    fn partial_rounds(state: &mut [Self; SPONGE_WIDTH], round_ctr: &mut usize) {
        Self::partial_first_constant_layer(state);
        *state = Self::mds_partial_layer_init(state);

        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = Self::sbox_monomial(state[0]);
            unsafe {
                state[0] = state[0].add_canonical_u64(Self::FAST_PARTIAL_ROUND_CONSTANTS[i]);
            }
            *state = Self::mds_partial_layer_fast(state, i);
        }
        *round_ctr += N_PARTIAL_ROUNDS;
    }

    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    fn poseidon(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut state = input;
        let mut round_ctr = 0;

        Self::full_rounds(&mut state, &mut round_ctr);
        Self::partial_rounds(&mut state, &mut round_ctr);
        Self::full_rounds(&mut state, &mut round_ctr);
        debug_assert_eq!(round_ctr, N_ROUNDS);

        state
    }

    #[cfg(target_feature = "avx2")]
    #[inline]
    fn poseidon(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        poseidon_avx(&input)
    }

    // For testing only, to ensure that various tricks are correct.
    #[inline]
    fn partial_rounds_naive(state: &mut [Self; SPONGE_WIDTH], round_ctr: &mut usize) {
        for _ in 0..N_PARTIAL_ROUNDS {
            Self::constant_layer(state, *round_ctr);
            state[0] = Self::sbox_monomial(state[0]);
            *state = Self::mds_layer(state);
            *round_ctr += 1;
        }
    }

    #[inline]
    fn poseidon_naive(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut state = input;
        let mut round_ctr = 0;

        Self::full_rounds(&mut state, &mut round_ctr);
        Self::partial_rounds_naive(&mut state, &mut round_ctr);
        Self::full_rounds(&mut state, &mut round_ctr);
        debug_assert_eq!(round_ctr, N_ROUNDS);

        state
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct PoseidonPermutation<T> {
    state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for PoseidonPermutation<T> {}

impl<T> AsRef<[T]> for PoseidonPermutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

trait Permuter: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

impl<F: Poseidon> Permuter for F {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        <F as Poseidon>::poseidon(input)
    }
}

impl Permuter for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}

impl<T: Copy + Debug + Default + Eq + Permuter + Send + Sync> PlonkyPermutation<T>
    for PoseidonPermutation<T>
{
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

/// Poseidon hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PoseidonHash;
impl<F: RichField> Hasher<F> for PoseidonHash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = PoseidonPermutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }
}

impl<F: RichField> AlgebraicHasher<F> for PoseidonHash {
    type AlgebraicPermutation = PoseidonPermutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        let gate_type = PoseidonGate::<F, D>::new();
        let gate = builder.add_gate(gate_type, vec![]);

        let swap_wire = PoseidonGate::<F, D>::WIRE_SWAP;
        let swap_wire = Target::wire(gate, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        let inputs = inputs.as_ref();
        for i in 0..SPONGE_WIDTH {
            let in_wire = PoseidonGate::<F, D>::wire_input(i);
            let in_wire = Target::wire(gate, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Collect output wires.
        Self::AlgebraicPermutation::new(
            (0..SPONGE_WIDTH).map(|i| Target::wire(gate, PoseidonGate::<F, D>::wire_output(i))),
        )
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    pub(crate) fn check_test_vectors<F>(
        test_vectors: Vec<([u64; SPONGE_WIDTH], [u64; SPONGE_WIDTH])>,
    ) where
        F: Poseidon,
    {
        for (input_, expected_output_) in test_vectors.into_iter() {
            let mut input = [F::ZERO; SPONGE_WIDTH];
            for i in 0..SPONGE_WIDTH {
                input[i] = F::from_canonical_u64(input_[i]);
            }
            let output = F::poseidon(input);
            for i in 0..SPONGE_WIDTH {
                let ex_output = F::from_canonical_u64(expected_output_[i]);
                assert_eq!(output[i], ex_output);
            }
        }
    }

    pub(crate) fn check_consistency<F>()
    where
        F: Poseidon,
    {
        let mut input = [F::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            input[i] = F::from_canonical_u64(i as u64);
        }
        let output = F::poseidon(input);
        let output_naive = F::poseidon_naive(input);
        for i in 0..SPONGE_WIDTH {
            assert_eq!(output[i], output_naive[i]);
        }
    }
}

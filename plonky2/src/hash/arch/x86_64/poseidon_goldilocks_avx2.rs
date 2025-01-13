use core::arch::x86_64::*;

use unroll::unroll_for_loops;

use super::goldilocks_avx2::{add64_no_carry, mul64_no_overflow, mult_avx_128, reduce_avx_96_64};
use crate::field::types::PrimeField64;
use crate::hash::arch::x86_64::goldilocks_avx2::{add_avx, mult_avx, reduce_avx_128_64, sbox_avx};
use crate::hash::poseidon::{
    add_u160_u128, reduce_u160, Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS,
    N_PARTIAL_ROUNDS, SPONGE_WIDTH,
};
use crate::hash::poseidon_goldilocks::poseidon12_mds::block2;

#[allow(dead_code)]
const MDS_MATRIX_CIRC: [u64; 12] = [17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20];

#[allow(dead_code)]
const MDS_MATRIX_DIAG: [u64; 12] = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [u64; 12] = [
    0x3cc3f892184df408,
    0xe993fd841e7e97f1,
    0xf2831d3575f0f3af,
    0xd2500e0a350994ca,
    0xc5571f35d7288633,
    0x91d89c5184109a02,
    0xf37f925d04e5667b,
    0x2d6e448371955a69,
    0x740ef19ce01398a1,
    0x694d24c0752fdf45,
    0x60936af96ee2f148,
    0xc33448feadc78f0c,
];

const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS] = [
    0x74cb2e819ae421ab,
    0xd2559d2370e7f663,
    0x62bf78acf843d17c,
    0xd5ab7b67e14d1fb4,
    0xb9fe2ae6e0969bdc,
    0xe33fdf79f92a10e8,
    0x0ea2bb4c2b25989b,
    0xca9121fbf9d38f06,
    0xbdd9b0aa81f58fa4,
    0x83079fa4ecf20d7e,
    0x650b838edfcc4ad3,
    0x77180c88583c76ac,
    0xaf8c20753143a180,
    0xb8ccfe9989a39175,
    0x954a1729f60cc9c5,
    0xdeb5b550c4dca53b,
    0xf01bb0b00f77011e,
    0xa1ebb404b676afd9,
    0x860b6e1597a0173e,
    0x308bb65a036acbce,
    0x1aca78f31c97c876,
    0x0,
];

const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[u64; 12]; 12] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0,
        0x80772dc2645b280b,
        0xdc927721da922cf8,
        0xc1978156516879ad,
        0x90e80c591f48b603,
        0x3a2432625475e3ae,
        0x00a2d4321cca94fe,
        0x77736f524010c932,
        0x904d3f2804a36c54,
        0xbf9b39e28a16f354,
        0x3a1ded54a6cd058b,
        0x42392870da5737cf,
    ],
    [
        0,
        0xe796d293a47a64cb,
        0xb124c33152a2421a,
        0x0ee5dc0ce131268a,
        0xa9032a52f930fae6,
        0x7e33ca8c814280de,
        0xad11180f69a8c29e,
        0xc75ac6d5b5a10ff3,
        0xf0674a8dc5a387ec,
        0xb36d43120eaa5e2b,
        0x6f232aab4b533a25,
        0x3a1ded54a6cd058b,
    ],
    [
        0,
        0xdcedab70f40718ba,
        0x14a4a64da0b2668f,
        0x4715b8e5ab34653b,
        0x1e8916a99c93a88e,
        0xbba4b5d86b9a3b2c,
        0xe76649f9bd5d5c2e,
        0xaf8e2518a1ece54d,
        0xdcda1344cdca873f,
        0xcd080204256088e5,
        0xb36d43120eaa5e2b,
        0xbf9b39e28a16f354,
    ],
    [
        0,
        0xf4a437f2888ae909,
        0xc537d44dc2875403,
        0x7f68007619fd8ba9,
        0xa4911db6a32612da,
        0x2f7e9aade3fdaec1,
        0xe7ffd578da4ea43d,
        0x43a608e7afa6b5c2,
        0xca46546aa99e1575,
        0xdcda1344cdca873f,
        0xf0674a8dc5a387ec,
        0x904d3f2804a36c54,
    ],
    [
        0,
        0xf97abba0dffb6c50,
        0x5e40f0c9bb82aab5,
        0x5996a80497e24a6b,
        0x07084430a7307c9a,
        0xad2f570a5b8545aa,
        0xab7f81fef4274770,
        0xcb81f535cf98c9e9,
        0x43a608e7afa6b5c2,
        0xaf8e2518a1ece54d,
        0xc75ac6d5b5a10ff3,
        0x77736f524010c932,
    ],
    [
        0,
        0x7f8e41e0b0a6cdff,
        0x4b1ba8d40afca97d,
        0x623708f28fca70e8,
        0xbf150dc4914d380f,
        0xc26a083554767106,
        0x753b8b1126665c22,
        0xab7f81fef4274770,
        0xe7ffd578da4ea43d,
        0xe76649f9bd5d5c2e,
        0xad11180f69a8c29e,
        0x00a2d4321cca94fe,
    ],
    [
        0,
        0x726af914971c1374,
        0x1d7f8a2cce1a9d00,
        0x18737784700c75cd,
        0x7fb45d605dd82838,
        0x862361aeab0f9b6e,
        0xc26a083554767106,
        0xad2f570a5b8545aa,
        0x2f7e9aade3fdaec1,
        0xbba4b5d86b9a3b2c,
        0x7e33ca8c814280de,
        0x3a2432625475e3ae,
    ],
    [
        0,
        0x64dd936da878404d,
        0x4db9a2ead2bd7262,
        0xbe2e19f6d07f1a83,
        0x02290fe23c20351a,
        0x7fb45d605dd82838,
        0xbf150dc4914d380f,
        0x07084430a7307c9a,
        0xa4911db6a32612da,
        0x1e8916a99c93a88e,
        0xa9032a52f930fae6,
        0x90e80c591f48b603,
    ],
    [
        0,
        0x85418a9fef8a9890,
        0xd8a2eb7ef5e707ad,
        0xbfe85ababed2d882,
        0xbe2e19f6d07f1a83,
        0x18737784700c75cd,
        0x623708f28fca70e8,
        0x5996a80497e24a6b,
        0x7f68007619fd8ba9,
        0x4715b8e5ab34653b,
        0x0ee5dc0ce131268a,
        0xc1978156516879ad,
    ],
    [
        0,
        0x156048ee7a738154,
        0x91f7562377e81df5,
        0xd8a2eb7ef5e707ad,
        0x4db9a2ead2bd7262,
        0x1d7f8a2cce1a9d00,
        0x4b1ba8d40afca97d,
        0x5e40f0c9bb82aab5,
        0xc537d44dc2875403,
        0x14a4a64da0b2668f,
        0xb124c33152a2421a,
        0xdc927721da922cf8,
    ],
    [
        0,
        0xd841e8ef9dde8ba0,
        0x156048ee7a738154,
        0x85418a9fef8a9890,
        0x64dd936da878404d,
        0x726af914971c1374,
        0x7f8e41e0b0a6cdff,
        0xf97abba0dffb6c50,
        0xf4a437f2888ae909,
        0xdcedab70f40718ba,
        0xe796d293a47a64cb,
        0x80772dc2645b280b,
    ],
];

const FAST_PARTIAL_ROUND_W_HATS: [[u64; 12 - 1]; N_PARTIAL_ROUNDS] = [
    [
        0x3d999c961b7c63b0,
        0x814e82efcd172529,
        0x2421e5d236704588,
        0x887af7d4dd482328,
        0xa5e9c291f6119b27,
        0xbdc52b2676a4b4aa,
        0x64832009d29bcf57,
        0x09c4155174a552cc,
        0x463f9ee03d290810,
        0xc810936e64982542,
        0x043b1c289f7bc3ac,
    ],
    [
        0x673655aae8be5a8b,
        0xd510fe714f39fa10,
        0x2c68a099b51c9e73,
        0xa667bfa9aa96999d,
        0x4d67e72f063e2108,
        0xf84dde3e6acda179,
        0x40f9cc8c08f80981,
        0x5ead032050097142,
        0x6591b02092d671bb,
        0x00e18c71963dd1b7,
        0x8a21bcd24a14218a,
    ],
    [
        0x202800f4addbdc87,
        0xe4b5bdb1cc3504ff,
        0xbe32b32a825596e7,
        0x8e0f68c5dc223b9a,
        0x58022d9e1c256ce3,
        0x584d29227aa073ac,
        0x8b9352ad04bef9e7,
        0xaead42a3f445ecbf,
        0x3c667a1d833a3cca,
        0xda6f61838efa1ffe,
        0xe8f749470bd7c446,
    ],
    [
        0xc5b85bab9e5b3869,
        0x45245258aec51cf7,
        0x16e6b8e68b931830,
        0xe2ae0f051418112c,
        0x0470e26a0093a65b,
        0x6bef71973a8146ed,
        0x119265be51812daf,
        0xb0be7356254bea2e,
        0x8584defff7589bd7,
        0x3c5fe4aeb1fb52ba,
        0x9e7cd88acf543a5e,
    ],
    [
        0x179be4bba87f0a8c,
        0xacf63d95d8887355,
        0x6696670196b0074f,
        0xd99ddf1fe75085f9,
        0xc2597881fef0283b,
        0xcf48395ee6c54f14,
        0x15226a8e4cd8d3b6,
        0xc053297389af5d3b,
        0x2c08893f0d1580e2,
        0x0ed3cbcff6fcc5ba,
        0xc82f510ecf81f6d0,
    ],
    [
        0x94b06183acb715cc,
        0x500392ed0d431137,
        0x861cc95ad5c86323,
        0x05830a443f86c4ac,
        0x3b68225874a20a7c,
        0x10b3309838e236fb,
        0x9b77fc8bcd559e2c,
        0xbdecf5e0cb9cb213,
        0x30276f1221ace5fa,
        0x7935dd342764a144,
        0xeac6db520bb03708,
    ],
    [
        0x7186a80551025f8f,
        0x622247557e9b5371,
        0xc4cbe326d1ad9742,
        0x55f1523ac6a23ea2,
        0xa13dfe77a3d52f53,
        0xe30750b6301c0452,
        0x08bd488070a3a32b,
        0xcd800caef5b72ae3,
        0x83329c90f04233ce,
        0xb5b99e6664a0a3ee,
        0x6b0731849e200a7f,
    ],
    [
        0xec3fabc192b01799,
        0x382b38cee8ee5375,
        0x3bfb6c3f0e616572,
        0x514abd0cf6c7bc86,
        0x47521b1361dcc546,
        0x178093843f863d14,
        0xad1003c5d28918e7,
        0x738450e42495bc81,
        0xaf947c59af5e4047,
        0x4653fb0685084ef2,
        0x057fde2062ae35bf,
    ],
    [
        0xe376678d843ce55e,
        0x66f3860d7514e7fc,
        0x7817f3dfff8b4ffa,
        0x3929624a9def725b,
        0x0126ca37f215a80a,
        0xfce2f5d02762a303,
        0x1bc927375febbad7,
        0x85b481e5243f60bf,
        0x2d3c5f42a39c91a0,
        0x0811719919351ae8,
        0xf669de0add993131,
    ],
    [
        0x7de38bae084da92d,
        0x5b848442237e8a9b,
        0xf6c705da84d57310,
        0x31e6a4bdb6a49017,
        0x889489706e5c5c0f,
        0x0e4a205459692a1b,
        0xbac3fa75ee26f299,
        0x5f5894f4057d755e,
        0xb0dc3ecd724bb076,
        0x5e34d8554a6452ba,
        0x04f78fd8c1fdcc5f,
    ],
    [
        0x4dd19c38779512ea,
        0xdb79ba02704620e9,
        0x92a29a3675a5d2be,
        0xd5177029fe495166,
        0xd32b3298a13330c1,
        0x251c4a3eb2c5f8fd,
        0xe1c48b26e0d98825,
        0x3301d3362a4ffccb,
        0x09bb6c88de8cd178,
        0xdc05b676564f538a,
        0x60192d883e473fee,
    ],
    [
        0x16b9774801ac44a0,
        0x3cb8411e786d3c8e,
        0xa86e9cf505072491,
        0x0178928152e109ae,
        0x5317b905a6e1ab7b,
        0xda20b3be7f53d59f,
        0xcb97dedecebee9ad,
        0x4bd545218c59f58d,
        0x77dc8d856c05a44a,
        0x87948589e4f243fd,
        0x7e5217af969952c2,
    ],
    [
        0xbc58987d06a84e4d,
        0x0b5d420244c9cae3,
        0xa3c4711b938c02c0,
        0x3aace640a3e03990,
        0x865a0f3249aacd8a,
        0x8d00b2a7dbed06c7,
        0x6eacb905beb7e2f8,
        0x045322b216ec3ec7,
        0xeb9de00d594828e6,
        0x088c5f20df9e5c26,
        0xf555f4112b19781f,
    ],
    [
        0xa8cedbff1813d3a7,
        0x50dcaee0fd27d164,
        0xf1cb02417e23bd82,
        0xfaf322786e2abe8b,
        0x937a4315beb5d9b6,
        0x1b18992921a11d85,
        0x7d66c4368b3c497b,
        0x0e7946317a6b4e99,
        0xbe4430134182978b,
        0x3771e82493ab262d,
        0xa671690d8095ce82,
    ],
    [
        0xb035585f6e929d9d,
        0xba1579c7e219b954,
        0xcb201cf846db4ba3,
        0x287bf9177372cf45,
        0xa350e4f61147d0a6,
        0xd5d0ecfb50bcff99,
        0x2e166aa6c776ed21,
        0xe1e66c991990e282,
        0x662b329b01e7bb38,
        0x8aa674b36144d9a9,
        0xcbabf78f97f95e65,
    ],
    [
        0xeec24b15a06b53fe,
        0xc8a7aa07c5633533,
        0xefe9c6fa4311ad51,
        0xb9173f13977109a1,
        0x69ce43c9cc94aedc,
        0xecf623c9cd118815,
        0x28625def198c33c7,
        0xccfc5f7de5c3636a,
        0xf5e6c40f1621c299,
        0xcec0e58c34cb64b1,
        0xa868ea113387939f,
    ],
    [
        0xd8dddbdc5ce4ef45,
        0xacfc51de8131458c,
        0x146bb3c0fe499ac0,
        0x9e65309f15943903,
        0x80d0ad980773aa70,
        0xf97817d4ddbf0607,
        0xe4626620a75ba276,
        0x0dfdc7fd6fc74f66,
        0xf464864ad6f2bb93,
        0x02d55e52a5d44414,
        0xdd8de62487c40925,
    ],
    [
        0xc15acf44759545a3,
        0xcbfdcf39869719d4,
        0x33f62042e2f80225,
        0x2599c5ead81d8fa3,
        0x0b306cb6c1d7c8d0,
        0x658c80d3df3729b1,
        0xe8d1b2b21b41429c,
        0xa1b67f09d4b3ccb8,
        0x0e1adf8b84437180,
        0x0d593a5e584af47b,
        0xa023d94c56e151c7,
    ],
    [
        0x49026cc3a4afc5a6,
        0xe06dff00ab25b91b,
        0x0ab38c561e8850ff,
        0x92c3c8275e105eeb,
        0xb65256e546889bd0,
        0x3c0468236ea142f6,
        0xee61766b889e18f2,
        0xa206f41b12c30415,
        0x02fe9d756c9f12d1,
        0xe9633210630cbf12,
        0x1ffea9fe85a0b0b1,
    ],
    [
        0x81d1ae8cc50240f3,
        0xf4c77a079a4607d7,
        0xed446b2315e3efc1,
        0x0b0a6b70915178c3,
        0xb11ff3e089f15d9a,
        0x1d4dba0b7ae9cc18,
        0x65d74e2f43b48d05,
        0xa2df8c6b8ae0804a,
        0xa4e6f0a8c33348a6,
        0xc0a26efc7be5669b,
        0xa6b6582c547d0d60,
    ],
    [
        0x84afc741f1c13213,
        0x2f8f43734fc906f3,
        0xde682d72da0a02d9,
        0x0bb005236adb9ef2,
        0x5bdf35c10a8b5624,
        0x0739a8a343950010,
        0x52f515f44785cfbc,
        0xcbaf4e5d82856c60,
        0xac9ea09074e3e150,
        0x8f0fa011a2035fb0,
        0x1a37905d8450904a,
    ],
    [
        0x3abeb80def61cc85,
        0x9d19c9dd4eac4133,
        0x075a652d9641a985,
        0x9daf69ae1b67e667,
        0x364f71da77920a18,
        0x50bd769f745c95b1,
        0xf223d1180dbbf3fc,
        0x2f885e584e04aa99,
        0xb69a0fa70aea684a,
        0x09584acaa6e062a0,
        0x0bc051640145b19b,
    ],
];

const FAST_PARTIAL_ROUND_VS: [[u64; 12]; N_PARTIAL_ROUNDS] = [
    [
        0x0,
        0x94877900674181c3,
        0xc6c67cc37a2a2bbd,
        0xd667c2055387940f,
        0x0ba63a63e94b5ff0,
        0x99460cc41b8f079f,
        0x7ff02375ed524bb3,
        0xea0870b47a8caf0e,
        0xabcad82633b7bc9d,
        0x3b8d135261052241,
        0xfb4515f5e5b0d539,
        0x3ee8011c2b37f77c,
    ],
    [
        0x0,
        0x0adef3740e71c726,
        0xa37bf67c6f986559,
        0xc6b16f7ed4fa1b00,
        0x6a065da88d8bfc3c,
        0x4cabc0916844b46f,
        0x407faac0f02e78d1,
        0x07a786d9cf0852cf,
        0x42433fb6949a629a,
        0x891682a147ce43b0,
        0x26cfd58e7b003b55,
        0x2bbf0ed7b657acb3,
    ],
    [
        0x0,
        0x481ac7746b159c67,
        0xe367de32f108e278,
        0x73f260087ad28bec,
        0x5cfc82216bc1bdca,
        0xcaccc870a2663a0e,
        0xdb69cd7b4298c45d,
        0x7bc9e0c57243e62d,
        0x3cc51c5d368693ae,
        0x366b4e8cc068895b,
        0x2bd18715cdabbca4,
        0xa752061c4f33b8cf,
    ],
    [
        0x0,
        0xb22d2432b72d5098,
        0x9e18a487f44d2fe4,
        0x4b39e14ce22abd3c,
        0x9e77fde2eb315e0d,
        0xca5e0385fe67014d,
        0x0c2cb99bf1b6bddb,
        0x99ec1cd2a4460bfe,
        0x8577a815a2ff843f,
        0x7d80a6b4fd6518a5,
        0xeb6c67123eab62cb,
        0x8f7851650eca21a5,
    ],
    [
        0x0,
        0x11ba9a1b81718c2a,
        0x9f7d798a3323410c,
        0xa821855c8c1cf5e5,
        0x535e8d6fac0031b2,
        0x404e7c751b634320,
        0xa729353f6e55d354,
        0x4db97d92e58bb831,
        0xb53926c27897bf7d,
        0x965040d52fe115c5,
        0x9565fa41ebd31fd7,
        0xaae4438c877ea8f4,
    ],
    [
        0x0,
        0x37f4e36af6073c6e,
        0x4edc0918210800e9,
        0xc44998e99eae4188,
        0x9f4310d05d068338,
        0x9ec7fe4350680f29,
        0xc5b2c1fdc0b50874,
        0xa01920c5ef8b2ebe,
        0x59fa6f8bd91d58ba,
        0x8bfc9eb89b515a82,
        0xbe86a7a2555ae775,
        0xcbb8bbaa3810babf,
    ],
    [
        0x0,
        0x577f9a9e7ee3f9c2,
        0x88c522b949ace7b1,
        0x82f07007c8b72106,
        0x8283d37c6675b50e,
        0x98b074d9bbac1123,
        0x75c56fb7758317c1,
        0xfed24e206052bc72,
        0x26d7c3d1bc07dae5,
        0xf88c5e441e28dbb4,
        0x4fe27f9f96615270,
        0x514d4ba49c2b14fe,
    ],
    [
        0x0,
        0xf02a3ac068ee110b,
        0x0a3630dafb8ae2d7,
        0xce0dc874eaf9b55c,
        0x9a95f6cff5b55c7e,
        0x626d76abfed00c7b,
        0xa0c1cf1251c204ad,
        0xdaebd3006321052c,
        0x3d4bd48b625a8065,
        0x7f1e584e071f6ed2,
        0x720574f0501caed3,
        0xe3260ba93d23540a,
    ],
    [
        0x0,
        0xab1cbd41d8c1e335,
        0x9322ed4c0bc2df01,
        0x51c3c0983d4284e5,
        0x94178e291145c231,
        0xfd0f1a973d6b2085,
        0xd427ad96e2b39719,
        0x8a52437fecaac06b,
        0xdc20ee4b8c4c9a80,
        0xa2c98e9549da2100,
        0x1603fe12613db5b6,
        0x0e174929433c5505,
    ],
    [
        0x0,
        0x3d4eab2b8ef5f796,
        0xcfff421583896e22,
        0x4143cb32d39ac3d9,
        0x22365051b78a5b65,
        0x6f7fd010d027c9b6,
        0xd9dd36fba77522ab,
        0xa44cf1cb33e37165,
        0x3fc83d3038c86417,
        0xc4588d418e88d270,
        0xce1320f10ab80fe2,
        0xdb5eadbbec18de5d,
    ],
    [
        0x0,
        0x1183dfce7c454afd,
        0x21cea4aa3d3ed949,
        0x0fce6f70303f2304,
        0x19557d34b55551be,
        0x4c56f689afc5bbc9,
        0xa1e920844334f944,
        0xbad66d423d2ec861,
        0xf318c785dc9e0479,
        0x99e2032e765ddd81,
        0x400ccc9906d66f45,
        0xe1197454db2e0dd9,
    ],
    [
        0x0,
        0x84d1ecc4d53d2ff1,
        0xd8af8b9ceb4e11b6,
        0x335856bb527b52f4,
        0xc756f17fb59be595,
        0xc0654e4ea5553a78,
        0x9e9a46b61f2ea942,
        0x14fc8b5b3b809127,
        0xd7009f0f103be413,
        0x3e0ee7b7a9fb4601,
        0xa74e888922085ed7,
        0xe80a7cde3d4ac526,
    ],
    [
        0x0,
        0x238aa6daa612186d,
        0x9137a5c630bad4b4,
        0xc7db3817870c5eda,
        0x217e4f04e5718dc9,
        0xcae814e2817bd99d,
        0xe3292e7ab770a8ba,
        0x7bb36ef70b6b9482,
        0x3c7835fb85bca2d3,
        0xfe2cdf8ee3c25e86,
        0x61b3915ad7274b20,
        0xeab75ca7c918e4ef,
    ],
    [
        0x0,
        0xd6e15ffc055e154e,
        0xec67881f381a32bf,
        0xfbb1196092bf409c,
        0xdc9d2e07830ba226,
        0x0698ef3245ff7988,
        0x194fae2974f8b576,
        0x7a5d9bea6ca4910e,
        0x7aebfea95ccdd1c9,
        0xf9bd38a67d5f0e86,
        0xfa65539de65492d8,
        0xf0dfcbe7653ff787,
    ],
    [
        0x0,
        0x0bd87ad390420258,
        0x0ad8617bca9e33c8,
        0x0c00ad377a1e2666,
        0x0ac6fc58b3f0518f,
        0x0c0cc8a892cc4173,
        0x0c210accb117bc21,
        0x0b73630dbb46ca18,
        0x0c8be4920cbd4a54,
        0x0bfe877a21be1690,
        0x0ae790559b0ded81,
        0x0bf50db2f8d6ce31,
    ],
    [
        0x0,
        0x000cf29427ff7c58,
        0x000bd9b3cf49eec8,
        0x000d1dc8aa81fb26,
        0x000bc792d5c394ef,
        0x000d2ae0b2266453,
        0x000d413f12c496c1,
        0x000c84128cfed618,
        0x000db5ebd48fc0d4,
        0x000d1b77326dcb90,
        0x000beb0ccc145421,
        0x000d10e5b22b11d1,
    ],
    [
        0x0,
        0x00000e24c99adad8,
        0x00000cf389ed4bc8,
        0x00000e580cbf6966,
        0x00000cde5fd7e04f,
        0x00000e63628041b3,
        0x00000e7e81a87361,
        0x00000dabe78f6d98,
        0x00000efb14cac554,
        0x00000e5574743b10,
        0x00000d05709f42c1,
        0x00000e4690c96af1,
    ],
    [
        0x0,
        0x0000000f7157bc98,
        0x0000000e3006d948,
        0x0000000fa65811e6,
        0x0000000e0d127e2f,
        0x0000000fc18bfe53,
        0x0000000fd002d901,
        0x0000000eed6461d8,
        0x0000001068562754,
        0x0000000fa0236f50,
        0x0000000e3af13ee1,
        0x0000000fa460f6d1,
    ],
    [
        0x0,
        0x0000000011131738,
        0x000000000f56d588,
        0x0000000011050f86,
        0x000000000f848f4f,
        0x00000000111527d3,
        0x00000000114369a1,
        0x00000000106f2f38,
        0x0000000011e2ca94,
        0x00000000110a29f0,
        0x000000000fa9f5c1,
        0x0000000010f625d1,
    ],
    [
        0x0,
        0x000000000011f718,
        0x000000000010b6c8,
        0x0000000000134a96,
        0x000000000010cf7f,
        0x0000000000124d03,
        0x000000000013f8a1,
        0x0000000000117c58,
        0x0000000000132c94,
        0x0000000000134fc0,
        0x000000000010a091,
        0x0000000000128961,
    ],
    [
        0x0,
        0x0000000000001300,
        0x0000000000001750,
        0x000000000000114e,
        0x000000000000131f,
        0x000000000000167b,
        0x0000000000001371,
        0x0000000000001230,
        0x000000000000182c,
        0x0000000000001368,
        0x0000000000000f31,
        0x00000000000015c9,
    ],
    [
        0x0,
        0x0000000000000014,
        0x0000000000000022,
        0x0000000000000012,
        0x0000000000000027,
        0x000000000000000d,
        0x000000000000000d,
        0x000000000000001c,
        0x0000000000000002,
        0x0000000000000010,
        0x0000000000000029,
        0x000000000000000f,
    ],
];

const MDS_FREQ_BLOCK_ONE: [i64; 3] = [16, 32, 16];
const MDS_FREQ_BLOCK_TWO: [(i64, i64); 3] = [(2, -1), (-4, 1), (16, 1)];
const MDS_FREQ_BLOCK_THREE: [i64; 3] = [-1, -8, 2];

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
fn mds_row_shf(r: usize, v: &[u64; SPONGE_WIDTH]) -> (u64, u64) {
    let mut res = 0u128;

    // This is a hacky way of fully unrolling the loop.
    for i in 0..12 {
        if i < SPONGE_WIDTH {
            res += (v[(i + r) % SPONGE_WIDTH] as u128) * (MDS_MATRIX_CIRC[i] as u128);
        }
    }
    res += (v[r] as u128) * (MDS_MATRIX_DIAG[r] as u128);

    ((res >> 64) as u64, res as u64)
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx_v1(
    s0: &__m256i,
    s1: &__m256i,
    s2: &__m256i,
) -> (__m256i, __m256i, __m256i) {
    let mut st64 = [0u64; SPONGE_WIDTH];

    _mm256_storeu_si256((&mut st64[0..4]).as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256((&mut st64[4..8]).as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256((&mut st64[8..12]).as_mut_ptr().cast::<__m256i>(), *s2);

    let mut sumh: [u64; 12] = [0; 12];
    let mut suml: [u64; 12] = [0; 12];
    for r in 0..12 {
        if r < SPONGE_WIDTH {
            (sumh[r], suml[r]) = mds_row_shf(r, &st64);
        }
    }

    let ss0h = _mm256_loadu_si256((&sumh[0..4]).as_ptr().cast::<__m256i>());
    let ss0l = _mm256_loadu_si256((&suml[0..4]).as_ptr().cast::<__m256i>());
    let ss1h = _mm256_loadu_si256((&sumh[4..8]).as_ptr().cast::<__m256i>());
    let ss1l = _mm256_loadu_si256((&suml[4..8]).as_ptr().cast::<__m256i>());
    let ss2h = _mm256_loadu_si256((&sumh[8..12]).as_ptr().cast::<__m256i>());
    let ss2l = _mm256_loadu_si256((&suml[8..12]).as_ptr().cast::<__m256i>());
    let r0 = reduce_avx_128_64(&ss0h, &ss0l);
    let r1 = reduce_avx_128_64(&ss1h, &ss1l);
    let r2 = reduce_avx_128_64(&ss2h, &ss2l);

    (r0, r1, r2)
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx_v2<F>(
    s0: &__m256i,
    s1: &__m256i,
    s2: &__m256i,
) -> (__m256i, __m256i, __m256i)
where
    F: PrimeField64,
{
    let mut st64 = [0u64; SPONGE_WIDTH];

    _mm256_storeu_si256((&mut st64[0..4]).as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256((&mut st64[4..8]).as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256((&mut st64[8..12]).as_mut_ptr().cast::<__m256i>(), *s2);

    let mut result = [F::ZERO; SPONGE_WIDTH];
    // This is a hacky way of fully unrolling the loop.
    for r in 0..12 {
        if r < SPONGE_WIDTH {
            let (sum_hi, sum_lo) = mds_row_shf(r, &st64);
            result[r] = F::from_noncanonical_u96((sum_lo, sum_hi.try_into().unwrap()));
        }
    }

    let r0 = _mm256_loadu_si256((&result[0..4]).as_ptr().cast::<__m256i>());
    let r1 = _mm256_loadu_si256((&result[4..8]).as_ptr().cast::<__m256i>());
    let r2 = _mm256_loadu_si256((&result[8..12]).as_ptr().cast::<__m256i>());

    (r0, r1, r2)
}

#[inline(always)]
unsafe fn block1_avx(x: &__m256i, y: [i64; 3]) -> __m256i {
    let x0 = _mm256_permute4x64_epi64(*x, 0x0);
    let x1 = _mm256_permute4x64_epi64(*x, 0x55);
    let x2 = _mm256_permute4x64_epi64(*x, 0xAA);

    let f0 = _mm256_set_epi64x(0, y[2], y[1], y[0]);
    let f1 = _mm256_set_epi64x(0, y[1], y[0], y[2]);
    let f2 = _mm256_set_epi64x(0, y[0], y[2], y[1]);

    let t0 = mul64_no_overflow(&x0, &f0);
    let t1 = mul64_no_overflow(&x1, &f1);
    let t2 = mul64_no_overflow(&x2, &f2);

    let t0 = _mm256_add_epi64(t0, t1);
    _mm256_add_epi64(t0, t2)
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn block2_full_avx(xr: &__m256i, xi: &__m256i, y: [(i64, i64); 3]) -> (__m256i, __m256i) {
    let yr = _mm256_set_epi64x(0, y[2].0, y[1].0, y[0].0);
    let yi = _mm256_set_epi64x(0, y[2].1, y[1].1, y[0].1);
    let ys = _mm256_add_epi64(yr, yi);
    let xs = _mm256_add_epi64(*xr, *xi);

    // z0
    // z0r = dif2[0] + prod[1] - sum[1] + prod[2] - sum[2]
    // z0i = prod[0] - sum[0] + dif1[1] + dif1[2]
    let yy = _mm256_permute4x64_epi64(yr, 0x18);
    let mr_z0 = mul64_no_overflow(xr, &yy);
    let yy = _mm256_permute4x64_epi64(yi, 0x18);
    let mi_z0 = mul64_no_overflow(xi, &yy);
    let sum = _mm256_add_epi64(mr_z0, mi_z0);
    let dif1 = _mm256_sub_epi64(mi_z0, mr_z0);
    let dif2 = _mm256_sub_epi64(mr_z0, mi_z0);
    let yy = _mm256_permute4x64_epi64(ys, 0x18);
    let prod = mul64_no_overflow(&xs, &yy);
    let dif3 = _mm256_sub_epi64(prod, sum);
    let dif3perm1 = _mm256_permute4x64_epi64(dif3, 0x1);
    let dif3perm2 = _mm256_permute4x64_epi64(dif3, 0x2);
    let z0r = _mm256_add_epi64(dif2, dif3perm1);
    let z0r = _mm256_add_epi64(z0r, dif3perm2);
    let dif1perm1 = _mm256_permute4x64_epi64(dif1, 0x1);
    let dif1perm2 = _mm256_permute4x64_epi64(dif1, 0x2);
    let z0i = _mm256_add_epi64(dif3, dif1perm1);
    let z0i = _mm256_add_epi64(z0i, dif1perm2);
    let mask = _mm256_set_epi64x(0, 0, 0, 0xFFFFFFFFFFFFFFFFu64 as i64);
    let z0r = _mm256_and_si256(z0r, mask);
    let z0i = _mm256_and_si256(z0i, mask);

    // z1
    // z1r = dif2[0] + dif2[1] + prod[2] - sum[2];
    // z1i = prod[0] - sum[0] + prod[1] - sum[1] + dif1[2];
    let yy = _mm256_permute4x64_epi64(yr, 0x21);
    let mr_z1 = mul64_no_overflow(xr, &yy);
    let yy = _mm256_permute4x64_epi64(yi, 0x21);
    let mi_z1 = mul64_no_overflow(xi, &yy);
    let sum = _mm256_add_epi64(mr_z1, mi_z1);
    let dif1 = _mm256_sub_epi64(mi_z1, mr_z1);
    let dif2 = _mm256_sub_epi64(mr_z1, mi_z1);
    let yy = _mm256_permute4x64_epi64(ys, 0x21);
    let prod = mul64_no_overflow(&xs, &yy);
    let dif3 = _mm256_sub_epi64(prod, sum);
    let dif2perm = _mm256_permute4x64_epi64(dif2, 0x0);
    let dif3perm = _mm256_permute4x64_epi64(dif3, 0x8);
    let z1r = _mm256_add_epi64(dif2, dif2perm);
    let z1r = _mm256_add_epi64(z1r, dif3perm);
    let dif3perm = _mm256_permute4x64_epi64(dif3, 0x0);
    let dif1perm = _mm256_permute4x64_epi64(dif1, 0x8);
    let z1i = _mm256_add_epi64(dif3, dif3perm);
    let z1i = _mm256_add_epi64(z1i, dif1perm);
    let mask = _mm256_set_epi64x(0, 0, 0xFFFFFFFFFFFFFFFFu64 as i64, 0);
    let z1r = _mm256_and_si256(z1r, mask);
    let z1i = _mm256_and_si256(z1i, mask);

    // z2
    // z2r = dif2[0] + dif2[1] + dif2[2];
    // z2i = prod[0] - sum[0] + prod[1] - sum[1] + prod[2] - sum[2]
    let yy = _mm256_permute4x64_epi64(yr, 0x6);
    let mr_z2 = mul64_no_overflow(xr, &yy);
    let yy = _mm256_permute4x64_epi64(yi, 0x6);
    let mi_z2 = mul64_no_overflow(xi, &yy);
    let sum = _mm256_add_epi64(mr_z2, mi_z2);
    let dif2 = _mm256_sub_epi64(mr_z2, mi_z2);
    let yy = _mm256_permute4x64_epi64(ys, 0x6);
    let prod = mul64_no_overflow(&xs, &yy);
    let dif3 = _mm256_sub_epi64(prod, sum);
    let dif2perm1 = _mm256_permute4x64_epi64(dif2, 0x0);
    let dif2perm2 = _mm256_permute4x64_epi64(dif2, 0x10);
    let z2r = _mm256_add_epi64(dif2, dif2perm1);
    let z2r = _mm256_add_epi64(z2r, dif2perm2);
    let dif3perm1 = _mm256_permute4x64_epi64(dif3, 0x0);
    let dif3perm2 = _mm256_permute4x64_epi64(dif3, 0x10);
    let z2i = _mm256_add_epi64(dif3, dif3perm1);
    let z2i = _mm256_add_epi64(z2i, dif3perm2);
    let mask = _mm256_set_epi64x(0, 0xFFFFFFFFFFFFFFFFu64 as i64, 0, 0);
    let z2r = _mm256_and_si256(z2r, mask);
    let z2i = _mm256_and_si256(z2i, mask);

    let zr = _mm256_or_si256(z0r, z1r);
    let zr = _mm256_or_si256(zr, z2r);
    let zi = _mm256_or_si256(z0i, z1i);
    let zi = _mm256_or_si256(zi, z2i);
    (zr, zi)
}

#[inline(always)]
unsafe fn block2_avx(xr: &__m256i, xi: &__m256i, y: [(i64, i64); 3]) -> (__m256i, __m256i) {
    let mut vxr: [i64; 4] = [0; 4];
    let mut vxi: [i64; 4] = [0; 4];
    _mm256_storeu_si256(vxr.as_mut_ptr().cast::<__m256i>(), *xr);
    _mm256_storeu_si256(vxi.as_mut_ptr().cast::<__m256i>(), *xi);
    let x: [(i64, i64); 3] = [(vxr[0], vxi[0]), (vxr[1], vxi[1]), (vxr[2], vxi[2])];
    let b = block2(x, y);
    vxr = [b[0].0, b[1].0, b[2].0, 0];
    vxi = [b[0].1, b[1].1, b[2].1, 0];
    let rr = _mm256_loadu_si256(vxr.as_ptr().cast::<__m256i>());
    let ri = _mm256_loadu_si256(vxi.as_ptr().cast::<__m256i>());
    (rr, ri)
}

#[inline(always)]
unsafe fn block3_avx(x: &__m256i, y: [i64; 3]) -> __m256i {
    let x0 = _mm256_permute4x64_epi64(*x, 0x0);
    let x1 = _mm256_permute4x64_epi64(*x, 0x55);
    let x2 = _mm256_permute4x64_epi64(*x, 0xAA);

    let f0 = _mm256_set_epi64x(0, y[2], y[1], y[0]);
    let f1 = _mm256_set_epi64x(0, y[1], y[0], -y[2]);
    let f2 = _mm256_set_epi64x(0, y[0], -y[2], -y[1]);

    let t0 = mul64_no_overflow(&x0, &f0);
    let t1 = mul64_no_overflow(&x1, &f1);
    let t2 = mul64_no_overflow(&x2, &f2);

    let t0 = _mm256_add_epi64(t0, t1);
    _mm256_add_epi64(t0, t2)
}

#[inline(always)]
unsafe fn fft2_real_avx(x0: &__m256i, x1: &__m256i) -> (__m256i, __m256i) {
    let y0 = _mm256_add_epi64(*x0, *x1);
    let y1 = _mm256_sub_epi64(*x0, *x1);
    (y0, y1)
}

#[inline(always)]
unsafe fn fft4_real_avx(
    x0: &__m256i,
    x1: &__m256i,
    x2: &__m256i,
    x3: &__m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (z0, z2) = fft2_real_avx(x0, x2);
    let (z1, z3) = fft2_real_avx(x1, x3);
    let y0 = _mm256_add_epi64(z0, z1);
    let y2 = _mm256_sub_epi64(z0, z1);
    let y3 = _mm256_sub_epi64(zeros, z3);
    (y0, z2, y3, y2)
}

#[inline(always)]
unsafe fn ifft2_real_unreduced_avx(y0: &__m256i, y1: &__m256i) -> (__m256i, __m256i) {
    let x0 = _mm256_add_epi64(*y0, *y1);
    let x1 = _mm256_sub_epi64(*y0, *y1);
    (x0, x1)
}

#[inline(always)]
unsafe fn ifft4_real_unreduced_avx(
    y: (__m256i, (__m256i, __m256i), __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let z0 = _mm256_add_epi64(y.0, y.2);
    let z1 = _mm256_sub_epi64(y.0, y.2);
    let z2 = y.1 .0;
    let z3 = _mm256_sub_epi64(zeros, y.1 .1);
    let (x0, x2) = ifft2_real_unreduced_avx(&z0, &z2);
    let (x1, x3) = ifft2_real_unreduced_avx(&z1, &z3);
    (x0, x1, x2, x3)
}

#[inline]
unsafe fn mds_multiply_freq_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i) {
    /*
    // Alternative code using store and set.
    let mut s: [i64; 12] = [0; 12];
    _mm256_storeu_si256(s[0..4].as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256(s[4..8].as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256(s[8..12].as_mut_ptr().cast::<__m256i>(), *s2);
    let f0 = _mm256_set_epi64x(0, s[2], s[1], s[0]);
    let f1 = _mm256_set_epi64x(0, s[5], s[4], s[3]);
    let f2 = _mm256_set_epi64x(0, s[8], s[7], s[6]);
    let f3 = _mm256_set_epi64x(0, s[11], s[10], s[9]);
    */

    // Alternative code using permute and blend (it is faster).
    let f0 = *s0;
    let f11 = _mm256_permute4x64_epi64(*s0, 0x3);
    let f12 = _mm256_permute4x64_epi64(*s1, 0x10);
    let f1 = _mm256_blend_epi32(f11, f12, 0x3C);
    let f21 = _mm256_permute4x64_epi64(*s1, 0xE);
    let f22 = _mm256_permute4x64_epi64(*s2, 0x0);
    let f2 = _mm256_blend_epi32(f21, f22, 0x30);
    let f3 = _mm256_permute4x64_epi64(*s2, 0x39);

    let (u0, u1, u2, u3) = fft4_real_avx(&f0, &f1, &f2, &f3);

    // let [v0, v4, v8] = block1_avx([u[0], u[1], u[2]], MDS_FREQ_BLOCK_ONE);
    // [u[0], u[1], u[2]] are all in u0
    let f0 = block1_avx(&u0, MDS_FREQ_BLOCK_ONE);

    // let [v1, v5, v9] = block2([(u[0], v[0]), (u[1], v[1]), (u[2], v[2])], MDS_FREQ_BLOCK_TWO);
    let (f1, f2) = block2_avx(&u1, &u2, MDS_FREQ_BLOCK_TWO);

    // let [v2, v6, v10] = block3_avx([u[0], u[1], u[2]], MDS_FREQ_BLOCK_ONE);
    // [u[0], u[1], u[2]] are all in u3
    let f3 = block3_avx(&u3, MDS_FREQ_BLOCK_THREE);

    let (r0, r3, r6, r9) = ifft4_real_unreduced_avx((f0, (f1, f2), f3));
    let t = _mm256_permute4x64_epi64(r3, 0x0);
    *s0 = _mm256_blend_epi32(r0, t, 0xC0);
    let t1 = _mm256_permute4x64_epi64(r3, 0x9);
    let t2 = _mm256_permute4x64_epi64(r6, 0x40);
    *s1 = _mm256_blend_epi32(t1, t2, 0xF0);
    let t1 = _mm256_permute4x64_epi64(r6, 0x2);
    let t2 = _mm256_permute4x64_epi64(r9, 0x90);
    *s2 = _mm256_blend_epi32(t1, t2, 0xFC);
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i) {
    let mask = _mm256_set_epi64x(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    let mut sl0 = _mm256_and_si256(*s0, mask);
    let mut sl1 = _mm256_and_si256(*s1, mask);
    let mut sl2 = _mm256_and_si256(*s2, mask);
    let mut sh0 = _mm256_srli_epi64(*s0, 32);
    let mut sh1 = _mm256_srli_epi64(*s1, 32);
    let mut sh2 = _mm256_srli_epi64(*s2, 32);

    mds_multiply_freq_avx(&mut sl0, &mut sl1, &mut sl2);
    mds_multiply_freq_avx(&mut sh0, &mut sh1, &mut sh2);

    let shl0 = _mm256_slli_epi64(sh0, 32);
    let shl1 = _mm256_slli_epi64(sh1, 32);
    let shl2 = _mm256_slli_epi64(sh2, 32);
    let shh0 = _mm256_srli_epi64(sh0, 32);
    let shh1 = _mm256_srli_epi64(sh1, 32);
    let shh2 = _mm256_srli_epi64(sh2, 32);

    let (rl0, c0) = add64_no_carry(&sl0, &shl0);
    let (rh0, _) = add64_no_carry(&shh0, &c0);
    let r0 = reduce_avx_128_64(&rh0, &rl0);

    let (rl1, c1) = add64_no_carry(&sl1, &shl1);
    let (rh1, _) = add64_no_carry(&shh1, &c1);
    *s1 = reduce_avx_128_64(&rh1, &rl1);

    let (rl2, c2) = add64_no_carry(&sl2, &shl2);
    let (rh2, _) = add64_no_carry(&shh2, &c2);
    *s2 = reduce_avx_128_64(&rh2, &rl2);

    let rl = _mm256_slli_epi64(*s0, 3); // * 8 (low part)
    let rh = _mm256_srli_epi64(*s0, 61); // * 8 (high part, only 3 bits)
    let rx = reduce_avx_96_64(&rh, &rl);
    let rx = add_avx(&r0, &rx);
    *s0 = _mm256_blend_epi32(r0, rx, 0x3);
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
fn mds_partial_layer_init_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    let mut result = [F::ZERO; SPONGE_WIDTH];
    let res0 = state[0];
    unsafe {
        let mut r0 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
        let mut r1 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
        let mut r2 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
        for r in 1..12 {
            let sr = _mm256_set_epi64x(
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
            );
            let t0 = _mm256_loadu_si256(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][0..4])
                    .as_ptr()
                    .cast::<__m256i>(),
            );
            let t1 = _mm256_loadu_si256(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][4..8])
                    .as_ptr()
                    .cast::<__m256i>(),
            );
            let t2 = _mm256_loadu_si256(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][8..12])
                    .as_ptr()
                    .cast::<__m256i>(),
            );
            let m0 = mult_avx(&sr, &t0);
            let m1 = mult_avx(&sr, &t1);
            let m2 = mult_avx(&sr, &t2);
            r0 = add_avx(&r0, &m0);
            r1 = add_avx(&r1, &m1);
            r2 = add_avx(&r2, &m2);
        }
        _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), r0);
        _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), r1);
        _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), r2);
        state[0] = res0;
    }
}

#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_partial_layer_fast_avx<F>(
    s0: &mut __m256i,
    s1: &mut __m256i,
    s2: &mut __m256i,
    state: &mut [F; SPONGE_WIDTH],
    r: usize,
) where
    F: PrimeField64,
{
    let mut d_sum = (0u128, 0u32); // u160 accumulator
    for i in 1..12 {
        if i < SPONGE_WIDTH {
            let t = FAST_PARTIAL_ROUND_W_HATS[r][i - 1] as u128;
            let si = state[i].to_noncanonical_u64() as u128;
            d_sum = add_u160_u128(d_sum, si * t);
        }
    }
    let x0 = state[0].to_noncanonical_u64() as u128;
    let mds0to0 = (MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0]) as u128;
    d_sum = add_u160_u128(d_sum, x0 * mds0to0);
    let d = reduce_u160::<F>(d_sum);

    // result = [d] concat [state[0] * v + state[shift up by 1]]
    let ss0 = _mm256_set_epi64x(
        state[0].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
    );
    let rc0 = _mm256_loadu_si256((&FAST_PARTIAL_ROUND_VS[r][0..4]).as_ptr().cast::<__m256i>());
    let rc1 = _mm256_loadu_si256((&FAST_PARTIAL_ROUND_VS[r][4..8]).as_ptr().cast::<__m256i>());
    let rc2 = _mm256_loadu_si256(
        (&FAST_PARTIAL_ROUND_VS[r][8..12])
            .as_ptr()
            .cast::<__m256i>(),
    );
    let (mh, ml) = mult_avx_128(&ss0, &rc0);
    let m = reduce_avx_128_64(&mh, &ml);
    let r0 = add_avx(s0, &m);
    let d0 = _mm256_set_epi64x(0, 0, 0, d.to_canonical_u64() as i64);
    *s0 = _mm256_blend_epi32(r0, d0, 0x3);

    let (mh, ml) = mult_avx_128(&ss0, &rc1);
    let m = reduce_avx_128_64(&mh, &ml);
    *s1 = add_avx(s1, &m);

    let (mh, ml) = mult_avx_128(&ss0, &rc2);
    let m = reduce_avx_128_64(&mh, &ml);
    *s2 = add_avx(s2, &m);

    _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), *s2);
}

#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_partial_layer_init_avx_m256i<F>(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i)
where
    F: PrimeField64,
{
    let mut result = [F::ZERO; SPONGE_WIDTH];
    let res0 = *s0;

    let mut r0 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
    let mut r1 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
    let mut r2 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
    for r in 1..12 {
        let sr = match r {
            1 => _mm256_permute4x64_epi64(*s0, 0x55),
            2 => _mm256_permute4x64_epi64(*s0, 0xAA),
            3 => _mm256_permute4x64_epi64(*s0, 0xFF),
            4 => _mm256_permute4x64_epi64(*s1, 0x0),
            5 => _mm256_permute4x64_epi64(*s1, 0x55),
            6 => _mm256_permute4x64_epi64(*s1, 0xAA),
            7 => _mm256_permute4x64_epi64(*s1, 0xFF),
            8 => _mm256_permute4x64_epi64(*s2, 0x0),
            9 => _mm256_permute4x64_epi64(*s2, 0x55),
            10 => _mm256_permute4x64_epi64(*s2, 0xAA),
            11 => _mm256_permute4x64_epi64(*s2, 0xFF),
            _ => _mm256_permute4x64_epi64(*s0, 0x55),
        };
        let t0 = _mm256_loadu_si256(
            (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][0..4])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let t1 = _mm256_loadu_si256(
            (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][4..8])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let t2 = _mm256_loadu_si256(
            (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][8..12])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let m0 = mult_avx(&sr, &t0);
        let m1 = mult_avx(&sr, &t1);
        let m2 = mult_avx(&sr, &t2);
        r0 = add_avx(&r0, &m0);
        r1 = add_avx(&r1, &m1);
        r2 = add_avx(&r2, &m2);
    }
    *s0 = _mm256_blend_epi32(r0, res0, 0x3);
    *s1 = r1;
    *s2 = r2;
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
fn partial_first_constant_layer_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    unsafe {
        let c0 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[0..4])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let c1 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[4..8])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let c2 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[8..12])
                .as_ptr()
                .cast::<__m256i>(),
        );

        let mut s0 = _mm256_loadu_si256((state[0..4]).as_ptr().cast::<__m256i>());
        let mut s1 = _mm256_loadu_si256((state[4..8]).as_ptr().cast::<__m256i>());
        let mut s2 = _mm256_loadu_si256((state[8..12]).as_ptr().cast::<__m256i>());
        s0 = add_avx(&s0, &c0);
        s1 = add_avx(&s1, &c1);
        s2 = add_avx(&s2, &c2);
        _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), s2);
    }
}

#[inline(always)]
fn sbox_monomial<F>(x: F) -> F
where
    F: PrimeField64,
{
    // x |--> x^7
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

pub fn poseidon_avx<F>(input: &[F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let mut state = &mut input.clone();
    let mut round_ctr = 0;

    unsafe {
        // load state
        let mut s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let mut s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let mut s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());

        for _ in 0..HALF_N_FULL_ROUNDS {
            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm256_loadu_si256((&rc[0..4]).as_ptr().cast::<__m256i>());
            let rc1 = _mm256_loadu_si256((&rc[4..8]).as_ptr().cast::<__m256i>());
            let rc2 = _mm256_loadu_si256((&rc[8..12]).as_ptr().cast::<__m256i>());
            s0 = add_avx(&s0, &rc0);
            s1 = add_avx(&s1, &rc1);
            s2 = add_avx(&s2, &rc2);
            sbox_avx(&mut s0, &mut s1, &mut s2);
            mds_layer_avx(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        // this does partial_first_constant_layer_avx(&mut state);
        let c0 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[0..4])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let c1 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[4..8])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let c2 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[8..12])
                .as_ptr()
                .cast::<__m256i>(),
        );
        s0 = add_avx(&s0, &c0);
        s1 = add_avx(&s1, &c1);
        s2 = add_avx(&s2, &c2);

        mds_partial_layer_init_avx_m256i::<F>(&mut s0, &mut s1, &mut s2);

        _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), s2);

        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = sbox_monomial(state[0]);
            state[0] = state[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
            mds_partial_layer_fast_avx(&mut s0, &mut s1, &mut s2, &mut state, i);
        }
        round_ctr += N_PARTIAL_ROUNDS;

        // here state is already loaded in s0, s1, s2
        for _ in 0..HALF_N_FULL_ROUNDS {
            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm256_loadu_si256((&rc[0..4]).as_ptr().cast::<__m256i>());
            let rc1 = _mm256_loadu_si256((&rc[4..8]).as_ptr().cast::<__m256i>());
            let rc2 = _mm256_loadu_si256((&rc[8..12]).as_ptr().cast::<__m256i>());
            s0 = add_avx(&s0, &rc0);
            s1 = add_avx(&s1, &rc1);
            s2 = add_avx(&s2, &rc2);
            sbox_avx(&mut s0, &mut s1, &mut s2);
            mds_layer_avx(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        // store state
        _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), s2);
    };
    *state
}

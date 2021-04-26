//! Implements Rescue Prime.

use unroll::unroll_for_loops;

use crate::field::field::Field;

const ROUNDS: usize = 8;

const W: usize = 12;

const MDS: [[u64; W]; W] = [
    [
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
        3074457345215605419,
        11068046442776179508,
        13835058053470224385,
        6148914690431210838,
        9223372035646816257,
        1,
    ],
    [
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
        3074457345215605419,
        11068046442776179508,
        13835058053470224385,
        6148914690431210838,
        9223372035646816257,
    ],
    [
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
        3074457345215605419,
        11068046442776179508,
        13835058053470224385,
        6148914690431210838,
    ],
    [
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
        3074457345215605419,
        11068046442776179508,
        13835058053470224385,
    ],
    [
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
        3074457345215605419,
        11068046442776179508,
    ],
    [
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
        3074457345215605419,
    ],
    [
        1024819115071868473,
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
        2635249153041947502,
    ],
    [
        9708812669101911849,
        1024819115071868473,
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
        16140901062381928449,
    ],
    [
        2767011610694044877,
        9708812669101911849,
        1024819115071868473,
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
        2049638230143736946,
    ],
    [
        878416384347315834,
        2767011610694044877,
        9708812669101911849,
        1024819115071868473,
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
        5534023221388089754,
    ],
    [
        17608255704416649217,
        878416384347315834,
        2767011610694044877,
        9708812669101911849,
        1024819115071868473,
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
        16769767337539665921,
    ],
    [
        15238614667590392076,
        17608255704416649217,
        878416384347315834,
        2767011610694044877,
        9708812669101911849,
        1024819115071868473,
        3255307777287111620,
        17293822566837780481,
        15987178195121148178,
        1317624576520973751,
        5675921252705733081,
        10760600708254618966,
    ],
];

const RESCUE_CONSTANTS: [[u64; W]; 16] = [
    [
        12050887499329086906,
        1748247961703512657,
        315780861775001585,
        2827656358919812970,
        13335864861236723579,
        3010729529365640897,
        8463534053828271146,
        2528500966106598845,
        8969871077123422281,
        1002624930202741107,
        599979829006456404,
        4386170815218774254,
    ],
    [
        5771413917591851532,
        11946802620311685142,
        4759792267858670262,
        6879094914431255667,
        3985911073214909073,
        1542850118294175816,
        5393560436452023029,
        8331250756632997735,
        3395511836281190608,
        17601255793194446503,
        12848459944475727152,
        11995465655754698601,
    ],
    [
        14063960046551560130,
        14790209580166185143,
        5509023472758717841,
        1274395897760495573,
        16719545989415697758,
        17865948122414223407,
        3919263713959798649,
        5633741078654387163,
        15665612362287352054,
        3418834727998553015,
        5324019631954832682,
        17962066557010997431,
    ],
    [
        3282193104189649752,
        18423507935939999211,
        9035104445528866459,
        30842260240043277,
        3896337933354935129,
        6615548113269323045,
        6625827707190475694,
        6677757329269550670,
        11419013193186889337,
        17111888851716383760,
        12075517898615128691,
        8139844272075088233,
    ],
    [
        8872892112814161072,
        17529364346566228604,
        7526576514327158912,
        850359069964902700,
        9679332912197531902,
        10591229741059812071,
        12759208863825924546,
        14552519355635838750,
        16066249893409806278,
        11283035366525176262,
        1047378652379935387,
        17032498397644511356,
    ],
    [
        2938626421478254042,
        10375267398354586672,
        13728514869380643947,
        16707318479225743731,
        9785828188762698567,
        8610686976269299752,
        5478372191917042178,
        12716344455538470365,
        9968276048553747246,
        14746805727771473956,
        4822070620124107028,
        9901161649549513416,
    ],
    [
        13458162407040644078,
        4045792126424269312,
        9709263167782315020,
        2163173014916005515,
        17079206331095671215,
        2556388076102629669,
        6582772486087242347,
        1239959540200663058,
        18268236910639895687,
        12499012548657350745,
        17213068585339946119,
        7641451088868756688,
    ],
    [
        14674555473338434116,
        14624532976317185113,
        13625541984298615970,
        7612892294159054770,
        12294028208969561574,
        6067206081581804358,
        5778082506883496792,
        7389487446513884800,
        12929525660730020877,
        18244350162788654296,
        15285920877034454694,
        3640669683987215349,
    ],
    [
        6737585134029996281,
        1826890539455248546,
        289376081355380231,
        10782622161517803787,
        12978425540147835172,
        9828233103297278473,
        16384075371934678711,
        3187492301890791304,
        12985433735185968457,
        9470935291631377473,
        16328323199113140151,
        16218490552434224203,
    ],
    [
        6188809977565251499,
        18437718710937437067,
        4530469469895539008,
        9596355277372723349,
        13602518824447658705,
        8759976068576854281,
        10504320064094929535,
        3980760429843656150,
        14609448298151012462,
        5839843841558860609,
        10283805260656050418,
        7239168159249274821,
    ],
    [
        3604243611640027441,
        5237321927316578323,
        5071861664926666316,
        13025405632646149705,
        3285281651566464074,
        12121596060272825779,
        1900602777802961569,
        8122527981264852045,
        6731303887159752901,
        9197659817406857040,
        844741616904786364,
        14249777686667858094,
    ],
    [
        8602844218963499297,
        10133401373828451640,
        11618292280328565166,
        8828272598402499582,
        4252246265076774689,
        9760449011955070998,
        10233981507028897480,
        10427510555228840014,
        1007817664531124790,
        4465396600980659145,
        7727267420665314215,
        7904022788946844554,
    ],
    [
        11418297156527169222,
        15865399053509010196,
        1727198235391450850,
        16557095577717348672,
        1524052121709169653,
        14531367160053894310,
        4071756280138432327,
        10333204220115446291,
        16584144375833061215,
        12237566480526488368,
        11090440024401607208,
        18281335018830792766,
    ],
    [
        16152169547074248135,
        18338155611216027761,
        15842640128213925612,
        14687926435880145351,
        13259626900273707210,
        6187877366876303234,
        10312881470701795438,
        1924945292721719446,
        2278209355262975917,
        3250749056007953206,
        11589006946114672195,
        241829012299953928,
    ],
    [
        11244459446597052449,
        7319043416418482137,
        8148526814449636806,
        9054933038587901070,
        550333919248348827,
        5513167392062632770,
        12644459803778263764,
        9903621375535446226,
        16390581784506871871,
        14586524717888286021,
        6975796306584548762,
        5200407948555191573,
    ],
    [
        2855794043288846965,
        1259443213892506318,
        6145351706926586935,
        3853784494234324998,
        5871277378086513850,
        9414363368707862566,
        11946957446931890832,
        308083693687568600,
        12712587722369770461,
        6792392698104204991,
        16465224002344550280,
        10282380383506806095,
    ],
];

pub fn rescue<F: Field>(mut xs: [F; W]) -> [F; W] {
    for r in 0..8 {
        xs = sbox_layer_a(xs);
        xs = mds_layer(xs);
        xs = constant_layer(xs, &RESCUE_CONSTANTS[r * 2]);

        xs = sbox_layer_b(xs);
        xs = mds_layer(xs);
        xs = constant_layer(xs, &RESCUE_CONSTANTS[r * 2 + 1]);
    }
    xs
}

#[unroll_for_loops]
fn sbox_layer_a<F: Field>(x: [F; W]) -> [F; W] {
    let mut result = [F::ZERO; W];
    for i in 0..W {
        result[i] = x[i].cube();
    }
    result
}

#[unroll_for_loops]
fn sbox_layer_b<F: Field>(x: [F; W]) -> [F; W] {
    let mut result = [F::ZERO; W];
    for i in 0..W {
        result[i] = x[i].cube_root();
    }
    result
}

#[unroll_for_loops]
fn mds_layer<F: Field>(x: [F; W]) -> [F; W] {
    let mut result = [F::ZERO; W];
    for r in 0..W {
        for c in 0..W {
            result[r] = result[r] + F::from_canonical_u64(MDS[r][c]) * x[c];
        }
    }
    result
}

#[unroll_for_loops]
fn constant_layer<F: Field>(xs: [F; W], con: &[u64; W]) -> [F; W] {
    let mut result = [F::ZERO; W];
    for i in 0..W {
        result[i] = xs[i] + F::from_canonical_u64(con[i]);
    }
    result
}

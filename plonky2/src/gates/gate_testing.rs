#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{ensure, Result};

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::{Field, Sample};
use crate::gates::gate::Gate;
use crate::hash::hash_types::{HashOut, RichField};
use crate::iop::witness::{PartialWitness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CircuitConfig;
use crate::plonk::config::GenericConfig;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBaseBatch};
use crate::plonk::verifier::verify;
use crate::util::{log2_ceil, transpose};

const WITNESS_SIZE: usize = 1 << 5;
const WITNESS_DEGREE: usize = WITNESS_SIZE - 1;

/// Tests that the constraints imposed by the given gate are low-degree by applying them to random
/// low-degree witness polynomials.
#[cfg(not(feature = "no_random"))]
pub fn test_low_degree<F: RichField + Extendable<D>, G: Gate<F, D>, const D: usize>(gate: G) {
    let rate_bits = log2_ceil(gate.degree() + 1);

    let wire_ldes = random_low_degree_matrix::<F::Extension>(gate.num_wires(), rate_bits);
    let constant_ldes = random_low_degree_matrix::<F::Extension>(gate.num_constants(), rate_bits);
    assert_eq!(wire_ldes.len(), constant_ldes.len());
    let public_inputs_hash = &HashOut::rand();

    let constraint_evals = wire_ldes
        .iter()
        .zip(constant_ldes.iter())
        .map(|(local_wires, local_constants)| EvaluationVars {
            local_constants,
            local_wires,
            public_inputs_hash,
        })
        .map(|vars| gate.eval_unfiltered(vars))
        .collect::<Vec<_>>();

    let constraint_eval_degrees = transpose(&constraint_evals)
        .into_iter()
        .map(PolynomialValues::new)
        .map(|p| p.degree())
        .collect::<Vec<_>>();

    assert_eq!(
        constraint_eval_degrees.len(),
        gate.num_constraints(),
        "eval should return num_constraints() constraints"
    );

    let expected_eval_degree = WITNESS_DEGREE * gate.degree();

    assert!(
        constraint_eval_degrees
            .iter()
            .all(|&deg| deg <= expected_eval_degree),
        "Expected degrees at most {} * {} = {}, actual {:?}",
        WITNESS_SIZE,
        gate.degree(),
        expected_eval_degree,
        constraint_eval_degrees
    );
}

#[cfg(not(feature = "no_random"))]
fn random_low_degree_matrix<F: Field>(num_polys: usize, rate_bits: usize) -> Vec<Vec<F>> {
    let polys = (0..num_polys)
        .map(|_| random_low_degree_values(rate_bits))
        .collect::<Vec<_>>();

    if polys.is_empty() {
        // We want a Vec of many empty Vecs, whereas transpose would just give an empty Vec.
        vec![Vec::new(); WITNESS_SIZE << rate_bits]
    } else {
        transpose(&polys)
    }
}

#[cfg(not(feature = "no_random"))]
fn random_low_degree_values<F: Field>(rate_bits: usize) -> Vec<F> {
    PolynomialCoeffs::new(F::rand_vec(WITNESS_SIZE))
        .lde(rate_bits)
        .fft()
        .values
}

#[cfg(not(feature = "no_random"))]
pub fn test_eval_fns<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    G: Gate<F, D>,
    const D: usize,
>(
    gate: G,
) -> Result<()> {
    // Test that `eval_unfiltered` and `eval_unfiltered_base` are coherent.
    let wires_base = F::rand_vec(gate.num_wires());
    let constants_base = F::rand_vec(gate.num_constants());
    let wires = wires_base
        .iter()
        .map(|&x| F::Extension::from_basefield(x))
        .collect::<Vec<_>>();
    let constants = constants_base
        .iter()
        .map(|&x| F::Extension::from_basefield(x))
        .collect::<Vec<_>>();
    let public_inputs_hash = HashOut::rand();

    // Batch of 1.
    let vars_base_batch =
        EvaluationVarsBaseBatch::new(1, &constants_base, &wires_base, &public_inputs_hash);
    let vars = EvaluationVars {
        local_constants: &constants,
        local_wires: &wires,
        public_inputs_hash: &public_inputs_hash,
    };

    let evals_base = gate.eval_unfiltered_base_batch(vars_base_batch);
    let evals = gate.eval_unfiltered(vars);
    // This works because we have a batch of 1.
    ensure!(
        evals
            == evals_base
                .into_iter()
                .map(F::Extension::from_basefield)
                .collect::<Vec<_>>()
    );

    // Test that `eval_unfiltered` and `eval_unfiltered_recursively` are coherent.
    let wires = F::Extension::rand_vec(gate.num_wires());
    let constants = F::Extension::rand_vec(gate.num_constants());

    let config = CircuitConfig::standard_recursion_config();
    let mut pw = PartialWitness::new();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let wires_t = builder.add_virtual_extension_targets(wires.len());
    let constants_t = builder.add_virtual_extension_targets(constants.len());
    pw.set_extension_targets(&wires_t, &wires)?;
    pw.set_extension_targets(&constants_t, &constants)?;
    let public_inputs_hash_t = builder.add_virtual_hash();
    pw.set_hash_target(public_inputs_hash_t, public_inputs_hash)?;

    let vars = EvaluationVars {
        local_constants: &constants,
        local_wires: &wires,
        public_inputs_hash: &public_inputs_hash,
    };
    let evals = gate.eval_unfiltered(vars);

    let vars_t = EvaluationTargets {
        local_constants: &constants_t,
        local_wires: &wires_t,
        public_inputs_hash: &public_inputs_hash_t,
    };
    let evals_t = gate.eval_unfiltered_circuit(&mut builder, vars_t);
    pw.set_extension_targets(&evals_t, &evals)?;

    let data = builder.build::<C>();
    let proof = data.prove(pw)?;
    verify::<F, C, D>(proof, &data.verifier_only, &data.common)
}

#[cfg(not(feature = "std"))]
use alloc::vec;

use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierCircuitTarget};
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::plonk::plonk_common::salt_size;
use crate::plonk::proof::{
    OpeningSetTarget, ProofChallengesTarget, ProofTarget, ProofWithPublicInputsTarget,
};
use crate::plonk::vanishing_poly::eval_vanishing_poly_circuit;
use crate::plonk::vars::EvaluationTargets;
use crate::util::reducing::ReducingFactorTarget;
use crate::with_context;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Recursively verifies an inner proof.
    pub fn verify_proof<C: GenericConfig<D, F = F>>(
        &mut self,
        proof_with_pis: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        assert_eq!(
            proof_with_pis.public_inputs.len(),
            inner_common_data.num_public_inputs
        );
        let public_inputs_hash =
            self.hash_n_to_hash_no_pad::<C::InnerHasher>(proof_with_pis.public_inputs.clone());
        let challenges = proof_with_pis.get_challenges::<F, C>(
            self,
            public_inputs_hash,
            inner_verifier_data.circuit_digest,
            inner_common_data,
        );

        self.verify_proof_with_challenges::<C>(
            &proof_with_pis.proof,
            public_inputs_hash,
            challenges,
            inner_verifier_data,
            inner_common_data,
        );
    }

    /// Recursively verifies an inner proof.
    fn verify_proof_with_challenges<C: GenericConfig<D, F = F>>(
        &mut self,
        proof: &ProofTarget<D>,
        public_inputs_hash: HashOutTarget,
        challenges: ProofChallengesTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        let one = self.one_extension();

        let local_constants = &proof.openings.constants;
        let local_wires = &proof.openings.wires;
        let vars = EvaluationTargets {
            local_constants,
            local_wires,
            public_inputs_hash: &public_inputs_hash,
        };
        let local_zs = &proof.openings.plonk_zs;
        let next_zs = &proof.openings.plonk_zs_next;
        let local_lookup_zs = &proof.openings.lookup_zs;
        let next_lookup_zs = &proof.openings.next_lookup_zs;
        let s_sigmas = &proof.openings.plonk_sigmas;
        let partial_products = &proof.openings.partial_products;

        let zeta_pow_deg =
            self.exp_power_of_2_extension(challenges.plonk_zeta, inner_common_data.degree_bits());
        let vanishing_polys_zeta = with_context!(
            self,
            "evaluate the vanishing polynomial at our challenge point, zeta.",
            eval_vanishing_poly_circuit::<F, D>(
                self,
                inner_common_data,
                challenges.plonk_zeta,
                zeta_pow_deg,
                vars,
                local_zs,
                next_zs,
                local_lookup_zs,
                next_lookup_zs,
                partial_products,
                s_sigmas,
                &challenges.plonk_betas,
                &challenges.plonk_gammas,
                &challenges.plonk_alphas,
                &challenges.plonk_deltas,
            )
        );

        with_context!(self, "check vanishing and quotient polynomials.", {
            let quotient_polys_zeta = &proof.openings.quotient_polys;
            let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
            let z_h_zeta = self.sub_extension(zeta_pow_deg, one);
            for (i, chunk) in quotient_polys_zeta
                .chunks(inner_common_data.quotient_degree_factor)
                .enumerate()
            {
                let recombined_quotient = scale.reduce(chunk, self);
                let computed_vanishing_poly = self.mul_extension(z_h_zeta, recombined_quotient);
                self.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
            }
        });

        let merkle_caps = &[
            inner_verifier_data.constants_sigmas_cap.clone(),
            proof.wires_cap.clone(),
            proof.plonk_zs_partial_products_cap.clone(),
            proof.quotient_polys_cap.clone(),
        ];

        let fri_instance = inner_common_data.get_fri_instance_target(self, challenges.plonk_zeta);
        with_context!(
            self,
            "verify FRI proof",
            self.verify_fri_proof::<C>(
                &fri_instance,
                &proof.openings.to_fri_openings(),
                &challenges.fri_challenges,
                merkle_caps,
                &proof.opening_proof,
                &inner_common_data.fri_params,
            )
        );
    }

    pub fn add_virtual_proof_with_pis(
        &mut self,
        common_data: &CommonCircuitData<F, D>,
    ) -> ProofWithPublicInputsTarget<D> {
        let proof = self.add_virtual_proof(common_data);
        let public_inputs = self.add_virtual_targets(common_data.num_public_inputs);
        ProofWithPublicInputsTarget {
            proof,
            public_inputs,
        }
    }

    fn add_virtual_proof(&mut self, common_data: &CommonCircuitData<F, D>) -> ProofTarget<D> {
        let config = &common_data.config;
        let fri_params = &common_data.fri_params;
        let cap_height = fri_params.config.cap_height;

        let salt = salt_size(common_data.fri_params.hiding);
        let num_leaves_per_oracle = &mut vec![
            common_data.num_preprocessed_polys(),
            config.num_wires + salt,
            common_data.num_zs_partial_products_polys() + common_data.num_all_lookup_polys() + salt,
        ];

        if common_data.num_quotient_polys() > 0 {
            num_leaves_per_oracle.push(common_data.num_quotient_polys() + salt);
        }

        ProofTarget {
            wires_cap: self.add_virtual_cap(cap_height),
            plonk_zs_partial_products_cap: self.add_virtual_cap(cap_height),
            quotient_polys_cap: self.add_virtual_cap(cap_height),
            openings: self.add_opening_set(common_data),
            opening_proof: self.add_virtual_fri_proof(num_leaves_per_oracle, fri_params),
        }
    }

    fn add_opening_set(&mut self, common_data: &CommonCircuitData<F, D>) -> OpeningSetTarget<D> {
        let config = &common_data.config;
        let num_challenges = config.num_challenges;
        let total_partial_products = num_challenges * common_data.num_partial_products;
        let has_lookup = common_data.num_lookup_polys != 0;
        let num_lookups = if has_lookup {
            common_data.num_all_lookup_polys()
        } else {
            0
        };
        OpeningSetTarget {
            constants: self.add_virtual_extension_targets(common_data.num_constants),
            plonk_sigmas: self.add_virtual_extension_targets(config.num_routed_wires),
            wires: self.add_virtual_extension_targets(config.num_wires),
            plonk_zs: self.add_virtual_extension_targets(num_challenges),
            plonk_zs_next: self.add_virtual_extension_targets(num_challenges),
            lookup_zs: self.add_virtual_extension_targets(num_lookups),
            next_lookup_zs: self.add_virtual_extension_targets(num_lookups),
            partial_products: self.add_virtual_extension_targets(total_partial_products),
            quotient_polys: self.add_virtual_extension_targets(common_data.num_quotient_polys()),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{sync::Arc, vec};
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use anyhow::Result;
    use itertools::Itertools;
    use log::{info, Level};

    use super::*;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::fri::FriConfig;
    use crate::gadgets::lookup::{OTHER_TABLE, TIP5_TABLE};
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::{CircuitConfig, VerifierOnlyCircuitData};
    use crate::plonk::config::{KeccakGoldilocksConfig, PoseidonGoldilocksConfig};
    use crate::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
    use crate::plonk::prover::prove;
    use crate::util::timing::TimingTree;

    #[test]
    #[cfg(not(feature = "no_random"))]
    fn test_recursive_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_zk_config();

        let (proof, vd, common_data) = dummy_proof::<F, C, D>(&config, 4_000)?;
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    #[cfg(not(feature = "no_random"))]
    fn test_recursive_verifier_one_lookup() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_zk_config();

        let (proof, vd, common_data) = dummy_lookup_proof::<F, C, D>(&config, 10)?;
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_two_luts() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_config();

        let (proof, vd, common_data) = dummy_two_luts_proof::<F, C, D>(&config)?;
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_too_many_rows() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_config();

        let (proof, vd, common_data) = dummy_too_many_rows_proof::<F, C, D>(&config)?;
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_recursive_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        // Start with a degree 2^14 proof
        let (proof, vd, common_data) = dummy_proof::<F, C, D>(&config, 16_000)?;
        assert_eq!(common_data.degree_bits(), 14);

        // Shrink it to 2^13.
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, Some(13), false, false)?;
        assert_eq!(common_data.degree_bits(), 13);

        // Shrink it to 2^12.
        let (proof, vd, common_data) =
            recursive_proof::<F, C, C, D>(proof, vd, common_data, &config, None, true, true)?;
        assert_eq!(common_data.degree_bits(), 12);

        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    /// Creates a chain of recursive proofs where the last proof is made as small as reasonably
    /// possible, using a high rate, high PoW bits, etc.
    #[test]
    #[ignore]
    fn test_size_optimized_recursion() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type KC = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let standard_config = CircuitConfig::standard_recursion_config();

        // An initial dummy proof.
        let (proof, vd, common_data) = dummy_proof::<F, C, D>(&standard_config, 4_000)?;
        assert_eq!(common_data.degree_bits(), 12);

        // A standard recursive proof.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D>(
            proof,
            vd,
            common_data,
            &standard_config,
            None,
            false,
            false,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        // A high-rate recursive proof, designed to be verifiable with fewer routed wires.
        let high_rate_config = CircuitConfig {
            fri_config: FriConfig {
                rate_bits: 7,
                proof_of_work_bits: 16,
                num_query_rounds: 12,
                ..standard_config.fri_config.clone()
            },
            ..standard_config
        };
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D>(
            proof,
            vd,
            common_data,
            &high_rate_config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        // A final proof, optimized for size.
        let final_config = CircuitConfig {
            num_routed_wires: 37,
            fri_config: FriConfig {
                rate_bits: 8,
                cap_height: 0,
                proof_of_work_bits: 20,
                reduction_strategy: FriReductionStrategy::MinSize(None),
                num_query_rounds: 10,
            },
            ..high_rate_config
        };
        let (proof, vd, common_data) = recursive_proof::<F, KC, C, D>(
            proof,
            vd,
            common_data,
            &final_config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12, "final proof too large");

        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_multi_hash() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type PC = PoseidonGoldilocksConfig;
        type KC = KeccakGoldilocksConfig;
        type F = <PC as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let (proof, vd, common_data) = dummy_proof::<F, PC, D>(&config, 4_000)?;

        let (proof, vd, common_data) =
            recursive_proof::<F, PC, PC, D>(proof, vd, common_data, &config, None, false, false)?;
        test_serialization(&proof, &vd, &common_data)?;

        let (proof, vd, common_data) =
            recursive_proof::<F, KC, PC, D>(proof, vd, common_data, &config, None, false, false)?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    type Proof<F, C, const D: usize> = (
        ProofWithPublicInputs<F, C, D>,
        VerifierOnlyCircuitData<C, D>,
        CommonCircuitData<F, D>,
    );

    /// Creates a dummy proof which should have roughly `num_dummy_gates` gates.
    fn dummy_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        config: &CircuitConfig,
        num_dummy_gates: u64,
    ) -> Result<Proof<F, C, D>> {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        for _ in 0..num_dummy_gates {
            builder.add_gate(NoopGate, vec![]);
        }

        let data = builder.build::<C>();
        let inputs = PartialWitness::new();
        let proof = data.prove(inputs)?;
        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    /// Creates a dummy lookup proof which does one lookup to one LUT.
    fn dummy_lookup_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        config: &CircuitConfig,
        num_dummy_gates: u64,
    ) -> Result<Proof<F, C, D>> {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let out_a = table[look_val_a].1;
        let out_b = table[look_val_b].1;

        let tip5_index = builder.add_lookup_table_from_pairs(table);

        let output_a = builder.add_lookup_from_index(initial_a, tip5_index);
        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);

        for _ in 0..num_dummy_gates + 1 {
            builder.add_gate(NoopGate, vec![]);
        }

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);

        let data = builder.build::<C>();
        let mut inputs = PartialWitness::new();
        inputs.set_target(initial_a, F::from_canonical_usize(look_val_a))?;
        inputs.set_target(initial_b, F::from_canonical_usize(look_val_b))?;

        let proof = data.prove(inputs)?;
        data.verify(proof.clone())?;

        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(out_a),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_b),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );

        Ok((proof, data.verifier_only, data.common))
    }

    /// Creates a dummy lookup proof which does one lookup to two different LUTs.
    fn dummy_two_luts_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        config: &CircuitConfig,
    ) -> Result<Proof<F, C, D>> {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();

        let first_out = tip5_table[look_val_a];
        let second_out = tip5_table[look_val_b];

        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let other_table = OTHER_TABLE.to_vec();

        let tip5_index = builder.add_lookup_table_from_pairs(table);
        let output_a = builder.add_lookup_from_index(initial_a, tip5_index);

        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);
        let sum = builder.add(output_a, output_b);

        let s = first_out + second_out;
        let final_out = other_table[s as usize];

        let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());

        let other_index = builder.add_lookup_table_from_pairs(table2);
        let output_final = builder.add_lookup_from_index(sum, other_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);

        builder.register_public_input(sum);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);
        builder.register_public_input(output_final);

        let mut pw = PartialWitness::new();
        pw.set_target(initial_a, F::ONE)?;
        pw.set_target(initial_b, F::TWO)?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof.clone())?;

        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(first_out),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[4] == F::from_canonical_u16(second_out),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(s),
            "Sum between the first two LUT outputs is incorrect."
        );
        assert!(
            proof.public_inputs[5] == F::from_canonical_u16(final_out),
            "Output of the second LUT at index {} is incorrect.",
            s
        );

        Ok((proof, data.verifier_only, data.common))
    }

    /// Creates a dummy proof which has more than 256 lookups to one LUT.
    fn dummy_too_many_rows_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        config: &CircuitConfig,
    ) -> Result<Proof<F, C, D>> {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let out_a = table[look_val_a].1;
        let out_b = table[look_val_b].1;

        let tip5_index = builder.add_lookup_table_from_pairs(table);
        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);
        let mut output = builder.add_lookup_from_index(initial_a, tip5_index);
        for _ in 0..514 {
            output = builder.add_lookup_from_index(initial_a, tip5_index);
        }

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(output_b);
        builder.register_public_input(output);

        let mut pw = PartialWitness::new();

        pw.set_target(initial_a, F::from_canonical_usize(look_val_a))?;
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b))?;

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(out_b),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_a),
            "Lookups at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    fn recursive_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        InnerC: GenericConfig<D, F = F>,
        const D: usize,
    >(
        inner_proof: ProofWithPublicInputs<F, InnerC, D>,
        inner_vd: VerifierOnlyCircuitData<InnerC, D>,
        inner_cd: CommonCircuitData<F, D>,
        config: &CircuitConfig,
        min_degree_bits: Option<usize>,
        print_gate_counts: bool,
        print_timing: bool,
    ) -> Result<Proof<F, C, D>>
    where
        InnerC::Hasher: AlgebraicHasher<F>,
    {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let mut pw = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis(&inner_cd);
        pw.set_proof_with_pis_target(&pt, &inner_proof)?;

        let inner_data = builder.add_virtual_verifier_data(inner_cd.config.fri_config.cap_height);
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_vd.constants_sigmas_cap,
        )?;
        pw.set_hash_target(inner_data.circuit_digest, inner_vd.circuit_digest)?;

        builder.verify_proof::<InnerC>(&pt, &inner_data, &inner_cd);

        if print_gate_counts {
            builder.print_gate_counts(0);
        }

        if let Some(min_degree_bits) = min_degree_bits {
            // We don't want to pad all the way up to 2^min_degree_bits, as the builder will add a
            // few special gates afterward. So just pad to 2^(min_degree_bits - 1) + 1. Then the
            // builder will pad to the next power of two, 2^min_degree_bits.
            let min_gates = (1 << (min_degree_bits - 1)) + 1;
            for _ in builder.num_gates()..min_gates {
                builder.add_gate(NoopGate, vec![]);
            }
        }

        let data = builder.build::<C>();

        let mut timing = TimingTree::new("prove", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        if print_timing {
            timing.print();
        }

        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    /// Test serialization and print some size info.
    fn test_serialization<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        proof: &ProofWithPublicInputs<F, C, D>,
        vd: &VerifierOnlyCircuitData<C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Result<()> {
        let proof_bytes = proof.to_bytes();
        info!("Proof length: {} bytes", proof_bytes.len());
        let proof_from_bytes = ProofWithPublicInputs::from_bytes(proof_bytes, common_data)?;
        assert_eq!(proof, &proof_from_bytes);

        #[cfg(feature = "std")]
        let now = std::time::Instant::now();

        let compressed_proof = proof.clone().compress(&vd.circuit_digest, common_data)?;
        let decompressed_compressed_proof = compressed_proof
            .clone()
            .decompress(&vd.circuit_digest, common_data)?;

        #[cfg(feature = "std")]
        info!("{:.4}s to compress proof", now.elapsed().as_secs_f64());

        assert_eq!(proof, &decompressed_compressed_proof);

        let compressed_proof_bytes = compressed_proof.to_bytes();
        info!(
            "Compressed proof length: {} bytes",
            compressed_proof_bytes.len()
        );
        let compressed_proof_from_bytes =
            CompressedProofWithPublicInputs::from_bytes(compressed_proof_bytes, common_data)?;
        assert_eq!(compressed_proof, compressed_proof_from_bytes);

        Ok(())
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}

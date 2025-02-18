#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::fri::proof::{
    FriInitialTreeProofTarget, FriProofTarget, FriQueryRoundTarget, FriQueryStepTarget,
};
use crate::gadgets::polynomial::PolynomialCoeffsExtTarget;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_proofs::MerkleProofTarget;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierCircuitTarget};
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::plonk::proof::{OpeningSetTarget, ProofTarget, ProofWithPublicInputsTarget};
use crate::with_context;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Verify `proof0` if `condition` else verify `proof1`.
    /// `proof0` and `proof1` are assumed to use the same `CommonCircuitData`.
    pub fn conditionally_verify_proof<C: GenericConfig<D, F = F>>(
        &mut self,
        condition: BoolTarget,
        proof_with_pis0: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data0: &VerifierCircuitTarget,
        proof_with_pis1: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data1: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        let selected_proof =
            self.select_proof_with_pis(condition, proof_with_pis0, proof_with_pis1);
        let selected_verifier_data =
            self.select_verifier_data(condition, inner_verifier_data0, inner_verifier_data1);

        self.verify_proof::<C>(&selected_proof, &selected_verifier_data, inner_common_data);
    }

    /// Conditionally verify a proof with a new generated dummy proof.
    pub fn conditionally_verify_proof_or_dummy<C: GenericConfig<D, F = F> + 'static>(
        &mut self,
        condition: BoolTarget,
        proof_with_pis: &ProofWithPublicInputsTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<()>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        let (dummy_proof_with_pis_target, dummy_verifier_data_target) =
            self.dummy_proof_and_vk::<C>(inner_common_data)?;
        self.conditionally_verify_proof::<C>(
            condition,
            proof_with_pis,
            inner_verifier_data,
            &dummy_proof_with_pis_target,
            &dummy_verifier_data_target,
            inner_common_data,
        );
        Ok(())
    }

    /// Computes `if b { proof_with_pis0 } else { proof_with_pis1 }`.
    pub fn select_proof_with_pis(
        &mut self,
        b: BoolTarget,
        proof_with_pis0: &ProofWithPublicInputsTarget<D>,
        proof_with_pis1: &ProofWithPublicInputsTarget<D>,
    ) -> ProofWithPublicInputsTarget<D> {
        let ProofWithPublicInputsTarget {
            proof:
                ProofTarget {
                    wires_cap: wires_cap0,
                    plonk_zs_partial_products_cap: plonk_zs_partial_products_cap0,
                    quotient_polys_cap: quotient_polys_cap0,
                    openings: openings0,
                    opening_proof: opening_proof0,
                },
            public_inputs: public_inputs0,
        } = proof_with_pis0;
        let ProofWithPublicInputsTarget {
            proof:
                ProofTarget {
                    wires_cap: wires_cap1,
                    plonk_zs_partial_products_cap: plonk_zs_partial_products_cap1,
                    quotient_polys_cap: quotient_polys_cap1,
                    openings: openings1,
                    opening_proof: opening_proof1,
                },
            public_inputs: public_inputs1,
        } = proof_with_pis1;
        with_context!(self, "select proof", {
            let selected_wires_cap = self.select_cap(b, wires_cap0, wires_cap1);
            let selected_plonk_zs_partial_products_cap = self.select_cap(
                b,
                plonk_zs_partial_products_cap0,
                plonk_zs_partial_products_cap1,
            );
            let selected_quotient_polys_cap =
                self.select_cap(b, quotient_polys_cap0, quotient_polys_cap1);
            let selected_openings = self.select_opening_set(b, openings0, openings1);
            let selected_opening_proof =
                self.select_opening_proof(b, opening_proof0, opening_proof1);
            let selected_public_inputs = self.select_vec(b, public_inputs0, public_inputs1);
            ProofWithPublicInputsTarget {
                proof: ProofTarget {
                    wires_cap: selected_wires_cap,
                    plonk_zs_partial_products_cap: selected_plonk_zs_partial_products_cap,
                    quotient_polys_cap: selected_quotient_polys_cap,
                    openings: selected_openings,
                    opening_proof: selected_opening_proof,
                },
                public_inputs: selected_public_inputs,
            }
        })
    }

    /// Computes `if b { v0 } else { v1 }`.
    fn select_vec(&mut self, b: BoolTarget, v0: &[Target], v1: &[Target]) -> Vec<Target> {
        v0.iter()
            .zip_eq(v1)
            .map(|(t0, t1)| self.select(b, *t0, *t1))
            .collect()
    }

    /// Computes `if b { h0 } else { h1 }`.
    pub(crate) fn select_hash(
        &mut self,
        b: BoolTarget,
        h0: HashOutTarget,
        h1: HashOutTarget,
    ) -> HashOutTarget {
        HashOutTarget {
            elements: core::array::from_fn(|i| self.select(b, h0.elements[i], h1.elements[i])),
        }
    }

    /// Computes `if b { cap0 } else { cap1 }`.
    fn select_cap(
        &mut self,
        b: BoolTarget,
        cap0: &MerkleCapTarget,
        cap1: &MerkleCapTarget,
    ) -> MerkleCapTarget {
        assert_eq!(cap0.0.len(), cap1.0.len());
        MerkleCapTarget(
            cap0.0
                .iter()
                .zip_eq(&cap1.0)
                .map(|(h0, h1)| self.select_hash(b, *h0, *h1))
                .collect(),
        )
    }

    /// Computes `if b { v0 } else { v1 }`.
    fn select_vec_cap(
        &mut self,
        b: BoolTarget,
        v0: &[MerkleCapTarget],
        v1: &[MerkleCapTarget],
    ) -> Vec<MerkleCapTarget> {
        v0.iter()
            .zip_eq(v1)
            .map(|(c0, c1)| self.select_cap(b, c0, c1))
            .collect()
    }

    /// Computes `if b { vk0 } else { vk1 }`.
    pub fn select_verifier_data(
        &mut self,
        b: BoolTarget,
        vk0: &VerifierCircuitTarget,
        vk1: &VerifierCircuitTarget,
    ) -> VerifierCircuitTarget {
        VerifierCircuitTarget {
            constants_sigmas_cap: self.select_cap(
                b,
                &vk0.constants_sigmas_cap,
                &vk1.constants_sigmas_cap,
            ),
            circuit_digest: self.select_hash(b, vk0.circuit_digest, vk1.circuit_digest),
        }
    }

    /// Computes `if b { os0 } else { os1 }`.
    fn select_opening_set(
        &mut self,
        b: BoolTarget,
        os0: &OpeningSetTarget<D>,
        os1: &OpeningSetTarget<D>,
    ) -> OpeningSetTarget<D> {
        OpeningSetTarget {
            constants: self.select_vec_ext(b, &os0.constants, &os1.constants),
            plonk_sigmas: self.select_vec_ext(b, &os0.plonk_sigmas, &os1.plonk_sigmas),
            wires: self.select_vec_ext(b, &os0.wires, &os1.wires),
            plonk_zs: self.select_vec_ext(b, &os0.plonk_zs, &os1.plonk_zs),
            plonk_zs_next: self.select_vec_ext(b, &os0.plonk_zs_next, &os1.plonk_zs_next),
            lookup_zs: self.select_vec_ext(b, &os0.lookup_zs, &os1.lookup_zs),
            next_lookup_zs: self.select_vec_ext(b, &os0.next_lookup_zs, &os1.next_lookup_zs),
            partial_products: self.select_vec_ext(b, &os0.partial_products, &os1.partial_products),
            quotient_polys: self.select_vec_ext(b, &os0.quotient_polys, &os1.quotient_polys),
        }
    }

    /// Computes `if b { v0 } else { v1 }`.
    fn select_vec_ext(
        &mut self,
        b: BoolTarget,
        v0: &[ExtensionTarget<D>],
        v1: &[ExtensionTarget<D>],
    ) -> Vec<ExtensionTarget<D>> {
        v0.iter()
            .zip_eq(v1)
            .map(|(e0, e1)| self.select_ext(b, *e0, *e1))
            .collect()
    }

    /// Computes `if b { proof0 } else { proof1 }`.
    fn select_opening_proof(
        &mut self,
        b: BoolTarget,
        proof0: &FriProofTarget<D>,
        proof1: &FriProofTarget<D>,
    ) -> FriProofTarget<D> {
        FriProofTarget {
            commit_phase_merkle_caps: self.select_vec_cap(
                b,
                &proof0.commit_phase_merkle_caps,
                &proof1.commit_phase_merkle_caps,
            ),
            query_round_proofs: self.select_vec_query_round(
                b,
                &proof0.query_round_proofs,
                &proof1.query_round_proofs,
            ),
            final_poly: PolynomialCoeffsExtTarget(self.select_vec_ext(
                b,
                &proof0.final_poly.0,
                &proof1.final_poly.0,
            )),
            pow_witness: self.select(b, proof0.pow_witness, proof1.pow_witness),
        }
    }

    /// Computes `if b { qr0 } else { qr1 }`.
    fn select_query_round(
        &mut self,
        b: BoolTarget,
        qr0: &FriQueryRoundTarget<D>,
        qr1: &FriQueryRoundTarget<D>,
    ) -> FriQueryRoundTarget<D> {
        FriQueryRoundTarget {
            initial_trees_proof: self.select_initial_tree_proof(
                b,
                &qr0.initial_trees_proof,
                &qr1.initial_trees_proof,
            ),
            steps: self.select_vec_query_step(b, &qr0.steps, &qr1.steps),
        }
    }

    /// Computes `if b { v0 } else { v1 }`.
    fn select_vec_query_round(
        &mut self,
        b: BoolTarget,
        v0: &[FriQueryRoundTarget<D>],
        v1: &[FriQueryRoundTarget<D>],
    ) -> Vec<FriQueryRoundTarget<D>> {
        v0.iter()
            .zip_eq(v1)
            .map(|(qr0, qr1)| self.select_query_round(b, qr0, qr1))
            .collect()
    }

    /// Computes `if b { proof0 } else { proof1 }`.
    fn select_initial_tree_proof(
        &mut self,
        b: BoolTarget,
        proof0: &FriInitialTreeProofTarget,
        proof1: &FriInitialTreeProofTarget,
    ) -> FriInitialTreeProofTarget {
        FriInitialTreeProofTarget {
            evals_proofs: proof0
                .evals_proofs
                .iter()
                .zip_eq(&proof1.evals_proofs)
                .map(|((v0, p0), (v1, p1))| {
                    (
                        self.select_vec(b, v0, v1),
                        self.select_merkle_proof(b, p0, p1),
                    )
                })
                .collect(),
        }
    }

    /// Computes `if b { proof0 } else { proof1 }`.
    fn select_merkle_proof(
        &mut self,
        b: BoolTarget,
        proof0: &MerkleProofTarget,
        proof1: &MerkleProofTarget,
    ) -> MerkleProofTarget {
        MerkleProofTarget {
            siblings: proof0
                .siblings
                .iter()
                .zip_eq(&proof1.siblings)
                .map(|(h0, h1)| self.select_hash(b, *h0, *h1))
                .collect(),
        }
    }

    /// Computes `if b { qs0 } else { qs01 }`.
    fn select_query_step(
        &mut self,
        b: BoolTarget,
        qs0: &FriQueryStepTarget<D>,
        qs1: &FriQueryStepTarget<D>,
    ) -> FriQueryStepTarget<D> {
        FriQueryStepTarget {
            evals: self.select_vec_ext(b, &qs0.evals, &qs1.evals),
            merkle_proof: self.select_merkle_proof(b, &qs0.merkle_proof, &qs1.merkle_proof),
        }
    }

    /// Computes `if b { v0 } else { v1 }`.
    fn select_vec_query_step(
        &mut self,
        b: BoolTarget,
        v0: &[FriQueryStepTarget<D>],
        v1: &[FriQueryStepTarget<D>],
    ) -> Vec<FriQueryStepTarget<D>> {
        v0.iter()
            .zip_eq(v1)
            .map(|(qs0, qs1)| self.select_query_step(b, qs0, qs1))
            .collect()
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use anyhow::Result;
    use hashbrown::HashMap;

    use super::*;
    use crate::field::types::Sample;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::recursion::dummy_circuit::{dummy_circuit, dummy_proof};

    #[test]
    fn test_conditional_recursive_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_config();

        // Generate proof.
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let mut pw = PartialWitness::new();
        let t = builder.add_virtual_target();
        pw.set_target(t, F::rand())?;
        builder.register_public_input(t);
        let _t2 = builder.square(t);
        for _ in 0..64 {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof.clone())?;

        // Generate dummy proof with the same `CommonCircuitData`.
        let dummy_data = dummy_circuit(&data.common);
        let dummy_proof = dummy_proof(&dummy_data, HashMap::new())?;

        // Conditionally verify the two proofs.
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let mut pw = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis(&data.common);
        pw.set_proof_with_pis_target(&pt, &proof)?;
        let dummy_pt = builder.add_virtual_proof_with_pis(&data.common);
        pw.set_proof_with_pis_target::<C, D>(&dummy_pt, &dummy_proof)?;
        let inner_data =
            builder.add_virtual_verifier_data(data.common.config.fri_config.cap_height);
        pw.set_verifier_data_target(&inner_data, &data.verifier_only)?;
        let dummy_inner_data =
            builder.add_virtual_verifier_data(data.common.config.fri_config.cap_height);
        pw.set_verifier_data_target(&dummy_inner_data, &dummy_data.verifier_only)?;
        let b = builder.constant_bool(F::rand().0 % 2 == 0);
        builder.conditionally_verify_proof::<C>(
            b,
            &pt,
            &inner_data,
            &dummy_pt,
            &dummy_inner_data,
            &data.common,
        );

        builder.print_gate_counts(100);
        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}

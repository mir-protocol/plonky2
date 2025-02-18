#[cfg(not(feature = "std"))]
use alloc::vec;

use plonky2_field::extension::Extendable;

use crate::gates::coset_interpolation::CosetInterpolationGate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Interpolates a polynomial, whose points are a coset of the multiplicative subgroup with the
    /// given size, and whose values are given. Returns the evaluation of the interpolant at
    /// `evaluation_point`.
    pub(crate) fn interpolate_coset(
        &mut self,
        gate: CosetInterpolationGate<F, D>,
        coset_shift: Target,
        values: &[ExtensionTarget<D>],
        evaluation_point: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let row = self.num_gates();
        self.connect(coset_shift, Target::wire(row, gate.wire_shift()));
        for (i, &v) in values.iter().enumerate() {
            self.connect_extension(v, ExtensionTarget::from_range(row, gate.wires_value(i)));
        }
        self.connect_extension(
            evaluation_point,
            ExtensionTarget::from_range(row, gate.wires_evaluation_point()),
        );

        let eval = ExtensionTarget::from_range(row, gate.wires_evaluation_value());
        self.add_gate(gate, vec![]);

        eval
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use anyhow::Result;

    use crate::field::extension::FieldExtension;
    use crate::field::interpolation::interpolant;
    use crate::field::types::{Field, Sample};
    use crate::gates::coset_interpolation::CosetInterpolationGate;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    #[test]
    fn test_interpolate() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;
        let config = CircuitConfig::standard_recursion_config();
        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let subgroup_bits = 2;
        let len = 1 << subgroup_bits;
        let coset_shift = F::rand();
        let g = F::primitive_root_of_unity(subgroup_bits);
        let points = F::cyclic_subgroup_coset_known_order(g, coset_shift, len);
        let values = FF::rand_vec(len);

        let homogeneous_points = points
            .iter()
            .zip(values.iter())
            .map(|(&a, &b)| (<FF as FieldExtension<D>>::from_basefield(a), b))
            .collect::<Vec<_>>();

        let true_interpolant = interpolant(&homogeneous_points);

        let z = FF::rand();
        let true_eval = true_interpolant.eval(z);

        let coset_shift_target = builder.constant(coset_shift);

        let value_targets = values
            .iter()
            .map(|&v| (builder.constant_extension(v)))
            .collect::<Vec<_>>();

        let zt = builder.constant_extension(z);

        let evals_coset_gates = (2..=4)
            .map(|max_degree| {
                builder.interpolate_coset(
                    CosetInterpolationGate::with_max_degree(subgroup_bits, max_degree),
                    coset_shift_target,
                    &value_targets,
                    zt,
                )
            })
            .collect::<Vec<_>>();
        let true_eval_target = builder.constant_extension(true_eval);
        for &eval_coset_gate in evals_coset_gates.iter() {
            builder.connect_extension(eval_coset_gate, true_eval_target);
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }
}

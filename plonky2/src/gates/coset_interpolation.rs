#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;
use core::ops::Range;

use anyhow::Result;

use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension, OEF};
use crate::field::interpolation::barycentric_weights;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// One of the instantiations of `InterpolationGate`: allows constraints of variable
/// degree, up to `1<<subgroup_bits`.
///
/// This gate has as routed wires
/// - the coset shift from subgroup H
/// - the values that the interpolated polynomial takes on the coset
/// - the evaluation point
///
/// The evaluation strategy is based on the observation that if $P(X)$ is the interpolant of some
/// values over a coset and $P'(X)$ is the interpolant of those values over the subgroup, then
/// $P(X) = P'(X \cdot \mathrm{shift}^{-1})$. Interpolating $P'(X)$ is preferable because when subgroup is fixed
/// then so are the Barycentric weights and both can be hardcoded into the constraint polynomials.
///
/// A full interpolation of N values corresponds to the evaluation of a degree-N polynomial. This
/// gate can however be configured with a bounded degree of at least 2 by introducing more
/// non-routed wires. Let $x[]$ be the domain points, $v[]$ be the values, $w[]$ be the Barycentric
/// weights and $z$ be the evaluation point. Define the sequences
///
/// $p\[0\] = 1,$
///
/// $p\[i\] = p[i - 1] \cdot (z - x[i - 1]),$
///
/// $e\[0\] = 0,$
///
/// $e\[i\] = e[i - 1] ] \cdot (z - x[i - 1]) + w[i - 1] \cdot v[i - 1] \cdot p[i - 1]$
///
/// Then $e\[N\]$ is the final interpolated value. The non-routed wires hold every $(d - 1)$'th
/// intermediate value of $p$ and $e$, starting at $p\[d\]$ and $e\[d\]$, where $d$ is the gate degree.
#[derive(Clone, Debug, Default)]
pub struct CosetInterpolationGate<F: RichField + Extendable<D>, const D: usize> {
    pub subgroup_bits: usize,
    pub degree: usize,
    pub barycentric_weights: Vec<F>,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> CosetInterpolationGate<F, D> {
    pub fn new(subgroup_bits: usize) -> Self {
        Self::with_max_degree(subgroup_bits, 1 << subgroup_bits)
    }

    pub(crate) fn with_max_degree(subgroup_bits: usize, max_degree: usize) -> Self {
        assert!(max_degree > 1, "need at least quadratic constraints");

        let n_points = 1 << subgroup_bits;

        // Number of intermediate values required to compute interpolation with degree bound
        let n_intermediates = (n_points - 2) / (max_degree - 1);

        // Find minimum degree such that (n_points - 2) / (degree - 1) < n_intermediates + 1
        // Minimizing the degree this way allows the gate to be in a larger selector group
        let degree = (n_points - 2) / (n_intermediates + 1) + 2;

        let barycentric_weights = barycentric_weights(
            &F::two_adic_subgroup(subgroup_bits)
                .into_iter()
                .map(|x| (x, F::ZERO))
                .collect::<Vec<_>>(),
        );

        Self {
            subgroup_bits,
            degree,
            barycentric_weights,
            _phantom: PhantomData,
        }
    }

    const fn num_points(&self) -> usize {
        1 << self.subgroup_bits
    }

    /// Wire index of the coset shift.
    pub(crate) const fn wire_shift(&self) -> usize {
        0
    }

    const fn start_values(&self) -> usize {
        1
    }

    /// Wire indices of the `i`th interpolant value.
    pub(crate) fn wires_value(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_points());
        let start = self.start_values() + i * D;
        start..start + D
    }

    const fn start_evaluation_point(&self) -> usize {
        self.start_values() + self.num_points() * D
    }

    /// Wire indices of the point to evaluate the interpolant at.
    pub(crate) const fn wires_evaluation_point(&self) -> Range<usize> {
        let start = self.start_evaluation_point();
        start..start + D
    }

    const fn start_evaluation_value(&self) -> usize {
        self.start_evaluation_point() + D
    }

    /// Wire indices of the interpolated value.
    pub(crate) const fn wires_evaluation_value(&self) -> Range<usize> {
        let start = self.start_evaluation_value();
        start..start + D
    }

    const fn start_intermediates(&self) -> usize {
        self.start_evaluation_value() + D
    }

    pub const fn num_routed_wires(&self) -> usize {
        self.start_intermediates()
    }

    const fn num_intermediates(&self) -> usize {
        (self.num_points() - 2) / (self.degree - 1)
    }

    /// The wires corresponding to the i'th intermediate evaluation.
    const fn wires_intermediate_eval(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_intermediates());
        let start = self.start_intermediates() + D * i;
        start..start + D
    }

    /// The wires corresponding to the i'th intermediate product.
    const fn wires_intermediate_prod(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_intermediates());
        let start = self.start_intermediates() + D * (self.num_intermediates() + i);
        start..start + D
    }

    /// End of wire indices, exclusive.
    const fn end(&self) -> usize {
        self.start_intermediates() + D * (2 * self.num_intermediates() + 1)
    }

    /// Wire indices of the shifted point to evaluate the interpolant at.
    const fn wires_shifted_evaluation_point(&self) -> Range<usize> {
        let start = self.start_intermediates() + D * 2 * self.num_intermediates();
        start..start + D
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for CosetInterpolationGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.subgroup_bits)?;
        dst.write_usize(self.degree)?;
        dst.write_usize(self.barycentric_weights.len())?;
        dst.write_field_vec(&self.barycentric_weights)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let subgroup_bits = src.read_usize()?;
        let degree = src.read_usize()?;
        let length = src.read_usize()?;
        let barycentric_weights: Vec<F> = src.read_field_vec(length)?;
        Ok(Self {
            subgroup_bits,
            degree,
            barycentric_weights,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        let shift = vars.local_wires[self.wire_shift()];
        let evaluation_point = vars.get_local_ext_algebra(self.wires_evaluation_point());
        let shifted_evaluation_point =
            vars.get_local_ext_algebra(self.wires_shifted_evaluation_point());
        constraints.extend(
            (evaluation_point - shifted_evaluation_point.scalar_mul(shift)).to_basefield_array(),
        );

        let domain = F::two_adic_subgroup(self.subgroup_bits);
        let values = (0..self.num_points())
            .map(|i| vars.get_local_ext_algebra(self.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.barycentric_weights;

        let (mut computed_eval, mut computed_prod) = partial_interpolate_ext_algebra(
            &domain[..self.degree()],
            &values[..self.degree()],
            &weights[..self.degree()],
            shifted_evaluation_point,
            ExtensionAlgebra::ZERO,
            ExtensionAlgebra::one(),
        );

        for i in 0..self.num_intermediates() {
            let intermediate_eval = vars.get_local_ext_algebra(self.wires_intermediate_eval(i));
            let intermediate_prod = vars.get_local_ext_algebra(self.wires_intermediate_prod(i));
            constraints.extend((intermediate_eval - computed_eval).to_basefield_array());
            constraints.extend((intermediate_prod - computed_prod).to_basefield_array());

            let start_index = 1 + (self.degree() - 1) * (i + 1);
            let end_index = (start_index + self.degree() - 1).min(self.num_points());
            (computed_eval, computed_prod) = partial_interpolate_ext_algebra(
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                intermediate_eval,
                intermediate_prod,
            );
        }

        let evaluation_value = vars.get_local_ext_algebra(self.wires_evaluation_value());
        constraints.extend((evaluation_value - computed_eval).to_basefield_array());

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let shift = vars.local_wires[self.wire_shift()];
        let evaluation_point = vars.get_local_ext(self.wires_evaluation_point());
        let shifted_evaluation_point = vars.get_local_ext(self.wires_shifted_evaluation_point());
        yield_constr.many(
            (evaluation_point - shifted_evaluation_point.scalar_mul(shift)).to_basefield_array(),
        );

        let domain = F::two_adic_subgroup(self.subgroup_bits);
        let values = (0..self.num_points())
            .map(|i| vars.get_local_ext(self.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.barycentric_weights;

        let (mut computed_eval, mut computed_prod) = partial_interpolate(
            &domain[..self.degree()],
            &values[..self.degree()],
            &weights[..self.degree()],
            shifted_evaluation_point,
            F::Extension::ZERO,
            F::Extension::ONE,
        );

        for i in 0..self.num_intermediates() {
            let intermediate_eval = vars.get_local_ext(self.wires_intermediate_eval(i));
            let intermediate_prod = vars.get_local_ext(self.wires_intermediate_prod(i));
            yield_constr.many((intermediate_eval - computed_eval).to_basefield_array());
            yield_constr.many((intermediate_prod - computed_prod).to_basefield_array());

            let start_index = 1 + (self.degree() - 1) * (i + 1);
            let end_index = (start_index + self.degree() - 1).min(self.num_points());
            (computed_eval, computed_prod) = partial_interpolate(
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                intermediate_eval,
                intermediate_prod,
            );
        }

        let evaluation_value = vars.get_local_ext(self.wires_evaluation_value());
        yield_constr.many((evaluation_value - computed_eval).to_basefield_array());
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        let shift = vars.local_wires[self.wire_shift()];
        let evaluation_point = vars.get_local_ext_algebra(self.wires_evaluation_point());
        let shifted_evaluation_point =
            vars.get_local_ext_algebra(self.wires_shifted_evaluation_point());

        let neg_one = builder.neg_one();
        let neg_shift = builder.scalar_mul_ext(neg_one, shift);
        constraints.extend(
            builder
                .scalar_mul_add_ext_algebra(neg_shift, shifted_evaluation_point, evaluation_point)
                .to_ext_target_array(),
        );

        let domain = F::two_adic_subgroup(self.subgroup_bits);
        let values = (0..self.num_points())
            .map(|i| vars.get_local_ext_algebra(self.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.barycentric_weights;

        let initial_eval = builder.zero_ext_algebra();
        let initial_prod = builder.constant_ext_algebra(F::Extension::ONE.into());
        let (mut computed_eval, mut computed_prod) = partial_interpolate_ext_algebra_target(
            builder,
            &domain[..self.degree()],
            &values[..self.degree()],
            &weights[..self.degree()],
            shifted_evaluation_point,
            initial_eval,
            initial_prod,
        );

        for i in 0..self.num_intermediates() {
            let intermediate_eval = vars.get_local_ext_algebra(self.wires_intermediate_eval(i));
            let intermediate_prod = vars.get_local_ext_algebra(self.wires_intermediate_prod(i));
            constraints.extend(
                builder
                    .sub_ext_algebra(intermediate_eval, computed_eval)
                    .to_ext_target_array(),
            );
            constraints.extend(
                builder
                    .sub_ext_algebra(intermediate_prod, computed_prod)
                    .to_ext_target_array(),
            );

            let start_index = 1 + (self.degree() - 1) * (i + 1);
            let end_index = (start_index + self.degree() - 1).min(self.num_points());
            (computed_eval, computed_prod) = partial_interpolate_ext_algebra_target(
                builder,
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                intermediate_eval,
                intermediate_prod,
            );
        }

        let evaluation_value = vars.get_local_ext_algebra(self.wires_evaluation_value());
        constraints.extend(
            builder
                .sub_ext_algebra(evaluation_value, computed_eval)
                .to_ext_target_array(),
        );

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = InterpolationGenerator::<F, D>::new(row, self.clone());
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        self.end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn num_constraints(&self) -> usize {
        // D constraints to check for consistency of the shifted evaluation point, plus D
        // constraints for the evaluation value.
        D + D + 2 * D * self.num_intermediates()
    }
}

#[derive(Debug, Default)]
pub struct InterpolationGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    gate: CosetInterpolationGate<F, D>,
    interpolation_domain: Vec<F>,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> InterpolationGenerator<F, D> {
    fn new(row: usize, gate: CosetInterpolationGate<F, D>) -> Self {
        let interpolation_domain = F::two_adic_subgroup(gate.subgroup_bits);
        InterpolationGenerator {
            row,
            gate,
            interpolation_domain,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for InterpolationGenerator<F, D>
{
    fn id(&self) -> String {
        "InterpolationGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| {
            Target::Wire(Wire {
                row: self.row,
                column,
            })
        };

        let local_targets = |columns: Range<usize>| columns.map(local_target);

        let num_points = self.gate.num_points();
        let mut deps = Vec::with_capacity(1 + D + num_points * D);

        deps.push(local_target(self.gate.wire_shift()));
        deps.extend(local_targets(self.gate.wires_evaluation_point()));
        for i in 0..num_points {
            deps.extend(local_targets(self.gate.wires_value(i)));
        }
        deps
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let get_local_wire = |column| witness.get_wire(local_wire(column));

        let get_local_ext = |wire_range: Range<usize>| {
            debug_assert_eq!(wire_range.len(), D);
            let values = wire_range.map(get_local_wire).collect::<Vec<_>>();
            let arr = values.try_into().unwrap();
            F::Extension::from_basefield_array(arr)
        };

        let evaluation_point = get_local_ext(self.gate.wires_evaluation_point());
        let shift = get_local_wire(self.gate.wire_shift());
        let shifted_evaluation_point = evaluation_point.scalar_mul(shift.inverse());
        let degree = self.gate.degree();

        out_buffer.set_ext_wires(
            self.gate.wires_shifted_evaluation_point().map(local_wire),
            shifted_evaluation_point,
        )?;

        let domain = &self.interpolation_domain;
        let values = (0..self.gate.num_points())
            .map(|i| get_local_ext(self.gate.wires_value(i)))
            .collect::<Vec<_>>();
        let weights = &self.gate.barycentric_weights;

        let (mut computed_eval, mut computed_prod) = partial_interpolate(
            &domain[..degree],
            &values[..degree],
            &weights[..degree],
            shifted_evaluation_point,
            F::Extension::ZERO,
            F::Extension::ONE,
        );

        for i in 0..self.gate.num_intermediates() {
            let intermediate_eval_wires = self.gate.wires_intermediate_eval(i).map(local_wire);
            let intermediate_prod_wires = self.gate.wires_intermediate_prod(i).map(local_wire);
            out_buffer.set_ext_wires(intermediate_eval_wires, computed_eval)?;
            out_buffer.set_ext_wires(intermediate_prod_wires, computed_prod)?;

            let start_index = 1 + (degree - 1) * (i + 1);
            let end_index = (start_index + degree - 1).min(self.gate.num_points());
            (computed_eval, computed_prod) = partial_interpolate(
                &domain[start_index..end_index],
                &values[start_index..end_index],
                &weights[start_index..end_index],
                shifted_evaluation_point,
                computed_eval,
                computed_prod,
            );
        }

        let evaluation_value_wires = self.gate.wires_evaluation_value().map(local_wire);
        out_buffer.set_ext_wires(evaluation_value_wires, computed_eval)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        self.gate.serialize(dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = CosetInterpolationGate::deserialize(src, _common_data)?;
        Ok(Self::new(row, gate))
    }
}

/// Interpolate the polynomial defined by its values on an arbitrary domain at the given point `x`.
///
/// The domain lies in a base field while the values and evaluation point may be from an extension
/// field. The Barycentric weights are precomputed and taken as arguments.
pub fn interpolate_over_base_domain<F: Field + Extendable<D>, const D: usize>(
    domain: &[F],
    values: &[F::Extension],
    barycentric_weights: &[F],
    x: F::Extension,
) -> F::Extension {
    let (result, _) = partial_interpolate(
        domain,
        values,
        barycentric_weights,
        x,
        F::Extension::ZERO,
        F::Extension::ONE,
    );
    result
}

/// Perform a partial interpolation of the polynomial defined by its values on an arbitrary domain.
///
/// The Barycentric algorithm to interpolate a polynomial at a given point `x` is a linear pass
/// over the sequence of domain points, values, and Barycentric weights which maintains two
/// accumulated values, a partial evaluation and a partial product. This partially updates the
/// accumulated values, so that starting with an initial evaluation of 0 and a partial evaluation
/// of 1 and running over the whole domain is a full interpolation.
fn partial_interpolate<F: Field + Extendable<D>, const D: usize>(
    domain: &[F],
    values: &[F::Extension],
    barycentric_weights: &[F],
    x: F::Extension,
    initial_eval: F::Extension,
    initial_partial_prod: F::Extension,
) -> (F::Extension, F::Extension) {
    let n = domain.len();
    assert_ne!(n, 0);
    assert_eq!(n, values.len());
    assert_eq!(n, barycentric_weights.len());

    let weighted_values = values
        .iter()
        .zip(barycentric_weights.iter())
        .map(|(&value, &weight)| value.scalar_mul(weight));

    weighted_values.zip(domain.iter()).fold(
        (initial_eval, initial_partial_prod),
        |(eval, terms_partial_prod), (val, &x_i)| {
            let term = x - x_i.into();
            let next_eval = eval * term + val * terms_partial_prod;
            let next_terms_partial_prod = terms_partial_prod * term;
            (next_eval, next_terms_partial_prod)
        },
    )
}

fn partial_interpolate_ext_algebra<F: OEF<D>, const D: usize>(
    domain: &[F::BaseField],
    values: &[ExtensionAlgebra<F, D>],
    barycentric_weights: &[F::BaseField],
    x: ExtensionAlgebra<F, D>,
    initial_eval: ExtensionAlgebra<F, D>,
    initial_partial_prod: ExtensionAlgebra<F, D>,
) -> (ExtensionAlgebra<F, D>, ExtensionAlgebra<F, D>) {
    let n = domain.len();
    assert_ne!(n, 0);
    assert_eq!(n, values.len());
    assert_eq!(n, barycentric_weights.len());

    let weighted_values = values
        .iter()
        .zip(barycentric_weights.iter())
        .map(|(&value, &weight)| value.scalar_mul(F::from_basefield(weight)));

    weighted_values.zip(domain.iter()).fold(
        (initial_eval, initial_partial_prod),
        |(eval, terms_partial_prod), (val, &x_i)| {
            let term = x - F::from_basefield(x_i).into();
            let next_eval = eval * term + val * terms_partial_prod;
            let next_terms_partial_prod = terms_partial_prod * term;
            (next_eval, next_terms_partial_prod)
        },
    )
}

fn partial_interpolate_ext_algebra_target<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    domain: &[F],
    values: &[ExtensionAlgebraTarget<D>],
    barycentric_weights: &[F],
    point: ExtensionAlgebraTarget<D>,
    initial_eval: ExtensionAlgebraTarget<D>,
    initial_partial_prod: ExtensionAlgebraTarget<D>,
) -> (ExtensionAlgebraTarget<D>, ExtensionAlgebraTarget<D>) {
    let n = values.len();
    debug_assert!(n != 0);
    debug_assert!(domain.len() == n);
    debug_assert!(barycentric_weights.len() == n);

    values
        .iter()
        .cloned()
        .zip(domain.iter().cloned())
        .zip(barycentric_weights.iter().cloned())
        .fold(
            (initial_eval, initial_partial_prod),
            |(eval, partial_prod), ((val, x), weight)| {
                let x_target = builder.constant_ext_algebra(F::Extension::from(x).into());
                let weight_target = builder.constant_extension(F::Extension::from(weight));
                let term = builder.sub_ext_algebra(point, x_target);
                let weighted_val = builder.scalar_mul_ext_algebra(weight_target, val);
                let new_eval = builder.mul_ext_algebra(eval, term);
                let new_eval = builder.mul_add_ext_algebra(weighted_val, partial_prod, new_eval);
                let new_partial_prod = builder.mul_ext_algebra(partial_prod, term);
                (new_eval, new_partial_prod)
            },
        )
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use anyhow::Result;
    use plonky2_field::polynomial::PolynomialValues;
    use plonky2_util::log2_strict;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn test_degree_and_wires_minimized() {
        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 2);
        assert_eq!(gate.num_intermediates(), 6);
        assert_eq!(gate.degree(), 2);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 3);
        assert_eq!(gate.num_intermediates(), 3);
        assert_eq!(gate.degree(), 3);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 4);
        assert_eq!(gate.num_intermediates(), 2);
        assert_eq!(gate.degree(), 4);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 5);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 5);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 6);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 5);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 7);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 5);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 3);
        assert_eq!(gate.num_intermediates(), 7);
        assert_eq!(gate.degree(), 3);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 6);
        assert_eq!(gate.num_intermediates(), 2);
        assert_eq!(gate.degree(), 6);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 8);
        assert_eq!(gate.num_intermediates(), 2);
        assert_eq!(gate.degree(), 6);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 9);
        assert_eq!(gate.num_intermediates(), 1);
        assert_eq!(gate.degree(), 9);
    }

    #[test]
    fn wire_indices_degree2() {
        let gate = CosetInterpolationGate::<GoldilocksField, 4> {
            subgroup_bits: 2,
            degree: 2,
            barycentric_weights: barycentric_weights(
                &GoldilocksField::two_adic_subgroup(2)
                    .into_iter()
                    .map(|x| (x, GoldilocksField::ZERO))
                    .collect::<Vec<_>>(),
            ),
            _phantom: PhantomData,
        };

        // The exact indices aren't really important, but we want to make sure we don't have any
        // overlaps or gaps.
        assert_eq!(gate.wire_shift(), 0);
        assert_eq!(gate.wires_value(0), 1..5);
        assert_eq!(gate.wires_value(1), 5..9);
        assert_eq!(gate.wires_value(2), 9..13);
        assert_eq!(gate.wires_value(3), 13..17);
        assert_eq!(gate.wires_evaluation_point(), 17..21);
        assert_eq!(gate.wires_evaluation_value(), 21..25);
        assert_eq!(gate.wires_intermediate_eval(0), 25..29);
        assert_eq!(gate.wires_intermediate_eval(1), 29..33);
        assert_eq!(gate.wires_intermediate_prod(0), 33..37);
        assert_eq!(gate.wires_intermediate_prod(1), 37..41);
        assert_eq!(gate.wires_shifted_evaluation_point(), 41..45);
        assert_eq!(gate.num_wires(), 45);
    }

    #[test]
    fn wire_indices_degree_3() {
        let gate = CosetInterpolationGate::<GoldilocksField, 4> {
            subgroup_bits: 2,
            degree: 3,
            barycentric_weights: barycentric_weights(
                &GoldilocksField::two_adic_subgroup(2)
                    .into_iter()
                    .map(|x| (x, GoldilocksField::ZERO))
                    .collect::<Vec<_>>(),
            ),
            _phantom: PhantomData,
        };

        // The exact indices aren't really important, but we want to make sure we don't have any
        // overlaps or gaps.
        assert_eq!(gate.wire_shift(), 0);
        assert_eq!(gate.wires_value(0), 1..5);
        assert_eq!(gate.wires_value(1), 5..9);
        assert_eq!(gate.wires_value(2), 9..13);
        assert_eq!(gate.wires_value(3), 13..17);
        assert_eq!(gate.wires_evaluation_point(), 17..21);
        assert_eq!(gate.wires_evaluation_value(), 21..25);
        assert_eq!(gate.wires_intermediate_eval(0), 25..29);
        assert_eq!(gate.wires_intermediate_prod(0), 29..33);
        assert_eq!(gate.wires_shifted_evaluation_point(), 33..37);
        assert_eq!(gate.num_wires(), 37);
    }

    #[test]
    fn wire_indices_degree_n() {
        let gate = CosetInterpolationGate::<GoldilocksField, 4> {
            subgroup_bits: 2,
            degree: 4,
            barycentric_weights: barycentric_weights(
                &GoldilocksField::two_adic_subgroup(2)
                    .into_iter()
                    .map(|x| (x, GoldilocksField::ZERO))
                    .collect::<Vec<_>>(),
            ),
            _phantom: PhantomData,
        };

        // The exact indices aren't really important, but we want to make sure we don't have any
        // overlaps or gaps.
        assert_eq!(gate.wire_shift(), 0);
        assert_eq!(gate.wires_value(0), 1..5);
        assert_eq!(gate.wires_value(1), 5..9);
        assert_eq!(gate.wires_value(2), 9..13);
        assert_eq!(gate.wires_value(3), 13..17);
        assert_eq!(gate.wires_evaluation_point(), 17..21);
        assert_eq!(gate.wires_evaluation_value(), 21..25);
        assert_eq!(gate.wires_shifted_evaluation_point(), 25..29);
        assert_eq!(gate.num_wires(), 29);
    }

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(CosetInterpolationGate::new(2));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        for degree in 2..=4 {
            test_eval_fns::<F, C, _, D>(CosetInterpolationGate::with_max_degree(2, degree))?;
        }
        Ok(())
    }

    #[test]
    fn test_gate_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        /// Returns the local wires for an interpolation gate for given coeffs, points and eval point.
        fn get_wires(shift: F, values: PolynomialValues<FF>, eval_point: FF) -> Vec<FF> {
            let domain = F::two_adic_subgroup(log2_strict(values.len()));
            let shifted_eval_point =
                <FF as FieldExtension<2>>::scalar_mul(&eval_point, shift.inverse());
            let weights =
                barycentric_weights(&domain.iter().map(|&x| (x, F::ZERO)).collect::<Vec<_>>());
            let (intermediate_eval, intermediate_prod) = partial_interpolate::<_, D>(
                &domain[..3],
                &values.values[..3],
                &weights[..3],
                shifted_eval_point,
                FF::ZERO,
                FF::ONE,
            );
            let eval = interpolate_over_base_domain::<_, D>(
                &domain,
                &values.values,
                &weights,
                shifted_eval_point,
            );
            let mut v = vec![shift];
            for val in values.values.iter() {
                v.extend(val.0);
            }
            v.extend(eval_point.0);
            v.extend(eval.0);
            v.extend(intermediate_eval.0);
            v.extend(intermediate_prod.0);
            v.extend(shifted_eval_point.0);
            v.iter().map(|&x| x.into()).collect()
        }

        // Get a working row for InterpolationGate.
        let shift = F::rand();
        let values = PolynomialValues::new(core::iter::repeat_with(FF::rand).take(4).collect());
        let eval_point = FF::rand();
        let gate = CosetInterpolationGate::<F, D>::with_max_degree(2, 3);
        let vars = EvaluationVars {
            local_constants: &[],
            local_wires: &get_wires(shift, values, eval_point),
            public_inputs_hash: &HashOut::rand(),
        };

        assert!(
            gate.eval_unfiltered(vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
    }

    #[test]
    fn test_num_wires_constraints() {
        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 8);
        assert_eq!(gate.num_wires(), 47);
        assert_eq!(gate.num_constraints(), 12);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(3, 8);
        assert_eq!(gate.num_wires(), 23);
        assert_eq!(gate.num_constraints(), 4);

        let gate = <CosetInterpolationGate<GoldilocksField, 2>>::with_max_degree(4, 16);
        assert_eq!(gate.num_wires(), 39);
        assert_eq!(gate.num_constraints(), 4);
    }
}

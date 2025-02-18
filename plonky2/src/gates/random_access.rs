#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use anyhow::Result;
use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate for checking that a particular element of a list matches a given value.
#[derive(Copy, Clone, Debug, Default)]
pub struct RandomAccessGate<F: RichField + Extendable<D>, const D: usize> {
    /// Number of bits in the index (log2 of the list size).
    pub bits: usize,

    /// How many separate copies are packed into one gate.
    pub num_copies: usize,

    /// Leftover wires are used as global scratch space to store constants.
    pub num_extra_constants: usize,

    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> RandomAccessGate<F, D> {
    const fn new(num_copies: usize, bits: usize, num_extra_constants: usize) -> Self {
        Self {
            bits,
            num_copies,
            num_extra_constants,
            _phantom: PhantomData,
        }
    }

    pub fn new_from_config(config: &CircuitConfig, bits: usize) -> Self {
        // We can access a list of 2^bits elements.
        let vec_size = 1 << bits;

        // We need `(2 + vec_size) * num_copies` routed wires.
        let max_copies = (config.num_routed_wires / (2 + vec_size)).min(
            // We need `(2 + vec_size + bits) * num_copies` wires in total.
            config.num_wires / (2 + vec_size + bits),
        );

        // Any leftover wires can be used for constants.
        let max_extra_constants = config.num_routed_wires - (2 + vec_size) * max_copies;

        Self::new(
            max_copies,
            bits,
            max_extra_constants.min(config.num_constants),
        )
    }

    /// Length of the list being accessed.
    const fn vec_size(&self) -> usize {
        1 << self.bits
    }

    /// For each copy, a wire containing the claimed index of the element.
    pub(crate) const fn wire_access_index(&self, copy: usize) -> usize {
        debug_assert!(copy < self.num_copies);
        (2 + self.vec_size()) * copy
    }

    /// For each copy, a wire containing the element claimed to be at the index.
    pub(crate) const fn wire_claimed_element(&self, copy: usize) -> usize {
        debug_assert!(copy < self.num_copies);
        (2 + self.vec_size()) * copy + 1
    }

    /// For each copy, wires containing the entire list.
    pub(crate) const fn wire_list_item(&self, i: usize, copy: usize) -> usize {
        debug_assert!(i < self.vec_size());
        debug_assert!(copy < self.num_copies);
        (2 + self.vec_size()) * copy + 2 + i
    }

    const fn start_extra_constants(&self) -> usize {
        (2 + self.vec_size()) * self.num_copies
    }

    const fn wire_extra_constant(&self, i: usize) -> usize {
        debug_assert!(i < self.num_extra_constants);
        self.start_extra_constants() + i
    }

    /// All above wires are routed.
    pub const fn num_routed_wires(&self) -> usize {
        self.start_extra_constants() + self.num_extra_constants
    }

    /// An intermediate wire where the prover gives the (purported) binary decomposition of the
    /// index.
    pub(crate) const fn wire_bit(&self, i: usize, copy: usize) -> usize {
        debug_assert!(i < self.bits);
        debug_assert!(copy < self.num_copies);
        self.num_routed_wires() + copy * self.bits + i
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for RandomAccessGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.bits)?;
        dst.write_usize(self.num_copies)?;
        dst.write_usize(self.num_extra_constants)?;
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let bits = src.read_usize()?;
        let num_copies = src.read_usize()?;
        let num_extra_constants = src.read_usize()?;
        Ok(Self::new(num_copies, bits, num_extra_constants))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        for copy in 0..self.num_copies {
            let access_index = vars.local_wires[self.wire_access_index(copy)];
            let mut list_items = (0..self.vec_size())
                .map(|i| vars.local_wires[self.wire_list_item(i, copy)])
                .collect::<Vec<_>>();
            let claimed_element = vars.local_wires[self.wire_claimed_element(copy)];
            let bits = (0..self.bits)
                .map(|i| vars.local_wires[self.wire_bit(i, copy)])
                .collect::<Vec<_>>();

            // Assert that each bit wire value is indeed boolean.
            for &b in &bits {
                constraints.push(b * (b - F::Extension::ONE));
            }

            // Assert that the binary decomposition was correct.
            let reconstructed_index = bits
                .iter()
                .rev()
                .fold(F::Extension::ZERO, |acc, &b| acc.double() + b);
            constraints.push(reconstructed_index - access_index);

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.
            for b in bits {
                list_items = list_items
                    .iter()
                    .tuples()
                    .map(|(&x, &y)| x + b * (y - x))
                    .collect()
            }

            debug_assert_eq!(list_items.len(), 1);
            constraints.push(list_items[0] - claimed_element);
        }

        constraints.extend(
            (0..self.num_extra_constants)
                .map(|i| vars.local_constants[i] - vars.local_wires[self.wire_extra_constant(i)]),
        );

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let zero = builder.zero_extension();
        let two = builder.two_extension();
        let mut constraints = Vec::with_capacity(self.num_constraints());

        for copy in 0..self.num_copies {
            let access_index = vars.local_wires[self.wire_access_index(copy)];
            let mut list_items = (0..self.vec_size())
                .map(|i| vars.local_wires[self.wire_list_item(i, copy)])
                .collect::<Vec<_>>();
            let claimed_element = vars.local_wires[self.wire_claimed_element(copy)];
            let bits = (0..self.bits)
                .map(|i| vars.local_wires[self.wire_bit(i, copy)])
                .collect::<Vec<_>>();

            // Assert that each bit wire value is indeed boolean.
            for &b in &bits {
                constraints.push(builder.mul_sub_extension(b, b, b));
            }

            // Assert that the binary decomposition was correct.
            let reconstructed_index = bits
                .iter()
                .rev()
                .fold(zero, |acc, &b| builder.mul_add_extension(acc, two, b));
            constraints.push(builder.sub_extension(reconstructed_index, access_index));

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.
            for b in bits {
                list_items = list_items
                    .iter()
                    .tuples()
                    .map(|(&x, &y)| builder.select_ext_generalized(b, y, x))
                    .collect()
            }

            // Check that the one remaining element after the folding is the claimed element.
            debug_assert_eq!(list_items.len(), 1);
            constraints.push(builder.sub_extension(list_items[0], claimed_element));
        }

        // Check the constant values.
        constraints.extend((0..self.num_extra_constants).map(|i| {
            builder.sub_extension(
                vars.local_constants[i],
                vars.local_wires[self.wire_extra_constant(i)],
            )
        }));

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_copies)
            .map(|copy| {
                WitnessGeneratorRef::new(
                    RandomAccessGenerator {
                        row,
                        gate: *self,
                        copy,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.wire_bit(self.bits - 1, self.num_copies - 1) + 1
    }

    fn num_constants(&self) -> usize {
        self.num_extra_constants
    }

    fn degree(&self) -> usize {
        self.bits + 1
    }

    fn num_constraints(&self) -> usize {
        let constraints_per_copy = self.bits + 2;
        self.num_copies * constraints_per_copy + self.num_extra_constants
    }

    fn extra_constant_wires(&self) -> Vec<(usize, usize)> {
        (0..self.num_extra_constants)
            .map(|i| (i, self.wire_extra_constant(i)))
            .collect()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for RandomAccessGate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for copy in 0..self.num_copies {
            let access_index = vars.local_wires[self.wire_access_index(copy)];
            let mut list_items = (0..self.vec_size())
                .map(|i| vars.local_wires[self.wire_list_item(i, copy)])
                .collect::<Vec<_>>();
            let claimed_element = vars.local_wires[self.wire_claimed_element(copy)];
            let bits = (0..self.bits)
                .map(|i| vars.local_wires[self.wire_bit(i, copy)])
                .collect::<Vec<_>>();

            // Assert that each bit wire value is indeed boolean.
            for &b in &bits {
                yield_constr.one(b * (b - F::ONE));
            }

            // Assert that the binary decomposition was correct.
            let reconstructed_index = bits.iter().rev().fold(P::ZEROS, |acc, &b| acc + acc + b);
            yield_constr.one(reconstructed_index - access_index);

            // Repeatedly fold the list, selecting the left or right item from each pair based on
            // the corresponding bit.
            for b in bits {
                list_items = list_items
                    .iter()
                    .tuples()
                    .map(|(&x, &y)| x + b * (y - x))
                    .collect()
            }

            debug_assert_eq!(list_items.len(), 1);
            yield_constr.one(list_items[0] - claimed_element);
        }
        yield_constr.many(
            (0..self.num_extra_constants)
                .map(|i| vars.local_constants[i] - vars.local_wires[self.wire_extra_constant(i)]),
        );
    }
}

#[derive(Debug, Default)]
pub struct RandomAccessGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    gate: RandomAccessGate<F, D>,
    copy: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for RandomAccessGenerator<F, D>
{
    fn id(&self) -> String {
        "RandomAccessGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        let mut deps = vec![local_target(self.gate.wire_access_index(self.copy))];
        for i in 0..self.gate.vec_size() {
            deps.push(local_target(self.gate.wire_list_item(i, self.copy)));
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
        let mut set_local_wire = |column, value| out_buffer.set_wire(local_wire(column), value);

        let copy = self.copy;
        let vec_size = self.gate.vec_size();

        let access_index_f = get_local_wire(self.gate.wire_access_index(copy));
        let access_index = access_index_f.to_canonical_u64() as usize;
        debug_assert!(
            access_index < vec_size,
            "Access index {} is larger than the vector size {}",
            access_index,
            vec_size
        );

        set_local_wire(
            self.gate.wire_claimed_element(copy),
            get_local_wire(self.gate.wire_list_item(access_index, copy)),
        )?;

        for i in 0..self.gate.bits {
            let bit = F::from_bool(((access_index >> i) & 1) != 0);
            set_local_wire(self.gate.wire_bit(i, copy), bit)?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.copy)?;
        self.gate.serialize(dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let copy = src.read_usize()?;
        let gate = RandomAccessGate::<F, D>::deserialize(src, _common_data)?;
        Ok(Self { row, gate, copy })
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use anyhow::Result;
    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(RandomAccessGate::new(4, 4, 1));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(RandomAccessGate::new(4, 4, 1))
    }

    #[test]
    fn test_gate_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        /// Returns the local wires for a random access gate given the vectors, elements to compare,
        /// and indices.
        fn get_wires(
            bits: usize,
            lists: Vec<Vec<F>>,
            access_indices: Vec<usize>,
            claimed_elements: Vec<F>,
            constants: &[F],
        ) -> Vec<FF> {
            let num_copies = lists.len();
            let vec_size = lists[0].len();

            let mut v = Vec::new();
            let mut bit_vals = Vec::new();
            for copy in 0..num_copies {
                let access_index = access_indices[copy];
                v.push(F::from_canonical_usize(access_index));
                v.push(claimed_elements[copy]);
                for j in 0..vec_size {
                    v.push(lists[copy][j]);
                }

                for i in 0..bits {
                    bit_vals.push(F::from_bool(((access_index >> i) & 1) != 0));
                }
            }
            v.extend(constants);
            v.extend(bit_vals);

            v.iter().map(|&x| x.into()).collect()
        }

        let bits = 3;
        let vec_size = 1 << bits;
        let num_copies = 4;
        let lists = (0..num_copies)
            .map(|_| F::rand_vec(vec_size))
            .collect::<Vec<_>>();
        let access_indices = (0..num_copies)
            .map(|_| OsRng.gen_range(0..vec_size))
            .collect::<Vec<_>>();
        let gate = RandomAccessGate::<F, D> {
            bits,
            num_copies,
            num_extra_constants: 1,
            _phantom: PhantomData,
        };
        let constants = F::rand_vec(gate.num_constants());

        let good_claimed_elements = lists
            .iter()
            .zip(&access_indices)
            .map(|(l, &i)| l[i])
            .collect();
        let good_vars = EvaluationVars {
            local_constants: &constants.iter().map(|&x| x.into()).collect::<Vec<_>>(),
            local_wires: &get_wires(
                bits,
                lists.clone(),
                access_indices.clone(),
                good_claimed_elements,
                &constants,
            ),
            public_inputs_hash: &HashOut::rand(),
        };
        let bad_claimed_elements = F::rand_vec(4);
        let bad_vars = EvaluationVars {
            local_constants: &constants.iter().map(|&x| x.into()).collect::<Vec<_>>(),
            local_wires: &get_wires(
                bits,
                lists,
                access_indices,
                bad_claimed_elements,
                &constants,
            ),
            public_inputs_hash: &HashOut::rand(),
        };

        assert!(
            gate.eval_unfiltered(good_vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
        assert!(
            !gate.eval_unfiltered(bad_vars).iter().all(|x| x.is_zero()),
            "Gate constraints are satisfied but should not be."
        );
    }
}

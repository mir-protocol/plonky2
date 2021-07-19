use std::convert::TryInto;
use std::marker::PhantomData;
use std::ops::Range;

use crate::circuit_builder::CircuitBuilder;
use crate::field::extension_field::target::ExtensionTarget;
use crate::field::extension_field::{Extendable, FieldExtension};
use crate::field::field::Field;
use crate::gates::gate::{Gate, GateRef};
use crate::generator::{SimpleGenerator, WitnessGenerator};
use crate::target::Target;
use crate::vars::{EvaluationTargets, EvaluationVars};
use crate::wire::Wire;
use crate::witness::PartialWitness;

/// A gate for inserting a value into a list at a non-deterministic location.
#[derive(Clone, Debug)]
pub(crate) struct InsertionGate<F: Extendable<D>, const D: usize> {
    pub vec_size: usize,
    pub _phantom: PhantomData<F>,
}

impl<F: Extendable<D>, const D: usize> InsertionGate<F, D> {
    pub fn new(vec_size: usize) -> GateRef<F, D> {
        let gate = Self {
            vec_size,
            _phantom: PhantomData,
        };
        GateRef::new(gate)
    }

    pub fn wires_insertion_index(&self) -> usize {
        0
    }

    pub fn wires_element_to_insert(&self) -> Range<usize> {
        1..D + 1
    }

    pub fn wires_original_list_item(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.vec_size);
        let start = (i + 1) * D + 1;
        start..start + D
    }

    fn start_of_output_wires(&self) -> usize {
        (self.vec_size + 1) * D + 1
    }

    pub fn wires_output_list_item(&self, i: usize) -> Range<usize> {
        debug_assert!(i <= self.vec_size);
        let start = self.start_of_output_wires() + i * D;
        start..start + D
    }

    fn start_of_intermediate_wires(&self) -> usize {
        self.start_of_output_wires() + (self.vec_size + 1) * D
    }

    /// An intermediate wire for a dummy variable used to show equality.
    /// The prover sets this to 1/(x-y) if x != y, or to an arbitrary value if
    /// x == y.
    pub fn wires_equality_dummy_for_round_r(&self, r: usize) -> usize {
        self.start_of_intermediate_wires() + r
    }

    // An intermediate wire for the "insert_here" variable (1 if the current index is the index at
    /// which to insert the new value, 0 otherwise).
    pub fn wires_insert_here_for_round_r(&self, r: usize) -> usize {
        self.start_of_intermediate_wires() + (self.vec_size + 1) + r
    }
}

impl<F: Extendable<D>, const D: usize> Gate<F, D> for InsertionGate<F, D> {
    fn id(&self) -> String {
        format!("{:?}<D={}>", self, D)
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let insertion_index = vars.local_wires[self.wires_insertion_index()];
        let list_items = (0..self.vec_size)
            .map(|i| vars.get_local_ext_algebra(self.wires_original_list_item(i)))
            .collect::<Vec<_>>();
        let output_list_items = (0..=self.vec_size)
            .map(|i| vars.get_local_ext_algebra(self.wires_output_list_item(i)))
            .collect::<Vec<_>>();
        let element_to_insert = vars.get_local_ext_algebra(self.wires_element_to_insert());

        let mut constraints = Vec::new();
        let mut already_inserted = F::Extension::ZERO;
        for r in 0..=self.vec_size {
            let cur_index = F::Extension::from_canonical_usize(r);
            let difference = cur_index - insertion_index;
            let equality_dummy = vars.local_wires[self.wires_equality_dummy_for_round_r(r)];
            let insert_here = vars.local_wires[self.wires_insert_here_for_round_r(r)];

            // The two equality constraints.
            constraints.push(difference * equality_dummy - (F::Extension::ONE - insert_here));
            constraints.push(insert_here * difference);

            let mut new_item = element_to_insert * insert_here.into();
            if r > 0 {
                new_item += list_items[r - 1] * already_inserted.into();
            }
            already_inserted += insert_here;
            if r < self.vec_size {
                new_item += list_items[r] * (F::Extension::ONE - already_inserted).into();
            }

            // Output constraint.
            constraints.extend((new_item - output_list_items[r]).to_basefield_array());
        }

        constraints
    }

    fn eval_unfiltered_recursively(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let insertion_index = vars.local_wires[self.wires_insertion_index()];
        let list_items = (0..self.vec_size)
            .map(|i| vars.get_local_ext_algebra(self.wires_original_list_item(i)))
            .collect::<Vec<_>>();
        let output_list_items = (0..=self.vec_size)
            .map(|i| vars.get_local_ext_algebra(self.wires_output_list_item(i)))
            .collect::<Vec<_>>();
        let element_to_insert = vars.get_local_ext_algebra(self.wires_element_to_insert());

        let mut constraints = Vec::new();
        let mut already_inserted = builder.constant_extension(F::Extension::ZERO);
        for r in 0..=self.vec_size {
            let cur_index_ext = F::Extension::from_canonical_usize(r);
            let cur_index = builder.constant_extension(cur_index_ext);

            let difference = builder.sub_extension(cur_index, insertion_index);
            let equality_dummy = vars.local_wires[self.wires_equality_dummy_for_round_r(r)];
            let insert_here = vars.local_wires[self.wires_insert_here_for_round_r(r)];

            // The two equality constraints.
            let prod = builder.mul_extension(difference, equality_dummy);
            let one = builder.constant_extension(F::Extension::ONE);
            let not_insert_here = builder.sub_extension(one, insert_here);
            let first_equality_constraint = builder.sub_extension(prod, not_insert_here);
            constraints.push(first_equality_constraint);

            let second_equality_constraint = builder.mul_extension(insert_here, difference);
            constraints.push(second_equality_constraint);

            let mut new_item = builder.scalar_mul_ext_algebra(insert_here, element_to_insert);
            if r > 0 {
                let to_add = builder.scalar_mul_ext_algebra(already_inserted, list_items[r - 1]);
                new_item = builder.add_ext_algebra(new_item, to_add);
            }
            already_inserted = builder.add_extension(already_inserted, insert_here);
            if r < self.vec_size {
                let not_already_inserted = builder.sub_extension(one, already_inserted);
                let to_add = builder.scalar_mul_ext_algebra(not_already_inserted, list_items[r]);
                new_item = builder.add_ext_algebra(new_item, to_add);
            }

            // Output constraint.
            let diff = builder.sub_ext_algebra(new_item, output_list_items[r]);
            constraints.extend(diff.to_ext_target_array());
        }

        constraints
    }

    fn generators(
        &self,
        gate_index: usize,
        _local_constants: &[F],
    ) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = InsertionGenerator::<F, D> {
            gate_index,
            gate: self.clone(),
        };
        vec![Box::new(gen)]
    }

    fn num_wires(&self) -> usize {
        self.wires_insert_here_for_round_r(self.vec_size) + 1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        (self.vec_size + 1) * (2 + D)
    }
}

#[derive(Debug)]
struct InsertionGenerator<F: Extendable<D>, const D: usize> {
    gate_index: usize,
    gate: InsertionGate<F, D>,
}

impl<F: Extendable<D>, const D: usize> SimpleGenerator<F> for InsertionGenerator<F, D> {
    fn dependencies(&self) -> Vec<Target> {
        let local_target = |input| Target::wire(self.gate_index, input);

        let local_targets = |inputs: Range<usize>| inputs.map(local_target);

        let mut deps = Vec::new();
        deps.push(local_target(self.gate.wires_insertion_index()));
        deps.extend(local_targets(self.gate.wires_element_to_insert()));
        for i in 0..self.gate.vec_size {
            deps.extend(local_targets(self.gate.wires_original_list_item(i)));
        }
        deps
    }

    fn run_once(&self, witness: &PartialWitness<F>) -> PartialWitness<F> {
        let local_wire = |input| Wire {
            gate: self.gate_index,
            input,
        };

        let get_local_wire = |input| witness.get_wire(local_wire(input));

        let get_local_ext = |wire_range: Range<usize>| {
            debug_assert_eq!(wire_range.len(), D);
            let values = wire_range.map(get_local_wire).collect::<Vec<_>>();
            let arr = values.try_into().unwrap();
            F::Extension::from_basefield_array(arr)
        };

        // Compute the new vector and the values for equality_dummy and insert_here
        let vec_size = self.gate.vec_size;
        let orig_vec = (0..vec_size)
            .map(|i| get_local_ext(self.gate.wires_original_list_item(i)))
            .collect::<Vec<_>>();
        let to_insert = get_local_ext(self.gate.wires_element_to_insert());
        let insertion_index_f = get_local_wire(self.gate.wires_insertion_index());

        let insertion_index = insertion_index_f.to_canonical_u64() as usize;
        debug_assert!(
            insertion_index <= vec_size,
            "Insertion index {} is larger than the vector size {}",
            insertion_index,
            vec_size
        );

        let mut new_vec = orig_vec.clone();
        new_vec.insert(insertion_index, to_insert);

        let mut equality_dummy_vals = Vec::new();
        for i in 0..=vec_size {
            equality_dummy_vals.push(if i == insertion_index {
                F::ONE
            } else {
                (F::from_canonical_usize(i) - insertion_index_f).inverse()
            });
        }

        let mut insert_here_vals = vec![F::ZERO; vec_size];
        insert_here_vals.insert(insertion_index, F::ONE);

        let mut result = PartialWitness::<F>::new();
        for i in 0..=vec_size {
            let output_wires = self.gate.wires_output_list_item(i).map(local_wire);
            result.set_ext_wires(output_wires, new_vec[i]);
            let equality_dummy_wire = local_wire(self.gate.wires_equality_dummy_for_round_r(i));
            result.set_wire(equality_dummy_wire, equality_dummy_vals[i]);
            let insert_here_wire = local_wire(self.gate.wires_insert_here_for_round_r(i));
            result.set_wire(insert_here_wire, insert_here_vals[i]);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use crate::field::crandall_field::CrandallField;
    use crate::field::extension_field::quartic::QuarticCrandallField;
    use crate::field::field::Field;
    use crate::gates::gate::Gate;
    use crate::gates::gate_testing::test_low_degree;
    use crate::gates::insertion::InsertionGate;
    use crate::vars::EvaluationVars;

    #[test]
    fn wire_indices() {
        let gate = InsertionGate::<CrandallField, 4> {
            vec_size: 3,
            _phantom: PhantomData,
        };

        assert_eq!(gate.wires_insertion_index(), 0);
        assert_eq!(gate.wires_element_to_insert(), 1..5);
        assert_eq!(gate.wires_original_list_item(0), 5..9);
        assert_eq!(gate.wires_original_list_item(2), 13..17);
        assert_eq!(gate.wires_output_list_item(0), 17..21);
        assert_eq!(gate.wires_output_list_item(3), 29..33);
        assert_eq!(gate.wires_equality_dummy_for_round_r(0), 33);
        assert_eq!(gate.wires_equality_dummy_for_round_r(3), 36);
        assert_eq!(gate.wires_insert_here_for_round_r(0), 37);
        assert_eq!(gate.wires_insert_here_for_round_r(3), 40);
    }

    #[test]
    fn low_degree() {
        type F = CrandallField;
        test_low_degree(InsertionGate::<F, 4>::new(4));
    }

    #[test]
    fn test_gate_constraint() {
        type F = CrandallField;
        type FF = QuarticCrandallField;
        const D: usize = 4;

        /// Returns the local wires for an insertion gate for given the original vector, element to
        /// insert, and index.
        fn get_wires(orig_vec: Vec<FF>, insertion_index: usize, element_to_insert: FF) -> Vec<FF> {
            let vec_size = orig_vec.len();

            let mut v = Vec::new();
            v.push(F::from_canonical_usize(insertion_index));
            v.extend(element_to_insert.0);
            for j in 0..vec_size {
                v.extend(orig_vec[j].0);
            }

            let mut new_vec = orig_vec.clone();
            new_vec.insert(insertion_index, element_to_insert);
            let mut equality_dummy_vals = Vec::new();
            for i in 0..=vec_size {
                equality_dummy_vals.push(if i == insertion_index {
                    F::ONE
                } else {
                    (F::from_canonical_usize(i) - F::from_canonical_usize(insertion_index))
                        .inverse()
                });
            }
            let mut insert_here_vals = vec![F::ZERO; vec_size];
            insert_here_vals.insert(insertion_index, F::ONE);

            for j in 0..=vec_size {
                v.extend(new_vec[j].0);
            }
            v.extend(equality_dummy_vals);
            v.extend(insert_here_vals);

            v.iter().map(|&x| x.into()).collect::<Vec<_>>()
        }

        let orig_vec = vec![FF::rand(); 3];
        let insertion_index = 1;
        let element_to_insert = FF::rand();
        let gate = InsertionGate::<F, D> {
            vec_size: 3,
            _phantom: PhantomData,
        };
        let vars = EvaluationVars {
            local_constants: &[],
            local_wires: &get_wires(orig_vec, insertion_index, element_to_insert),
        };

        assert!(
            gate.eval_unfiltered(vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
    }
}

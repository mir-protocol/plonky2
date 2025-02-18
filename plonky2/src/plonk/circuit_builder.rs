//! Logic for building plonky2 circuits.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::cmp::max;
#[cfg(feature = "std")]
use std::{collections::BTreeMap, sync::Arc};

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use log::{debug, info, warn, Level};
#[cfg(feature = "timing")]
use web_time::Instant;

use crate::field::cosets::get_unique_coset_shifts;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::fft::fft_root_table;
use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::{FriConfig, FriParams};
use crate::gadgets::arithmetic::BaseArithmeticOperation;
use crate::gadgets::arithmetic_extension::ExtensionArithmeticOperation;
use crate::gadgets::polynomial::PolynomialCoeffsExtTarget;
use crate::gates::arithmetic_base::ArithmeticGate;
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::constant::ConstantGate;
use crate::gates::gate::{CurrentSlot, Gate, GateInstance, GateRef};
use crate::gates::lookup::{Lookup, LookupGate};
use crate::gates::lookup_table::LookupTable;
use crate::gates::noop::NoopGate;
use crate::gates::public_input::PublicInputGate;
use crate::gates::selectors::{selector_ends_lookups, selector_polynomials, selectors_lookup};
use crate::hash::hash_types::{HashOut, HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_proofs::MerkleProofTarget;
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{
    ConstantGenerator, CopyGenerator, SimpleGenerator, WitnessGeneratorRef,
};
#[cfg(not(feature = "no_random"))]
use crate::iop::generator::RandomValueGenerator;
use crate::iop::target::{BoolTarget, Target};
use crate::iop::wire::Wire;
use crate::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, MockCircuitData, ProverCircuitData,
    ProverOnlyCircuitData, VerifierCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use crate::plonk::config::{AlgebraicHasher, GenericConfig, GenericHashOut, Hasher};
use crate::plonk::copy_constraint::CopyConstraint;
use crate::plonk::permutation_argument::Forest;
use crate::plonk::plonk_common::PlonkOracle;
use crate::timed;
use crate::util::context_tree::ContextTree;
use crate::util::partial_products::num_partial_products;
use crate::util::timing::TimingTree;
use crate::util::{log2_ceil, log2_strict, transpose, transpose_poly_values};

/// Number of random coins needed for lookups (for each challenge).
/// A coin is a randomly sampled extension field element from the verifier,
/// consisting internally of `CircuitConfig::num_challenges` field elements.
pub const NUM_COINS_LOOKUP: usize = 4;

/// Enum listing the different types of lookup challenges.
/// `ChallengeA` is used for the linear combination of input and output pairs in Sum and LDC.
/// `ChallengeB` is used for the linear combination of input and output pairs in the polynomial RE.
/// `ChallengeAlpha` is used for the running sums: 1/(alpha - combo_i).
/// `ChallengeDelta` is a challenge on which to evaluate the interpolated LUT function.
#[derive(Debug)]
pub enum LookupChallenges {
    ChallengeA = 0,
    ChallengeB = 1,
    ChallengeAlpha = 2,
    ChallengeDelta = 3,
}

/// Structure containing, for each lookup table, the indices of the last lookup row,
/// the last lookup table row and the first lookup table row. Since the rows are in
/// reverse order in the trace, they actually correspond, respectively, to: the indices
/// of the first `LookupGate`, the first `LookupTableGate` and the last `LookupTableGate`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LookupWire {
    /// Index of the last lookup row (i.e. the first `LookupGate`).
    pub last_lu_gate: usize,
    /// Index of the last lookup table row (i.e. the first `LookupTableGate`).
    pub last_lut_gate: usize,
    /// Index of the first lookup table row (i.e. the last `LookupTableGate`).
    pub first_lut_gate: usize,
}

/// Structure used to construct a plonky2 circuit. It provides all the necessary toolkit that,
/// from an initial circuit configuration, will enable one to design a circuit and its associated
/// prover/verifier data.
///
/// # Usage
///
/// ```rust
/// use plonky2::plonk::circuit_data::CircuitConfig;
/// use plonky2::iop::witness::PartialWitness;
/// use plonky2::plonk::circuit_builder::CircuitBuilder;
/// use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
/// use plonky2::field::types::Field;
///
/// // Define parameters for this circuit
/// const D: usize = 2;
/// type C = PoseidonGoldilocksConfig;
/// type F = <C as GenericConfig<D>>::F;
///
/// let config = CircuitConfig::standard_recursion_config();
/// let mut builder = CircuitBuilder::<F, D>::new(config);
///
/// // Build a circuit for the statement: "I know the 100th term
/// // of the Fibonacci sequence, starting from 0 and 1".
/// let initial_a = builder.constant(F::ZERO);
/// let initial_b = builder.constant(F::ONE);
/// let mut prev_target = initial_a;
/// let mut cur_target = initial_b;
/// for _ in 0..99 {
///     // Encode an addition of the two previous terms
///     let temp = builder.add(prev_target, cur_target);
///     // Shift the two previous terms with the new value
///     prev_target = cur_target;
///     cur_target = temp;
/// }
///
/// // The only public input is the result (which is generated).
/// builder.register_public_input(cur_target);
///
/// // Build the circuit
/// let circuit_data = builder.build::<C>();
///
/// // Now compute the witness and generate a proof
/// let mut pw = PartialWitness::new();
///
/// // There are no public inputs to register, as the only one
/// // will be generated while proving the statement.
/// let proof = circuit_data.prove(pw).unwrap();
///
/// // Verify the proof
/// assert!(circuit_data.verify(proof).is_ok());
/// ```
#[derive(Debug)]
pub struct CircuitBuilder<F: RichField + Extendable<D>, const D: usize> {
    /// Circuit configuration to be used by this [`CircuitBuilder`].
    pub config: CircuitConfig,

    /// A domain separator, which is included in the initial Fiat-Shamir seed. This is generally not
    /// needed, but can be used to ensure that proofs for one application are not valid for another.
    /// Defaults to the empty vector.
    domain_separator: Option<Vec<F>>,

    /// The types of gates used in this circuit.
    gates: HashSet<GateRef<F, D>>,

    /// The concrete placement of each gate.
    pub(crate) gate_instances: Vec<GateInstance<F, D>>,

    /// Targets to be made public.
    public_inputs: Vec<Target>,

    /// The next available index for a `VirtualTarget`.
    virtual_target_index: usize,

    copy_constraints: Vec<CopyConstraint>,

    /// A tree of named scopes, used for debugging.
    context_log: ContextTree,

    /// Generators used to generate the witness.
    generators: Vec<WitnessGeneratorRef<F, D>>,

    constants_to_targets: HashMap<F, Target>,
    targets_to_constants: HashMap<Target, F>,

    /// Memoized results of `arithmetic` calls.
    pub(crate) base_arithmetic_results: HashMap<BaseArithmeticOperation<F>, Target>,

    /// Memoized results of `arithmetic_extension` calls.
    pub(crate) arithmetic_results: HashMap<ExtensionArithmeticOperation<F, D>, ExtensionTarget<D>>,

    /// Map between gate type and the current gate of this type with available slots.
    current_slots: HashMap<GateRef<F, D>, CurrentSlot<F, D>>,

    /// List of constant generators used to fill the constant wires.
    constant_generators: Vec<ConstantGenerator<F>>,

    /// Rows for each LUT: [`LookupWire`] contains: first [`LookupGate`], first and last
    /// [LookupTableGate](crate::gates::lookup_table::LookupTableGate).
    lookup_rows: Vec<LookupWire>,

    /// For each LUT index, vector of `(looking_in, looking_out)` pairs.
    lut_to_lookups: Vec<Lookup>,

    // Lookup tables in the form of `Vec<(input_value, output_value)>`.
    luts: Vec<LookupTable>,

    /// Optional common data. When it is `Some(goal_data)`, the `build` function panics if the resulting
    /// common data doesn't equal `goal_data`.
    /// This is used in cyclic recursion.
    pub(crate) goal_common_data: Option<CommonCircuitData<F, D>>,

    /// Optional verifier data that is registered as public inputs.
    /// This is used in cyclic recursion to hold the circuit's own verifier key.
    pub(crate) verifier_data_public_input: Option<VerifierCircuitTarget>,
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Given a [`CircuitConfig`], generate a new [`CircuitBuilder`] instance.
    /// It will also check that the configuration provided is consistent, i.e.
    /// that the different parameters provided can achieve the targeted security
    /// level.
    pub fn new(config: CircuitConfig) -> Self {
        let builder = CircuitBuilder {
            config,
            domain_separator: None,
            gates: HashSet::new(),
            gate_instances: Vec::new(),
            public_inputs: Vec::new(),
            virtual_target_index: 0,
            copy_constraints: Vec::new(),
            context_log: ContextTree::new(),
            generators: Vec::new(),
            constants_to_targets: HashMap::new(),
            targets_to_constants: HashMap::new(),
            base_arithmetic_results: HashMap::new(),
            arithmetic_results: HashMap::new(),
            current_slots: HashMap::new(),
            constant_generators: Vec::new(),
            lookup_rows: Vec::new(),
            lut_to_lookups: Vec::new(),
            luts: Vec::new(),
            goal_common_data: None,
            verifier_data_public_input: None,
        };
        builder.check_config();
        builder
    }

    /// Assert that the configuration used to create this `CircuitBuilder` is consistent,
    /// i.e. that the different parameters meet the targeted security level.
    fn check_config(&self) {
        let &CircuitConfig {
            security_bits,
            fri_config:
                FriConfig {
                    rate_bits,
                    proof_of_work_bits,
                    num_query_rounds,
                    ..
                },
            ..
        } = &self.config;

        // Conjectured FRI security; see the ethSTARK paper.
        let fri_field_bits = F::Extension::order().bits() as usize;
        let fri_query_security_bits = num_query_rounds * rate_bits + proof_of_work_bits as usize;
        let fri_security_bits = fri_field_bits.min(fri_query_security_bits);
        assert!(
            fri_security_bits >= security_bits,
            "FRI params fall short of target security"
        );
    }

    pub fn set_domain_separator(&mut self, separator: Vec<F>) {
        assert!(self.domain_separator.is_none());
        self.domain_separator = Some(separator);
    }

    /// Outputs the number of gates in this circuit.
    pub fn num_gates(&self) -> usize {
        self.gate_instances.len()
    }

    /// Registers the given target as a public input.
    pub fn register_public_input(&mut self, target: Target) {
        self.public_inputs.push(target);
    }

    /// Registers the given targets as public inputs.
    pub fn register_public_inputs(&mut self, targets: &[Target]) {
        targets.iter().for_each(|&t| self.register_public_input(t));
    }

    /// Outputs the number of public inputs in this circuit.
    pub fn num_public_inputs(&self) -> usize {
        self.public_inputs.len()
    }

    /// Adds lookup rows for a lookup table.
    pub fn add_lookup_rows(
        &mut self,
        last_lu_gate: usize,
        last_lut_gate: usize,
        first_lut_gate: usize,
    ) {
        self.lookup_rows.push(LookupWire {
            last_lu_gate,
            last_lut_gate,
            first_lut_gate,
        });
    }

    /// Adds a looking (input, output) pair to the corresponding LUT.
    pub fn update_lookups(&mut self, looking_in: Target, looking_out: Target, lut_index: usize) {
        assert!(
            lut_index < self.lut_to_lookups.len(),
            "The LUT with index {} has not been created. The last LUT is at index {}",
            lut_index,
            self.lut_to_lookups.len() - 1
        );
        self.lut_to_lookups[lut_index].push((looking_in, looking_out));
    }

    /// Outputs the number of lookup tables in this circuit.
    pub fn num_luts(&self) -> usize {
        self.lut_to_lookups.len()
    }

    /// Given an index, outputs the corresponding looking table in the set of tables
    /// used in this circuit, as a sequence of target tuples `(input, output)`.
    pub fn get_lut_lookups(&self, lut_index: usize) -> &[(Target, Target)] {
        &self.lut_to_lookups[lut_index]
    }

    /// Adds a new "virtual" target. This is not an actual wire in the witness, but just a target
    /// that help facilitate witness generation. In particular, a generator can assign a values to a
    /// virtual target, which can then be copied to other (virtual or concrete) targets. When we
    /// generate the final witness (a grid of wire values), these virtual targets will go away.
    pub fn add_virtual_target(&mut self) -> Target {
        let index = self.virtual_target_index;
        self.virtual_target_index += 1;
        Target::VirtualTarget { index }
    }

    /// Adds `n` new "virtual" targets.
    pub fn add_virtual_targets(&mut self, n: usize) -> Vec<Target> {
        (0..n).map(|_i| self.add_virtual_target()).collect()
    }

    /// Adds `N` new "virtual" targets, arranged as an array.
    pub fn add_virtual_target_arr<const N: usize>(&mut self) -> [Target; N] {
        [0; N].map(|_| self.add_virtual_target())
    }

    /// Adds a new `HashOutTarget`.
    pub fn add_virtual_hash(&mut self) -> HashOutTarget {
        HashOutTarget::from(self.add_virtual_target_arr::<4>())
    }

    /// Registers a new `HashOutTarget` as a public input, adding
    /// internally `NUM_HASH_OUT_ELTS` virtual targets.
    pub fn add_virtual_hash_public_input(&mut self) -> HashOutTarget {
        HashOutTarget::from(self.add_virtual_public_input_arr::<4>())
    }

    /// Adds a new `MerkleCapTarget`, consisting in `1 << cap_height` `HashOutTarget`.
    pub fn add_virtual_cap(&mut self, cap_height: usize) -> MerkleCapTarget {
        MerkleCapTarget(self.add_virtual_hashes(1 << cap_height))
    }

    /// Adds `n` new `HashOutTarget` in a vector fashion.
    pub fn add_virtual_hashes(&mut self, n: usize) -> Vec<HashOutTarget> {
        (0..n).map(|_i| self.add_virtual_hash()).collect()
    }

    /// Registers `n` new `HashOutTarget` as public inputs, in a vector fashion.
    pub fn add_virtual_hashes_public_input(&mut self, n: usize) -> Vec<HashOutTarget> {
        (0..n)
            .map(|_i| self.add_virtual_hash_public_input())
            .collect()
    }

    pub(crate) fn add_virtual_merkle_proof(&mut self, len: usize) -> MerkleProofTarget {
        MerkleProofTarget {
            siblings: self.add_virtual_hashes(len),
        }
    }

    pub fn add_virtual_extension_target(&mut self) -> ExtensionTarget<D> {
        ExtensionTarget(self.add_virtual_targets(D).try_into().unwrap())
    }

    pub fn add_virtual_extension_targets(&mut self, n: usize) -> Vec<ExtensionTarget<D>> {
        (0..n)
            .map(|_i| self.add_virtual_extension_target())
            .collect()
    }

    pub(crate) fn add_virtual_poly_coeff_ext(
        &mut self,
        num_coeffs: usize,
    ) -> PolynomialCoeffsExtTarget<D> {
        let coeffs = self.add_virtual_extension_targets(num_coeffs);
        PolynomialCoeffsExtTarget(coeffs)
    }

    pub fn add_virtual_bool_target_unsafe(&mut self) -> BoolTarget {
        BoolTarget::new_unsafe(self.add_virtual_target())
    }

    pub fn add_virtual_bool_target_safe(&mut self) -> BoolTarget {
        let b = BoolTarget::new_unsafe(self.add_virtual_target());
        self.assert_bool(b);
        b
    }

    /// Add a virtual target and register it as a public input.
    pub fn add_virtual_public_input(&mut self) -> Target {
        let t = self.add_virtual_target();
        self.register_public_input(t);
        t
    }

    pub fn add_virtual_public_input_arr<const N: usize>(&mut self) -> [Target; N] {
        let ts = [0; N].map(|_| self.add_virtual_target());
        self.register_public_inputs(&ts);
        ts
    }

    pub fn add_virtual_verifier_data(&mut self, cap_height: usize) -> VerifierCircuitTarget {
        VerifierCircuitTarget {
            constants_sigmas_cap: self.add_virtual_cap(cap_height),
            circuit_digest: self.add_virtual_hash(),
        }
    }

    /// Add a virtual verifier data, register it as a public input and set it to `self.verifier_data_public_input`.
    ///
    /// **WARNING**: Do not register any public input after calling this!
    // TODO: relax this
    pub fn add_verifier_data_public_inputs(&mut self) -> VerifierCircuitTarget {
        assert!(
            self.verifier_data_public_input.is_none(),
            "add_verifier_data_public_inputs only needs to be called once"
        );

        let verifier_data = self.add_virtual_verifier_data(self.config.fri_config.cap_height);
        // The verifier data are public inputs.
        self.register_public_inputs(&verifier_data.circuit_digest.elements);
        for i in 0..self.config.fri_config.num_cap_elements() {
            self.register_public_inputs(&verifier_data.constants_sigmas_cap.0[i].elements);
        }

        self.verifier_data_public_input = Some(verifier_data.clone());
        verifier_data
    }

    /// Adds a gate to the circuit, and returns its index.
    pub fn add_gate<G: Gate<F, D>>(&mut self, gate_type: G, mut constants: Vec<F>) -> usize {
        self.check_gate_compatibility(&gate_type);

        assert!(
            constants.len() <= gate_type.num_constants(),
            "Too many constants."
        );
        constants.resize(gate_type.num_constants(), F::ZERO);

        let row = self.gate_instances.len();

        self.constant_generators
            .extend(gate_type.extra_constant_wires().into_iter().map(
                |(constant_index, wire_index)| ConstantGenerator {
                    row,
                    constant_index,
                    wire_index,
                    constant: F::ZERO, // Placeholder; will be replaced later.
                },
            ));

        // Note that we can't immediately add this gate's generators, because the list of constants
        // could be modified later, i.e. in the case of `ConstantGate`. We will add them later in
        // `build` instead.

        // Register this gate type if we haven't seen it before.
        let gate_ref = GateRef::new(gate_type);
        self.gates.insert(gate_ref.clone());

        self.gate_instances.push(GateInstance {
            gate_ref,
            constants,
        });

        row
    }

    fn check_gate_compatibility<G: Gate<F, D>>(&self, gate: &G) {
        assert!(
            gate.num_wires() <= self.config.num_wires,
            "{:?} requires {} wires, but our CircuitConfig has only {}",
            gate.id(),
            gate.num_wires(),
            self.config.num_wires
        );
        assert!(
            gate.num_constants() <= self.config.num_constants,
            "{:?} requires {} constants, but our CircuitConfig has only {}",
            gate.id(),
            gate.num_constants(),
            self.config.num_constants
        );
    }

    /// Adds a gate type to the set of gates to be used in this circuit. This can be useful
    /// in conditional recursion to uniformize the set of gates of the different circuits.
    pub fn add_gate_to_gate_set(&mut self, gate: GateRef<F, D>) {
        self.gates.insert(gate);
    }

    /// Adds a generator which will copy `src` to `dst`.
    pub fn generate_copy(&mut self, src: Target, dst: Target) {
        self.add_simple_generator(CopyGenerator { src, dst });
    }

    /// Uses Plonk's permutation argument to require that two elements be equal.
    /// Both elements must be routable, otherwise this method will panic.
    ///
    /// For an example of usage, see [`CircuitBuilder::assert_one()`].
    pub fn connect(&mut self, x: Target, y: Target) {
        assert!(
            x.is_routable(&self.config),
            "Tried to route a wire that isn't routable"
        );
        assert!(
            y.is_routable(&self.config),
            "Tried to route a wire that isn't routable"
        );
        self.copy_constraints
            .push(CopyConstraint::new((x, y), self.context_log.open_stack()));
    }

    /// Enforces that the underlying values of two [`Target`] arrays are equal.
    pub fn connect_array<const N: usize>(&mut self, x: [Target; N], y: [Target; N]) {
        for i in 0..N {
            self.connect(x[i], y[i]);
        }
    }

    /// Enforces that two [`ExtensionTarget<D>`] underlying values are equal.
    pub fn connect_extension(&mut self, src: ExtensionTarget<D>, dst: ExtensionTarget<D>) {
        for i in 0..D {
            self.connect(src.0[i], dst.0[i]);
        }
    }

    /// If `condition`, enforces that two routable `Target` values are equal, using Plonk's permutation argument.
    pub fn conditional_assert_eq(&mut self, condition: Target, x: Target, y: Target) {
        let zero = self.zero();
        let diff = self.sub(x, y);
        let constr = self.mul(condition, diff);
        self.connect(constr, zero);
    }

    /// If `condition`, enforces that two `ExtensionTarget<D>` values are equal.
    pub fn conditional_assert_eq_ext(
        &mut self,
        condition: Target,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) {
        for i in 0..D {
            self.conditional_assert_eq(condition, x.0[i], y.0[i]);
        }
    }

    /// Enforces that a routable `Target` value is 0, using Plonk's permutation argument.
    pub fn assert_zero(&mut self, x: Target) {
        let zero = self.zero();
        self.connect(x, zero);
    }

    /// Enforces that a routable `Target` value is 1, using Plonk's permutation argument.
    ///
    /// # Example
    ///
    /// Let say the circuit contains a target `a`, and a target `b` as public input so that the
    /// prover can non-deterministically compute the multiplicative inverse of `a` when generating
    /// a proof.
    ///
    /// One can then add the following constraint in the circuit to enforce that the value provided
    /// by the prover is correct:
    ///
    /// ```ignore
    /// let c = builder.mul(a, b);
    /// builder.assert_one(c);
    /// ```
    pub fn assert_one(&mut self, x: Target) {
        let one = self.one();
        self.connect(x, one);
    }

    pub fn add_generators(&mut self, generators: Vec<WitnessGeneratorRef<F, D>>) {
        self.generators.extend(generators);
    }

    pub fn add_simple_generator<G: SimpleGenerator<F, D>>(&mut self, generator: G) {
        self.generators
            .push(WitnessGeneratorRef::new(generator.adapter()));
    }

    /// Returns a routable target with a value of 0.
    pub fn zero(&mut self) -> Target {
        self.constant(F::ZERO)
    }

    /// Returns a routable target with a value of 1.
    pub fn one(&mut self) -> Target {
        self.constant(F::ONE)
    }

    /// Returns a routable target with a value of 2.
    pub fn two(&mut self) -> Target {
        self.constant(F::TWO)
    }

    /// Returns a routable target with a value of `order() - 1`.
    pub fn neg_one(&mut self) -> Target {
        self.constant(F::NEG_ONE)
    }

    /// Returns a routable boolean target set to false.
    pub fn _false(&mut self) -> BoolTarget {
        BoolTarget::new_unsafe(self.zero())
    }

    /// Returns a routable boolean target set to true.
    pub fn _true(&mut self) -> BoolTarget {
        BoolTarget::new_unsafe(self.one())
    }

    /// Returns a routable target with the given constant value.
    pub fn constant(&mut self, c: F) -> Target {
        if let Some(&target) = self.constants_to_targets.get(&c) {
            // We already have a wire for this constant.
            return target;
        }

        let target = self.add_virtual_target();
        self.constants_to_targets.insert(c, target);
        self.targets_to_constants.insert(target, c);

        target
    }

    /// Returns a vector of routable targets with the given constant values.
    pub fn constants(&mut self, constants: &[F]) -> Vec<Target> {
        constants.iter().map(|&c| self.constant(c)).collect()
    }

    /// Returns a routable target with the given constant boolean value.
    pub fn constant_bool(&mut self, b: bool) -> BoolTarget {
        if b {
            self._true()
        } else {
            self._false()
        }
    }

    /// Returns a routable [`HashOutTarget`].
    pub fn constant_hash(&mut self, h: HashOut<F>) -> HashOutTarget {
        HashOutTarget {
            elements: h.elements.map(|x| self.constant(x)),
        }
    }

    /// Returns a routable [`MerkleCapTarget`].
    pub fn constant_merkle_cap<H: Hasher<F, Hash = HashOut<F>>>(
        &mut self,
        cap: &MerkleCap<F, H>,
    ) -> MerkleCapTarget {
        MerkleCapTarget(cap.0.iter().map(|h| self.constant_hash(*h)).collect())
    }

    pub fn constant_verifier_data<C: GenericConfig<D, F = F>>(
        &mut self,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
    ) -> VerifierCircuitTarget
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        VerifierCircuitTarget {
            constants_sigmas_cap: self.constant_merkle_cap(&verifier_data.constants_sigmas_cap),
            circuit_digest: self.constant_hash(verifier_data.circuit_digest),
        }
    }

    /// If the given target is a constant (i.e. it was created by the `constant(F)` method), returns
    /// its constant value. Otherwise, returns `None`.
    pub fn target_as_constant(&self, target: Target) -> Option<F> {
        self.targets_to_constants.get(&target).cloned()
    }

    /// If the given [`ExtensionTarget`] is a constant (i.e. it was created by the
    /// `constant_extension(F)` method), returns its constant value. Otherwise, returns `None`.
    pub fn target_as_constant_ext(&self, target: ExtensionTarget<D>) -> Option<F::Extension> {
        // Get a Vec of any coefficients that are constant. If we end up with exactly D of them,
        // then the `ExtensionTarget` as a whole is constant.
        let const_coeffs: Vec<F> = target
            .0
            .iter()
            .filter_map(|&t| self.target_as_constant(t))
            .collect();

        if let Ok(d_const_coeffs) = const_coeffs.try_into() {
            Some(F::Extension::from_basefield_array(d_const_coeffs))
        } else {
            None
        }
    }

    pub fn push_context(&mut self, level: log::Level, ctx: &str) {
        self.context_log.push(ctx, level, self.num_gates());
    }

    pub fn pop_context(&mut self) {
        self.context_log.pop(self.num_gates());
    }

    /// Returns the total number of LUTs.
    pub fn get_luts_length(&self) -> usize {
        self.luts.len()
    }

    /// Gets the length of the LUT at index `idx`.
    pub fn get_luts_idx_length(&self, idx: usize) -> usize {
        assert!(
            idx < self.luts.len(),
            "index idx: {} greater than the total number of created LUTS: {}",
            idx,
            self.luts.len()
        );
        self.luts[idx].len()
    }

    /// Checks whether a LUT is already stored in `self.luts`
    pub fn is_stored(&self, lut: LookupTable) -> Option<usize> {
        self.luts.iter().position(|elt| *elt == lut)
    }

    /// Returns the LUT at index `idx`.
    pub fn get_lut(&self, idx: usize) -> LookupTable {
        assert!(
            idx < self.luts.len(),
            "index idx: {} greater than the total number of created LUTS: {}",
            idx,
            self.luts.len()
        );
        self.luts[idx].clone()
    }

    /// Generates a LUT from a function.
    pub fn get_lut_from_fn<T>(f: fn(T) -> T, inputs: &[T]) -> Vec<(T, T)>
    where
        T: Copy,
    {
        inputs.iter().map(|&input| (input, f(input))).collect()
    }

    /// Given a function `f: fn(u16) -> u16`, adds a LUT to the circuit builder.
    pub fn update_luts_from_fn(&mut self, f: fn(u16) -> u16, inputs: &[u16]) -> usize {
        let lut = Arc::new(Self::get_lut_from_fn::<u16>(f, inputs));

        // If the LUT `lut` is already stored in `self.luts`, return its index. Otherwise, append `table` to `self.luts` and return its index.
        if let Some(idx) = self.is_stored(lut.clone()) {
            idx
        } else {
            self.luts.push(lut);
            self.lut_to_lookups.push(vec![]);
            assert!(self.luts.len() == self.lut_to_lookups.len());
            self.luts.len() - 1
        }
    }

    /// Adds a table to the vector of LUTs in the circuit builder, given a list of inputs and table values.
    pub fn update_luts_from_table(&mut self, inputs: &[u16], table: &[u16]) -> usize {
        assert!(
            inputs.len() == table.len(),
            "Inputs and table have incompatible lengths: {} and {}",
            inputs.len(),
            table.len()
        );
        let pairs = inputs
            .iter()
            .copied()
            .zip_eq(table.iter().copied())
            .collect();
        let lut: LookupTable = Arc::new(pairs);

        // If the LUT `lut` is already stored in `self.luts`, return its index. Otherwise, append `table` to `self.luts` and return its index.
        if let Some(idx) = self.is_stored(lut.clone()) {
            idx
        } else {
            self.luts.push(lut);
            self.lut_to_lookups.push(vec![]);
            assert!(self.luts.len() == self.lut_to_lookups.len());
            self.luts.len() - 1
        }
    }

    /// Adds a table to the vector of LUTs in the circuit builder.
    pub fn update_luts_from_pairs(&mut self, table: LookupTable) -> usize {
        // If the LUT `table` is already stored in `self.luts`, return its index. Otherwise, append `table` to `self.luts` and return its index.
        if let Some(idx) = self.is_stored(table.clone()) {
            idx
        } else {
            self.luts.push(table);
            self.lut_to_lookups.push(vec![]);
            assert!(self.luts.len() == self.lut_to_lookups.len());
            self.luts.len() - 1
        }
    }

    /// Find an available slot, of the form `(row, op)` for gate `G` using parameters `params`
    /// and constants `constants`. Parameters are any data used to differentiate which gate should be
    /// used for the given operation.
    pub fn find_slot<G: Gate<F, D> + Clone>(
        &mut self,
        gate: G,
        params: &[F],
        constants: &[F],
    ) -> (usize, usize) {
        let num_gates = self.num_gates();
        let num_ops = gate.num_ops();
        let gate_ref = GateRef::new(gate.clone());
        let gate_slot = self.current_slots.entry(gate_ref.clone()).or_default();
        let slot = gate_slot.current_slot.get(params);
        let (gate_idx, slot_idx) = if let Some(&s) = slot {
            s
        } else {
            self.add_gate(gate, constants.to_vec());
            (num_gates, 0)
        };
        let current_slot = &mut self.current_slots.get_mut(&gate_ref).unwrap().current_slot;
        if slot_idx == num_ops - 1 {
            // We've filled up the slots at this index.
            current_slot.remove(params);
        } else {
            // Increment the slot operation index.
            current_slot.insert(params.to_vec(), (gate_idx, slot_idx + 1));
        }

        (gate_idx, slot_idx)
    }

    fn fri_params(&self, degree_bits: usize) -> FriParams {
        self.config
            .fri_config
            .fri_params(degree_bits, self.config.zero_knowledge)
    }

    /// The number of (base field) `arithmetic` operations that can be performed in a single gate.
    pub(crate) const fn num_base_arithmetic_ops_per_gate(&self) -> usize {
        if self.config.use_base_arithmetic_gate {
            ArithmeticGate::new_from_config(&self.config).num_ops
        } else {
            self.num_ext_arithmetic_ops_per_gate()
        }
    }

    /// The number of `arithmetic_extension` operations that can be performed in a single gate.
    pub(crate) const fn num_ext_arithmetic_ops_per_gate(&self) -> usize {
        ArithmeticExtensionGate::<D>::new_from_config(&self.config).num_ops
    }

    /// The number of polynomial values that will be revealed per opening, both for the "regular"
    /// polynomials and for the Z polynomials. Because calculating these values involves a recursive
    /// dependence (the amount of blinding depends on the degree, which depends on the blinding),
    /// this function takes in an estimate of the degree.
    fn num_blinding_gates(&self, degree_estimate: usize) -> (usize, usize) {
        let degree_bits_estimate = log2_strict(degree_estimate);
        let fri_queries = self.config.fri_config.num_query_rounds;
        let arities: Vec<usize> = self
            .fri_params(degree_bits_estimate)
            .reduction_arity_bits
            .iter()
            .map(|x| 1 << x)
            .collect();
        let total_fri_folding_points: usize = arities.iter().map(|x| x - 1).sum::<usize>();
        let final_poly_coeffs: usize = degree_estimate / arities.iter().product::<usize>();
        let fri_openings = fri_queries * (1 + D * total_fri_folding_points + D * final_poly_coeffs);

        // We add D for openings at zeta.
        let regular_poly_openings = D + fri_openings;
        // We add 2 * D for openings at zeta and g * zeta.
        let z_openings = 2 * D + fri_openings;

        (regular_poly_openings, z_openings)
    }

    /// The number of polynomial values that will be revealed per opening, both for the "regular"
    /// polynomials (which are opened at only one location) and for the Z polynomials (which are
    /// opened at two).
    fn blinding_counts(&self) -> (usize, usize) {
        let num_gates = self.gate_instances.len();
        let mut degree_estimate = 1 << log2_ceil(num_gates);

        loop {
            let (regular_poly_openings, z_openings) = self.num_blinding_gates(degree_estimate);

            // For most polynomials, we add one random element to offset each opened value.
            // But blinding Z is separate. For that, we add two random elements with a copy
            // constraint between them.
            let total_blinding_count = regular_poly_openings + 2 * z_openings;

            if num_gates + total_blinding_count <= degree_estimate {
                return (regular_poly_openings, z_openings);
            }

            // The blinding gates do not fit within our estimated degree; increase our estimate.
            degree_estimate *= 2;
        }
    }

    fn blind_and_pad(&mut self) {
        if self.config.zero_knowledge {
            #[cfg(not(feature = "no_random"))]
            self.blind();
            #[cfg(feature = "no_random")]
            assert!(false, "Cannot use no_random feature with config.zero_knowledge");
        }

        while !self.gate_instances.len().is_power_of_two() {
            self.add_gate(NoopGate, vec![]);
        }
    }

    #[cfg(not(feature = "no_random"))]
    fn blind(&mut self) {
        let (regular_poly_openings, z_openings) = self.blinding_counts();
        info!(
            "Adding {} blinding terms for witness polynomials, and {}*2 for Z polynomials",
            regular_poly_openings, z_openings
        );

        let num_routed_wires = self.config.num_routed_wires;
        let num_wires = self.config.num_wires;

        // For each "regular" blinding factor, we simply add a no-op gate, and insert a random value
        // for each wire.
        for _ in 0..regular_poly_openings {
            let row = self.add_gate(NoopGate, vec![]);
            for w in 0..num_wires {
                self.add_simple_generator(RandomValueGenerator {
                    target: Target::Wire(Wire { row, column: w }),
                });
            }
        }

        // For each z poly blinding factor, we add two new gates with the same random value, and
        // enforce a copy constraint between them.
        // See https://mirprotocol.org/blog/Adding-zero-knowledge-to-Plonk-Halo
        for _ in 0..z_openings {
            let gate_1 = self.add_gate(NoopGate, vec![]);
            let gate_2 = self.add_gate(NoopGate, vec![]);

            for w in 0..num_routed_wires {
                self.add_simple_generator(RandomValueGenerator {
                    target: Target::Wire(Wire {
                        row: gate_1,
                        column: w,
                    }),
                });
                self.generate_copy(
                    Target::Wire(Wire {
                        row: gate_1,
                        column: w,
                    }),
                    Target::Wire(Wire {
                        row: gate_2,
                        column: w,
                    }),
                );
            }
        }
    }

    fn constant_polys(&self) -> Vec<PolynomialValues<F>> {
        let max_constants = self
            .gates
            .iter()
            .map(|g| g.0.num_constants())
            .max()
            .unwrap();
        transpose(
            &self
                .gate_instances
                .iter()
                .map(|g| {
                    let mut consts = g.constants.clone();
                    consts.resize(max_constants, F::ZERO);
                    consts
                })
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(PolynomialValues::new)
        .collect()
    }

    fn sigma_vecs(&self, k_is: &[F], subgroup: &[F]) -> (Vec<PolynomialValues<F>>, Forest) {
        let degree = self.gate_instances.len();
        let degree_log = log2_strict(degree);
        let config = &self.config;
        let mut forest = Forest::new(
            config.num_wires,
            config.num_routed_wires,
            degree,
            self.virtual_target_index,
        );

        for gate in 0..degree {
            for input in 0..config.num_wires {
                forest.add(Target::Wire(Wire {
                    row: gate,
                    column: input,
                }));
            }
        }

        for index in 0..self.virtual_target_index {
            forest.add(Target::VirtualTarget { index });
        }

        for &CopyConstraint { pair: (a, b), .. } in &self.copy_constraints {
            forest.merge(a, b);
        }

        forest.compress_paths();

        let wire_partition = forest.wire_partition();
        (
            wire_partition.get_sigma_polys(degree_log, k_is, subgroup),
            forest,
        )
    }

    pub fn print_gate_counts(&self, min_delta: usize) {
        // Print gate counts for each context.
        self.context_log
            .filter(self.num_gates(), min_delta)
            .print(self.num_gates());

        // Print total count of each gate type.
        debug!("Total gate counts:");
        for gate in self.gates.iter().cloned() {
            let count = self
                .gate_instances
                .iter()
                .filter(|inst| inst.gate_ref == gate)
                .count();
            debug!("- {} instances of {}", count, gate.0.id());
        }
    }

    /// In PLONK's permutation argument, there's a slight chance of division by zero. We can
    /// mitigate this by randomizing some unused witness elements, so if proving fails with
    /// division by zero, the next attempt will have an (almost) independent chance of success.
    /// See <https://github.com/0xPolygonZero/plonky2/issues/456>.
    #[cfg(not(feature = "no_random"))]
    fn randomize_unused_pi_wires(&mut self, pi_gate: usize) {
        for wire in PublicInputGate::wires_public_inputs_hash().end..self.config.num_wires {
            self.add_simple_generator(RandomValueGenerator {
                target: Target::wire(pi_gate, wire),
            });
        }
    }

    /// Builds a "full circuit", with both prover and verifier data.
    pub fn build_with_options<C: GenericConfig<D, F = F>>(
        self,
        commit_to_sigma: bool,
    ) -> CircuitData<F, C, D> {
        let (circuit_data, success) = self.try_build_with_options(commit_to_sigma);
        if !success {
            panic!("Failed to build circuit");
        }
        circuit_data
    }

    pub fn try_build_with_options<C: GenericConfig<D, F = F>>(
        mut self,
        commit_to_sigma: bool,
    ) -> (CircuitData<F, C, D>, bool) {
        let mut timing = TimingTree::new("preprocess", Level::Trace);

        #[cfg(feature = "timing")]
        let start = Instant::now();

        let rate_bits = self.config.fri_config.rate_bits;
        let cap_height = self.config.fri_config.cap_height;
        // Total number of LUTs.
        let num_luts = self.get_luts_length();
        // Hash the public inputs, and route them to a `PublicInputGate` which will enforce that
        // those hash wires match the claimed public inputs.
        let num_public_inputs = self.public_inputs.len();
        let public_inputs_hash =
            self.hash_n_to_hash_no_pad::<C::InnerHasher>(self.public_inputs.clone());
        let pi_gate = self.add_gate(PublicInputGate, vec![]);
        for (&hash_part, wire) in public_inputs_hash
            .elements
            .iter()
            .zip(PublicInputGate::wires_public_inputs_hash())
        {
            self.connect(hash_part, Target::wire(pi_gate, wire))
        }
        #[cfg(not(feature = "no_random"))]
        self.randomize_unused_pi_wires(pi_gate);

        // Place LUT-related gates.
        self.add_all_lookups();

        // Make sure we have enough constant generators. If not, add a `ConstantGate`.
        while self.constants_to_targets.len() > self.constant_generators.len() {
            self.add_gate(
                ConstantGate {
                    num_consts: self.config.num_constants,
                },
                vec![],
            );
        }

        // For each constant-target pair used in the circuit, use a constant generator to fill this target.
        for ((c, t), mut const_gen) in self
            .constants_to_targets
            .clone()
            .into_iter()
            // We need to enumerate constants_to_targets in some deterministic order to ensure that
            // building a circuit is deterministic.
            .sorted_by_key(|(c, _t)| c.to_canonical_u64())
            .zip(self.constant_generators.clone())
        {
            // Set the constant in the constant polynomial.
            self.gate_instances[const_gen.row].constants[const_gen.constant_index] = c;
            // Generate a copy between the target and the routable wire.
            self.connect(Target::wire(const_gen.row, const_gen.wire_index), t);
            // Set the constant in the generator (it's initially set with a dummy value).
            const_gen.set_constant(c);
            self.add_simple_generator(const_gen);
        }

        debug!(
            "Degree before blinding & padding: {}",
            self.gate_instances.len()
        );
        self.blind_and_pad();
        let degree = self.gate_instances.len();
        debug!("Degree after blinding & padding: {}", degree);
        let degree_bits = log2_strict(degree);
        let fri_params = self.fri_params(degree_bits);
        assert!(
            fri_params.total_arities() <= degree_bits + rate_bits - cap_height,
            "FRI total reduction arity is too large.",
        );

        let quotient_degree_factor = self.config.max_quotient_degree_factor;
        let mut gates = self.gates.iter().cloned().collect::<Vec<_>>();
        // Gates need to be sorted by their degrees (and ID to make the ordering deterministic) to compute the selector polynomials.
        gates.sort_unstable_by_key(|g| (g.0.degree(), g.0.id()));
        let (mut constant_vecs, selectors_info) =
            selector_polynomials(&gates, &self.gate_instances, quotient_degree_factor + 1);

        // Get the lookup selectors.
        let num_lookup_selectors = if num_luts != 0 {
            let selector_lookups =
                selectors_lookup(&gates, &self.gate_instances, &self.lookup_rows);
            let selector_ends = selector_ends_lookups(&self.lookup_rows, &self.gate_instances);
            let all_lookup_selectors = [selector_lookups, selector_ends].concat();
            let num_lookup_selectors = all_lookup_selectors.len();
            constant_vecs.extend(all_lookup_selectors);
            num_lookup_selectors
        } else {
            0
        };

        constant_vecs.extend(self.constant_polys());
        let num_constants = constant_vecs.len();

        let subgroup = F::two_adic_subgroup(degree_bits);

        let k_is = get_unique_coset_shifts(degree, self.config.num_routed_wires);
        let (sigma_vecs, forest) = timed!(
            timing,
            "generate sigma polynomials",
            self.sigma_vecs(&k_is, &subgroup)
        );

        // Precompute FFT roots.
        let max_fft_points = 1 << (degree_bits + max(rate_bits, log2_ceil(quotient_degree_factor)));
        let fft_root_table = fft_root_table(max_fft_points);

        let constants_sigmas_commitment = if commit_to_sigma {
            let constants_sigmas_vecs = [constant_vecs, sigma_vecs.clone()].concat();
            PolynomialBatch::<F, C, D>::from_values(
                constants_sigmas_vecs,
                rate_bits,
                PlonkOracle::CONSTANTS_SIGMAS.blinding,
                cap_height,
                &mut timing,
                Some(&fft_root_table),
            )
        } else {
            PolynomialBatch::<F, C, D>::default()
        };

        // Map between gates where not all generators are used and the gate's number of used generators.
        let incomplete_gates = self
            .current_slots
            .values()
            .flat_map(|current_slot| current_slot.current_slot.values().copied())
            .collect::<HashMap<_, _>>();

        // Add gate generators.
        self.add_generators(
            self.gate_instances
                .iter()
                .enumerate()
                .flat_map(|(index, gate)| {
                    let mut gens = gate.gate_ref.0.generators(index, &gate.constants);
                    // Remove unused generators, if any.
                    if let Some(&op) = incomplete_gates.get(&index) {
                        gens.drain(op..);
                    }
                    gens
                })
                .collect(),
        );

        // Index generator indices by their watched targets.
        let mut generator_indices_by_watches = BTreeMap::new();
        for (i, generator) in self.generators.iter().enumerate() {
            for watch in generator.0.watch_list() {
                let watch_index = forest.target_index(watch);
                let watch_rep_index = forest.parents[watch_index];
                generator_indices_by_watches
                    .entry(watch_rep_index)
                    .or_insert_with(Vec::new)
                    .push(i);
            }
        }
        for indices in generator_indices_by_watches.values_mut() {
            indices.dedup();
            indices.shrink_to_fit();
        }

        let num_gate_constraints = gates
            .iter()
            .map(|gate| gate.0.num_constraints())
            .max()
            .expect("No gates?");

        let num_partial_products =
            num_partial_products(self.config.num_routed_wires, quotient_degree_factor);

        let lookup_degree = self.config.max_quotient_degree_factor - 1;
        let num_lookup_polys = if num_luts == 0 {
            0
        } else {
            // There is 1 RE polynomial and multiple Sum/LDC polynomials.
            LookupGate::num_slots(&self.config).div_ceil(lookup_degree) + 1
        };
        let constants_sigmas_cap = constants_sigmas_commitment.merkle_tree.cap.clone();
        let domain_separator = self.domain_separator.unwrap_or_default();
        let domain_separator_digest = C::Hasher::hash_pad(&domain_separator);
        // TODO: This should also include an encoding of gate constraints.
        let circuit_digest_parts = [
            constants_sigmas_cap.flatten(),
            domain_separator_digest.to_vec(),
            vec![
                F::from_canonical_usize(degree_bits),
                /* Add other circuit data here */
            ],
        ];
        let circuit_digest = C::Hasher::hash_no_pad(&circuit_digest_parts.concat());

        let common = CommonCircuitData {
            config: self.config,
            fri_params,
            gates,
            selectors_info,
            quotient_degree_factor,
            num_gate_constraints,
            num_constants,
            num_public_inputs,
            k_is,
            num_partial_products,
            num_lookup_polys,
            num_lookup_selectors,
            luts: self.luts,
        };

        let mut success = true;

        if let Some(goal_data) = self.goal_common_data {
            if goal_data != common {
                warn!("The expected circuit data passed to cyclic recursion method did not match the actual circuit");
                success = false;
            }
        }

        let prover_only = ProverOnlyCircuitData::<F, C, D> {
            generators: self.generators,
            generator_indices_by_watches,
            constants_sigmas_commitment,
            sigmas: transpose_poly_values(sigma_vecs),
            subgroup,
            public_inputs: self.public_inputs,
            representative_map: forest.parents,
            fft_root_table: Some(fft_root_table),
            circuit_digest,
            lookup_rows: self.lookup_rows.clone(),
            lut_to_lookups: self.lut_to_lookups.clone(),
        };

        let verifier_only = VerifierOnlyCircuitData::<C, D> {
            constants_sigmas_cap,
            circuit_digest,
        };

        timing.print();
        #[cfg(feature = "timing")]
        debug!("Building circuit took {}s", start.elapsed().as_secs_f32());
        (
            CircuitData {
                prover_only,
                verifier_only,
                common,
            },
            success,
        )
    }

    /// Builds a "full circuit", with both prover and verifier data.
    pub fn build<C: GenericConfig<D, F = F>>(self) -> CircuitData<F, C, D> {
        self.build_with_options(true)
    }

    pub fn mock_build<C: GenericConfig<D, F = F>>(self) -> MockCircuitData<F, C, D> {
        let circuit_data = self.build_with_options(false);
        MockCircuitData {
            prover_only: circuit_data.prover_only,
            common: circuit_data.common,
        }
    }
    /// Builds a "prover circuit", with data needed to generate proofs but not verify them.
    pub fn build_prover<C: GenericConfig<D, F = F>>(self) -> ProverCircuitData<F, C, D> {
        // TODO: Can skip parts of this.
        let circuit_data = self.build::<C>();
        circuit_data.prover_data()
    }

    /// Builds a "verifier circuit", with data needed to verify proofs but not generate them.
    pub fn build_verifier<C: GenericConfig<D, F = F>>(self) -> VerifierCircuitData<F, C, D> {
        // TODO: Can skip parts of this.
        let circuit_data = self.build::<C>();
        circuit_data.verifier_data()
    }
}

#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::borrow::Borrow;

use anyhow::Result;

use crate::field::extension::{Extendable, FieldExtension, OEF};
use crate::field::types::{Field, Field64};
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::multiplication_extension::MulExtensionGate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator};
use crate::iop::target::{BoolTarget, Target};
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::util::bits_u64;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub fn arithmetic_extension(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
        addend: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        // See if we can determine the result without adding an `ArithmeticGate`.
        if let Some(result) = self.arithmetic_extension_special_cases(
            const_0,
            const_1,
            multiplicand_0,
            multiplicand_1,
            addend,
        ) {
            return result;
        }

        // See if we've already computed the same operation.
        let operation = ExtensionArithmeticOperation {
            const_0,
            const_1,
            multiplicand_0,
            multiplicand_1,
            addend,
        };
        if let Some(&result) = self.arithmetic_results.get(&operation) {
            return result;
        }

        let result = if self.target_as_constant_ext(addend) == Some(F::Extension::ZERO) {
            // If the addend is zero, we use a multiplication gate.
            self.compute_mul_extension_operation(operation)
        } else {
            // Otherwise, we use an arithmetic gate.
            self.compute_arithmetic_extension_operation(operation)
        };
        // Otherwise, we must actually perform the operation using an ArithmeticExtensionGate slot.
        self.arithmetic_results.insert(operation, result);
        result
    }

    fn compute_arithmetic_extension_operation(
        &mut self,
        operation: ExtensionArithmeticOperation<F, D>,
    ) -> ExtensionTarget<D> {
        let gate = ArithmeticExtensionGate::new_from_config(&self.config);
        let constants = vec![operation.const_0, operation.const_1];
        let (gate, i) = self.find_slot(gate, &constants, &constants);
        let wires_multiplicand_0 = ExtensionTarget::from_range(
            gate,
            ArithmeticExtensionGate::<D>::wires_ith_multiplicand_0(i),
        );
        let wires_multiplicand_1 = ExtensionTarget::from_range(
            gate,
            ArithmeticExtensionGate::<D>::wires_ith_multiplicand_1(i),
        );
        let wires_addend =
            ExtensionTarget::from_range(gate, ArithmeticExtensionGate::<D>::wires_ith_addend(i));

        self.connect_extension(operation.multiplicand_0, wires_multiplicand_0);
        self.connect_extension(operation.multiplicand_1, wires_multiplicand_1);
        self.connect_extension(operation.addend, wires_addend);

        ExtensionTarget::from_range(gate, ArithmeticExtensionGate::<D>::wires_ith_output(i))
    }

    fn compute_mul_extension_operation(
        &mut self,
        operation: ExtensionArithmeticOperation<F, D>,
    ) -> ExtensionTarget<D> {
        let gate = MulExtensionGate::new_from_config(&self.config);
        let constants = vec![operation.const_0];
        let (gate, i) = self.find_slot(gate, &constants, &constants);
        let wires_multiplicand_0 =
            ExtensionTarget::from_range(gate, MulExtensionGate::<D>::wires_ith_multiplicand_0(i));
        let wires_multiplicand_1 =
            ExtensionTarget::from_range(gate, MulExtensionGate::<D>::wires_ith_multiplicand_1(i));

        self.connect_extension(operation.multiplicand_0, wires_multiplicand_0);
        self.connect_extension(operation.multiplicand_1, wires_multiplicand_1);

        ExtensionTarget::from_range(gate, MulExtensionGate::<D>::wires_ith_output(i))
    }

    /// Checks for special cases where the value of
    /// `const_0 * multiplicand_0 * multiplicand_1 + const_1 * addend`
    /// can be determined without adding an `ArithmeticGate`.
    fn arithmetic_extension_special_cases(
        &mut self,
        const_0: F,
        const_1: F,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
        addend: ExtensionTarget<D>,
    ) -> Option<ExtensionTarget<D>> {
        let zero = self.zero_extension();

        let mul_0_const = self.target_as_constant_ext(multiplicand_0);
        let mul_1_const = self.target_as_constant_ext(multiplicand_1);
        let addend_const = self.target_as_constant_ext(addend);

        let first_term_zero =
            const_0 == F::ZERO || multiplicand_0 == zero || multiplicand_1 == zero;
        let second_term_zero = const_1 == F::ZERO || addend == zero;

        // If both terms are constant, return their (constant) sum.
        let first_term_const = if first_term_zero {
            Some(F::Extension::ZERO)
        } else if let (Some(x), Some(y)) = (mul_0_const, mul_1_const) {
            Some((x * y).scalar_mul(const_0))
        } else {
            None
        };
        let second_term_const = if second_term_zero {
            Some(F::Extension::ZERO)
        } else {
            addend_const.map(|x| x.scalar_mul(const_1))
        };
        if let (Some(x), Some(y)) = (first_term_const, second_term_const) {
            return Some(self.constant_extension(x + y));
        }

        if first_term_zero && const_1.is_one() {
            return Some(addend);
        }

        if second_term_zero {
            if let Some(x) = mul_0_const {
                if x.scalar_mul(const_0).is_one() {
                    return Some(multiplicand_1);
                }
            }
            if let Some(x) = mul_1_const {
                if x.scalar_mul(const_0).is_one() {
                    return Some(multiplicand_0);
                }
            }
        }

        None
    }

    /// Returns `a*b + c*d + e`.
    pub fn wide_arithmetic_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
        d: ExtensionTarget<D>,
        e: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.inner_product_extension(F::ONE, e, vec![(a, b), (c, d)])
    }

    /// Returns `sum_{(a,b) in vecs} constant * a * b`.
    pub fn inner_product_extension(
        &mut self,
        constant: F,
        starting_acc: ExtensionTarget<D>,
        pairs: Vec<(ExtensionTarget<D>, ExtensionTarget<D>)>,
    ) -> ExtensionTarget<D> {
        let mut acc = starting_acc;
        for (a, b) in pairs {
            acc = self.arithmetic_extension(constant, F::ONE, a, b, acc);
        }
        acc
    }

    pub fn add_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let one = self.one_extension();
        self.arithmetic_extension(F::ONE, F::ONE, one, a, b)
    }

    pub fn add_ext_algebra(
        &mut self,
        mut a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        for i in 0..D {
            a.0[i] = self.add_extension(a.0[i], b.0[i]);
        }
        a
    }

    /// Add `n` `ExtensionTarget`s.
    pub fn add_many_extension<T>(
        &mut self,
        terms: impl IntoIterator<Item = T>,
    ) -> ExtensionTarget<D>
    where
        T: Borrow<ExtensionTarget<D>>,
    {
        terms.into_iter().fold(self.zero_extension(), |acc, t| {
            self.add_extension(acc, *t.borrow())
        })
    }

    pub fn sub_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let one = self.one_extension();
        self.arithmetic_extension(F::ONE, F::NEG_ONE, one, a, b)
    }

    pub fn sub_ext_algebra(
        &mut self,
        mut a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        for i in 0..D {
            a.0[i] = self.sub_extension(a.0[i], b.0[i]);
        }
        a
    }

    pub fn mul_extension_with_const(
        &mut self,
        const_0: F,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let zero = self.zero_extension();
        self.arithmetic_extension(const_0, F::ZERO, multiplicand_0, multiplicand_1, zero)
    }

    pub fn mul_extension(
        &mut self,
        multiplicand_0: ExtensionTarget<D>,
        multiplicand_1: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.mul_extension_with_const(F::ONE, multiplicand_0, multiplicand_1)
    }

    /// Computes `x^2`.
    pub fn square_extension(&mut self, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.mul_extension(x, x)
    }

    /// Computes `x^3`.
    pub fn cube_extension(&mut self, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.mul_many_extension([x, x, x])
    }

    /// Returns `a * b + c`.
    pub fn mul_add_ext_algebra(
        &mut self,
        a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
        c: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        let mut inner = vec![vec![]; D];
        let mut inner_w = vec![vec![]; D];
        for i in 0..D {
            for j in 0..D - i {
                inner[(i + j) % D].push((a.0[i], b.0[j]));
            }
            for j in D - i..D {
                inner_w[(i + j) % D].push((a.0[i], b.0[j]));
            }
        }
        let res = inner_w
            .into_iter()
            .zip(inner)
            .zip(c.0)
            .map(|((pairs_w, pairs), ci)| {
                let acc = self.inner_product_extension(F::Extension::W, ci, pairs_w);
                self.inner_product_extension(F::ONE, acc, pairs)
            })
            .collect::<Vec<_>>();

        ExtensionAlgebraTarget(res.try_into().unwrap())
    }

    /// Returns `a * b`.
    pub fn mul_ext_algebra(
        &mut self,
        a: ExtensionAlgebraTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        let zero = self.zero_ext_algebra();
        self.mul_add_ext_algebra(a, b, zero)
    }

    /// Multiply `n` `ExtensionTarget`s.
    pub fn mul_many_extension<T>(
        &mut self,
        terms: impl IntoIterator<Item = T>,
    ) -> ExtensionTarget<D>
    where
        T: Borrow<ExtensionTarget<D>>,
    {
        terms.into_iter().fold(self.one_extension(), |acc, t| {
            self.mul_extension(acc, *t.borrow())
        })
    }

    /// Like `mul_add`, but for `ExtensionTarget`s.
    pub fn mul_add_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.arithmetic_extension(F::ONE, F::ONE, a, b, c)
    }

    /// Like `add_const`, but for `ExtensionTarget`s.
    pub fn add_const_extension(&mut self, x: ExtensionTarget<D>, c: F) -> ExtensionTarget<D> {
        let c = self.constant_extension(c.into());
        self.add_extension(x, c)
    }

    /// Like `mul_const`, but for `ExtensionTarget`s.
    pub fn mul_const_extension(&mut self, c: F, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        let c = self.constant_extension(c.into());
        self.mul_extension(c, x)
    }

    /// Like `mul_const_add`, but for `ExtensionTarget`s.
    pub fn mul_const_add_extension(
        &mut self,
        c: F,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let c = self.constant_extension(c.into());
        self.mul_add_extension(c, x, y)
    }

    /// Like `mul_add`, but for `ExtensionTarget`s.
    pub fn scalar_mul_add_extension(
        &mut self,
        a: Target,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let a_ext = self.convert_to_ext(a);
        self.arithmetic_extension(F::ONE, F::ONE, a_ext, b, c)
    }

    /// Like `mul_sub`, but for `ExtensionTarget`s.
    pub fn mul_sub_extension(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        self.arithmetic_extension(F::ONE, F::NEG_ONE, a, b, c)
    }

    /// Like `mul_sub`, but for `ExtensionTarget`s.
    pub fn scalar_mul_sub_extension(
        &mut self,
        a: Target,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let a_ext = self.convert_to_ext(a);
        self.arithmetic_extension(F::ONE, F::NEG_ONE, a_ext, b, c)
    }

    /// Returns `a * b`, where `b` is in the extension field and `a` is in the base field.
    pub fn scalar_mul_ext(&mut self, a: Target, b: ExtensionTarget<D>) -> ExtensionTarget<D> {
        let a_ext = self.convert_to_ext(a);
        self.mul_extension(a_ext, b)
    }

    /// Returns `a * b + c`, where `b, c` are in the extension algebra and `a` in the extension field.
    pub fn scalar_mul_add_ext_algebra(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionAlgebraTarget<D>,
        mut c: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        for i in 0..D {
            c.0[i] = self.mul_add_extension(a, b.0[i], c.0[i]);
        }
        c
    }

    /// Returns `a * b`, where `b` is in the extension algebra and `a` in the extension field.
    pub fn scalar_mul_ext_algebra(
        &mut self,
        a: ExtensionTarget<D>,
        b: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D> {
        let zero = self.zero_ext_algebra();
        self.scalar_mul_add_ext_algebra(a, b, zero)
    }

    /// Exponentiates `base` to the power of exponent expressed as `exponent_bits`.
    pub fn exp_extension_from_bits(
        &mut self,
        mut base: ExtensionTarget<D>,
        exponent_bits: &[BoolTarget],
    ) -> ExtensionTarget<D> {
        let mut res = self.one_extension();
        for i in 0..exponent_bits.len() {
            let new_res = self.mul_extension(res, base);
            res = self.select_ext(exponent_bits[i], new_res, res);
            base = self.mul_extension(base, base);
        }
        res
    }

    /// Exponentiate `base` to the power of `2^power_log`.
    // TODO: Test
    pub fn exp_power_of_2_extension(
        &mut self,
        mut base: ExtensionTarget<D>,
        power_log: usize,
    ) -> ExtensionTarget<D> {
        for _ in 0..power_log {
            base = self.square_extension(base);
        }
        base
    }

    /// Exponentiate `base` to the power of a known `exponent`.
    // TODO: Test
    pub fn exp_u64_extension(
        &mut self,
        base: ExtensionTarget<D>,
        exponent: u64,
    ) -> ExtensionTarget<D> {
        match exponent {
            0 => return self.one_extension(),
            1 => return base,
            2 => return self.square_extension(base),
            3 => return self.cube_extension(base),
            _ => (),
        }
        let mut current = base;
        let mut product = self.one_extension();

        for j in 0..bits_u64(exponent) {
            if j != 0 {
                current = self.square_extension(current);
            }
            if ((exponent >> j) & 1) != 0 {
                product = self.mul_extension(product, current);
            }
        }
        product
    }

    /// Computes `x / y`. Results in an unsatisfiable instance if `y = 0`.
    pub fn div_extension(
        &mut self,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let zero = self.zero_extension();
        self.div_add_extension(x, y, zero)
    }

    /// Computes ` x / y + z`.
    pub fn div_add_extension(
        &mut self,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
        z: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let inv = self.add_virtual_extension_target();
        let one = self.one_extension();
        self.add_simple_generator(QuotientGeneratorExtension {
            numerator: one,
            denominator: y,
            quotient: inv,
        });

        // Enforce that y times its purported inverse equals 1.
        let y_inv = self.mul_extension(y, inv);
        self.connect_extension(y_inv, one);

        self.mul_add_extension(x, inv, z)
    }

    /// Computes `1 / x`. Results in an unsatisfiable instance if `x = 0`.
    pub fn inverse_extension(&mut self, x: ExtensionTarget<D>) -> ExtensionTarget<D> {
        let one = self.one_extension();
        self.div_extension(one, x)
    }
}

#[derive(Debug, Default)]
pub struct QuotientGeneratorExtension<const D: usize> {
    numerator: ExtensionTarget<D>,
    denominator: ExtensionTarget<D>,
    quotient: ExtensionTarget<D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for QuotientGeneratorExtension<D>
{
    fn id(&self) -> String {
        "QuotientGeneratorExtension".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let mut deps = self.numerator.to_target_array().to_vec();
        deps.extend(self.denominator.to_target_array());
        deps
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let num = witness.get_extension_target(self.numerator);
        let dem = witness.get_extension_target(self.denominator);
        let quotient = num / dem;
        out_buffer.set_extension_target(self.quotient, quotient)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target_ext(self.numerator)?;
        dst.write_target_ext(self.denominator)?;
        dst.write_target_ext(self.quotient)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let numerator = src.read_target_ext()?;
        let denominator = src.read_target_ext()?;
        let quotient = src.read_target_ext()?;
        Ok(Self {
            numerator,
            denominator,
            quotient,
        })
    }
}

/// An iterator over the powers of a certain base element `b`: `b^0, b^1, b^2, ...`.
#[derive(Clone, Debug)]
pub struct PowersTarget<const D: usize> {
    base: ExtensionTarget<D>,
    current: ExtensionTarget<D>,
}

impl<const D: usize> PowersTarget<D> {
    pub fn next<F: RichField + Extendable<D>>(
        &mut self,
        builder: &mut CircuitBuilder<F, D>,
    ) -> ExtensionTarget<D> {
        let result = self.current;
        self.current = builder.mul_extension(self.base, self.current);
        result
    }

    pub fn repeated_frobenius<F: RichField + Extendable<D>>(
        self,
        k: usize,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self {
        let Self { base, current } = self;
        Self {
            base: base.repeated_frobenius(k, builder),
            current: current.repeated_frobenius(k, builder),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub fn powers(&mut self, base: ExtensionTarget<D>) -> PowersTarget<D> {
        PowersTarget {
            base,
            current: self.one_extension(),
        }
    }
}

/// Represents an extension arithmetic operation in the circuit. Used to memoize results.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct ExtensionArithmeticOperation<F: Field64 + Extendable<D>, const D: usize> {
    const_0: F,
    const_1: F,
    multiplicand_0: ExtensionTarget<D>,
    multiplicand_1: ExtensionTarget<D>,
    addend: ExtensionTarget<D>,
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use anyhow::Result;

    use crate::field::extension::algebra::ExtensionAlgebra;
    use crate::field::types::Sample;
    use crate::iop::ext_target::ExtensionAlgebraTarget;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    #[test]
    fn test_mul_many() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_config();

        let mut pw = PartialWitness::<F>::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let vs = FF::rand_vec(3);
        let ts = builder.add_virtual_extension_targets(3);
        for (&v, &t) in vs.iter().zip(&ts) {
            pw.set_extension_target(t, v)?;
        }
        let mul0 = builder.mul_many_extension(&ts);
        let mul1 = {
            let mut acc = builder.one_extension();
            for &t in &ts {
                acc = builder.mul_extension(acc, t);
            }
            acc
        };
        let mul2 = builder.constant_extension(vs.into_iter().product());

        builder.connect_extension(mul0, mul1);
        builder.connect_extension(mul1, mul2);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    #[test]
    fn test_div_extension() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_zk_config();

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let x = FF::rand();
        let y = FF::rand();
        let z = x / y;
        let xt = builder.constant_extension(x);
        let yt = builder.constant_extension(y);
        let zt = builder.constant_extension(z);
        let comp_zt = builder.div_extension(xt, yt);
        builder.connect_extension(zt, comp_zt);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    #[test]
    fn test_mul_algebra() -> Result<()> {
        const D: usize = 2;
        type C = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        let config = CircuitConfig::standard_recursion_config();

        let mut pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let xt =
            ExtensionAlgebraTarget(builder.add_virtual_extension_targets(D).try_into().unwrap());
        let yt =
            ExtensionAlgebraTarget(builder.add_virtual_extension_targets(D).try_into().unwrap());
        let zt =
            ExtensionAlgebraTarget(builder.add_virtual_extension_targets(D).try_into().unwrap());
        let comp_zt = builder.mul_ext_algebra(xt, yt);
        for i in 0..D {
            builder.connect_extension(zt.0[i], comp_zt.0[i]);
        }

        let x = ExtensionAlgebra::<FF, D>(FF::rand_array());
        let y = ExtensionAlgebra::<FF, D>(FF::rand_array());
        let z = x * y;
        for i in 0..D {
            pw.set_extension_target(xt.0[i], x.0[i])?;
            pw.set_extension_target(yt.0[i], y.0[i])?;
            pw.set_extension_target(zt.0[i], z.0[i])?;
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }
}

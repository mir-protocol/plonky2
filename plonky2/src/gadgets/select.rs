use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Selects `x` or `y` based on `b`, i.e., this returns `if b { x } else { y }`.
    pub fn select_ext(
        &mut self,
        b: BoolTarget,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let b_ext = self.convert_to_ext(b.target);
        self.select_ext_generalized(b_ext, x, y)
    }

    /// Like `select_ext`, but accepts a condition input which does not necessarily have to be
    /// binary. In this case, it computes the arithmetic generalization of `if b { x } else { y }`,
    /// i.e. `bx - (by-y)`.
    pub fn select_ext_generalized(
        &mut self,
        b: ExtensionTarget<D>,
        x: ExtensionTarget<D>,
        y: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let tmp = self.mul_sub_extension(b, y, y);
        self.mul_sub_extension(b, x, tmp)
    }

    /// See `select_ext`.
    pub fn select(&mut self, b: BoolTarget, x: Target, y: Target) -> Target {
        let tmp = self.mul_sub(b.target, y, y);
        self.mul_sub(b.target, x, tmp)
    }
}

#[cfg(test)]
#[cfg(not(feature = "no_random"))]
mod tests {
    use anyhow::Result;

    use crate::field::types::Sample;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    #[test]
    fn test_select() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;
        let config = CircuitConfig::standard_recursion_config();
        let mut pw = PartialWitness::<F>::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let (x, y) = (FF::rand(), FF::rand());
        let xt = builder.add_virtual_extension_target();
        let yt = builder.add_virtual_extension_target();
        let truet = builder._true();
        let falset = builder._false();

        pw.set_extension_target(xt, x)?;
        pw.set_extension_target(yt, y)?;

        let should_be_x = builder.select_ext(truet, xt, yt);
        let should_be_y = builder.select_ext(falset, xt, yt);

        builder.connect_extension(should_be_x, xt);
        builder.connect_extension(should_be_y, yt);

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }
}

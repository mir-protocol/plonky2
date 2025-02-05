use core::ops::Mul;

pub trait Square {
    fn square(&self) -> Self;
}

#[cfg(nightly)]
impl<F: Mul<F, Output = Self> + Copy> Square for F {
    default fn square(&self) -> Self {
        *self * *self
    }
}

#[cfg(not(nightly))]
impl<F: Mul<F, Output = Self> + Copy> Square for F {
    fn square(&self) -> Self {
        *self * *self
    }
}

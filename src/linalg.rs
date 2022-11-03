pub trait Dot<Rhs> {
    type Output;
    fn dot(&self, rhs: Rhs) -> Self::Output;
}

pub trait Dot<'a, Rhs> {
    type Output;
    fn dot(&'a self, rhs: Rhs) -> Self::Output;
}

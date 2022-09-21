use core::any::{Any, TypeId};

pub fn type_eq<T, U>() -> bool
where
    T: Any + ?Sized,
    U: Any + ?Sized,
{
    TypeId::of::<T>() == TypeId::of::<U>()
}

pub trait True {}

pub struct If<const B: bool>;

impl True for If<true> {}

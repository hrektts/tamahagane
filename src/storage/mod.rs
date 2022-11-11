mod cow;
mod owned;
mod routine;
mod shared;
mod view;

#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, sync::Arc, vec::Vec};
use core::iter::FromIterator;
#[cfg(feature = "std")]
use std::{borrow::Cow, sync::Arc};

use num_traits::{One, Zero};

pub trait Storage {
    type Elem: Clone;
    type Cow<'a>: FromIterator<<Self as Storage>::Elem> + StorageMut<Elem = <Self as Storage>::Elem>
    where
        Self: 'a,
        <Self as Storage>::Elem: 'a,
    = StorageBase<Cow<'a, [<Self as Storage>::Elem]>>;
    type Owned: FromIterator<<Self as Storage>::Elem>
        + StorageMut<Elem = <Self as Storage>::Elem>
        + StorageOwned<Elem = <Self as Storage>::Elem> = StorageBase<Vec<<Self as Storage>::Elem>>;
    type Shared: StorageOwned<Elem = <Self as Storage>::Elem> =
        StorageBase<Arc<Vec<<Self as Storage>::Elem>>>;
    type View<'a>: Storage<Elem = <Self as Storage>::Elem>
    where
        Self: 'a,
        <Self as Storage>::Elem: 'a,
    = StorageBase<&'a [<Self as Storage>::Elem]>;
    fn as_ptr(&self) -> *const <Self as Storage>::Elem;
    fn as_slice(&self) -> &[<Self as Storage>::Elem];
    fn cow(&self) -> <Self as Storage>::Cow<'_>;
    fn view(&self) -> <Self as Storage>::View<'_>;
}

pub trait StorageMut: Storage {
    type ViewMut<'a>: Storage<Elem = <Self as Storage>::Elem>
        + StorageMut<Elem = <Self as Storage>::Elem>
    where
        Self: 'a,
        <Self as Storage>::Elem: 'a,
    = StorageBase<&'a mut [<Self as Storage>::Elem]>;
    fn as_mut_ptr(&mut self) -> *mut <Self as Storage>::Elem;
    fn as_mut_slice(&mut self) -> &mut [<Self as Storage>::Elem];
    fn view_mut(&mut self) -> <Self as StorageMut>::ViewMut<'_>;
}

pub trait StorageOwned: FromIterator<<Self as Storage>::Elem> + Storage {
    fn allocate_uninitialized(len: usize) -> Self;
    fn ones(len: usize) -> Self
    where
        <Self as Storage>::Elem: One;
    fn zeros(len: usize) -> Self
    where
        <Self as Storage>::Elem: Zero;
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct StorageBase<B>(B);

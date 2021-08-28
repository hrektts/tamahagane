mod cow;
mod owned;
mod routine;
mod shared;
mod view;

use core::iter::FromIterator;

use num_traits::{One, Zero};

pub trait Storage<T>
where
    T: Clone,
{
    type Cow<'a, U>: FromIterator<U> + StorageMut<U>
    where
        U: Clone + 'a;
    type Owned<U>: FromIterator<U> + StorageMut<U> + StorageOwned<U>
    where
        U: Clone;
    type Shared<U>: StorageOwned<U>
    where
        U: Clone;
    type View<'a, U>: Storage<U>
    where
        U: Clone + 'a;
    fn as_ptr(&self) -> *const T;
    fn as_slice(&self) -> &[T];
    fn cow(&self) -> Self::Cow<'_, T>;
    fn view(&self) -> Self::View<'_, T>;
}

pub trait StorageMut<T>: Storage<T>
where
    T: Clone,
{
    type ViewMut<'a, U>: Storage<U> + StorageMut<U>
    where
        U: Clone + 'a;
    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_mut_slice(&mut self) -> &mut [T];
    fn view_mut(&mut self) -> Self::ViewMut<'_, T>;
}

pub trait StorageOwned<T>: FromIterator<T> + Storage<T>
where
    T: Clone,
{
    fn allocate_uninitialized(len: usize) -> Self;
    fn ones(len: usize) -> Self
    where
        T: One;
    fn zeros(len: usize) -> Self
    where
        T: Zero;
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct StorageImpl<B>(B);

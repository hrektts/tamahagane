#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, vec::Vec};
use core::iter::FromIterator;
#[cfg(feature = "std")]
use std::borrow::Cow;

use num_traits::{One, Zero};

use super::{routine, Storage, StorageImpl, StorageMut, StorageOwned};

impl<T> From<Vec<T>> for StorageImpl<Vec<T>> {
    fn from(data: Vec<T>) -> Self {
        Self(data)
    }
}

impl<T> FromIterator<T> for StorageImpl<Vec<T>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(FromIterator::from_iter(iter))
    }
}

impl<T> Storage for StorageImpl<Vec<T>>
where
    T: Clone,
{
    type Elem = T;

    fn as_ptr(&self) -> *const <Self as Storage>::Elem {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[<Self as Storage>::Elem] {
        self.0.as_slice()
    }

    fn cow(&self) -> <Self as Storage>::Cow<'_> {
        StorageImpl(Cow::Borrowed(self.0.as_slice()))
    }

    fn view(&self) -> <Self as Storage>::View<'_> {
        StorageImpl(self.0.as_slice())
    }
}

impl<T> StorageMut for StorageImpl<Vec<T>>
where
    T: Clone,
{
    fn as_mut_ptr(&mut self) -> *mut <Self as Storage>::Elem {
        self.0.as_mut_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [<Self as Storage>::Elem] {
        self.0.as_mut_slice()
    }

    fn view_mut(&mut self) -> <Self as StorageMut>::ViewMut<'_> {
        StorageImpl(self.0.as_mut_slice())
    }
}

impl<T> StorageOwned for StorageImpl<Vec<T>>
where
    T: Clone,
{
    fn allocate_uninitialized(len: usize) -> Self {
        let buf = routine::create_uninitialized_buf(len);
        StorageImpl(buf)
    }

    fn ones(len: usize) -> Self
    where
        <Self as Storage>::Elem: One,
    {
        let buf = routine::create_buf(len, T::one());
        StorageImpl(buf)
    }

    fn zeros(len: usize) -> Self
    where
        <Self as Storage>::Elem: Zero,
    {
        let buf = routine::create_buf(len, T::zero());
        StorageImpl(buf)
    }
}

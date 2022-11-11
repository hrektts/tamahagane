#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, sync::Arc, vec::Vec};
use core::iter::FromIterator;
#[cfg(feature = "std")]
use std::{borrow::Cow, sync::Arc};

use num_traits::{One, Zero};

use super::{routine, Storage, StorageBase, StorageOwned};

impl<T> From<Vec<T>> for StorageBase<Arc<Vec<T>>> {
    fn from(data: Vec<T>) -> Self {
        Self(Arc::new(data))
    }
}

impl<T> FromIterator<T> for StorageBase<Arc<Vec<T>>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(Arc::new(FromIterator::from_iter(iter)))
    }
}

impl<T> Storage for StorageBase<Arc<Vec<T>>>
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
        StorageBase(Cow::Borrowed(self.0.as_slice()))
    }

    fn view(&self) -> <Self as Storage>::View<'_> {
        StorageBase(self.0.as_slice())
    }
}

impl<T> StorageOwned for StorageBase<Arc<Vec<T>>>
where
    T: Clone,
{
    fn allocate_uninitialized(len: usize) -> Self {
        let buf = routine::create_uninitialized_buf(len);
        StorageBase(Arc::new(buf))
    }

    fn ones(len: usize) -> Self
    where
        <Self as Storage>::Elem: One,
    {
        let buf = routine::create_buf(len, <Self as Storage>::Elem::one());
        StorageBase(Arc::new(buf))
    }

    fn zeros(len: usize) -> Self
    where
        <Self as Storage>::Elem: Zero,
    {
        let buf = routine::create_buf(len, <Self as Storage>::Elem::zero());
        StorageBase(Arc::new(buf))
    }
}

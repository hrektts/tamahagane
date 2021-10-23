#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, sync::Arc, vec::Vec};
use core::iter::FromIterator;
#[cfg(feature = "std")]
use std::{borrow::Cow, sync::Arc};

use num_traits::{One, Zero};

use super::{routine, Storage, StorageImpl, StorageOwned};

impl<T> From<Vec<T>> for StorageImpl<Arc<Vec<T>>> {
    fn from(data: Vec<T>) -> Self {
        Self(Arc::new(data))
    }
}

impl<T> FromIterator<T> for StorageImpl<Arc<Vec<T>>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(Arc::new(FromIterator::from_iter(iter)))
    }
}

impl<T> Storage<T> for StorageImpl<Arc<Vec<T>>>
where
    T: Clone,
{
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    fn cow(&self) -> Self::Cow<'_, T> {
        StorageImpl(Cow::Borrowed(self.0.as_slice()))
    }

    fn view(&self) -> Self::View<'_, T> {
        StorageImpl(self.0.as_slice())
    }
}

impl<T> StorageOwned<T> for StorageImpl<Arc<Vec<T>>>
where
    T: Clone,
{
    fn allocate_uninitialized(len: usize) -> Self {
        let buf = routine::create_uninitialized_buf(len);
        StorageImpl(Arc::new(buf))
    }

    fn ones(len: usize) -> Self
    where
        T: One,
    {
        let buf = routine::create_buf(len, T::one());
        StorageImpl(Arc::new(buf))
    }

    fn zeros(len: usize) -> Self
    where
        T: Zero,
    {
        let buf = routine::create_buf(len, T::zero());
        StorageImpl(Arc::new(buf))
    }
}

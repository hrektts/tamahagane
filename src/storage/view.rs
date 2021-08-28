#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::{borrow::Cow, sync::Arc};

use super::{Storage, StorageImpl, StorageMut};

impl<T> Storage<T> for StorageImpl<&[T]>
where
    T: Clone,
{
    type Cow<'a, U>
    where
        U: Clone + 'a,
    = StorageImpl<Cow<'a, [U]>>;
    type Owned<U>
    where
        U: Clone,
    = StorageImpl<Vec<U>>;
    type Shared<U>
    where
        U: Clone,
    = StorageImpl<Arc<Vec<U>>>;
    type View<'a, U>
    where
        U: Clone + 'a,
    = StorageImpl<&'a [U]>;

    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[T] {
        self.0
    }

    fn cow(&self) -> Self::Cow<'_, T> {
        StorageImpl(Cow::Borrowed(self.0))
    }

    fn view(&self) -> Self::View<'_, T> {
        StorageImpl(self.0)
    }
}

impl<T> Storage<T> for StorageImpl<&mut [T]>
where
    T: Clone,
{
    type Cow<'a, U>
    where
        U: Clone + 'a,
    = StorageImpl<Cow<'a, [U]>>;
    type Owned<U>
    where
        U: Clone,
    = StorageImpl<Vec<U>>;
    type Shared<U>
    where
        U: Clone,
    = StorageImpl<Arc<Vec<U>>>;
    type View<'a, U>
    where
        U: Clone + 'a,
    = StorageImpl<&'a [U]>;

    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[T] {
        self.0
    }

    fn cow(&self) -> Self::Cow<'_, T> {
        StorageImpl(Cow::Borrowed(self.0))
    }

    fn view(&self) -> Self::View<'_, T> {
        StorageImpl(self.0)
    }
}

impl<T> StorageMut<T> for StorageImpl<&mut [T]>
where
    T: Clone,
{
    type ViewMut<'a, U>
    where
        U: Clone + 'a,
    = StorageImpl<&'a mut [U]>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_, T> {
        StorageImpl(self.0)
    }
}

#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
#[cfg(feature = "std")]
use std::borrow::Cow;

use super::{Storage, StorageImpl, StorageMut};

impl<T> Storage<T> for StorageImpl<&[T]>
where
    T: Clone,
{
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

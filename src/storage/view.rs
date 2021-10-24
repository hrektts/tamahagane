#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
#[cfg(feature = "std")]
use std::borrow::Cow;

use super::{Storage, StorageImpl, StorageMut};

impl<T> Storage for StorageImpl<&[T]>
where
    T: Clone,
{
    type Elem = T;

    fn as_ptr(&self) -> *const <Self as Storage>::Elem {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[<Self as Storage>::Elem] {
        self.0
    }

    fn cow(&self) -> <Self as Storage>::Cow<'_> {
        StorageImpl(Cow::Borrowed(self.0))
    }

    fn view(&self) -> <Self as Storage>::View<'_> {
        StorageImpl(self.0)
    }
}

impl<T> Storage for StorageImpl<&mut [T]>
where
    T: Clone,
{
    type Elem = T;

    fn as_ptr(&self) -> *const <Self as Storage>::Elem {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[<Self as Storage>::Elem] {
        self.0
    }

    fn cow(&self) -> <Self as Storage>::Cow<'_> {
        StorageImpl(Cow::Borrowed(self.0))
    }

    fn view(&self) -> <Self as Storage>::View<'_> {
        StorageImpl(self.0)
    }
}

impl<T> StorageMut for StorageImpl<&mut [T]>
where
    T: Clone,
{
    fn as_mut_ptr(&mut self) -> *mut <Self as Storage>::Elem {
        self.0.as_mut_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [<Self as Storage>::Elem] {
        self.0
    }

    fn view_mut(&mut self) -> <Self as StorageMut>::ViewMut<'_> {
        StorageImpl(self.0)
    }
}

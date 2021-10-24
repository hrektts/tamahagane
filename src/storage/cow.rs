#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
use core::iter::FromIterator;
#[cfg(feature = "std")]
use std::borrow::Cow;

use super::{Storage, StorageImpl, StorageMut};

impl<T> FromIterator<T> for StorageImpl<Cow<'_, [T]>>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(Cow::Owned(FromIterator::from_iter(iter)))
    }
}

impl<T> Storage for StorageImpl<Cow<'_, [T]>>
where
    T: Clone,
{
    type Elem = T;

    fn as_ptr(&self) -> *const <Self as Storage>::Elem {
        self.0.as_ptr()
    }

    fn as_slice(&self) -> &[<Self as Storage>::Elem] {
        &self.0
    }

    fn cow(&self) -> <Self as Storage>::Cow<'_> {
        let inner = match &self.0 {
            Cow::Borrowed(b) => Cow::Borrowed(*b),
            Cow::Owned(o) => Cow::Borrowed(o.as_slice()),
        };
        StorageImpl(inner)
    }

    fn view(&self) -> <Self as Storage>::View<'_> {
        StorageImpl(&self.0)
    }
}

impl<T> StorageMut for StorageImpl<Cow<'_, [T]>>
where
    T: Clone,
{
    fn as_mut_ptr(&mut self) -> *mut <Self as Storage>::Elem {
        self.0.to_mut().as_mut_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [<Self as Storage>::Elem] {
        self.0.to_mut()
    }

    fn view_mut(&mut self) -> <Self as StorageMut>::ViewMut<'_> {
        StorageImpl(self.0.to_mut())
    }
}

use core::{iter::FusedIterator, marker::PhantomData, ptr::NonNull};

use super::Array;
use crate::{
    storage::{Storage, StorageMut},
    Dimensionality, Shape,
};

macro_rules! iterator {
    ($name:ident, $elem:ty, [$( $mutability:tt )*]) => {
        impl<T, D> ExactSizeIterator for $name<'_, T, D>
        where
            D: Dimensionality,
        {
            fn len(&self) -> usize {
                self.len
            }

            fn is_empty(&self) -> bool {
                self.len == 0
            }
        }

        impl<T, D> FusedIterator for $name<'_, T, D> where D: Dimensionality {}

        impl<'a, T, D> Iterator for $name<'a, T, D>
        where
            D: Dimensionality,
        {
            type Item = $elem;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(indices) = &mut self.indices {
                    let offset = indices
                        .as_ref()
                        .iter()
                        .zip(self.strides.as_ref())
                        .fold(0_isize, |acc, (&index, &stride)| {
                            acc + index as isize * stride
                        });

                    increment_indices::<D>(&mut self.indices, &self.shape, &mut self.len);

                    unsafe { Some(& $( $mutability )* *(self.ptr.as_ptr().offset(offset))) }
                } else {
                    None
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.len, Some(self.len))
            }
        }
    };
}

#[inline]
fn increment_indices<D>(
    maybe_indices: &mut Option<<D as Dimensionality>::Shape>,
    shape: &<D as Dimensionality>::Shape,
    len: &mut usize,
) where
    D: Dimensionality,
{
    if let Some(indices) = maybe_indices {
        for (index, &dim) in indices
            .as_mut()
            .iter_mut()
            .rev()
            .zip(shape.as_ref().iter().rev())
        {
            *index += 1;
            if *index == dim {
                *index = 0;
            } else {
                break;
            }
        }

        *len -= 1;
    }

    if *len == 0 {
        maybe_indices.take();
    }
}

pub struct Iter<'a, T: 'a, D>
where
    D: Dimensionality,
{
    ptr: NonNull<T>,
    indices: Option<<D as Dimensionality>::Shape>,
    len: usize,
    shape: <D as Dimensionality>::Shape,
    strides: <<D as Dimensionality>::Shape as Shape>::Strides,
    phantom: PhantomData<&'a T>,
}

iterator!(Iter, &'a T, []);

impl<'a, T, D> Iter<'a, T, D>
where
    D: Dimensionality,
{
    pub(super) fn new<S, O>(a: &Array<T, S, D, O>) -> Self
    where
        D: Dimensionality,
        S: Storage<T>,
        T: Clone,
    {
        let ptr = unsafe { a.storage.as_ptr().add(a.offset) };
        debug_assert!(!ptr.is_null());

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            indices: D::first_indices(&a.shape),
            len: a.shape.array_len(),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            phantom: PhantomData,
        }
    }
}

pub struct IterMut<'a, T: 'a, D>
where
    D: Dimensionality,
{
    ptr: NonNull<T>,
    indices: Option<<D as Dimensionality>::Shape>,
    len: usize,
    shape: <D as Dimensionality>::Shape,
    strides: <<D as Dimensionality>::Shape as Shape>::Strides,
    phantom: PhantomData<&'a mut T>,
}

iterator!(IterMut, &'a mut T, [mut]);

impl<'a, T, D> IterMut<'a, T, D>
where
    D: Dimensionality,
{
    pub(super) fn new<S, O>(a: &mut Array<T, S, D, O>) -> Self
    where
        D: Dimensionality,
        S: StorageMut<T>,
        T: Clone,
    {
        let ptr = unsafe { a.storage.as_mut_ptr().add(a.offset) };
        debug_assert!(!ptr.is_null());

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            indices: D::first_indices(&a.shape),
            len: a.shape.array_len(),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use super::{Iter, IterMut};
    use crate::{dyn_s, storage::Storage, Array, NDArray, NDArrayMut, NDArrayOwned, Result};

    #[test]
    fn iterate_1d_storage() {
        let data = vec![1, 2, 3, 4];
        let a1 = Array::from(data.clone());
        {
            let iter = Iter::new(&a1);

            assert_eq!(iter.ptr.as_ptr() as *const _, a1.storage.as_ptr());
            assert_eq!(iter.indices, Some([0]));
            assert_eq!(iter.shape, a1.shape);
            assert_eq!(iter.strides, a1.strides);
            assert_eq!(iter.len(), data.len());
            assert!(!iter.is_empty());

            for (i, (&actual, &expected)) in iter.zip(&data).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }

        let a2 = a1.slice(dyn_s!(..;-1));
        {
            let iter = Iter::new(&a2);

            assert_eq!(iter.ptr.as_ptr() as *const _, unsafe {
                a2.storage.as_ptr().add(a2.storage.as_slice().len() - 1)
            });
            assert_eq!(iter.indices, Some(vec![0]));
            assert_eq!(iter.shape, a2.shape);
            assert_eq!(iter.strides, a2.strides);
            assert_eq!(iter.len(), data.len());
            assert!(!iter.is_empty());

            for (i, (&actual, &expected)) in iter.zip(data.iter().rev()).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }

    #[test]
    fn iterate_1d_storage_mutably() {
        let data = vec![1, 2, 3, 4];
        let mut a1 = Array::from(data.clone());
        {
            let iter = IterMut::new(&mut a1);

            assert_eq!(iter.ptr.as_ptr() as *const _, a1.storage.as_ptr());
            assert_eq!(iter.indices, Some([0]));
            assert_eq!(iter.shape, a1.shape);
            assert_eq!(iter.strides, a1.strides);
            assert_eq!(iter.len(), data.len());
            assert!(!iter.is_empty());

            for (i, (&mut actual, &expected)) in iter.zip(&data).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }

        let mut a2 = a1.slice_mut(dyn_s!(..;-1));
        {
            let iter = IterMut::new(&mut a2);

            assert_eq!(iter.ptr.as_ptr() as *const _, unsafe {
                a2.storage.as_ptr().add(a2.storage.as_slice().len() - 1)
            });
            assert_eq!(iter.indices, Some(vec![0]));
            assert_eq!(iter.shape, a2.shape);
            assert_eq!(iter.strides, a2.strides);
            assert_eq!(iter.len(), data.len());
            assert!(!iter.is_empty());

            for (i, (&mut actual, &expected)) in iter.zip(data.iter().rev()).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }

    #[test]
    fn iterate_nd_storage() -> Result<()> {
        let data = (1..25).collect::<Vec<usize>>();
        let a3 = Array::from(data.clone()).into_shape([2, 3, 4])?;
        let a3s = a3.slice(dyn_s!(.., .., ..;2));
        {
            let iter = Iter::new(&a3s);

            assert!(iter.ptr.as_ptr() as *const _ >= a3s.storage.as_ptr());
            assert!(
                iter.ptr.as_ptr() as *const _
                    < unsafe { a3s.storage.as_ptr().add(a3s.storage.as_slice().len()) }
            );
            assert_eq!(iter.indices, Some(vec![0; 3]));
            assert_eq!(iter.shape, a3s.shape);
            assert_eq!(iter.strides, a3s.strides);
            assert_eq!(iter.len(), data.len() / 2);
            assert!(!iter.is_empty());

            for (i, (&actual, &expected)) in iter.zip(data.iter().step_by(2)).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }

        let a3s = a3.slice(dyn_s!(.., .., ..;-2));
        {
            let iter = Iter::new(&a3s);

            assert!(iter.ptr.as_ptr() as *const _ >= a3s.storage.as_ptr());
            assert!(
                iter.ptr.as_ptr() as *const _
                    < unsafe { a3s.storage.as_ptr().add(a3s.storage.as_slice().len()) }
            );
            assert_eq!(iter.indices, Some(vec![0; 3]));
            assert_eq!(iter.shape, a3s.shape);
            assert_eq!(iter.strides, a3s.strides);
            assert_eq!(iter.len(), data.len() / 2);
            assert!(!iter.is_empty());

            for (i, (&actual, expected)) in iter
                .zip((1..).step_by(4).flat_map(|x| (x..x + 4).rev().step_by(2)))
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }

        Ok(())
    }

    #[test]
    fn iterate_nd_storage_mutably() -> Result<()> {
        let data = (1..25).collect::<Vec<usize>>();
        let mut a3 = Array::from(data.clone()).into_shape([2, 3, 4])?;
        let mut a3s = a3.slice_mut(dyn_s!(.., .., ..;2));
        {
            let iter = IterMut::new(&mut a3s);

            assert!(iter.ptr.as_ptr() as *const _ >= a3s.storage.as_ptr());
            assert!(
                iter.ptr.as_ptr() as *const _
                    < unsafe { a3s.storage.as_ptr().add(a3s.storage.as_slice().len()) }
            );
            assert_eq!(iter.indices, Some(vec![0; 3]));
            assert_eq!(iter.shape, a3s.shape);
            assert_eq!(iter.strides, a3s.strides);
            assert_eq!(iter.len(), data.len() / 2);
            assert!(!iter.is_empty());

            for (i, (&mut actual, &expected)) in iter.zip(data.iter().step_by(2)).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }

        let mut a3s = a3.slice_mut(dyn_s!(.., .., ..;-2));
        {
            let iter = IterMut::new(&mut a3s);

            assert!(iter.ptr.as_ptr() as *const _ >= a3s.storage.as_ptr());
            assert!(
                iter.ptr.as_ptr() as *const _
                    < unsafe { a3s.storage.as_ptr().add(a3s.storage.as_slice().len()) }
            );
            assert_eq!(iter.indices, Some(vec![0; 3]));
            assert_eq!(iter.shape, a3s.shape);
            assert_eq!(iter.strides, a3s.strides);
            assert_eq!(iter.len(), data.len() / 2);
            assert!(!iter.is_empty());

            for (i, (&mut actual, expected)) in iter
                .zip((1..).step_by(4).flat_map(|x| (x..x + 4).rev().step_by(2)))
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }

        Ok(())
    }
}

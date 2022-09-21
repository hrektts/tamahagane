mod sequence;
pub use sequence::SequenceIter;

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
            #[inline(always)]
            fn len(&self) -> usize {
                self.len
            }

            #[inline(always)]
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

            #[inline]
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

                    unsafe { Some(& $( $mutability )* *(self.ptr.as_ptr().wrapping_offset(offset))) }
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
            *index = index.wrapping_add(1);
            if *index >= dim {
                *index = 0;
            } else {
                break;
            }
        }

        *len = len.wrapping_sub(1);
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
    pub(super) fn new<O, S>(a: &Array<S, D, O>) -> Self
    where
        D: Dimensionality,
        S: Storage<Elem = T>,
    {
        let ptr = a.storage.as_ptr().wrapping_add(a.offset);
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
    pub(super) fn new<S, O>(a: &mut Array<S, D, O>) -> Self
    where
        D: Dimensionality,
        S: StorageMut<Elem = T>,
    {
        let ptr = a.storage.as_mut_ptr().wrapping_add(a.offset);
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
    use crate::{
        s,
        storage::{Storage, StorageImpl},
        Array, NDArray, NDArrayMut, NDArrayOwned, NDims, Result,
    };

    macro_rules! test_iterator_of_1d_array {
        ($subject:ident, $slice_method:ident, [$( $mutability:tt )*], $array:expr, $data:expr) => {
            {
                let iter = $subject::new(& $( $mutability )* $array);

                assert_eq!(iter.ptr.as_ptr() as *const _, $array.storage.as_ptr());
                assert_eq!(iter.indices, Some([0]));
                assert_eq!(iter.shape, $array.shape);
                assert_eq!(iter.strides, $array.strides);
                assert_eq!(iter.len(), $data.len());
                assert!(!iter.is_empty());

                for (i, (& $( $mutability )* actual, &expected)) in iter.zip(&$data).enumerate() {
                    assert_eq!(actual, expected, "{}th element is not equal", i);
                }
            }

            let $( $mutability )* view = $array.$slice_method(s!(..;-1));
            {
                let iter = $subject::new(& $( $mutability )* view);

                assert_eq!(iter.ptr.as_ptr() as *const _, unsafe {
                    view.storage.as_ptr().add(view.storage.as_slice().len() - 1)
                });
                assert_eq!(iter.indices, Some([0]));
                assert_eq!(iter.shape, view.shape);
                assert_eq!(iter.strides, view.strides);
                assert_eq!(iter.len(), $data.len());
                assert!(!iter.is_empty());

                for (i, (& $( $mutability )* actual, &expected)) in
                    iter.zip($data.iter().rev()).enumerate()
                {
                    assert_eq!(actual, expected, "{}th element is not equal", i);
                }
            }
        };
    }

    #[test]
    fn iterate_1d_array() {
        let data = vec![1, 2, 3, 4];
        let a1 = Array::from(data.clone());

        test_iterator_of_1d_array!(Iter, slice, [], a1, data);
    }

    #[test]
    fn iterate_1d_zst_array() {
        let data = vec![(), (), (), ()];
        let a1 = Array::from(data.clone());
        {
            let mut counter = 0;
            for _ in Iter::new(&a1) {
                counter += 1;
            }

            assert_eq!(counter, data.len());
        }
    }

    #[test]
    fn iterate_1d_array_mutably() {
        let data = vec![1, 2, 3, 4];
        let mut a1 = Array::from(data.clone());

        test_iterator_of_1d_array!(IterMut, slice_mut, [mut], a1, data);
    }

    #[test]
    fn iterate_empty_array() {
        let a = Array::<StorageImpl<Vec<u64>>, NDims<3>>::zeros(&[1_usize, 0, 3]);

        assert!(a.iter().next().is_none());
    }

    macro_rules! test_iterator_of_nd_array {
        ($subject:ident, $slice_method:ident, [$( $mutability:tt )*], $array:expr, $data:expr) => {
            let $( $mutability )* view = $array.$slice_method(s!(.., .., ..;2));
            {
                let iter = $subject::new(& $( $mutability )* view);

                assert!(iter.ptr.as_ptr() as *const _ >= view.storage.as_ptr());
                assert!(
                    iter.ptr.as_ptr() as *const _
                        < unsafe { view.storage.as_ptr().add(view.storage.as_slice().len()) }
                );
                assert_eq!(iter.indices, Some([0; 3]));
                assert_eq!(iter.shape, view.shape);
                assert_eq!(iter.strides, view.strides);
                assert_eq!(iter.len(), $data.len() / 2);
                assert!(!iter.is_empty());

                for (i, (& $( $mutability)* actual, &expected)) in iter.zip($data.iter().step_by(2)).enumerate() {
                    assert_eq!(actual, expected, "{}th element is not equal", i);
                }
            }

            let $( $mutability )* view = $array.$slice_method(s!(.., .., ..;-2));
            {
                let iter = $subject::new(& $( $mutability )* view);

                assert!(iter.ptr.as_ptr() as *const _ >= view.storage.as_ptr());
                assert!(
                    iter.ptr.as_ptr() as *const _
                        < unsafe { view.storage.as_ptr().add(view.storage.as_slice().len()) }
                );
                assert_eq!(iter.indices, Some([0; 3]));
                assert_eq!(iter.shape, view.shape);
                assert_eq!(iter.strides, view.strides);
                assert_eq!(iter.len(), $data.len() / 2);
                assert!(!iter.is_empty());

                for (i, (& $( $mutability)* actual, expected)) in iter
                    .zip((1..).step_by(4).flat_map(|x| (x..x + 4).rev().step_by(2)))
                    .enumerate()
                {
                    assert_eq!(actual, expected, "{}th element is not equal", i);
                }
            }
        };
    }

    #[test]
    fn iterate_nd_array() -> Result<()> {
        let data = (1..25).collect::<Vec<usize>>();
        let a3 = Array::from(data.clone()).into_shape([2, 3, 4])?;

        test_iterator_of_nd_array!(Iter, slice, [], a3, data);

        Ok(())
    }

    #[test]
    fn iterate_nd_array_mutably() -> Result<()> {
        let data = (1..25).collect::<Vec<usize>>();
        let mut a3 = Array::from(data.clone()).into_shape([2, 3, 4])?;

        test_iterator_of_nd_array!(IterMut, slice_mut, [mut], a3, data);

        Ok(())
    }
}

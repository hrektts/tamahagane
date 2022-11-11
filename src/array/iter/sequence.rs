use core::{intrinsics, iter::FusedIterator, marker::PhantomData, mem, ptr::NonNull};

use super::ArrayBase;
use crate::{storage::Storage, Dimensionality, Shape};

pub struct SequenceIter<'a, T: 'a, D>
where
    D: Dimensionality,
{
    ptr: NonNull<T>,
    indices: Option<<D as Dimensionality>::Shape>,
    len: usize,
    shape: <D as Dimensionality>::Shape,
    strides: <<D as Dimensionality>::Shape as Shape>::Strides,
    axis: usize,
    sequence_dim: usize,
    phantom: PhantomData<&'a T>,
}

impl<T, D> ExactSizeIterator for SequenceIter<'_, T, D>
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

impl<T, D> FusedIterator for SequenceIter<'_, T, D> where D: Dimensionality {}

impl<'a, T, D> Iterator for SequenceIter<'a, T, D>
where
    D: Dimensionality,
{
    type Item = ElementIterator<'a, T>;

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

            super::increment_indices::<D>(&mut self.indices, &self.shape, &mut self.len);

            Some(ElementIterator::new(
                unsafe { NonNull::new_unchecked(self.ptr.as_ptr().wrapping_offset(offset)) },
                self.sequence_dim,
                self.strides[self.axis],
            ))
        } else {
            None
        }
    }
}

impl<'a, T, D> SequenceIter<'a, T, D>
where
    D: Dimensionality,
{
    #[inline]
    pub fn new<O, S>(a: &ArrayBase<S, D, O>, axis: usize) -> Self
    where
        D: Dimensionality,
        S: Storage<Elem = T>,
    {
        let ptr = a.storage.as_ptr().wrapping_add(a.offset);
        let len = a.shape.array_len() / a.shape[axis];
        let mut shape = a.shape.clone();
        shape[axis] = 0;

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            indices: D::first_indices(&a.shape),
            len,
            shape,
            strides: a.strides.clone(),
            axis,
            sequence_dim: a.shape[axis],
            phantom: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct ElementIterator<'a, T: 'a> {
    ptr: NonNull<T>,
    end: *const T,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

impl<T> ExactSizeIterator for ElementIterator<'_, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        unsafe {
            let diff = (self.end as usize).unchecked_sub(self.ptr.as_ptr() as usize);
            intrinsics::exact_div(diff, mem::size_of::<T>() * self.stride as usize)
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.ptr.as_ptr() as *const _ == self.end
    }
}

impl<T> FusedIterator for ElementIterator<'_, T> {}

impl<'a, T> Iterator for ElementIterator<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr.as_ptr();
        if ptr as *const T == self.end {
            None
        } else {
            if mem::size_of::<T>() == 0 {
                self.end = self.end.wrapping_byte_sub(1);
            } else {
                self.ptr = unsafe { NonNull::new_unchecked(ptr.wrapping_offset(self.stride)) };
            }
            Some(unsafe { &*ptr })
        }
    }
}

impl<'a, T> ElementIterator<'a, T> {
    #[inline]
    fn new(ptr: NonNull<T>, dim: usize, stride: isize) -> Self {
        let end = if mem::size_of::<T>() == 0 {
            (ptr.as_ptr() as *mut u8).wrapping_add(dim) as *const T
        } else {
            ptr.as_ptr().wrapping_offset(stride * (dim as isize))
        };

        Self {
            ptr,
            end,
            stride,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::SequenceIter;
    use crate::{ArrayBase, NDArray, NDArrayOwned, Result, Storage};

    #[test]
    fn iterate_2d_array() -> Result<()> {
        macro_rules! test_iterator {
            ($iter:expr, $array:expr, $axis_to_exclude:expr) => {
                assert_eq!($iter.ptr.as_ptr() as *const _, $array.storage.as_ptr());
                assert_eq!($iter.indices, Some([0, 0]));

                let mut shape_expected = $array.shape;
                shape_expected[$axis_to_exclude] = 0;
                assert_eq!($iter.shape, shape_expected);

                assert_eq!($iter.strides, $array.strides);
                assert_eq!($iter.axis, $axis_to_exclude);

                assert_eq!($iter.len(), $array.len() / $array.shape[$axis_to_exclude]);
            };
        }

        let data = (1..7).collect::<Vec<usize>>();
        let a2 = ArrayBase::from(data.clone()).into_shape([2, 3])?;
        {
            let axis_to_exclude = 0;
            let seq_iter = SequenceIter::new(&a2, axis_to_exclude);

            test_iterator!(seq_iter, a2, axis_to_exclude);

            for (i, elem_iter) in seq_iter.enumerate() {
                assert_eq!(elem_iter.len(), a2.shape[axis_to_exclude]);
                for (&actual, expected) in elem_iter.zip([1_usize, 4].map(|x| x + i)) {
                    assert_eq!(actual, expected);
                }
            }
        }
        {
            let axis_to_exclude = 1;
            let seq_iter = SequenceIter::new(&a2, axis_to_exclude);

            test_iterator!(seq_iter, a2, axis_to_exclude);

            for (i, elem_iter) in seq_iter.enumerate() {
                assert_eq!(elem_iter.len(), a2.shape[axis_to_exclude]);
                for (&actual, expected) in elem_iter.zip([1_usize, 2, 3].map(|x| x + 3 * i)) {
                    assert_eq!(actual, expected);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn iterate_2d_zst_array() -> Result<()> {
        let data = [(); 6].to_vec();
        let a2 = ArrayBase::from(data.clone()).into_shape([2, 3])?;
        {
            let axis_to_exclude = 0;
            let seq_iter = SequenceIter::new(&a2, axis_to_exclude);
            let mut counter1 = 0;
            for elem_iter in seq_iter {
                counter1 += 1;
                let mut counter2 = 0;
                for _ in elem_iter {
                    counter2 += 1;
                }

                assert_eq!(counter2, a2.shape[axis_to_exclude]);
            }

            assert_eq!(counter1, a2.shape[1]);

            Ok(())
        }
    }
}

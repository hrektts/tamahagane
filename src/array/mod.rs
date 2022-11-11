mod fmt;

mod iter;
pub use iter::{Iter, IterMut, SequenceIter};

mod linarg;

mod ops;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::{iter::FromIterator, marker::PhantomData};

use num_traits::{One, Zero};

use crate::{
    storage::{Storage, StorageBase, StorageMut, StorageOwned},
    util, ArrayIndex, Dimensionality, DimensionalityAdd, DimensionalityDiff, NDArray, NDArrayMut,
    NDArrayOwned, NDims, NewShape, Order, Result, RowMajor, Shape, ShapeError, SliceInfo,
};

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct ArrayBase<S, D, O = RowMajor>
where
    D: Dimensionality,
{
    shape: <D as Dimensionality>::Shape,
    strides: <<D as Dimensionality>::Shape as Shape>::Strides,
    storage: S,
    offset: usize,
    phantom: PhantomData<O>,
}

impl<T> From<Vec<T>> for ArrayBase<StorageBase<Vec<T>>, NDims<1>> {
    fn from(data: Vec<T>) -> Self {
        Self {
            shape: [data.len()],
            strides: [1],
            storage: StorageBase::from(data),
            offset: 0,
            phantom: PhantomData,
        }
    }
}

impl<T> FromIterator<T> for ArrayBase<StorageBase<Vec<T>>, NDims<1>>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let storage = iter.into_iter().collect::<StorageBase<_>>();
        Self {
            shape: [storage.as_slice().len()],
            strides: [1],
            storage,
            offset: 0,
            phantom: PhantomData,
        }
    }
}

macro_rules! impl_ndarray {
    ($type:ty) => {
        impl<D, O, S> NDArray for $type
        where
            D: Dimensionality,
            O: Order,
            S: Storage,
        {
            type D = D;
            type O = O;
            type S = S;

            fn broadcast_to<BD>(
                &self,
                shape: &<BD as Dimensionality>::Shape,
            ) -> Result<ArrayBase<<Self::S as Storage>::View<'_>, BD, Self::O>>
            where
                BD: Dimensionality,
            {
                if shape.ndims() < self.shape.ndims() {
                    return Err(ShapeError::IncompatibleDimension(
                        "cannot broadcast to smaller dimensions".into(),
                    )
                    .into());
                }

                Ok(ArrayBase {
                    shape: shape.clone(),
                    strides: self.compute_strides_broadcasted::<BD>(shape)?,
                    storage: self.storage.view(),
                    offset: self.offset,
                    phantom: PhantomData,
                })
            }

            #[inline]
            fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #[inline]
            fn iter<'a>(&self) -> Iter<'a, <Self::S as Storage>::Elem, Self::D> {
                Iter::new(self)
            }

            #[inline]
            fn iter_sequence<'a>(&self, axis: usize) -> SequenceIter<'a, <S as Storage>::Elem, D> {
                SequenceIter::new(self, axis)
            }

            #[inline]
            fn len(&self) -> usize {
                self.shape.array_len()
            }

            #[inline]
            fn ndims(&self) -> usize {
                self.shape.ndims()
            }

            fn permute(
                &self,
                axes: <Self::D as Dimensionality>::Shape,
            ) -> Result<ArrayBase<<Self::S as Storage>::View<'_>, Self::D, Self::O>> {
                if axes.ndims() != self.ndims() {
                    return Err(
                        ShapeError::IncompatibleAxis("axes do not match array".into()).into(),
                    );
                }

                let mut counts = <D as Dimensionality>::shape_zeroed(axes.ndims());
                for &axis in axes.as_ref() {
                    if axis >= axes.ndims() {
                        return Err(ShapeError::IncompatibleAxis(format!(
                            "axis {} is out of bounds for array of dimension {}",
                            axis,
                            self.ndims()
                        ))
                        .into());
                    }
                    counts[axis] += 1;
                }
                for &count in counts.as_ref() {
                    if count != 1 {
                        return Err(ShapeError::IncompatibleAxis(
                            "repeated axis in permutation".into(),
                        )
                        .into());
                    }
                }

                let mut out_shape = counts;
                let mut out_strides = <D as Dimensionality>::strides_zeroed(axes.ndims());
                for (i, &axis) in axes.as_ref().iter().enumerate() {
                    out_shape[i] = self.shape[axis];
                    out_strides[i] = self.strides[axis];
                }

                Ok(ArrayBase {
                    shape: out_shape,
                    strides: out_strides,
                    storage: self.storage.view(),
                    offset: self.offset,
                    phantom: PhantomData,
                })
            }

            fn slice<ST, SD>(
                &self,
                info: SliceInfo<ST, SD>,
            ) -> ArrayBase<
                <Self::S as Storage>::View<'_>,
                <Self::D as DimensionalityAdd<SD>>::Output,
                Self::O,
            >
            where
                Self::D: DimensionalityAdd<SD>,
                SD: DimensionalityDiff,
                ST: AsRef<[ArrayIndex]>,
            {
                let (offset, shape, strides) = self.compute_sliced_parts(info);
                ArrayBase {
                    shape,
                    strides,
                    storage: self.storage.view(),
                    offset,
                    phantom: PhantomData,
                }
            }

            #[inline]
            fn shape(&self) -> &<Self::D as Dimensionality>::Shape {
                &self.shape
            }

            #[inline]
            fn strides(&self) -> &<<Self::D as Dimensionality>::Shape as Shape>::Strides {
                &self.strides
            }

            fn to_owned_array(&self) -> ArrayBase<<Self::S as Storage>::Owned, Self::D, Self::O> {
                ArrayBase {
                    shape: self.shape.clone(),
                    strides: self.shape.to_default_strides::<Self::O>(),
                    storage: self.iter().cloned().collect(),
                    offset: 0,
                    phantom: PhantomData,
                }
            }

            fn to_shape<NS>(
                &self,
                shape: NS,
            ) -> Result<
                ArrayBase<<Self::S as Storage>::Cow<'_>, <NS as NewShape>::Dimensionality, Self::O>,
            >
            where
                NS: NewShape,
            {
                self.to_shape_with_order::<_, O>(shape)
            }

            fn to_shape_with_order<NS, NO>(
                &self,
                shape: NS,
            ) -> Result<
                ArrayBase<<Self::S as Storage>::Cow<'_>, <NS as NewShape>::Dimensionality, NO>,
            >
            where
                NO: Order,
                NS: NewShape,
            {
                if self.ndims() == shape.ndims()
                    && util::type_eq::<O, NO>()
                    && self
                        .shape
                        .as_ref()
                        .iter()
                        .zip(shape.as_ref().iter())
                        .all(|(&dim, &new_dim)| dim as isize == new_dim)
                {
                    let (out_shape, out_strides) = self.convert_shape_and_strides(&shape);
                    return Ok(ArrayBase {
                        shape: out_shape,
                        strides: out_strides,
                        storage: self.storage.cow(),
                        offset: self.offset,
                        phantom: PhantomData,
                    });
                }

                let out_shape = self.infer_shape(shape)?;
                if let Some(out_strides) = self.compute_strides_reshaped::<NO, _>(&out_shape) {
                    Ok(ArrayBase {
                        shape: out_shape,
                        strides: out_strides,
                        storage: self.storage.cow(),
                        offset: self.offset,
                        phantom: PhantomData,
                    })
                } else {
                    Ok(ArrayBase {
                        strides: out_shape.to_default_strides::<O>(),
                        shape: out_shape,
                        storage: self.iter().cloned().collect(),
                        offset: 0,
                        phantom: PhantomData,
                    })
                }
            }

            fn transpose(&self) -> ArrayBase<<Self::S as Storage>::View<'_>, Self::D, Self::O> {
                let mut shape = self.shape.clone();
                shape.as_mut().reverse();
                let mut strides = self.strides.clone();
                strides.as_mut().reverse();

                ArrayBase {
                    shape,
                    strides,
                    storage: self.storage.view(),
                    offset: self.offset,
                    phantom: PhantomData,
                }
            }
        }
    };
}

impl_ndarray!(ArrayBase<S, D, O>);
impl_ndarray!(&ArrayBase<S, D, O>);
impl_ndarray!(&mut ArrayBase<S, D, O>);

macro_rules! impl_ndarray_mut {
    ($type:ty) => {
        impl<D, O, S> NDArrayMut for $type
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut,
        {
            type SM = S;

            fn fill(&mut self, value: <Self::S as Storage>::Elem) {
                for elem in self.iter_mut() {
                    *elem = value.clone();
                }
            }

            #[inline]
            fn iter_mut<'a>(&mut self) -> IterMut<'a, <Self::S as Storage>::Elem, Self::D> {
                IterMut::new(self)
            }

            fn slice_mut<ST, SD>(
                &mut self,
                info: SliceInfo<ST, SD>,
            ) -> ArrayBase<
                <Self::SM as StorageMut>::ViewMut<'_>,
                <Self::D as DimensionalityAdd<SD>>::Output,
                Self::O,
            >
            where
                Self::D: DimensionalityAdd<SD>,
                SD: DimensionalityDiff,
                ST: AsRef<[ArrayIndex]>,
            {
                let (offset, shape, strides) = self.compute_sliced_parts(info);
                ArrayBase {
                    shape,
                    strides,
                    storage: self.storage.view_mut(),
                    offset,
                    phantom: PhantomData,
                }
            }
        }
    };
}

impl_ndarray_mut!(ArrayBase<S, D, O>);
impl_ndarray_mut!(&mut ArrayBase<S, D, O>);

impl<D, O, S> NDArrayOwned for ArrayBase<S, D, O>
where
    D: Dimensionality,
    O: Order,
    S: StorageOwned,
{
    type SO = S;

    fn allocate_uninitialized<Sh>(shape: &Sh) -> Self
    where
        Sh: Shape<Dimensionality = Self::D>,
    {
        ArrayBase {
            shape: shape.as_associated_shape().clone(),
            strides: shape.as_associated_shape().to_default_strides::<O>(),
            storage: Self::SO::allocate_uninitialized(shape.as_associated_shape().array_len()),
            offset: 0,
            phantom: PhantomData,
        }
    }

    fn into_shape<NS>(
        self,
        shape: NS,
    ) -> Result<ArrayBase<Self::S, <NS as NewShape>::Dimensionality, Self::O>>
    where
        NS: NewShape,
    {
        self.into_shape_with_order::<_, O>(shape)
    }

    fn into_shape_with_order<NS, NO>(
        self,
        shape: NS,
    ) -> Result<ArrayBase<Self::S, <NS as NewShape>::Dimensionality, NO>>
    where
        NS: NewShape,
        NO: Order,
    {
        if self.ndims() == shape.ndims()
            && util::type_eq::<O, NO>()
            && self
                .shape
                .as_ref()
                .iter()
                .zip(shape.as_ref().iter())
                .all(|(&dim, &new_dim)| dim as isize == new_dim)
        {
            let out_shape = self.convert_shape(&shape);
            return Ok(ArrayBase {
                strides: out_shape.to_default_strides::<NO>(),
                shape: out_shape,
                storage: self.storage,
                offset: 0,
                phantom: PhantomData,
            });
        }

        let out_shape = self.infer_shape(shape)?;
        if let Some(out_strides) = self.compute_strides_reshaped::<NO, _>(&out_shape) {
            Ok(ArrayBase {
                shape: out_shape,
                strides: out_strides,
                storage: self.storage,
                offset: 0,
                phantom: PhantomData,
            })
        } else {
            Ok(ArrayBase {
                strides: out_shape.to_default_strides::<O>(),
                shape: out_shape,
                storage: self.iter().cloned().collect(),
                offset: 0,
                phantom: PhantomData,
            })
        }
    }

    fn ones<Sh>(shape: &Sh) -> Self
    where
        <Self::S as Storage>::Elem: One,
        Sh: Shape<Dimensionality = Self::D>,
    {
        ArrayBase {
            shape: shape.as_associated_shape().clone(),
            strides: shape.as_associated_shape().to_default_strides::<O>(),
            storage: S::ones(shape.as_associated_shape().array_len()),
            offset: 0,
            phantom: PhantomData,
        }
    }

    fn zeros<Sh>(shape: &Sh) -> Self
    where
        <Self::S as Storage>::Elem: Zero,
        Sh: Shape<Dimensionality = Self::D>,
    {
        ArrayBase {
            shape: shape.as_associated_shape().clone(),
            strides: shape.as_associated_shape().to_default_strides::<O>(),
            storage: S::zeros(shape.as_associated_shape().array_len()),
            offset: 0,
            phantom: PhantomData,
        }
    }
}

impl<D, O, S> ArrayBase<S, D, O>
where
    D: Dimensionality,
    O: Order,
    S: Storage,
{
    fn convert_shape<NS>(
        &self,
        shape: &NS,
    ) -> <<NS as NewShape>::Dimensionality as Dimensionality>::Shape
    where
        NS: NewShape,
    {
        debug_assert_eq!(self.ndims(), shape.ndims());

        let mut out_shape =
            <<NS as NewShape>::Dimensionality as Dimensionality>::shape_zeroed(shape.ndims());
        for (dest, src) in out_shape.as_mut().iter_mut().zip(self.shape.as_ref()) {
            *dest = *src;
        }
        out_shape
    }

    fn convert_shape_and_strides<NS>(
        &self,
        shape: &NS,
    ) -> (
        <<NS as NewShape>::Dimensionality as Dimensionality>::Shape,
        <<<NS as NewShape>::Dimensionality as Dimensionality>::Shape as Shape>::Strides,
    )
    where
        NS: NewShape,
    {
        debug_assert_eq!(self.ndims(), shape.ndims());

        let out_shape = self.convert_shape::<NS>(shape);

        let mut out_strides =
            <<NS as NewShape>::Dimensionality as Dimensionality>::strides_zeroed(shape.ndims());
        for (dest, src) in out_strides.as_mut().iter_mut().zip(self.strides.as_ref()) {
            *dest = *src;
        }

        (out_shape, out_strides)
    }

    #[allow(clippy::type_complexity)]
    fn compute_sliced_parts<ST, SD>(
        &self,
        info: SliceInfo<ST, SD>,
    ) -> (
        usize,
        <<D as DimensionalityAdd<SD>>::Output as Dimensionality>::Shape,
        <<<D as DimensionalityAdd<SD>>::Output as Dimensionality>::Shape as Shape>::Strides,
    )
    where
        D: Dimensionality + DimensionalityAdd<SD>,
        <D as DimensionalityAdd<SD>>::Output: Dimensionality,
        S: Storage,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>,
    {
        let out_n_dims = {
            let n_dims = self.ndims();
            let diff = info.dim_diff;
            let n_dims_indexed = info
                .as_ref()
                .iter()
                .filter(|&idx| idx.is_index() || idx.is_slice())
                .count();
            assert!(
                n_dims_indexed <= n_dims,
                "too many indices for array: array is {}-dimensional, but {} were indexed",
                n_dims,
                n_dims_indexed,
            );
            (n_dims as isize + diff) as usize
        };
        let mut out_shape =
            <<D as DimensionalityAdd<SD>>::Output as Dimensionality>::shape_zeroed(out_n_dims as _);
        let mut out_strides =
            <<D as DimensionalityAdd<SD>>::Output as Dimensionality>::strides_zeroed(
                out_n_dims as _,
            );
        let mut in_idx = 0_usize;
        let mut out_idx = 0_usize;
        let mut out_offset = self.offset as isize;
        for array_index in info.as_ref() {
            match array_index {
                ArrayIndex::Index(index) => {
                    let dim = self.shape.as_ref()[in_idx] as isize;
                    assert!(
                        -dim <= *index && *index < dim,
                        "index {} is out of bounds for axis {} with size {}",
                        index,
                        in_idx,
                        dim,
                    );
                    out_offset += self.strides.as_ref()[in_idx] * ((index + dim) % dim);
                    in_idx += 1;
                }
                ArrayIndex::Slice(slice) => {
                    let dim = self.shape.as_ref()[in_idx];
                    out_shape.as_mut()[out_idx] = slice.len_with_dim(dim);

                    let stride = self.strides.as_ref()[in_idx];
                    out_strides.as_mut()[out_idx] = stride * slice.step;

                    let start = slice.start_with_dim(dim);
                    if start > 0 {
                        out_offset += stride * start;
                    }

                    in_idx += 1;
                    out_idx += 1;
                }
                ArrayIndex::NewAxis => {
                    out_shape.as_mut()[out_idx] = 1;
                    out_strides.as_mut()[out_idx] = 0;
                    out_idx += 1;
                }
            }
        }

        for (i, o) in (in_idx..self.shape.as_ref().len()).zip(out_idx..) {
            out_shape.as_mut()[o] = self.shape.as_ref()[i];
            out_strides.as_mut()[o] = self.strides.as_ref()[i];
        }

        debug_assert!(0 <= out_offset && (out_offset as usize) < self.storage.as_slice().len());
        (out_offset as usize, out_shape, out_strides)
    }

    fn compute_strides_broadcasted<BD>(
        &self,
        shape: &<BD as Dimensionality>::Shape,
    ) -> Result<<<BD as Dimensionality>::Shape as Shape>::Strides>
    where
        BD: Dimensionality,
    {
        let mut strides = <BD as Dimensionality>::strides_zeroed(shape.ndims());
        for ((stride, dim), (in_stride, in_dim)) in strides
            .as_mut()
            .iter_mut()
            .rev()
            .zip(shape.as_ref().iter().rev())
            .zip(
                self.strides
                    .as_ref()
                    .iter()
                    .rev()
                    .zip(self.shape.as_ref().iter().rev()),
            )
        {
            if *in_dim == 1 {
                // do nothing; broadcast
            } else if *in_dim == *dim {
                *stride = *in_stride;
            } else {
                return Err(ShapeError::IncompatibleDimension("invalid broadcast".into()).into());
            }
        }
        Ok(strides)
    }

    fn compute_strides_reshaped<NO, RS>(&self, shape: &RS) -> Option<<RS as Shape>::Strides>
    where
        NO: Order,
        RS: Shape,
    {
        let mut strides = shape.to_default_strides::<NO>();
        let array_len = self.len();
        if array_len > 0 {
            let (reduced_shape, reduced_strides, reduced_n_dims) = {
                let mut r_shape = <D as Dimensionality>::shape_zeroed(self.ndims());
                let mut r_strides = <D as Dimensionality>::strides_zeroed(self.ndims());

                if array_len == 1 {
                    r_shape[0] = 1;
                    r_strides[0] = 1;
                    (r_shape, r_strides, 1)
                } else {
                    let mut i = 0;
                    for (&dim, &stride) in self
                        .shape
                        .as_ref()
                        .iter()
                        .zip(self.strides.as_ref())
                        .filter(|(&dim, _)| dim != 1)
                    {
                        r_shape[i] = dim;
                        r_strides[i] = stride;
                        i += 1;
                    }
                    (r_shape, r_strides, i)
                }
            };

            if !NO::is_data_aligned_monotonically(reduced_shape.as_ref(), reduced_strides.as_ref())
            {
                None
            } else {
                NO::convert_shape_to_strides(
                    shape,
                    reduced_strides[reduced_n_dims - 1],
                    &mut strides,
                );
                Some(strides)
            }
        } else {
            Some(strides)
        }
    }

    fn infer_shape<NS>(
        &self,
        shape: NS,
    ) -> Result<<<NS as NewShape>::Dimensionality as Dimensionality>::Shape>
    where
        NS: NewShape,
    {
        let mut inferred =
            <<NS as NewShape>::Dimensionality as Dimensionality>::shape_zeroed(shape.ndims());

        for (i, &dim) in shape.as_ref().iter().enumerate() {
            if dim < 0 {
                if shape.as_ref().iter().skip(i + 1).any(|&x| x < 0) {
                    return Err(ShapeError::IncompatibleShape(
                        "can only specify one unknown dimension".into(),
                    )
                    .into());
                }

                let rest_dim: isize = shape.as_ref()[..i]
                    .iter()
                    .chain(shape.as_ref().iter().skip(i + 1))
                    .product::<isize>();
                if rest_dim == 0 {
                    return Err(ShapeError::IncompatibleShape(format!(
                        "cannot transform array of length {} into shape {:?}",
                        self.len(),
                        shape
                    ))
                    .into());
                }
                inferred.as_mut()[i] = self.len() / rest_dim as usize;
            } else {
                inferred.as_mut()[i] = dim as usize;
            }
        }
        if inferred.array_len() != self.len() {
            return Err(ShapeError::IncompatibleShape(format!(
                "cannot transform array of length {} into shape {:?}",
                self.len(),
                shape
            ))
            .into());
        }

        Ok(inferred)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::ArrayBase;
    use crate::{
        storage::{Storage, StorageBase},
        ArrayIndex, ColumnMajor, DynDim, NDArray, NDArrayMut, NDArrayOwned, NDims, NewAxis, Result,
        RowMajor, Shape, Slice, SliceInfo,
    };

    #[test]
    fn allocate_uninitialized() {
        let shape = [2, 3, 4];
        let a3 = ArrayBase::<StorageBase<Vec<f64>>, _>::allocate_uninitialized(&shape);

        assert_eq!(a3.shape(), &shape);
    }

    #[test]
    #[should_panic]
    fn broadcast_to_smaller_dimensions() {
        let a3 = (1..)
            .take(6)
            .collect::<ArrayBase<_, _>>()
            .into_shape([1, 2, 3])
            .unwrap();
        a3.broadcast_to::<NDims<2>>(&[2, 3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn broadcast_to_invalid_shape() {
        let a3 = (1..)
            .take(6)
            .collect::<ArrayBase<_, _>>()
            .into_shape([1, 2, 3])
            .unwrap();
        a3.broadcast_to::<NDims<3>>(&[2, 3, 3]).unwrap();
    }

    #[test]
    fn broadcast_to() -> Result<()> {
        let a1 = ArrayBase::from(vec![1; 6]);
        {
            let a3 = a1.to_shape([1, 2, 3])?;
            {
                let a = a3.broadcast_to::<NDims<3>>(&[2, 2, 3])?;

                assert_eq!(a.shape, [2, 2, 3]);
                assert_eq!(a.strides, [0, 3, 1]);
                assert_eq!(a.offset, a3.offset);
            }
            {
                let a = a3.broadcast_to::<DynDim>(&vec![4, 2, 2, 3])?;

                assert_eq!(a.shape, [4, 2, 2, 3]);
                assert_eq!(a.strides, [0, 0, 3, 1]);
                assert_eq!(a.offset, a3.offset);
            }
        }
        {
            let a3 = a1.to_shape_with_order::<_, ColumnMajor>([1, 2, 3])?;
            {
                {
                    let a = a3.broadcast_to::<NDims<3>>(&[2, 2, 3])?;

                    assert_eq!(a.shape, [2, 2, 3]);
                    assert_eq!(a.strides, [0, 1, 2]);
                    assert_eq!(a.offset, a3.offset);
                }
                {
                    let a = a3.broadcast_to::<DynDim>(&vec![4, 2, 2, 3])?;

                    assert_eq!(a.shape, [4, 2, 2, 3]);
                    assert_eq!(a.strides, [0, 0, 1, 2]);
                    assert_eq!(a.offset, a3.offset);
                }
            }
        }

        Ok(())
    }

    #[test]
    #[should_panic]
    fn convert_empty_array_to_ambiguous_shape() {
        let a = ArrayBase::from(Vec::<usize>::new());
        a.to_shape([2, 0, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_empty_array_into_ambiguous_shape() {
        let a = ArrayBase::from(Vec::<usize>::new());
        a.into_shape([2, 0, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_to_ambiguous_shape() {
        let a = ArrayBase::from(vec![1_usize, 2, 3, 4]);
        a.to_shape([2, -1, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_into_ambiguous_shape() {
        let a = ArrayBase::from(vec![1_usize, 2, 3, 4]);
        a.into_shape([2, -1, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_to_incompatible_shape() {
        let a = ArrayBase::from(vec![1_usize, 2, 3, 4]);
        a.to_shape([2, 2, 2]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_into_incompatible_shape() {
        let a = ArrayBase::from(vec![1_usize, 2, 3, 4]);
        a.into_shape([2, 2, 2]).unwrap();
    }

    #[test]
    fn convert_to_new_shape() -> Result<()> {
        let a1 = (1..).take(24).collect::<ArrayBase<_, _>>();
        {
            let new_shape = [2_isize, 3, 4];
            let shape = new_shape.map(|x| x as usize);
            let a2 = a1.to_shape(new_shape)?;

            assert_eq!(a2.shape, shape);
            assert_eq!(a2.strides, shape.to_default_strides::<RowMajor>());
            assert_eq!(a2.storage.as_ptr(), a1.storage.as_ptr());
            assert_eq!(a2.offset, 0);
        }
        {
            let new_shape = [2_isize, 3, 4];
            let shape = new_shape.map(|x| x as usize);
            let a2 = a1.to_shape_with_order::<_, ColumnMajor>(new_shape)?;

            assert_eq!(a2.shape, shape);
            assert_eq!(a2.strides, shape.to_default_strides::<ColumnMajor>());
            assert_eq!(a2.storage.as_ptr(), a1.storage.as_ptr());
            assert_eq!(a2.offset, 0);
        }

        Ok(())
    }

    #[test]
    fn convert_into_new_shape() -> Result<()> {
        {
            let a1 = (1..).take(24).collect::<ArrayBase<_, _>>();
            let ptr = a1.storage.as_ptr();
            let new_shape = [2_isize, 3, 4];
            let shape = new_shape.map(|x| x as usize);
            let a2 = a1.into_shape(new_shape)?;

            assert_eq!(a2.shape, shape);
            assert_eq!(a2.strides, shape.to_default_strides::<RowMajor>());
            assert_eq!(a2.storage.as_ptr(), ptr);
            assert_eq!(a2.offset, 0);
        }
        {
            let a1 = (1..).take(24).collect::<ArrayBase<_, _>>();
            let ptr = a1.storage.as_ptr();
            let new_shape = [2_isize, 3, 4];
            let shape = new_shape.map(|x| x as usize);
            let a2 = a1.into_shape_with_order::<_, ColumnMajor>(new_shape)?;

            assert_eq!(a2.shape, shape);
            assert_eq!(a2.strides, shape.to_default_strides::<ColumnMajor>());
            assert_eq!(a2.storage.as_ptr(), ptr);
            assert_eq!(a2.offset, 0);
        }

        Ok(())
    }

    #[test]
    fn convert_to_the_same_shape() -> Result<()> {
        let data = vec![1_usize, 2, 3, 4];
        let a = ArrayBase::from(data.clone());
        let subject = a.to_shape([4])?;

        assert_eq!(subject.shape, a.shape);
        assert_eq!(subject.strides, a.strides);
        assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data).cow());
        assert_eq!(subject.offset, a.offset);

        Ok(())
    }

    #[test]
    fn convert_into_the_same_shape() -> Result<()> {
        let data = vec![1_usize, 2, 3, 4];
        let a = ArrayBase::from(data.clone());
        let subject = a.into_shape([4])?;

        assert_eq!(subject.shape, [4]);
        assert_eq!(subject.strides, [1]);
        assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data));
        assert_eq!(subject.offset, 0);

        Ok(())
    }

    #[test]
    fn fill() -> Result<()> {
        let mut a3 = (1..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
            .into_shape([2, 3, 4])?;
        a3.fill(7);

        assert!(a3.iter().all(|&x| x == 7));

        Ok(())
    }

    #[test]
    fn is_empty() -> Result<()> {
        let a1 = ArrayBase::from(Vec::<usize>::new());
        let a3 = ArrayBase::from(Vec::<usize>::new()).into_shape([3, 2, 0])?;

        assert!(a1.is_empty());
        assert!(a3.is_empty());

        Ok(())
    }

    #[test]
    fn len() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
            .into_shape(shape)?;

        assert_eq!(a3.len(), shape.iter().map(|&x| x as usize).product());

        Ok(())
    }

    #[test]
    fn ndims() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
            .into_shape(shape)?;

        assert_eq!(a3.ndims(), shape.len());

        Ok(())
    }

    #[test]
    fn ones() {
        let shape = [2, 3, 4];
        let a3 = ArrayBase::<StorageBase<Vec<u64>>, _>::ones(&shape);

        assert_eq!(a3.shape(), &shape);
        assert!(a3.iter().all(|&x| x == 1));
    }

    #[test]
    fn permute() -> Result<()> {
        let a3 = ArrayBase::from(vec![1; 24]).into_shape([2, 3, 4])?;
        let a3p = a3.permute([2, 0, 1])?;

        assert_eq!(a3p.shape, [4, 2, 3]);
        assert_eq!(a3p.strides, [1, 12, 4]);
        assert_eq!(a3p.offset, a3.offset);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn permute_by_axis_out_of_bounds() {
        let a3 = ArrayBase::from(vec![1; 24]).into_shape([2, 3, 4]).unwrap();
        a3.permute([0, 1, 100]).unwrap();
    }

    #[test]
    #[should_panic]
    fn permute_by_repeated_axes() {
        let a3 = ArrayBase::from(vec![1; 24]).into_shape([2, 3, 4]).unwrap();
        a3.permute([0, 1, 0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn permute_by_wrong_number_of_axes() {
        let a3 = ArrayBase::from(vec![1; 24]).into_shape([1, 2]).unwrap();
        a3.permute([1, 2]).unwrap();
    }

    #[test]
    fn shape() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
            .into_shape(shape)?;

        assert_eq!(a3.shape(), &shape.map(|x| x as usize));

        Ok(())
    }

    #[test]
    fn slice_by_index() {
        macro_rules! test {
            ($index:expr, $offset:expr) => {
                let data = vec![1_usize, 2, 3, 4];
                let a = ArrayBase::from(data.clone());
                let subject = a.slice(SliceInfo::from(vec![ArrayIndex::from($index)]));

                assert_eq!(subject.shape, vec![]);
                assert_eq!(subject.strides, vec![]);
                assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data).view());
                assert_eq!(subject.offset, $offset);
            };
        }

        test!(1, 1);
        test!(-1, 3);
    }

    #[test]
    #[should_panic]
    fn slice_by_invalid_index() {
        let a = ArrayBase::from(vec![1_usize, 2, 3, 4]);
        a.slice(SliceInfo::from(vec![10.into()]));
    }

    #[test]
    fn slice_by_new_axis() {
        let data = vec![1_usize, 2, 3, 4];
        let a = ArrayBase::from(data.clone());
        let subject = a.slice(SliceInfo::from(vec![NewAxis.into()]));

        assert_eq!(subject.shape, vec![1, 4]);
        assert_eq!(subject.strides, vec![0, 1]);
        assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data).view());
        assert_eq!(subject.offset, 0);
    }

    #[test]
    fn slice_by_slice() -> Result<()> {
        macro_rules! test {
            ($r:expr, $step:expr, $len:expr, $offset:expr) => {
                let data = vec![1_usize, 2, 3, 4, 5, 6, 7, 8];
                let a = ArrayBase::from(data.clone());
                let subject = a.slice(SliceInfo::from(vec![ArrayIndex::from(
                    Slice::from($r).step_by($step.try_into()?),
                )]));

                assert_eq!(subject.shape, vec![$len]);
                assert_eq!(subject.strides, vec![$step]);
                assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data).view());
                assert_eq!(subject.offset, $offset);
            };
        }

        test!(.., 1, 8, 0);
        test!(.., -1, 8, 7);
        test!(1..3, 1, 2, 1);
        test!(3..1, -1, 2, 3);
        test!(1.., 1, 7, 1);
        test!(-2.., -1, 7, 6);
        test!(..3, 1, 3, 0);
        test!(..-3, -1, 2, 7);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn slice_by_too_many_indices() {
        let a = ArrayBase::from(vec![1_usize, 2, 3, 4]);
        a.slice(SliceInfo::from(vec![
            NewAxis.into(),
            1.into(),
            Slice::from(..2).into(),
        ]));
    }

    #[test]
    fn slice_sliced_array_by_slice() {
        let a = ArrayBase::from(vec![1_usize; 32]);

        assert_eq!(a.offset, 0);

        let info = SliceInfo::from(vec![ArrayIndex::from(Slice::from(10..))]);
        let a1 = a.slice(info.clone());

        assert_eq!(a1.offset, 10);

        let a2 = a1.slice(info);

        assert_eq!(a2.offset, 20);
    }

    #[test]
    fn strides() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
            .into_shape(shape)?;

        assert_eq!(
            a3.strides(),
            &shape.map(|x| x as usize).to_default_strides::<RowMajor>()
        );

        Ok(())
    }

    #[test]
    fn transpose() -> Result<()> {
        let a2 = (1..)
            .take(6)
            .collect::<ArrayBase<_, _>>()
            .into_shape([2, 3])?;
        let a2t = a2.transpose();

        assert_eq!(a2t.shape(), &[3, 2]);
        for (&actual, expected) in a2t.iter().zip([1_usize, 4, 2, 5, 3, 6]) {
            assert_eq!(actual, expected);
        }

        Ok(())
    }

    #[test]
    fn zeros() {
        let shape = [2, 3, 4];
        let a3 = ArrayBase::<StorageBase<Vec<u64>>, _>::zeros(&shape);

        assert_eq!(a3.shape(), &shape);
        assert!(a3.iter().all(|&x| x == 0));
    }
}

mod fmt;

mod iter;
pub use iter::{Iter, IterMut};

mod linarg;

mod ops;

mod routine;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::{iter::FromIterator, marker::PhantomData, mem};

use num_traits::{One, Zero};

use crate::{
    storage::{Storage, StorageBase, StorageMut, StorageOwned},
    util, ArrayIndex, DimDiff, Dimensionality, DimensionalityAdd, DimensionalityDiff, Error,
    NDArray, NDArrayMut, NDArrayOwned, NDims, Order, Result, RowMajor, Shape, ShapeError,
    SignedShape, SliceInfo,
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

pub type Array<T, D, O = RowMajor> = ArrayBase<StorageBase<Vec<T>>, D, O>;

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

#[rustfmt::skip]
macro_rules! impl_ndarray {
    ($type:ty) => {
        impl<D, O, S> NDArray for $type
        where
            D: Dimensionality,
            O: Order,
            S: Storage,
        {
            type Dimensionality = D;
            type Order = O;
            type Storage = S;
            type Iter<'a> = Iter<'a, <Self::Storage as Storage>::Elem, Self::Dimensionality>
            where
                <Self::Storage as Storage>::Elem: 'a;
            type CowWithD<'a, D2> = ArrayBase<<Self::Storage as Storage>::Cow<'a>, D2, Self::Order>
            where
                Self: 'a,
                D2: Dimensionality;
            type CowWithDO<'a, D2, O2> = ArrayBase<<Self::Storage as Storage>::Cow<'a>, D2, O2>
            where
                Self: 'a,
                D2: Dimensionality,
                O2: Order;
            type Owned =
                ArrayBase<<Self::Storage as Storage>::Owned, Self::Dimensionality, Self::Order>;
            type View<'a> =
                ArrayBase<<Self::Storage as Storage>::View<'a>, Self::Dimensionality, Self::Order>
            where
                Self: 'a;
            type ViewWithD<'a, D2> =
                ArrayBase<<Self::Storage as Storage>::View<'a>, D2, Self::Order>
            where
                Self: 'a,
                D2: Dimensionality;

            fn as_ptr(&self) -> *const <Self::Storage as Storage>::Elem {
                let ptr = if self.storage.as_slice().is_empty()
                    || mem::size_of::<<Self::Storage as Storage>::Elem>() == 0
                {
                    self.storage.as_ptr()
                } else {
                    self.storage.as_ptr().wrapping_add(self.offset)
                };

                debug_assert!(
                    ptr >= self.storage.as_ptr()
                        && (ptr
                            < self
                                .storage
                                .as_ptr()
                                .wrapping_add(self.storage.as_slice().len())
                            || ((self.is_empty()
                                || mem::size_of::<<Self::Storage as Storage>::Elem>() == 0)
                                && ptr == self.storage.as_ptr()))
                );
                ptr
            }

            fn broadcast_to<BD>(
                &self,
                shape: &<BD as Dimensionality>::Shape,
            ) -> Result<Self::ViewWithD<'_, BD>>
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

            fn expand_shape(
                &self,
                axis: isize,
            ) -> Result<Self::CowWithD<'_, <<
                <Self::Dimensionality as DimensionalityAdd<DimDiff<1>>>::Output
                    as Dimensionality>::SignedShape as SignedShape>::Dimensionality>>
            where
                Self::Dimensionality: DimensionalityAdd<DimDiff<1>>,
            {
                let n_dims = self.ndims();
                let axis_normalized = routine::normalize_axis(axis, n_dims + 1)?;
                let mut out_shape =
                    <Self::Dimensionality
                        as DimensionalityAdd<DimDiff<1>>>::Output::signed_shape_zeroed(n_dims + 1);
                for (out_dim, dim) in out_shape.as_mut().iter_mut().zip(
                    self.shape()
                        .as_ref()
                        .iter()
                        .take(axis_normalized)
                        .chain([1].iter())
                        .chain(self.shape().as_ref().iter().skip(axis_normalized)),
                ) {
                    *out_dim = *dim as isize;
                }

                self.to_shape(out_shape)
            }

            fn flip(&self) -> Result<Self::View<'_>> {
                let mut axes = Self::Dimensionality::signed_shape_zeroed(self.ndims());
                for (i, axis) in axes.as_mut().iter_mut().enumerate() {
                    *axis = i as isize;
                }
                self.flip_along_axes(&axes.as_ref())
            }

            fn flip_along_axes(&self, axes: &[isize]) -> Result<Self::View<'_>> {
                let out_shape = self.shape.clone();
                let mut out_strides = self.strides.clone();
                let mut out_offset = self.offset as isize;
                let n_dims = self.ndims();
                for &axis in axes {
                    let axis_normalized = routine::normalize_axis(axis, n_dims)?;
                    out_offset += 0.max(out_shape[axis_normalized] as isize - 1) * out_strides[axis_normalized];
                    if out_shape[axis_normalized] != 0 {
                        out_strides[axis_normalized] *= -1;
                    }
                }

                Ok(ArrayBase {
                    shape: out_shape,
                    strides: out_strides,
                    storage: self.storage.view(),
                    offset: out_offset as usize,
                    phantom: PhantomData
                })
            }

            #[inline]
            fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #[inline]
            fn iter<'a>(&self) -> Self::Iter<'a> {
                Iter::new(self)
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
                axes: <Self::Dimensionality as Dimensionality>::Shape,
            ) -> Result<Self::View<'_>> {
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
            ) -> Self::ViewWithD<'_, <Self::Dimensionality as DimensionalityAdd<SD>>::Output>
            where
                Self::Dimensionality: DimensionalityAdd<SD>,
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
            fn shape(&self) -> &<Self::Dimensionality as Dimensionality>::Shape {
                &self.shape
            }

            #[inline]
            fn strides(
                &self,
            ) -> &<<Self::Dimensionality as Dimensionality>::Shape as Shape>::Strides {
                &self.strides
            }

            fn to_owned_array(&self) -> Self::Owned {
                ArrayBase {
                    shape: self.shape.clone(),
                    strides: self.shape.to_default_strides::<Self::Order>(),
                    storage: self.iter().cloned().collect(),
                    offset: 0,
                    phantom: PhantomData,
                }
            }

            fn to_shape<Sh>(
                &self,
                shape: Sh,
            ) -> Result<Self::CowWithD<'_, <Sh as SignedShape>::Dimensionality>>
            where
                Sh: SignedShape,
            {
                self.to_shape_with_order::<_, O>(shape)
            }

            fn to_shape_with_order<Sh, O2>(
                &self,
                shape: Sh,
            ) -> Result<Self::CowWithDO<'_, <Sh as SignedShape>::Dimensionality, O2>>
            where
                O2: Order,
                Sh: SignedShape,
            {
                if self.ndims() == shape.ndims()
                    && util::type_eq::<O, O2>()
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
                if let Some(out_strides) = self.compute_strides_reshaped::<O2, _>(&out_shape) {
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

            fn transpose(&self) -> Self::View<'_> {
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

            fn view(&self) -> Self::View<'_> {
                ArrayBase {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
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

#[rustfmt::skip]
macro_rules! impl_ndarray_mut {
    ($type:ty) => {
        impl<D, O, S> NDArrayMut for $type
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut,
        {
            type ViewMutWithD<'a, D2> =
                ArrayBase<<Self::Storage as StorageMut>::ViewMut<'a>, D2, Self::Order>
            where
                Self: 'a,
                D2: Dimensionality;

            fn fill(&mut self, value: <Self::Storage as Storage>::Elem) {
                for elem in self.iter_mut() {
                    *elem = value.clone();
                }
            }

            #[inline]
            fn iter_mut<'a>(
                &mut self,
            ) -> IterMut<'a, <Self::Storage as Storage>::Elem, Self::Dimensionality> {
                IterMut::new(self)
            }

            fn slice_mut<ST, SD>(
                &mut self,
                info: SliceInfo<ST, SD>,
            ) -> Self::ViewMutWithD<'_, <Self::Dimensionality as DimensionalityAdd<SD>>::Output>
            where
                Self::Dimensionality: DimensionalityAdd<SD>,
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
    S: StorageMut + StorageOwned,
{
    type WithD<D2> = ArrayBase<Self::Storage, D2, Self::Order> where D2: Dimensionality;

    fn allocate_uninitialized<Sh>(shape: &Sh) -> Self
    where
        Sh: Shape<Dimensionality = Self::Dimensionality>,
    {
        ArrayBase {
            shape: shape.as_associated_shape().clone(),
            strides: shape.as_associated_shape().to_default_strides::<O>(),
            storage: Self::Storage::allocate_uninitialized(shape.as_associated_shape().array_len()),
            offset: 0,
            phantom: PhantomData,
        }
    }

    fn concatenate<T>(arrays: &[T], axis: isize) -> Result<Self>
    where
        Self: Sized,
        T: NDArray,
        <<T as NDArray>::Dimensionality as Dimensionality>::Shape:
            Shape<Dimensionality = Self::Dimensionality>,
        <T as NDArray>::Storage: Storage<Elem = <Self::Storage as Storage>::Elem>,
    {
        if arrays.is_empty() {
            return Err(Error::Value(
                "need at least one array to concatenate".into(),
            ));
        }

        let n_dims = arrays[0].ndims();
        if n_dims == 0 {
            return Err(ShapeError::IncompatibleDimension(
                "zero-dimensional arrays cannot be concatenated".into(),
            )
            .into());
        }

        let axis_normalized = routine::normalize_axis(axis, n_dims)?;
        let mut shape = arrays[0].shape().clone();
        if arrays.len() > 1 {
            for array in arrays[1..].iter() {
                if array.ndims() != n_dims {
                    return Err(ShapeError::IncompatibleDimension(
                        "all the input arrays must have same number of dimensions".into(),
                    )
                    .into());
                }
                for (i, (dim, d)) in shape
                    .as_mut()
                    .iter_mut()
                    .zip(array.shape().as_ref().iter())
                    .enumerate()
                {
                    if i == axis_normalized {
                        *dim += *d;
                    } else if dim != d {
                        return Err(ShapeError::IncompatibleDimension(format!(
                            "{} {}",
                            "all the input array dimensions except for the",
                            "concatenation axis must match exactly"
                        ))
                        .into());
                    }
                }
            }
        }

        let mut out = Self::allocate_uninitialized(&shape);
        let mut slice_idx = 0_isize;
        for array in arrays {
            let dim = array.shape()[axis_normalized] as isize;
            let info = SliceInfo::from(
                (0..n_dims)
                    .map(|axis_idx| {
                        if axis_idx == axis_normalized {
                            ArrayIndex::from(slice_idx..slice_idx + dim)
                        } else {
                            ArrayIndex::from(..)
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let mut view = out.slice_mut(info);
            for (dst, src) in view.iter_mut().zip(array.iter()) {
                *dst = src.clone();
            }
            slice_idx += dim;
        }

        Ok(out)
    }

    fn into_shape<Sh>(self, shape: Sh) -> Result<Self::WithD<<Sh as SignedShape>::Dimensionality>>
    where
        Sh: SignedShape,
    {
        self.into_shape_with_order::<_, O>(shape)
    }

    fn into_shape_with_order<Sh, O2>(
        self,
        shape: Sh,
    ) -> Result<Self::WithD<<Sh as SignedShape>::Dimensionality>>
    where
        O2: Order,
        Sh: SignedShape,
    {
        if self.ndims() == shape.ndims()
            && util::type_eq::<O, O2>()
            && self
                .shape
                .as_ref()
                .iter()
                .zip(shape.as_ref().iter())
                .all(|(&dim, &new_dim)| dim as isize == new_dim)
        {
            let out_shape = self.convert_shape(&shape);
            return Ok(ArrayBase {
                strides: out_shape.to_default_strides::<O2>(),
                shape: out_shape,
                storage: self.storage,
                offset: 0,
                phantom: PhantomData,
            });
        }

        let out_shape = self.infer_shape(shape)?;
        if let Some(out_strides) = self.compute_strides_reshaped::<O2, _>(&out_shape) {
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
        <Self::Storage as Storage>::Elem: One,
        Sh: Shape<Dimensionality = Self::Dimensionality>,
    {
        ArrayBase {
            shape: shape.as_associated_shape().clone(),
            strides: shape.as_associated_shape().to_default_strides::<O>(),
            storage: S::ones(shape.as_associated_shape().array_len()),
            offset: 0,
            phantom: PhantomData,
        }
    }

    fn stack<T>(arrays: &[T], axis: isize) -> Result<Self>
    where
        Self: Sized,
        T: NDArray,
        <T as NDArray>::Dimensionality: DimensionalityAdd<DimDiff<1>>,
        <<<<<T as NDArray>::Dimensionality as DimensionalityAdd<DimDiff<1>>>::Output
            as Dimensionality>::SignedShape as SignedShape>::Dimensionality
            as Dimensionality>::Shape: Shape<Dimensionality = Self::Dimensionality>,
        <T as NDArray>::Storage: Storage<Elem = <Self::Storage as Storage>::Elem>,
    {
        let expanded_arrays = arrays
            .iter()
            .map(|x| x.expand_shape(axis))
            .collect::<Result<Vec<_>>>()?;
        Self::WithD::concatenate(&expanded_arrays, axis)
    }

    fn zeros<Sh>(shape: &Sh) -> Self
    where
        <Self::Storage as Storage>::Elem: Zero,
        Sh: Shape<Dimensionality = Self::Dimensionality>,
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
    fn convert_shape<Sh>(
        &self,
        shape: &Sh,
    ) -> <<Sh as SignedShape>::Dimensionality as Dimensionality>::Shape
    where
        Sh: SignedShape,
    {
        debug_assert_eq!(self.ndims(), shape.ndims());

        let mut out_shape =
            <<Sh as SignedShape>::Dimensionality as Dimensionality>::shape_zeroed(shape.ndims());
        for (dest, src) in out_shape.as_mut().iter_mut().zip(self.shape.as_ref()) {
            *dest = *src;
        }
        out_shape
    }

    fn convert_shape_and_strides<Sh>(
        &self,
        shape: &Sh,
    ) -> (
        <<Sh as SignedShape>::Dimensionality as Dimensionality>::Shape,
        <<<Sh as SignedShape>::Dimensionality as Dimensionality>::Shape as Shape>::Strides,
    )
    where
        Sh: SignedShape,
    {
        debug_assert_eq!(self.ndims(), shape.ndims());

        let out_shape = self.convert_shape::<Sh>(shape);

        let mut out_strides =
            <<Sh as SignedShape>::Dimensionality as Dimensionality>::strides_zeroed(shape.ndims());
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
                "too many indices for array: array is {n_dims}-dimensional, but {n_dims_indexed} were indexed",
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
                        "index {index} is out of bounds for axis {in_idx} with size {dim}",
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

        debug_assert!(
            self.storage.as_slice().is_empty()
                || (0 <= out_offset && (out_offset as usize) < self.storage.as_slice().len())
        );
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

    fn compute_strides_reshaped<O2, Sh>(&self, shape: &Sh) -> Option<<Sh as Shape>::Strides>
    where
        O2: Order,
        Sh: Shape,
    {
        let mut strides = shape.to_default_strides::<O2>();
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

            if !O2::is_data_aligned_monotonically(reduced_shape.as_ref(), reduced_strides.as_ref())
            {
                None
            } else {
                O2::convert_shape_to_strides(
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

    fn infer_shape<Sh>(
        &self,
        shape: Sh,
    ) -> Result<<<Sh as SignedShape>::Dimensionality as Dimensionality>::Shape>
    where
        Sh: SignedShape,
    {
        let mut inferred =
            <<Sh as SignedShape>::Dimensionality as Dimensionality>::shape_zeroed(shape.ndims());

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

#[macro_export]
macro_rules! array {
    (@count $x:tt, $($y:tt),+) => {
        array!(@count $($y),+) + 1
    };
    (@count $x:tt) => {
        1_isize
    };
    (@flatten []) => {
        core::iter::empty()
    };
    (@flatten [], $([]),+) => {
        core::iter::empty()
    };
    (@flatten [$($x:tt),+]) => {
        array!(@flatten $($x),+)
    };
    (@flatten [$($x:tt),+], $([$($y:tt),*]),+) => {
        (array!(@flatten $($x),+)).chain(array!(@flatten $([$($y),*]),+))
    };
    (@flatten $($x:tt),*) => {
        [$($x),*].into_iter()
    };
    (@shape []; [$($n:tt)*] $dim:expr) => {
        array!(@shape _; [$($n)* 0] $dim + 1)
    };
    (@shape [$($x:tt),+]; [$($n:tt)*] $dim:expr) => {
        array!(@shape $($x),*; [$($n)* (array!(@count $($x),*))] $dim + 1)
    };
    (@shape $x:tt, $($y:tt),+; [$($n:tt)*] $dim:expr) => {
        array!(@shape $x; [$($n)*] $dim)
    };
    (@shape $x:tt; [$($n:tt)*] $dim:expr) => {
        [$($n),*] as [isize; $dim]
    };
    ($($x:tt),*) => {{
        use $crate::NDArrayOwned;

        let shape = array!(@shape $($x),*; [] 0);
        let iter = array!(@flatten $($x),*);
        iter.collect::<$crate::Array<_, _>>().into_shape(shape).unwrap()
    }}
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::Array;
    use crate::{
        s,
        storage::{Storage, StorageBase},
        ColumnMajor, DynDim, NDArray, NDArrayMut, NDArrayOwned, NDims, NewAxis, Result, RowMajor,
        Shape,
    };

    #[test]
    fn allocate_uninitialized() {
        let shape = [2, 3, 4];
        let a3 = Array::<f64, _>::allocate_uninitialized(&shape);

        assert_eq!(a3.shape(), &shape);
    }

    #[test]
    fn array() {
        {
            let a = array!(1);
            assert_eq!(a.iter().cloned().collect::<Vec<_>>(), vec![1]);
            assert_eq!(a.ndims(), 0);
            assert_eq!(a.shape(), &[]);
        }
        {
            let a: Array<usize, _> = array!([]);
            assert!(a.iter().is_empty());
            assert_eq!(a.ndims(), 1);
            assert_eq!(a.shape(), &[0]);
        }
        {
            let a: Array<usize, _> = array!([[]]);
            assert!(a.iter().is_empty());
            assert_eq!(a.ndims(), 2);
            assert_eq!(a.shape(), &[1, 0]);
        }
        {
            let a: Array<usize, _> = array!([[], []]);
            assert!(a.iter().is_empty());
            assert_eq!(a.ndims(), 2);
            assert_eq!(a.shape(), &[2, 0]);
        }
        {
            let a = array!([1, 2, 3]);
            assert_eq!(a.iter().cloned().collect::<Vec<_>>(), vec![1, 2, 3]);
            assert_eq!(a.shape(), &[3]);
        }
        {
            let a = array!([[1, 2, 3], [2, 3, 4]]);
            assert_eq!(
                a.iter().cloned().collect::<Vec<_>>(),
                vec![1, 2, 3, 2, 3, 4]
            );
            assert_eq!(a.shape(), &[2, 3]);
        }
    }

    #[test]
    fn array_count() {
        assert_eq!(array!(@count []), 1);
        assert_eq!(array!(@count [], []), 2);
        assert_eq!(array!(@count 1), 1);
        assert_eq!(array!(@count 1, 1), 2);
    }

    #[test]
    fn array_flatten() {
        {
            let x: Vec<usize> = array!(@flatten [[]]).collect();
            assert_eq!(x, vec![]);
        }
        {
            let x: Vec<usize> = array!(@flatten [[], []]).collect();
            assert_eq!(x, vec![]);
        }
        assert_eq!(array!(@flatten 1).collect::<Vec<_>>(), vec![1]);
        assert_eq!(array!(@flatten [1,2,3]).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            array!(@flatten [[1,2,3],[2,3,4]]).collect::<Vec<_>>(),
            vec![1, 2, 3, 2, 3, 4]
        );
    }

    #[test]
    fn array_shape() {
        assert_eq!(array!(@shape 1; [] 0), []);
        assert_eq!(array!(@shape [[]]; [] 0), [1, 0]);
        assert_eq!(array!(@shape [[], []]; [] 0), [2, 0]);
        assert_eq!(array!(@shape [1, 2, 3]; [] 0), [3]);
        assert_eq!(array!(@shape [[1, 2, 3], [2, 3, 4]]; [] 0), [2, 3]);
    }

    #[test]
    #[should_panic]
    fn broadcast_to_smaller_dimensions() {
        let a3 = (1..)
            .take(6)
            .collect::<Array<_, _>>()
            .into_shape([1, 2, 3])
            .unwrap();
        a3.broadcast_to::<NDims<2>>(&[2, 3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn broadcast_to_invalid_shape() {
        let a3 = (1..)
            .take(6)
            .collect::<Array<_, _>>()
            .into_shape([1, 2, 3])
            .unwrap();
        a3.broadcast_to::<NDims<3>>(&[2, 3, 3]).unwrap();
    }

    #[test]
    fn broadcast_to() -> Result<()> {
        let a1 = Array::from(vec![1; 6]);
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
    fn concatenate() {
        let a = (0_usize..6)
            .collect::<Array<_, _>>()
            .into_shape([1, 2, 3])
            .unwrap();
        {
            let actual = Array::concatenate(&[a.view(), a.view()], 0).unwrap();
            let expected = array!([[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]);

            assert_eq!(actual, expected);
        }
        {
            let actual = Array::concatenate(&[a.view(), a.view()], 1).unwrap();
            let expected = array!([[[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]]]);

            assert_eq!(actual, expected);
        }
        {
            let actual = Array::concatenate(&[a.view(), a.view()], 2).unwrap();
            let expected = array!([[[0, 1, 2, 0, 1, 2], [3, 4, 5, 3, 4, 5]]]);

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn concatenate_arrays_of_different_order() {
        let a = array!([[1, 2], [3, 4]]);
        let concatenated: Array<_, _, ColumnMajor> =
            Array::concatenate(&[a.view(), a.view()], 0).unwrap();
        let actual = concatenated.storage;
        let expected = crate::storage::StorageBase::<Vec<_>>::from(vec![1, 3, 1, 3, 2, 4, 2, 4]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn concatenate_zero_length_arrays() {
        let a: Array<usize, _> = array!([]).into_shape([1, 0, 2]).unwrap();
        {
            let actual = Array::concatenate(&[a.view(), a.view()], 0).unwrap();
            let expected = array!([]).into_shape([2, 0, 2]).unwrap();

            assert_eq!(actual, expected);
        }
        {
            let actual = Array::concatenate(&[a.view(), a.view()], 1).unwrap();
            let expected = array!([]).into_shape([1, 0, 2]).unwrap();

            assert_eq!(actual, expected);
        }
        {
            let actual = Array::concatenate(&[a.view(), a.view()], 2).unwrap();
            let expected = array!([]).into_shape([1, 0, 4]).unwrap();

            assert_eq!(actual, expected);
        }
    }

    #[test]
    #[should_panic]
    fn convert_empty_array_to_ambiguous_shape() {
        let a = Array::from(Vec::<usize>::new());
        a.to_shape([2, 0, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_empty_array_into_ambiguous_shape() {
        let a = Array::from(Vec::<usize>::new());
        a.into_shape([2, 0, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_to_ambiguous_shape() {
        let a = array!([1, 2, 3, 4]);
        a.to_shape([2, -1, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_into_ambiguous_shape() {
        let a = array!([1, 2, 3, 4]);
        a.into_shape([2, -1, -1]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_to_incompatible_shape() {
        let a = array!([1, 2, 3, 4]);
        a.to_shape([2, 2, 2]).unwrap();
    }

    #[test]
    #[should_panic]
    fn convert_into_incompatible_shape() {
        let a = array!([1, 2, 3, 4]);
        a.into_shape([2, 2, 2]).unwrap();
    }

    #[test]
    fn convert_to_new_shape() -> Result<()> {
        let a1 = (1..).take(24).collect::<Array<_, _>>();
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
            let a1 = (1..).take(24).collect::<Array<_, _>>();
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
            let a1 = (1..).take(24).collect::<Array<_, _>>();
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
        let a = Array::from(data.clone());
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
        let a = Array::from(data.clone());
        let subject = a.into_shape([4])?;

        assert_eq!(subject.shape, [4]);
        assert_eq!(subject.strides, [1]);
        assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data));
        assert_eq!(subject.offset, 0);

        Ok(())
    }

    #[test]
    fn expand_shape() {
        let a = array!([[1, 2], [3, 4]]);
        {
            let subject = a.expand_shape(0).unwrap();

            assert_eq!(
                subject.iter().cloned().collect::<Vec<_>>(),
                vec![1, 2, 3, 4]
            );
            assert_eq!(subject.shape(), &[1, 2, 2]);
        }
        {
            let subject = a.expand_shape(2).unwrap();

            assert_eq!(
                subject.iter().cloned().collect::<Vec<_>>(),
                vec![1, 2, 3, 4]
            );
            assert_eq!(subject.shape(), &[2, 2, 1]);
        }
        {
            let subject = a.expand_shape(-1).unwrap();

            assert_eq!(
                subject.iter().cloned().collect::<Vec<_>>(),
                vec![1, 2, 3, 4]
            );
            assert_eq!(subject.shape(), &[2, 2, 1]);
        }
        {
            let subject = a.expand_shape(-3).unwrap();

            assert_eq!(
                subject.iter().cloned().collect::<Vec<_>>(),
                vec![1, 2, 3, 4]
            );
            assert_eq!(subject.shape(), &[1, 2, 2]);
        }
    }

    #[test]
    fn fill() -> Result<()> {
        let mut a3 = (1..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape([2, 3, 4])?;
        a3.fill(7);

        assert!(a3.iter().all(|&x| x == 7));

        Ok(())
    }

    #[test]
    fn flip_along_axes() {
        let a = (0..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape([2, 3, 4])
            .unwrap();

        macro_rules! check {
            ($axis:expr) => {
                let subject = a.flip_along_axes(&[$axis]).unwrap();

                assert_eq!(subject.storage, a.storage.view());
                assert_eq!(a.shape(), subject.shape());

                let mut actual_strides = subject.strides().clone();
                actual_strides[$axis as usize] *= -1;

                assert_eq!(&actual_strides, a.strides());
            };
        }

        check!(0);
        check!(1);
        check!(2);
    }

    #[test]
    fn flip_twice() -> Result<()> {
        let a = (0..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape([2, 3, 4])
            .unwrap();

        assert_eq!(a.flip()?.flip()?, a.view());

        Ok(())
    }

    #[test]
    fn is_empty() -> Result<()> {
        let a1 = Array::from(Vec::<usize>::new());
        let a3 = Array::from(Vec::<usize>::new()).into_shape([3, 2, 0])?;

        assert!(a1.is_empty());
        assert!(a3.is_empty());

        Ok(())
    }

    #[test]
    fn len() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..).take(24).collect::<Array<_, _>>().into_shape(shape)?;

        assert_eq!(a3.len(), shape.iter().map(|&x| x as usize).product());

        Ok(())
    }

    #[test]
    fn ndims() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..).take(24).collect::<Array<_, _>>().into_shape(shape)?;

        assert_eq!(a3.ndims(), shape.len());

        Ok(())
    }

    #[test]
    fn ones() {
        let shape = [2, 3, 4];
        let a3 = Array::<u64, _>::ones(&shape);

        assert_eq!(a3.shape(), &shape);
        assert!(a3.iter().all(|&x| x == 1));
    }

    #[test]
    fn permute() -> Result<()> {
        let a3 = Array::from(vec![1; 24]).into_shape([2, 3, 4])?;
        let a3p = a3.permute([2, 0, 1])?;

        assert_eq!(a3p.shape, [4, 2, 3]);
        assert_eq!(a3p.strides, [1, 12, 4]);
        assert_eq!(a3p.offset, a3.offset);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn permute_by_axis_out_of_bounds() {
        let a3 = Array::from(vec![1; 24]).into_shape([2, 3, 4]).unwrap();
        a3.permute([0, 1, 100]).unwrap();
    }

    #[test]
    #[should_panic]
    fn permute_by_repeated_axes() {
        let a3 = Array::from(vec![1; 24]).into_shape([2, 3, 4]).unwrap();
        a3.permute([0, 1, 0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn permute_by_wrong_number_of_axes() {
        let a3 = Array::from(vec![1; 24]).into_shape([1, 2]).unwrap();
        a3.permute([1, 2]).unwrap();
    }

    #[test]
    fn shape() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..).take(24).collect::<Array<_, _>>().into_shape(shape)?;

        assert_eq!(a3.shape(), &shape.map(|x| x as usize));

        Ok(())
    }

    #[test]
    fn slice_by_index() {
        macro_rules! test {
            ($index:expr, $offset:expr) => {
                let data = vec![1_usize, 2, 3, 4];
                let a = Array::from(data.clone());
                let subject = a.slice(s!($index));

                assert_eq!(subject.shape, []);
                assert_eq!(subject.strides, []);
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
        let a = array!([1, 2, 3, 4]);
        a.slice(s!(10));
    }

    #[test]
    fn slice_by_new_axis() {
        let data = vec![1_usize, 2, 3, 4];
        let a = Array::from(data.clone());
        let subject = a.slice(s!(NewAxis));

        assert_eq!(subject.shape, [1, 4]);
        assert_eq!(subject.strides, [0, 1]);
        assert_eq!(subject.storage, StorageBase::<Vec<_>>::from(data).view());
        assert_eq!(subject.offset, 0);
    }

    #[test]
    fn slice_by_slice() -> Result<()> {
        macro_rules! test {
            ($r:expr, $step:expr, $len:expr, $offset:expr) => {
                let data = vec![1_usize, 2, 3, 4, 5, 6, 7, 8];
                let a = Array::from(data.clone());
                let subject = a.slice(s!($r;$step));

                assert_eq!(subject.shape, [$len]);
                assert_eq!(subject.strides, [$step]);
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
        let a = array!([1, 2, 3, 4]);
        a.slice(s!(NewAxis, 1, ..2));
    }

    #[test]
    fn slice_sliced_array_by_slice() {
        let a = Array::from(vec![1_usize; 32]);

        assert_eq!(a.offset, 0);

        let info = s!(10..);
        let a1 = a.slice(info.clone());

        assert_eq!(a1.offset, 10);

        let a2 = a1.slice(info);

        assert_eq!(a2.offset, 20);
    }

    #[test]
    fn slice_zero_length_array() {
        let a: Array<usize, _> = array!([]).into_shape([0, 1, 2]).unwrap();
        let subject = a.slice(s!(.., .., ..));

        assert_eq!(subject.shape(), &[0, 1, 2]);
        assert_eq!(subject.strides(), &[2, 2, 1]);
    }

    #[test]
    fn stack() {
        let a = (0_usize..6)
            .collect::<Array<_, _>>()
            .into_shape([1, 2, 3])
            .unwrap();
        {
            let actual = Array::stack(&[a.view(), a.view()], 0).unwrap();
            let expected = array!([[[[0, 1, 2], [3, 4, 5]]], [[[0, 1, 2], [3, 4, 5]]]]);

            assert_eq!(actual, expected);
        }
        {
            let actual = Array::stack(&[a.view(), a.view()], 1).unwrap();
            let expected = array!([[[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]]);

            assert_eq!(actual, expected);
        }
        {
            let actual = Array::stack(&[a.view(), a.view()], 2).unwrap();
            let expected = array!([[[[0, 1, 2], [0, 1, 2]], [[3, 4, 5], [3, 4, 5]]]]);

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn strides() -> Result<()> {
        let shape = [2_isize, 3, 4];
        let a3 = (1..).take(24).collect::<Array<_, _>>().into_shape(shape)?;

        assert_eq!(
            a3.strides(),
            &shape.map(|x| x as usize).to_default_strides::<RowMajor>()
        );

        Ok(())
    }

    #[test]
    fn transpose() -> Result<()> {
        let a2 = array!([[1, 2, 3], [4, 5, 6]]);
        let a2t = a2.transpose();

        assert_eq!(a2t.shape(), &[3, 2]);
        for (&actual, expected) in a2t.iter().zip([1_usize, 4, 2, 5, 3, 6]) {
            assert_eq!(actual, expected);
        }

        Ok(())
    }

    #[test]
    fn view() {
        let a = array!([1, 2, 3]);
        let subject = a.view();

        assert_eq!(subject.storage, a.storage.view());
    }

    #[test]
    fn zeros() {
        let shape = [2, 3, 4];
        let a3 = Array::<u64, _>::zeros(&shape);

        assert_eq!(a3.shape(), &shape);
        assert!(a3.iter().all(|&x| x == 0));
    }
}

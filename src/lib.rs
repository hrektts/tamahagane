#![allow(incomplete_features)]
#![feature(
    associated_type_defaults,
    const_option,
    const_trait_impl,
    core_intrinsics,
    exact_size_is_empty,
    generic_arg_infer,
    generic_const_exprs,
    pointer_byte_offsets,
    unchecked_math
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

mod array;
pub use array::{Array, ArrayBase, Iter, IterMut};

mod array_index;
pub use array_index::{ArrayIndex, NewAxis};

mod dimension;
pub use dimension::{
    DimDiff, Dimensionality, DimensionalityAdd, DimensionalityAfterDot, DimensionalityDiff,
    DimensionalityMax, DynDim, DynDimDiff, NDims,
};

mod error;
pub use error::{Error, Result, ShapeError};

mod linalg;
pub use linalg::Dot;

mod order;
pub use order::{ColumnMajor, Order, RowMajor};

mod routine;

mod slice;
pub use slice::Slice;

mod slice_info;
pub use slice_info::SliceInfo;

mod shape;
pub use shape::{Shape, SignedShape};

mod util;

pub mod storage;

use num_traits::{One, Zero};

use storage::{Storage, StorageMut, StorageOwned};

pub trait NDArray {
    type D: Dimensionality;
    type O: Order;
    type S: Storage;
    type Iter<'a>: Iterator<Item = &'a <Self::S as Storage>::Elem>
    where
        <Self::S as Storage>::Elem: 'a;
    type CowWithD<'a, D2>: NDArray<D = D2, O = Self::O, S = <Self::S as Storage>::Cow<'a>>
    where
        Self: 'a,
        D2: Dimensionality;
    type CowWithDO<'a, D2, O2>: NDArray<D = D2, O = O2, S = <Self::S as Storage>::Cow<'a>>
    where
        Self: 'a,
        D2: Dimensionality,
        O2: Order;
    type Owned: NDArray<D = Self::D, O = Self::O, S = <Self::S as Storage>::Owned>;
    type View<'a>: NDArray<D = Self::D, O = Self::O, S = <Self::S as Storage>::View<'a>>
    where
        Self: 'a;
    type ViewWithD<'a, D2>: NDArray<D = D2, O = Self::O, S = <Self::S as Storage>::View<'a>>
    where
        Self: 'a,
        D2: Dimensionality;

    fn as_ptr(&self) -> *const <Self::S as Storage>::Elem;
    fn broadcast_to<BD>(
        &self,
        shape: &<BD as Dimensionality>::Shape,
    ) -> Result<Self::ViewWithD<'_, BD>>
    where
        BD: Dimensionality;
    #[allow(clippy::type_complexity)]
    fn expand_shape(
        &self,
        axis: isize,
    ) -> Result<Self::CowWithD<'_, <<
        <Self::D as DimensionalityAdd<NDims<1>>>::Output
            as Dimensionality>::SignedShape as SignedShape>::Dimensionality>>
    where
        Self::D: DimensionalityAdd<NDims<1>>;
    fn is_empty(&self) -> bool;
    fn iter<'a>(&self) -> Self::Iter<'a>;
    fn len(&self) -> usize;
    fn ndims(&self) -> usize;
    fn permute(&self, axes: <Self::D as Dimensionality>::Shape) -> Result<Self::View<'_>>;
    fn shape(&self) -> &<Self::D as Dimensionality>::Shape;
    #[allow(clippy::type_complexity)]
    fn slice<ST, SD>(
        &self,
        info: SliceInfo<ST, SD>,
    ) -> Self::ViewWithD<'_, <Self::D as DimensionalityAdd<SD>>::Output>
    where
        Self::D: DimensionalityAdd<SD>,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>;
    fn strides(&self) -> &<<Self::D as Dimensionality>::Shape as Shape>::Strides;
    fn to_owned_array(&self) -> Self::Owned;
    fn to_shape<Sh2>(
        &self,
        shape: Sh2,
    ) -> Result<Self::CowWithD<'_, <Sh2 as SignedShape>::Dimensionality>>
    where
        Sh2: SignedShape;
    fn to_shape_with_order<Sh2, O2>(
        &self,
        shape: Sh2,
    ) -> Result<Self::CowWithDO<'_, <Sh2 as SignedShape>::Dimensionality, O2>>
    where
        O2: Order,
        Sh2: SignedShape;
    fn transpose(&self) -> Self::View<'_>;
    fn view(&self) -> Self::View<'_>;
}

pub trait NDArrayMut: NDArray
where
    <Self as NDArray>::S: StorageMut,
{
    type ViewMutWithD<'a, D2>: NDArrayMut<
        D = D2,
        O = Self::O,
        S = <Self::S as StorageMut>::ViewMut<'a>,
    >
    where
        Self: 'a,
        D2: Dimensionality;

    fn fill(&mut self, value: <Self::S as Storage>::Elem);
    fn iter_mut<'a>(&mut self) -> IterMut<'a, <Self::S as Storage>::Elem, Self::D>;
    #[allow(clippy::type_complexity)]
    fn slice_mut<ST, SD>(
        &mut self,
        info: SliceInfo<ST, SD>,
    ) -> Self::ViewMutWithD<'_, <Self::D as DimensionalityAdd<SD>>::Output>
    where
        Self::D: DimensionalityAdd<SD>,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>;
}

pub trait NDArrayOwned: NDArray
where
    <Self as NDArray>::S: StorageOwned,
{
    type WithD<D2>: NDArrayOwned<D = D2, O = Self::O, S = Self::S>
    where
        D2: Dimensionality;

    fn allocate_uninitialized<Sh>(shape: &Sh) -> Self
    where
        Sh: Shape<Dimensionality = Self::D>;
    fn concatenate<T>(arrays: &[T], axis: isize) -> Result<Self>
    where
        Self: Sized,
        T: NDArray,
        <<T as NDArray>::D as Dimensionality>::Shape: Shape<Dimensionality = Self::D>,
        <T as NDArray>::S: Storage<Elem = <Self::S as Storage>::Elem>;
    fn into_shape<Sh2>(
        self,
        shape: Sh2,
    ) -> Result<Self::WithD<<Sh2 as SignedShape>::Dimensionality>>
    where
        Sh2: SignedShape;
    fn into_shape_with_order<Sh2, O2>(
        self,
        shape: Sh2,
    ) -> Result<Self::WithD<<Sh2 as SignedShape>::Dimensionality>>
    where
        O2: Order,
        Sh2: SignedShape;
    fn ones<Sh>(shape: &Sh) -> Self
    where
        <Self::S as Storage>::Elem: One,
        Sh: Shape<Dimensionality = Self::D>;
    fn zeros<Sh>(shape: &Sh) -> Self
    where
        <Self::S as Storage>::Elem: Zero,
        Sh: Shape<Dimensionality = Self::D>;
}

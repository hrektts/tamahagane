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
    type Dimensionality: Dimensionality;
    type Order: Order;
    type Storage: Storage;
    type Iter<'a>: Iterator<Item = &'a <Self::Storage as Storage>::Elem>
    where
        <Self::Storage as Storage>::Elem: 'a;
    type CowWithD<'a, D2>: NDArray<
        Dimensionality = D2,
        Order = Self::Order,
        Storage = <Self::Storage as Storage>::Cow<'a>,
    >
    where
        Self: 'a,
        D2: Dimensionality;
    type CowWithDO<'a, D2, O2>: NDArray<
        Dimensionality = D2,
        Order = O2,
        Storage = <Self::Storage as Storage>::Cow<'a>,
    >
    where
        Self: 'a,
        D2: Dimensionality,
        O2: Order;
    type Owned: NDArray<
        Dimensionality = Self::Dimensionality,
        Order = Self::Order,
        Storage = <Self::Storage as Storage>::Owned,
    >;
    type View<'a>: NDArray<
        Dimensionality = Self::Dimensionality,
        Order = Self::Order,
        Storage = <Self::Storage as Storage>::View<'a>,
    >
    where
        Self: 'a;
    type ViewWithD<'a, D2>: NDArray<
        Dimensionality = D2,
        Order = Self::Order,
        Storage = <Self::Storage as Storage>::View<'a>,
    >
    where
        Self: 'a,
        D2: Dimensionality;
    fn as_ptr(&self) -> *const <Self::Storage as Storage>::Elem;
    fn broadcast_to<BD>(
        &self,
        shape: &<BD as Dimensionality>::Shape,
    ) -> Result<Self::ViewWithD<'_, BD>>
    where
        BD: Dimensionality;
    fn flip(&self) -> Result<Self::View<'_>>;
    fn flip_along_axes(&self, axes: &[isize]) -> Result<Self::View<'_>>;
    #[allow(clippy::type_complexity)]
    fn expand_shape(
        &self,
        axis: isize,
    ) -> Result<Self::CowWithD<'_, <<
        <Self::Dimensionality as DimensionalityAdd<DimDiff<1>>>::Output
            as Dimensionality>::SignedShape as SignedShape>::Dimensionality>>
    where
        Self::Dimensionality: DimensionalityAdd<DimDiff<1>>;
    fn is_empty(&self) -> bool;
    fn iter<'a>(&self) -> Self::Iter<'a>;
    fn len(&self) -> usize;
    fn ndims(&self) -> usize;
    fn permute(
        &self,
        axes: <Self::Dimensionality as Dimensionality>::Shape,
    ) -> Result<Self::View<'_>>;
    fn shape(&self) -> &<Self::Dimensionality as Dimensionality>::Shape;
    fn slice<ST, SD>(
        &self,
        info: SliceInfo<ST, SD>,
    ) -> Self::ViewWithD<'_, <Self::Dimensionality as DimensionalityAdd<SD>>::Output>
    where
        Self::Dimensionality: DimensionalityAdd<SD>,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>;
    fn strides(&self) -> &<<Self::Dimensionality as Dimensionality>::Shape as Shape>::Strides;
    fn to_owned_array(&self) -> Self::Owned;
    fn to_shape<Sh>(
        &self,
        shape: Sh,
    ) -> Result<Self::CowWithD<'_, <Sh as SignedShape>::Dimensionality>>
    where
        Sh: SignedShape;
    fn to_shape_with_order<Sh, O2>(
        &self,
        shape: Sh,
    ) -> Result<Self::CowWithDO<'_, <Sh as SignedShape>::Dimensionality, O2>>
    where
        O2: Order,
        Sh: SignedShape;
    fn transpose(&self) -> Self::View<'_>;
    fn view(&self) -> Self::View<'_>;
}

pub trait NDArrayMut: NDArray
where
    <Self as NDArray>::Storage: StorageMut,
{
    type ViewMutWithD<'a, D2>: NDArrayMut<
        Dimensionality = D2,
        Order = Self::Order,
        Storage = <Self::Storage as StorageMut>::ViewMut<'a>,
    >
    where
        Self: 'a,
        D2: Dimensionality;
    fn fill(&mut self, value: <Self::Storage as Storage>::Elem);
    fn iter_mut<'a>(
        &mut self,
    ) -> IterMut<'a, <Self::Storage as Storage>::Elem, Self::Dimensionality>;
    fn slice_mut<ST, SD>(
        &mut self,
        info: SliceInfo<ST, SD>,
    ) -> Self::ViewMutWithD<'_, <Self::Dimensionality as DimensionalityAdd<SD>>::Output>
    where
        Self::Dimensionality: DimensionalityAdd<SD>,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>;
}

pub trait NDArrayOwned: NDArray
where
    <Self as NDArray>::Storage: StorageOwned,
{
    type WithD<D2>: NDArrayOwned<Dimensionality = D2, Order = Self::Order, Storage = Self::Storage>
    where
        D2: Dimensionality;
    fn allocate_uninitialized<Sh>(shape: &Sh) -> Self
    where
        Sh: Shape<Dimensionality = Self::Dimensionality>;
    fn concatenate<T>(arrays: &[T], axis: isize) -> Result<Self>
    where
        Self: Sized,
        T: NDArray,
        <<T as NDArray>::Dimensionality as Dimensionality>::Shape:
            Shape<Dimensionality = Self::Dimensionality>,
        <T as NDArray>::Storage: Storage<Elem = <Self::Storage as Storage>::Elem>;
    fn into_shape<Sh>(self, shape: Sh) -> Result<Self::WithD<<Sh as SignedShape>::Dimensionality>>
    where
        Sh: SignedShape;
    fn into_shape_with_order<Sh, O2>(
        self,
        shape: Sh,
    ) -> Result<Self::WithD<<Sh as SignedShape>::Dimensionality>>
    where
        Sh: SignedShape,
        O2: Order;
    fn ones<Sh>(shape: &Sh) -> Self
    where
        <Self::Storage as Storage>::Elem: One,
        Sh: Shape<Dimensionality = Self::Dimensionality>;
    fn stack<T>(arrays: &[T], axis: isize) -> Result<Self>
    where
        Self: Sized,
        T: NDArray,
        <T as NDArray>::Dimensionality: DimensionalityAdd<DimDiff<1>>,
        <<<<<T as NDArray>::Dimensionality as DimensionalityAdd<DimDiff<1>>>::Output
            as Dimensionality>::SignedShape as SignedShape>::Dimensionality
            as Dimensionality>::Shape: Shape<Dimensionality = Self::Dimensionality>,
        <T as NDArray>::Storage: Storage<Elem = <Self::Storage as Storage>::Elem>;
    fn zeros<Sh>(shape: &Sh) -> Self
    where
        <Self::Storage as Storage>::Elem: Zero,
        Sh: Shape<Dimensionality = Self::Dimensionality>;
}

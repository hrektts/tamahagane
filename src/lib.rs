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
pub use array::{Array, ArrayBase, Iter, IterMut, SequenceIter};

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
pub use shape::{NewShape, Shape};

mod util;

pub mod storage;

use num_traits::{One, Zero};

use storage::{Storage, StorageMut, StorageOwned};

pub trait NDArray {
    type D: Dimensionality;
    type O: Order;
    type S: Storage;

    fn broadcast_to<BD>(
        &self,
        shape: &<BD as Dimensionality>::Shape,
    ) -> Result<ArrayBase<<Self::S as Storage>::View<'_>, BD, Self::O>>
    where
        BD: Dimensionality;
    fn is_empty(&self) -> bool;
    fn iter<'a>(&self) -> Iter<'a, <Self::S as Storage>::Elem, Self::D>;
    fn iter_sequence<'a>(
        &self,
        axis: usize,
    ) -> SequenceIter<'a, <Self::S as Storage>::Elem, Self::D>;
    fn len(&self) -> usize;
    fn ndims(&self) -> usize;
    #[allow(clippy::type_complexity)]
    fn permute(
        &self,
        axes: <Self::D as Dimensionality>::Shape,
    ) -> Result<ArrayBase<<Self::S as Storage>::View<'_>, Self::D, Self::O>>;
    fn shape(&self) -> &<Self::D as Dimensionality>::Shape;
    #[allow(clippy::type_complexity)]
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
        ST: AsRef<[ArrayIndex]>;
    fn strides(&self) -> &<<Self::D as Dimensionality>::Shape as Shape>::Strides;
    fn to_owned_array(&self) -> ArrayBase<<Self::S as Storage>::Owned, Self::D, Self::O>;
    #[allow(clippy::type_complexity)]
    fn to_shape<NS>(
        &self,
        shape: NS,
    ) -> Result<ArrayBase<<Self::S as Storage>::Cow<'_>, <NS as NewShape>::Dimensionality, Self::O>>
    where
        NS: NewShape;
    fn to_shape_with_order<NS, NO>(
        &self,
        shape: NS,
    ) -> Result<ArrayBase<<Self::S as Storage>::Cow<'_>, <NS as NewShape>::Dimensionality, NO>>
    where
        NO: Order,
        NS: NewShape;
    fn transpose(&self) -> ArrayBase<<Self::S as Storage>::View<'_>, Self::D, Self::O>;
    fn view(&self) -> ArrayBase<<Self::S as Storage>::View<'_>, Self::D, Self::O>;
}

pub trait NDArrayMut: NDArray {
    type SM: StorageMut;

    fn fill(&mut self, value: <Self::S as Storage>::Elem);
    fn iter_mut<'a>(&mut self) -> IterMut<'a, <Self::S as Storage>::Elem, Self::D>;
    #[allow(clippy::type_complexity)]
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
        ST: AsRef<[ArrayIndex]>;
}

pub trait NDArrayOwned: NDArray {
    type SO: StorageOwned;

    fn allocate_uninitialized<Sh>(shape: &Sh) -> Self
    where
        Sh: Shape<Dimensionality = Self::D>;
    fn into_shape<NS>(
        self,
        shape: NS,
    ) -> Result<ArrayBase<Self::S, <NS as NewShape>::Dimensionality, Self::O>>
    where
        NS: NewShape;
    fn into_shape_with_order<NS, NO>(
        self,
        shape: NS,
    ) -> Result<ArrayBase<Self::S, <NS as NewShape>::Dimensionality, NO>>
    where
        NO: Order,
        NS: NewShape;
    fn ones<Sh>(shape: &Sh) -> Self
    where
        <Self::S as Storage>::Elem: One,
        Sh: Shape<Dimensionality = Self::D>;
    fn zeros<Sh>(shape: &Sh) -> Self
    where
        <Self::S as Storage>::Elem: Zero,
        Sh: Shape<Dimensionality = Self::D>;
}

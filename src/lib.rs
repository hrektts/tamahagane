#![feature(
    associated_type_defaults,
    const_option,
    const_trait_impl,
    exact_size_is_empty,
    generic_associated_types,
    generic_const_exprs
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

mod array;
pub use array::{Array, Iter, IterMut, Scalar};

mod array_index;
pub use array_index::{ArrayIndex, NewAxis};

mod dimension;
pub use dimension::{
    DimDiff, Dimensionality, DimensionalityAdd, DimensionalityDiff, DimensionalityMax, DynDim,
    DynDimDiff, NDims,
};

mod error;
pub use error::{Error, Result, ShapeError};

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

pub trait NDArray<S, D, O>
where
    D: Dimensionality,
    O: Order,
    S: Storage,
{
    #[allow(clippy::type_complexity)]
    fn broadcast_to<BD>(
        &self,
        shape: &<BD as Dimensionality>::Shape,
    ) -> Result<Array<<S as Storage>::View<'_>, BD, O>>
    where
        BD: Dimensionality;
    fn is_empty(&self) -> bool;
    fn iter(&self) -> Iter<'_, <S as Storage>::Elem, D>;
    fn len(&self) -> usize;
    fn n_dims(&self) -> usize;
    #[allow(clippy::type_complexity)]
    fn permute(
        &self,
        axes: <D as Dimensionality>::Shape,
    ) -> Result<Array<<S as Storage>::View<'_>, D, O>>;
    fn shape(&self) -> &<D as Dimensionality>::Shape;
    fn slice<ST, SD>(
        &self,
        info: SliceInfo<ST, SD>,
    ) -> Array<<S as Storage>::View<'_>, <D as DimensionalityAdd<SD>>::Output, O>
    where
        D: DimensionalityAdd<SD>,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>;
    fn strides(&self) -> &<<D as Dimensionality>::Shape as Shape>::Strides;
    fn to_owned_array(&self) -> Array<<S as Storage>::Owned, D, O>;
    #[allow(clippy::type_complexity)]
    fn to_shape<NS>(
        &self,
        shape: NS,
    ) -> Result<Array<<S as Storage>::Cow<'_>, <NS as NewShape>::Dimensionality, O>>
    where
        NS: NewShape;
    #[allow(clippy::type_complexity)]
    fn to_shape_with_order<NS, NO>(
        &self,
        shape: NS,
    ) -> Result<Array<<S as Storage>::Cow<'_>, <NS as NewShape>::Dimensionality, NO>>
    where
        NS: NewShape,
        NO: Order;
    fn transpose(&self) -> Array<<S as Storage>::View<'_>, D, O>;
}

pub trait NDArrayMut<S, D, O>: NDArray<S, D, O>
where
    D: Dimensionality,
    O: Order,
    S: StorageMut,
{
    fn fill(&mut self, value: <S as Storage>::Elem);
    fn iter_mut(&mut self) -> IterMut<'_, <S as Storage>::Elem, D>;
    fn slice_mut<ST, SD>(
        &mut self,
        info: SliceInfo<ST, SD>,
    ) -> Array<<S as StorageMut>::ViewMut<'_>, <D as DimensionalityAdd<SD>>::Output, O>
    where
        D: DimensionalityAdd<SD>,
        SD: DimensionalityDiff,
        ST: AsRef<[ArrayIndex]>;
}

pub trait NDArrayOwned<S, D, O>: NDArray<S, D, O>
where
    D: Dimensionality,
    O: Order,
    S: StorageOwned,
{
    fn allocate_uninitialized(shape: &<D as Dimensionality>::Shape) -> Self;
    fn into_shape<NS>(self, shape: NS) -> Result<Array<S, <NS as NewShape>::Dimensionality, O>>
    where
        NS: NewShape;
    fn into_shape_with_order<NS, NO>(
        self,
        shape: NS,
    ) -> Result<Array<S, <NS as NewShape>::Dimensionality, NO>>
    where
        NS: NewShape,
        NO: Order;
    fn ones(shape: &<D as Dimensionality>::Shape) -> Self
    where
        <S as Storage>::Elem: One;
    fn zeros(shape: &<D as Dimensionality>::Shape) -> Self
    where
        <S as Storage>::Elem: Zero;
}

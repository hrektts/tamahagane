mod new_shape;
pub use new_shape::NewShape;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::{
    fmt::Debug,
    hash::Hash,
    ops::{Index, IndexMut},
};

use crate::{Dimensionality, DynDim, NDims, Order};

pub trait Shape:
    AsRef<[usize]>
    + AsMut<[usize]>
    + Clone
    + Debug
    + Eq
    + Hash
    + Index<usize, Output = usize>
    + IndexMut<usize, Output = usize>
    + PartialEq
{
    type Dimensionality: Dimensionality;
    type Strides: AsRef<[isize]>
        + AsMut<[isize]>
        + Clone
        + Debug
        + Eq
        + Hash
        + Index<usize, Output = isize>
        + IndexMut<usize, Output = isize>
        + PartialEq;
    fn array_len(&self) -> usize;
    fn as_associated_shape(&self) -> &<Self::Dimensionality as Dimensionality>::Shape;
    fn n_dims(&self) -> usize;
    fn to_default_strides<O>(&self) -> Self::Strides
    where
        O: Order;
}

impl<const N: usize> Shape for [usize; N] {
    type Dimensionality = NDims<N>;
    type Strides = [isize; N];

    fn array_len(&self) -> usize {
        self.iter().product()
    }

    fn as_associated_shape(&self) -> &<Self::Dimensionality as Dimensionality>::Shape {
        self
    }

    fn n_dims(&self) -> usize {
        N
    }

    fn to_default_strides<O>(&self) -> Self::Strides
    where
        O: Order,
    {
        let mut strides = [0; N];
        O::convert_shape_to_default_strides(self, &mut strides);
        strides
    }
}

impl Shape for Vec<usize> {
    type Dimensionality = DynDim;
    type Strides = Vec<isize>;

    fn array_len(&self) -> usize {
        self.iter().product()
    }

    fn as_associated_shape(&self) -> &<Self::Dimensionality as Dimensionality>::Shape {
        self
    }

    fn n_dims(&self) -> usize {
        self.len()
    }

    fn to_default_strides<O>(&self) -> Self::Strides
    where
        O: Order,
    {
        let mut strides = Vec::<isize>::new();
        strides.reserve(self.len());
        strides.resize(self.len(), 0);
        O::convert_shape_to_default_strides(self, &mut strides);
        strides
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::Shape;
    use crate::RowMajor;

    #[test]
    fn shape_by_array() {
        let shape = [2_usize, 3, 4];

        assert_eq!(shape.array_len(), shape.iter().product());
        assert_eq!(shape.n_dims(), shape.len());
        assert_eq!(shape.to_default_strides::<RowMajor>(), [12, 4, 1]);
    }

    #[test]
    fn shape_by_vec() {
        let shape = vec![2_usize, 3, 4];

        assert_eq!(shape.array_len(), shape.iter().product());
        assert_eq!(shape.n_dims(), shape.len());
        assert_eq!(shape.to_default_strides::<RowMajor>(), [12, 4, 1]);
    }
}

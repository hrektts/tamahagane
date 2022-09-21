mod ops;
pub use ops::{DimensionalityAdd, DimensionalityAfterDot, DimensionalityMax};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::Shape;

pub trait Dimensionality: DimensionalityAdd<DynDimDiff> {
    type Shape: Shape;
    fn shape_zeroed(n_dims: usize) -> Self::Shape;
    fn strides_zeroed(n_dims: usize) -> <Self::Shape as Shape>::Strides;

    fn first_indices(shape: &Self::Shape) -> Option<Self::Shape> {
        if shape.as_ref().iter().any(|&x| x == 0) {
            None
        } else {
            Some(Self::shape_zeroed(shape.n_dims()))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NDims<const N: usize>;

impl<const N: usize> Dimensionality for NDims<N> {
    type Shape = [usize; N];

    fn shape_zeroed(n_dims: usize) -> Self::Shape {
        assert_eq!(n_dims, N);
        [0; N]
    }

    fn strides_zeroed(n_dims: usize) -> <Self::Shape as Shape>::Strides {
        assert_eq!(n_dims, N);
        [0; N]
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DynDim;

impl Dimensionality for DynDim {
    type Shape = Vec<usize>;

    fn shape_zeroed(n_dims: usize) -> Self::Shape {
        let mut shape = Vec::<usize>::new();
        shape.reserve(n_dims);
        shape.resize(n_dims, 0);
        shape
    }

    fn strides_zeroed(n_dims: usize) -> <Self::Shape as Shape>::Strides {
        let mut strides = Vec::<isize>::new();
        strides.reserve(n_dims);
        strides.resize(n_dims, 0);
        strides
    }
}

pub trait DimensionalityDiff {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DimDiff<const N: isize>;

impl<const N: isize> DimensionalityDiff for DimDiff<N> {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DynDimDiff;

impl DimensionalityDiff for DynDimDiff {}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::{Dimensionality, DynDim, NDims};

    #[test]
    fn get_shape_zeroed() {
        assert_eq!(NDims::<3>::shape_zeroed(3), [0; 3]);
        assert_eq!(DynDim::shape_zeroed(3), vec![0; 3]);
    }

    #[test]
    #[should_panic]
    fn get_shape_zeroed_with_invalid_n_dims() {
        NDims::<3>::shape_zeroed(4);
    }

    #[test]
    fn get_strides_zeroed() {
        assert_eq!(NDims::<3>::strides_zeroed(3), [0; 3]);
        assert_eq!(DynDim::strides_zeroed(3), vec![0; 3]);
    }

    #[test]
    #[should_panic]
    fn get_strides_zeroed_with_invalid_n_dims() {
        NDims::<3>::strides_zeroed(4);
    }

    #[test]
    fn get_first_indices() {
        {
            let indices = NDims::<3>::first_indices(&[2, 0, 4]);

            assert!(indices.is_none());
        }
        {
            let indices = DynDim::first_indices(&vec![2, 0, 4]);

            assert!(indices.is_none());
        }
        {
            let indices = NDims::<1>::first_indices(&[4]).unwrap();

            assert_eq!(indices, [0]);
        }
        {
            let indices = DynDim::first_indices(&vec![4]).unwrap();

            assert_eq!(indices, vec![0]);
        }
        {
            let indices = NDims::<3>::first_indices(&[2, 3, 4]).unwrap();

            assert_eq!(indices, [0; 3]);
        }
        {
            let indices = DynDim::first_indices(&vec![2, 3, 4]).unwrap();

            assert_eq!(indices, vec![0; 3]);
        }
    }
}

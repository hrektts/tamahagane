#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::{
    fmt::Debug,
    hash::Hash,
    ops::{Index, IndexMut},
};

use crate::{Dimensionality, DynDim, NDims};
pub trait NewShape:
    AsRef<[isize]>
    + AsMut<[isize]>
    + Clone
    + Debug
    + Eq
    + Hash
    + Index<usize, Output = isize>
    + IndexMut<usize, Output = isize>
    + PartialEq
{
    type Dimensionality: Dimensionality;
    fn ndims(&self) -> usize;
}

impl<const N: usize> NewShape for [isize; N] {
    type Dimensionality = NDims<N>;

    fn ndims(&self) -> usize {
        N
    }
}

impl NewShape for Vec<isize> {
    type Dimensionality = DynDim;

    fn ndims(&self) -> usize {
        self.len()
    }
}

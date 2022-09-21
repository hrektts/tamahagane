use crate::{
    util::{If, True},
    Dimensionality,
};

use super::{DimDiff, DynDim, DynDimDiff, NDims};

pub trait DimensionalityAdd<Rhs> {
    type Output: Dimensionality;
}

impl<const M: isize, const N: usize> DimensionalityAdd<DimDiff<M>> for NDims<N>
where
    If<{ (N as isize + M) >= 0 }>: True,
    NDims<{ (N as isize + M) as usize }>: Sized,
{
    type Output = NDims<{ (N as isize + M) as usize }>;
}

impl<const N: usize> DimensionalityAdd<DynDimDiff> for NDims<N> {
    type Output = DynDim;
}

impl<const N: isize> DimensionalityAdd<DimDiff<N>> for DynDim {
    type Output = DynDim;
}

impl DimensionalityAdd<DynDimDiff> for DynDim {
    type Output = DynDim;
}

pub trait DimensionalityAfterDot<Rhs> {
    type Output: Dimensionality;
}

impl<const M: usize, const N: usize> DimensionalityAfterDot<NDims<M>> for NDims<N>
where
    If<{ M > 0 }>: True,
    If<{ N > 0 }>: True,
    NDims<{ M + N - 2 }>: Sized,
{
    type Output = NDims<{ M + N - 2 }>;
}

impl<const N: usize> DimensionalityAfterDot<NDims<0>> for NDims<N>
where
    If<{ N > 0 }>: True,
{
    type Output = NDims<N>;
}

impl<const N: usize> DimensionalityAfterDot<NDims<N>> for NDims<0> {
    type Output = NDims<N>;
}

impl<const N: usize> DimensionalityAfterDot<DynDim> for NDims<N> {
    type Output = DynDim;
}

impl<const N: usize> DimensionalityAfterDot<NDims<N>> for DynDim {
    type Output = DynDim;
}

impl DimensionalityAfterDot<DynDim> for DynDim {
    type Output = DynDim;
}

pub trait DimensionalityMax<Rhs> {
    type Output: Dimensionality;
}

pub const fn max(v1: usize, v2: usize) -> usize {
    if v1 > v2 {
        v1
    } else {
        v2
    }
}

impl<const M: usize, const N: usize> DimensionalityMax<NDims<M>> for NDims<N>
where
    NDims<{ max(M, N) }>: Sized,
{
    type Output = NDims<{ max(M, N) }>;
}

impl<const N: usize> DimensionalityMax<DynDim> for NDims<N> {
    type Output = DynDim;
}

impl<const N: usize> DimensionalityMax<NDims<N>> for DynDim {
    type Output = DynDim;
}

impl DimensionalityMax<DynDim> for DynDim {
    type Output = DynDim;
}

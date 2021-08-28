use core::{
    marker::PhantomData,
    num::Wrapping,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    },
};

use num_complex::Complex;

use super::Array;
use crate::{
    routine,
    storage::{Storage, StorageMut, StorageOwned},
    Dimensionality, DimensionalityMax, NDArray, NDArrayMut, NDArrayOwned, Order, Shape,
};

macro_rules! impl_unary_op {
    ($trait:ident, $op:ident) => {
        impl<D, O, S, T> $trait for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            T: $trait<Output = T> + Clone,
        {
            type Output = Array<T, S, D, O>;

            fn $op(mut self) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op();
                }
                self
            }
        }

        impl<D, O, S, T> $trait for &Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: Storage<T>,
            T: $trait<Output = T> + Clone,
        {
            type Output = Array<T, <S as Storage<T>>::Owned<T>, D, O>;

            fn $op(self) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.clone().$op();
                }
                out
            }
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Not, not);

fn convert_strides<D, D1>(
    strides: &<<D as Dimensionality>::Shape as Shape>::Strides,
    n_dims: usize,
) -> <<<D as DimensionalityMax<D1>>::Output as Dimensionality>::Shape as Shape>::Strides
where
    D: Dimensionality + DimensionalityMax<D1>,
    D1: Dimensionality,
{
    let mut out_strides =
        <<D as DimensionalityMax<D1>>::Output as Dimensionality>::strides_zeroed(n_dims);
    for (out_stride, stride) in out_strides.as_mut().iter_mut().zip(strides.as_ref()) {
        *out_stride = *stride;
    }
    out_strides
}

macro_rules! impl_binary_op {
    ($trait:ident, $op:ident) => {
        impl<D, D1, O, S, S1, T, T1> $trait<Array<T1, S1, D1, O>> for Array<T, S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            S1: Storage<T1>,
            T: $trait<T1, Output = T> + Clone,
            T1: Clone,
        {
            type Output = Array<T, S, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(self, rhs: Array<T1, S1, D1, O>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<D, D1, O, S, S1, T, T1> $trait<&Array<T1, S1, D1, O>> for Array<T, S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            S1: Storage<T1>,
            T: $trait<T1, Output = T> + Clone,
            T1: Clone,
        {
            type Output = Array<T, S, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(mut self, rhs: &Array<T1, S1, D1, O>) -> Self::Output {
                let out_shape = routine::broadcast_shape::<D, D1>(&self.shape, &rhs.shape).unwrap();

                if self.shape.as_ref() == rhs.shape.as_ref() {
                    for (dst, src) in self.iter_mut().zip(rhs.iter()) {
                        *dst = dst.clone().$op(src.clone());
                    }
                    Array {
                        strides: convert_strides::<D, D1>(&self.strides, out_shape.n_dims()),
                        shape: out_shape,
                        storage: self.storage,
                        offset: self.offset,
                        phantom: PhantomData,
                    }
                } else if self.shape.as_ref() == out_shape.as_ref() {
                    for (dst, src) in self.iter_mut().zip(
                        rhs.broadcast_to::<<D as DimensionalityMax<D1>>::Output>(&out_shape)
                            .unwrap()
                            .iter(),
                    ) {
                        *dst = dst.clone().$op(src.clone());
                    }
                    Array {
                        strides: convert_strides::<D, D1>(&self.strides, out_shape.n_dims()),
                        shape: out_shape,
                        storage: self.storage,
                        offset: self.offset,
                        phantom: PhantomData,
                    }
                } else {
                    let mut out = Self::Output::allocate_uninitialized(&out_shape);
                    for (dst, (l, r)) in out.iter_mut().zip(self.iter().zip(rhs.iter())) {
                        *dst = l.clone().$op(r.clone());
                    }
                    out
                }
            }
        }

        impl<D, D1, O, S, S1, T, T1> $trait<Array<T1, S1, D1, O>> for &Array<T, S, D, O>
        where
            D: Dimensionality,
            D1: Dimensionality + DimensionalityMax<D>,
            O: Order,
            S: Storage<T>,
            S1: StorageMut<T1> + StorageOwned<T1>,
            T: Clone,
            T1: $trait<T, Output = T1> + Clone,
        {
            type Output = Array<T1, S1, <D1 as DimensionalityMax<D>>::Output, O>;

            fn $op(self, rhs: Array<T1, S1, D1, O>) -> Self::Output {
                rhs.$op(self)
            }
        }

        impl<D, D1, O, S, S1, T, T1> $trait<&Array<T1, S1, D1, O>> for &Array<T, S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: Storage<T>,
            S1: Storage<T1>,
            T: $trait<T1, Output = T> + Clone,
            T1: Clone,
        {
            type Output =
                Array<T, <S as Storage<T>>::Owned<T>, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(self, rhs: &Array<T1, S1, D1, O>) -> Self::Output {
                let out_shape = routine::broadcast_shape::<D, D1>(&self.shape, &rhs.shape).unwrap();
                let mut out = Self::Output::allocate_uninitialized(&out_shape);
                for (dst, (l, r)) in out.iter_mut().zip(self.iter().zip(rhs.iter())) {
                    *dst = l.clone().$op(r.clone());
                }
                out
            }
        }

        impl<D, O, S, T, T1> $trait<Wrapping<T1>> for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            T: $trait<Wrapping<T1>, Output = T> + Clone,
            T1: Copy,
        {
            type Output = Array<T, S, D, O>;

            fn $op(mut self, rhs: Wrapping<T1>) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op(rhs);
                }
                self
            }
        }

        impl<D, O, S, T, T1> $trait<Wrapping<T1>> for &Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: Storage<T>,
            T: $trait<Wrapping<T1>, Output = T> + Clone,
            T1: Copy,
        {
            type Output = Array<T, <S as Storage<T>>::Owned<T>, D, O>;

            fn $op(self, rhs: Wrapping<T1>) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.clone().$op(rhs);
                }
                out
            }
        }

        impl<D, O, S, T, T1> $trait<Complex<T1>> for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            T: $trait<Complex<T1>, Output = T> + Clone,
            T1: Copy,
        {
            type Output = Array<T, S, D, O>;

            fn $op(mut self, rhs: Complex<T1>) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op(rhs);
                }
                self
            }
        }

        impl<D, O, S, T, T1> $trait<Complex<T1>> for &Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: Storage<T>,
            T: $trait<Complex<T1>, Output = T> + Clone,
            T1: Copy,
        {
            type Output = Array<T, <S as Storage<T>>::Owned<T>, D, O>;

            fn $op(self, rhs: Complex<T1>) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.clone().$op(rhs);
                }
                out
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(BitAnd, bitand);
impl_binary_op!(BitOr, bitor);
impl_binary_op!(BitXor, bitxor);
impl_binary_op!(Div, div);
impl_binary_op!(Mul, mul);
impl_binary_op!(Rem, rem);
impl_binary_op!(Shl, shl);
impl_binary_op!(Shr, shr);
impl_binary_op!(Sub, sub);

macro_rules! impl_binary_op_with_scalar {
    ($trait:ident, $op:ident, $scalar_type:ty) => {
        impl<D, O, S, T> $trait<$scalar_type> for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            T: $trait<$scalar_type, Output = T> + Clone,
        {
            type Output = Array<T, S, D, O>;

            fn $op(mut self, rhs: $scalar_type) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op(rhs);
                }
                self
            }
        }

        impl<D, O, S, T> $trait<$scalar_type> for &Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: Storage<T>,
            T: $trait<$scalar_type, Output = T> + Clone,
        {
            type Output = Array<T, <S as Storage<T>>::Owned<T>, D, O>;

            fn $op(self, rhs: $scalar_type) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.clone().$op(rhs);
                }
                out
            }
        }
    };
}

macro_rules! impl_all_binary_op_with_scalar {
    ($scalar_type:ty) => {
        impl_binary_op_with_scalar!(Add, add, $scalar_type);
        impl_binary_op_with_scalar!(BitAnd, bitand, $scalar_type);
        impl_binary_op_with_scalar!(BitOr, bitor, $scalar_type);
        impl_binary_op_with_scalar!(BitXor, bitxor, $scalar_type);
        impl_binary_op_with_scalar!(Div, div, $scalar_type);
        impl_binary_op_with_scalar!(Mul, mul, $scalar_type);
        impl_binary_op_with_scalar!(Rem, rem, $scalar_type);
        impl_binary_op_with_scalar!(Shl, shl, $scalar_type);
        impl_binary_op_with_scalar!(Shr, shr, $scalar_type);
        impl_binary_op_with_scalar!(Sub, sub, $scalar_type);
    };
}

impl_all_binary_op_with_scalar!(bool);
impl_all_binary_op_with_scalar!(usize);
impl_all_binary_op_with_scalar!(u8);
impl_all_binary_op_with_scalar!(u16);
impl_all_binary_op_with_scalar!(u32);
impl_all_binary_op_with_scalar!(u64);
#[cfg(has_i128)]
impl_all_binary_op_with_scalar!(u128);
impl_all_binary_op_with_scalar!(isize);
impl_all_binary_op_with_scalar!(i8);
impl_all_binary_op_with_scalar!(i16);
impl_all_binary_op_with_scalar!(i32);
impl_all_binary_op_with_scalar!(i64);
#[cfg(has_i128)]
impl_all_binary_op_with_scalar!(i128);
impl_all_binary_op_with_scalar!(f32);
impl_all_binary_op_with_scalar!(f64);

macro_rules! impl_binary_op_for_scalar {
    ($trait:ident, $op:ident, $scalar_type:ty) => {
        impl<D, O, S, T> $trait<Array<T, S, D, O>> for $scalar_type
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T> + StorageOwned<T>,
            T: $trait<$scalar_type, Output = T> + Clone,
        {
            type Output = Array<T, S, D, O>;

            fn $op(self, rhs: Array<T, S, D, O>) -> Self::Output {
                rhs.$op(self)
            }
        }

        impl<D, O, S, T> $trait<&Array<T, S, D, O>> for $scalar_type
        where
            D: Dimensionality,
            O: Order,
            S: Storage<T>,
            T: $trait<$scalar_type, Output = T> + Clone,
        {
            type Output = Array<T, <S as Storage<T>>::Owned<T>, D, O>;

            fn $op(self, rhs: &Array<T, S, D, O>) -> Self::Output {
                rhs.$op(self)
            }
        }
    };
}

macro_rules! impl_all_binary_op_for_scalar {
    ($scalar_type:ty) => {
        impl_binary_op_for_scalar!(Add, add, $scalar_type);
        impl_binary_op_for_scalar!(BitAnd, bitand, $scalar_type);
        impl_binary_op_for_scalar!(BitOr, bitor, $scalar_type);
        impl_binary_op_for_scalar!(BitXor, bitxor, $scalar_type);
        impl_binary_op_for_scalar!(Div, div, $scalar_type);
        impl_binary_op_for_scalar!(Mul, mul, $scalar_type);
        impl_binary_op_for_scalar!(Rem, rem, $scalar_type);
        impl_binary_op_for_scalar!(Shl, shl, $scalar_type);
        impl_binary_op_for_scalar!(Shr, shr, $scalar_type);
        impl_binary_op_for_scalar!(Sub, sub, $scalar_type);
    };
}

impl_all_binary_op_for_scalar!(bool);
impl_all_binary_op_for_scalar!(usize);
impl_all_binary_op_for_scalar!(u8);
impl_all_binary_op_for_scalar!(u16);
impl_all_binary_op_for_scalar!(u32);
impl_all_binary_op_for_scalar!(u64);
#[cfg(has_i128)]
impl_all_binary_op_for_scalar!(u128);
impl_all_binary_op_for_scalar!(isize);
impl_all_binary_op_for_scalar!(i8);
impl_all_binary_op_for_scalar!(i16);
impl_all_binary_op_for_scalar!(i32);
impl_all_binary_op_for_scalar!(i64);
#[cfg(has_i128)]
impl_all_binary_op_for_scalar!(i128);
impl_all_binary_op_for_scalar!(f32);
impl_all_binary_op_for_scalar!(f64);

macro_rules! impl_binary_assign_op {
    ($trait:ident, $op:ident) => {
        impl<D, D1, O, S, S1, T, T1> $trait<&Array<T1, S1, D1, O>> for Array<T, S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut<T>,
            S1: Storage<T1>,
            T: $trait<T1> + Clone,
            T1: Clone,
        {
            fn $op(&mut self, rhs: &Array<T1, S1, D1, O>) {
                if self.shape.as_ref() == rhs.shape.as_ref() {
                    for (dst, src) in self.iter_mut().zip(rhs.iter()) {
                        dst.$op(src.clone());
                    }
                } else {
                    let out_shape =
                        routine::broadcast_shape::<D, D1>(&self.shape, &rhs.shape).unwrap();
                    if self.shape.as_ref() == out_shape.as_ref() {
                        for (dst, src) in self.iter_mut().zip(
                            rhs.broadcast_to::<<D as DimensionalityMax<D1>>::Output>(&out_shape)
                                .unwrap()
                                .iter(),
                        ) {
                            dst.$op(src.clone());
                        }
                    } else {
                        panic!(
                            "cannot broadcast array from shape {:?} to {:?}",
                            rhs.shape, self.shape
                        );
                    }
                }
            }
        }

        impl<D, O, S, T, T1> $trait<Wrapping<T1>> for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T>,
            T: $trait<Wrapping<T1>> + Clone,
            T1: Copy,
        {
            fn $op(&mut self, rhs: Wrapping<T1>) {
                for elem in self.iter_mut() {
                    elem.$op(rhs)
                }
            }
        }

        impl<D, O, S, T, T1> $trait<Complex<T1>> for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T>,
            T: $trait<Complex<T1>> + Clone,
            T1: Copy,
        {
            fn $op(&mut self, rhs: Complex<T1>) {
                for elem in self.iter_mut() {
                    elem.$op(rhs)
                }
            }
        }
    };
}

impl_binary_assign_op!(AddAssign, add_assign);
impl_binary_assign_op!(BitAndAssign, bitand_assign);
impl_binary_assign_op!(BitOrAssign, bitor_assign);
impl_binary_assign_op!(BitXorAssign, bitxor_assign);
impl_binary_assign_op!(DivAssign, div_assign);
impl_binary_assign_op!(MulAssign, mul_assign);
impl_binary_assign_op!(RemAssign, rem_assign);
impl_binary_assign_op!(ShlAssign, shl_assign);
impl_binary_assign_op!(ShrAssign, shr_assign);
impl_binary_assign_op!(SubAssign, sub_assign);

macro_rules! impl_binary_assign_op_with_scalar {
    ($trait:ident, $op:ident, $scalar_type:ty) => {
        impl<D, O, S, T> $trait<$scalar_type> for Array<T, S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<T>,
            T: $trait<$scalar_type> + Clone,
        {
            fn $op(&mut self, rhs: $scalar_type) {
                for elem in self.iter_mut() {
                    elem.$op(rhs)
                }
            }
        }
    };
}

macro_rules! impl_all_binary_assign_op_with_scalar {
    ($scalar_type:ty) => {
        impl_binary_assign_op_with_scalar!(AddAssign, add_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(BitAndAssign, bitand_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(BitOrAssign, bitor_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(BitXorAssign, bitxor_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(DivAssign, div_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(MulAssign, mul_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(RemAssign, rem_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(ShlAssign, shl_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(ShrAssign, shr_assign, $scalar_type);
        impl_binary_assign_op_with_scalar!(SubAssign, sub_assign, $scalar_type);
    };
}

impl_all_binary_assign_op_with_scalar!(bool);
impl_all_binary_assign_op_with_scalar!(usize);
impl_all_binary_assign_op_with_scalar!(u8);
impl_all_binary_assign_op_with_scalar!(u16);
impl_all_binary_assign_op_with_scalar!(u32);
impl_all_binary_assign_op_with_scalar!(u64);
#[cfg(has_i128)]
impl_all_binary_assign_op_with_scalar!(u128);
impl_all_binary_assign_op_with_scalar!(isize);
impl_all_binary_assign_op_with_scalar!(i8);
impl_all_binary_assign_op_with_scalar!(i16);
impl_all_binary_assign_op_with_scalar!(i32);
impl_all_binary_assign_op_with_scalar!(i64);
#[cfg(has_i128)]
impl_all_binary_assign_op_with_scalar!(i128);
impl_all_binary_assign_op_with_scalar!(f32);
impl_all_binary_assign_op_with_scalar!(f64);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use crate::{Array, NDArray, NDArrayOwned};

    #[test]
    fn unary_ops() {
        let a3 = (0_isize..)
            .take(24)
            .collect::<Array<_, _, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        {
            let subject = -a3.to_owned_array();

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((0..).map(|x| -x)).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = -&a3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((0..).map(|x| -x)).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }

    #[test]
    fn binary_ops() {
        let a3 = (0_usize..)
            .take(24)
            .collect::<Array<_, _, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let b3 = (10_usize..)
            .take(24)
            .collect::<Array<_, _, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        {
            let subject = a3.to_owned_array() + b3.to_owned_array();

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((10_usize..).step_by(2)).enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = a3.to_owned_array() + &b3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((10_usize..).step_by(2)).enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = &a3 + b3.to_owned_array();

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((10_usize..).step_by(2)).enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = &a3 + &b3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((10_usize..).step_by(2)).enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = a3.to_owned_array() + 3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip(3_usize..).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = &a3 + 3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip(3_usize..).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = 3 + b3.to_owned_array();

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip(13_usize..).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = 3 + &b3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip(13_usize..).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }

    #[test]
    fn binary_assign_ops() {
        let a3 = (0_usize..)
            .take(24)
            .collect::<Array<_, _, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let b3 = (10_usize..)
            .take(24)
            .collect::<Array<_, _, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        {
            let mut subject = a3.to_owned_array();
            subject += &b3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip((10_usize..).step_by(2)).enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let mut subject = a3.to_owned_array();
            subject += 3;

            assert_eq!(subject.len(), 24);
            for (i, (&actual, expected)) in subject.iter().zip(3_usize..).enumerate() {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }
}

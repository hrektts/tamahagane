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

pub trait Scalar: Copy {}

impl Scalar for bool {}
impl Scalar for usize {}
impl Scalar for u8 {}
impl Scalar for u16 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
#[cfg(has_i128)]
impl Scalar for u128 {}
impl Scalar for isize {}
impl Scalar for i8 {}
impl Scalar for i16 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
#[cfg(has_i128)]
impl Scalar for i128 {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl<T> Scalar for Complex<T> where T: Copy {}
impl<T> Scalar for Wrapping<T> where T: Copy {}

macro_rules! impl_unary_op {
    ($trait:ident, $op:ident) => {
        impl<D, O, S> $trait for Array<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<Output = <S as Storage>::Elem>,
        {
            type Output = Array<S, D, O>;

            fn $op(mut self) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op();
                }
                self
            }
        }

        impl<D, O, S> $trait for &Array<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: $trait<Output = <S as Storage>::Elem>,
        {
            type Output = Array<<S as Storage>::Owned, D, O>;

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
        impl<D, D1, O, S, S1> $trait<Array<S1, D1, O>> for Array<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<<S1 as Storage>::Elem, Output = <S as Storage>::Elem>,
            S1: Storage,
        {
            type Output = Array<S, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(self, rhs: Array<S1, D1, O>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<D, D1, O, S, S1> $trait<&Array<S1, D1, O>> for Array<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<<S1 as Storage>::Elem, Output = <S as Storage>::Elem>,
            S1: Storage,
        {
            type Output = Array<S, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(mut self, rhs: &Array<S1, D1, O>) -> Self::Output {
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

        impl<D, D1, O, S, S1> $trait<Array<S1, D1, O>> for &Array<S, D, O>
        where
            D: Dimensionality,
            D1: Dimensionality + DimensionalityMax<D>,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: $trait<<S1 as Storage>::Elem, Output = <S1 as Storage>::Elem>,
            S1: StorageMut + StorageOwned,
        {
            type Output = Array<S1, <D1 as DimensionalityMax<D>>::Output, O>;

            fn $op(self, mut rhs: Array<S1, D1, O>) -> Self::Output {
                let out_shape = routine::broadcast_shape::<D1, D>(&rhs.shape, &self.shape).unwrap();

                if self.shape.as_ref() == rhs.shape.as_ref() {
                    for (dst, src) in rhs.iter_mut().zip(self.iter()) {
                        *dst = src.clone().$op(dst.clone());
                    }
                    Array {
                        strides: convert_strides::<D1, D>(&rhs.strides, out_shape.n_dims()),
                        shape: out_shape,
                        storage: rhs.storage,
                        offset: rhs.offset,
                        phantom: PhantomData,
                    }
                } else if self.shape.as_ref() == out_shape.as_ref() {
                    for (dst, src) in rhs.iter_mut().zip(
                        self.broadcast_to::<<D1 as DimensionalityMax<D>>::Output>(&out_shape)
                            .unwrap()
                            .iter(),
                    ) {
                        *dst = src.clone().$op(dst.clone());
                    }
                    Array {
                        strides: convert_strides::<D1, D>(&rhs.strides, out_shape.n_dims()),
                        shape: out_shape,
                        storage: rhs.storage,
                        offset: rhs.offset,
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

        impl<D, D1, O, S, S1> $trait<&Array<S1, D1, O>> for &Array<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: $trait<<S1 as Storage>::Elem, Output = <S as Storage>::Elem>,
            S1: Storage,
        {
            type Output = Array<<S as Storage>::Owned, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(self, rhs: &Array<S1, D1, O>) -> Self::Output {
                let out_shape = routine::broadcast_shape::<D, D1>(&self.shape, &rhs.shape).unwrap();
                let mut out = Self::Output::allocate_uninitialized(&out_shape);
                for (dst, (l, r)) in out.iter_mut().zip(self.iter().zip(rhs.iter())) {
                    *dst = l.clone().$op(r.clone());
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
    ($trait:ident, $op:ident) => {
        impl<D, O, S, T> $trait<T> for Array<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<T, Output = <S as Storage>::Elem>,
            T: Scalar,
        {
            type Output = Array<S, D, O>;

            fn $op(mut self, rhs: T) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op(rhs);
                }
                self
            }
        }

        impl<D, O, S, T> $trait<T> for &Array<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: $trait<T, Output = <S as Storage>::Elem>,
            T: Scalar,
        {
            type Output = Array<<S as Storage>::Owned, D, O>;

            fn $op(self, rhs: T) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.clone().$op(rhs);
                }
                out
            }
        }
    };
}

impl_binary_op_with_scalar!(Add, add);
impl_binary_op_with_scalar!(BitAnd, bitand);
impl_binary_op_with_scalar!(BitOr, bitor);
impl_binary_op_with_scalar!(BitXor, bitxor);
impl_binary_op_with_scalar!(Div, div);
impl_binary_op_with_scalar!(Mul, mul);
impl_binary_op_with_scalar!(Rem, rem);
impl_binary_op_with_scalar!(Shl, shl);
impl_binary_op_with_scalar!(Shr, shr);
impl_binary_op_with_scalar!(Sub, sub);

macro_rules! impl_binary_op_for_scalar {
    ($trait:ident, $op:ident, $scalar_type:ty) => {
        impl<D, O, S> $trait<Array<S, D, O>> for $scalar_type
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<Elem = $scalar_type> + StorageOwned<Elem = $scalar_type>,
        {
            type Output = Array<S, D, O>;

            fn $op(self, mut rhs: Array<S, D, O>) -> Self::Output {
                for elem in rhs.iter_mut() {
                    *elem = self.$op(elem.clone())
                }
                rhs
            }
        }

        impl<D, O, S> $trait<&Array<S, D, O>> for $scalar_type
        where
            D: Dimensionality,
            O: Order,
            S: Storage<Elem = $scalar_type>,
        {
            type Output = Array<<S as Storage>::Owned, D, O>;

            fn $op(self, rhs: &Array<S, D, O>) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&rhs.shape);
                for (dst, src) in out.iter_mut().zip(rhs.iter()) {
                    *dst = self.$op(src.clone())
                }
                out
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

impl_binary_op_for_scalar!(BitAnd, bitand, bool);
impl_binary_op_for_scalar!(BitOr, bitor, bool);
impl_binary_op_for_scalar!(BitXor, bitxor, bool);

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

impl_binary_op_for_scalar!(Add, add, f32);
impl_binary_op_for_scalar!(Div, div, f32);
impl_binary_op_for_scalar!(Mul, mul, f32);
impl_binary_op_for_scalar!(Rem, rem, f32);
impl_binary_op_for_scalar!(Sub, sub, f32);

impl_binary_op_for_scalar!(Add, add, f64);
impl_binary_op_for_scalar!(Div, div, f64);
impl_binary_op_for_scalar!(Mul, mul, f64);
impl_binary_op_for_scalar!(Rem, rem, f64);
impl_binary_op_for_scalar!(Sub, sub, f64);

macro_rules! impl_binary_op_for_scalar_wrapped {
    ($trait:ident, $op:ident, $wrapper:ident) => {
        impl<D, O, S, T> $trait<Array<S, D, O>> for $wrapper<T>
        where
            Self: $trait<<S as Storage>::Elem, Output = <S as Storage>::Elem>,
            D: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            T: Copy,
        {
            type Output = Array<S, D, O>;

            fn $op(self, mut rhs: Array<S, D, O>) -> Self::Output {
                for elem in rhs.iter_mut() {
                    *elem = self.$op(elem.clone())
                }
                rhs
            }
        }

        impl<D, O, S, T> $trait<&Array<S, D, O>> for $wrapper<T>
        where
            Self: $trait<<S as Storage>::Elem, Output = <S as Storage>::Elem>,
            D: Dimensionality,
            O: Order,
            S: Storage,
            T: Copy,
        {
            type Output = Array<<S as Storage>::Owned, D, O>;

            fn $op(self, rhs: &Array<S, D, O>) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&rhs.shape);
                for (dst, src) in out.iter_mut().zip(rhs.iter()) {
                    *dst = self.$op(src.clone())
                }
                out
            }
        }
    };
}

macro_rules! impl_all_binary_op_for_scalar_wrapped {
    ($wrapper:ident) => {
        impl_binary_op_for_scalar_wrapped!(Add, add, $wrapper);
        impl_binary_op_for_scalar_wrapped!(BitAnd, bitand, $wrapper);
        impl_binary_op_for_scalar_wrapped!(BitOr, bitor, $wrapper);
        impl_binary_op_for_scalar_wrapped!(BitXor, bitxor, $wrapper);
        impl_binary_op_for_scalar_wrapped!(Div, div, $wrapper);
        impl_binary_op_for_scalar_wrapped!(Mul, mul, $wrapper);
        impl_binary_op_for_scalar_wrapped!(Rem, rem, $wrapper);
        impl_binary_op_for_scalar_wrapped!(Shl, shl, $wrapper);
        impl_binary_op_for_scalar_wrapped!(Shr, shr, $wrapper);
        impl_binary_op_for_scalar_wrapped!(Sub, sub, $wrapper);
    };
}

impl_all_binary_op_for_scalar_wrapped!(Complex);
impl_all_binary_op_for_scalar_wrapped!(Wrapping);

macro_rules! impl_binary_assign_op {
    ($trait:ident, $op:ident) => {
        impl<D, D1, O, S, S1> $trait<&Array<S1, D1, O>> for Array<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut,
            <S as Storage>::Elem: $trait<<S1 as Storage>::Elem> + Clone,
            S1: Storage,
        {
            fn $op(&mut self, rhs: &Array<S1, D1, O>) {
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
    ($trait:ident, $op:ident) => {
        impl<D, O, S, T> $trait<T> for Array<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut,
            <S as Storage>::Elem: $trait<T>,
            T: Scalar,
        {
            fn $op(&mut self, rhs: T) {
                for elem in self.iter_mut() {
                    elem.$op(rhs)
                }
            }
        }
    };
}

impl_binary_assign_op_with_scalar!(AddAssign, add_assign);
impl_binary_assign_op_with_scalar!(BitAndAssign, bitand_assign);
impl_binary_assign_op_with_scalar!(BitOrAssign, bitor_assign);
impl_binary_assign_op_with_scalar!(BitXorAssign, bitxor_assign);
impl_binary_assign_op_with_scalar!(DivAssign, div_assign);
impl_binary_assign_op_with_scalar!(MulAssign, mul_assign);
impl_binary_assign_op_with_scalar!(RemAssign, rem_assign);
impl_binary_assign_op_with_scalar!(ShlAssign, shl_assign);
impl_binary_assign_op_with_scalar!(ShrAssign, shr_assign);
impl_binary_assign_op_with_scalar!(SubAssign, sub_assign);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use crate::{Array, NDArray, NDArrayOwned};

    #[test]
    fn unary_ops() {
        let a3 = (0_isize..)
            .take(24)
            .collect::<Array<_, _>>()
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
        let a3_ = (10_usize..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let a3 = a3_.slice(crate::s!(.., .., 2..));
        let b3_ = (0_usize..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let b3 = b3_.slice(crate::s!(.., .., 2..));
        {
            let subject = a3.to_owned_array() - b3.to_owned_array();

            assert_eq!(subject.len(), 12);
            for (i, &elem) in subject.iter().enumerate() {
                assert_eq!(elem, 10, "{}th element is not equal", i);
            }
        }
        {
            let subject = a3.to_owned_array() - &b3;

            assert_eq!(subject.len(), 12);
            for (i, &elem) in subject.iter().enumerate() {
                assert_eq!(elem, 10, "{}th element is not equal", i);
            }
        }
        {
            let subject = &a3 - b3.to_owned_array();

            assert_eq!(subject.len(), 12);
            for (i, &elem) in subject.iter().enumerate() {
                assert_eq!(elem, 10, "{}th element is not equal", i);
            }
        }
        {
            let subject = &a3 - &b3;

            assert_eq!(subject.len(), 12);
            for (i, &elem) in subject.iter().enumerate() {
                assert_eq!(elem, 10, "{}th element is not equal", i);
            }
        }
        {
            let subject = a3.to_owned_array() - 3;

            assert_eq!(subject.len(), 12);
            for (i, (&actual, expected)) in subject
                .iter()
                .zip([9, 10, 13, 14, 17, 18, 21, 22])
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = &a3 - 3;

            assert_eq!(subject.len(), 12);
            for (i, (&actual, expected)) in subject
                .iter()
                .zip([9, 10, 13, 14, 17, 18, 21, 22])
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = 23 - b3.to_owned_array();

            assert_eq!(subject.len(), 12);
            for (i, (&actual, expected)) in subject
                .iter()
                .zip([21, 20, 17, 16, 13, 12, 9, 8, 5, 4, 1, 0])
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
        {
            let subject = 23 - &b3;

            assert_eq!(subject.len(), 12);
            for (i, (&actual, expected)) in subject
                .iter()
                .zip([21, 20, 17, 16, 13, 12, 9, 8, 5, 4, 1, 0])
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }

    #[test]
    fn binary_assign_ops() {
        let a3_ = (10_usize..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let a3 = a3_.slice(crate::s!(.., .., 2..));
        let b3_ = (0_usize..)
            .take(24)
            .collect::<Array<_, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let b3 = b3_.slice(crate::s!(.., .., 2..));
        {
            let mut subject = a3.to_owned_array();
            subject -= &b3;

            assert_eq!(subject.len(), 12);
            for (i, &elem) in subject.iter().enumerate() {
                assert_eq!(elem, 10, "{}th element is not equal", i);
            }
        }
        {
            let mut subject = a3.to_owned_array();
            subject -= 3;

            assert_eq!(subject.len(), 12);
            for (i, (&actual, expected)) in subject
                .iter()
                .zip([9, 10, 13, 14, 17, 18, 21, 22])
                .enumerate()
            {
                assert_eq!(actual, expected, "{}th element is not equal", i);
            }
        }
    }
}

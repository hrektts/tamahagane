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

use super::ArrayBase;
use crate::{
    routine,
    storage::{Storage, StorageMut, StorageOwned},
    Dimensionality, DimensionalityMax, NDArray, NDArrayMut, NDArrayOwned, Order, Shape,
};

macro_rules! impl_unary_op {
    ($trait:ident, $op:ident) => {
        impl<D, O, S> $trait for ArrayBase<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<Output = <S as Storage>::Elem>,
        {
            type Output = ArrayBase<S, D, O>;

            fn $op(mut self) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op();
                }
                self
            }
        }

        impl<'a, D, O, S> $trait for &'a ArrayBase<S, D, O>
        where
            D: Dimensionality,
            <D as Dimensionality>::Shape: Shape<Dimensionality = D>,
            O: Order,
            S: Storage,
            &'a <S as Storage>::Elem: $trait<Output = <S as Storage>::Elem>,
        {
            type Output = ArrayBase<<S as Storage>::Owned, D, O>;

            fn $op(self) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.$op();
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
        impl<'a, D, D1, O, S, S1, T> $trait<T> for ArrayBase<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            <<D as DimensionalityMax<D1>>::Output as Dimensionality>::Shape:
                Shape<Dimensionality = <D as DimensionalityMax<D1>>::Output>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<&'a <S1 as Storage>::Elem, Output = <S as Storage>::Elem>,
            S1: Storage,
            <S1 as Storage>::Elem: 'a,
            T: NDArray<D = D1, O = O, S = S1>,
        {
            type Output = ArrayBase<S, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(mut self, rhs: T) -> Self::Output {
                let out_shape =
                    routine::broadcast_shape::<D, D1>(&self.shape, rhs.shape()).unwrap();

                if self.shape.as_ref() == rhs.shape().as_ref() {
                    for (dst, src) in self.iter_mut().zip(rhs.iter()) {
                        *dst = dst.clone().$op(src);
                    }
                    ArrayBase {
                        strides: convert_strides::<D, D1>(&self.strides, out_shape.ndims()),
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
                        *dst = dst.clone().$op(src);
                    }
                    ArrayBase {
                        strides: convert_strides::<D, D1>(&self.strides, out_shape.ndims()),
                        shape: out_shape,
                        storage: self.storage,
                        offset: self.offset,
                        phantom: PhantomData,
                    }
                } else {
                    let mut out = Self::Output::allocate_uninitialized(&out_shape);
                    for (dst, (l, r)) in out.iter_mut().zip(self.iter().zip(rhs.iter())) {
                        *dst = l.clone().$op(r);
                    }
                    out
                }
            }
        }

        impl<'a, 'b, D, D1, O, S, S1, T> $trait<T> for &'a ArrayBase<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            <<D as DimensionalityMax<D1>>::Output as Dimensionality>::Shape:
                Shape<Dimensionality = <D as DimensionalityMax<D1>>::Output>,
            D1: Dimensionality,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: 'a,
            &'a <S as Storage>::Elem:
                $trait<&'b <S1 as Storage>::Elem, Output = <S as Storage>::Elem>,
            S1: Storage,
            <S1 as Storage>::Elem: 'b,
            T: NDArray<D = D1, O = O, S = S1>,
        {
            type Output = ArrayBase<<S as Storage>::Owned, <D as DimensionalityMax<D1>>::Output, O>;

            fn $op(self, rhs: T) -> Self::Output {
                let out_shape =
                    routine::broadcast_shape::<D, D1>(&self.shape, rhs.shape()).unwrap();
                let mut out = Self::Output::allocate_uninitialized(&out_shape);
                for (dst, (l, r)) in out.iter_mut().zip(self.iter().zip(rhs.iter())) {
                    *dst = l.$op(r);
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

macro_rules! impl_binary_op_with_type {
    (<$( $param:ident ),*>, $trait:ident, $op:ident, $type:ty) => {
        impl<D, O, S, $( $param ),*> $trait<$type> for ArrayBase<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            <S as Storage>::Elem: $trait<$type, Output = <S as Storage>::Elem>,
            $( $param: Copy ),*
        {
            type Output = ArrayBase<S, D, O>;

            fn $op(mut self, rhs: $type) -> Self::Output {
                for elem in self.iter_mut() {
                    *elem = elem.clone().$op(rhs);
                }
                self
            }
        }

        impl<'a, D, O, S, $( $param ),*> $trait<$type> for &'a ArrayBase<S, D, O>
        where
            D: Dimensionality,
            <D as Dimensionality>::Shape: Shape<Dimensionality = D>,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: 'a,
            &'a <S as Storage>::Elem: $trait<$type, Output = <S as Storage>::Elem>,
            $( $param: Copy ),*
        {
            type Output = ArrayBase<<S as Storage>::Owned, D, O>;

            fn $op(self, rhs: $type) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&self.shape);
                for (dst, src) in out.iter_mut().zip(self.iter()) {
                    *dst = src.$op(rhs);
                }
                out
            }
        }
    };
}

macro_rules! impl_all_binary_op_with_type {
    (< $( $param:ident ),* >, $type:ty) => {
        impl_binary_op_with_type!(<$( $param ),*>, Add, add, $type);
        impl_binary_op_with_type!(<$( $param ),*>, BitAnd, bitand, $type);
        impl_binary_op_with_type!(<$( $param ),*>, BitOr, bitor, $type);
        impl_binary_op_with_type!(<$( $param ),*>, BitXor, bitxor, $type);
        impl_binary_op_with_type!(<$( $param ),*>, Div, div, $type);
        impl_binary_op_with_type!(<$( $param ),*>, Mul, mul, $type);
        impl_binary_op_with_type!(<$( $param ),*>, Rem, rem, $type);
        impl_binary_op_with_type!(<$( $param ),*>, Shl, shl, $type);
        impl_binary_op_with_type!(<$( $param ),*>, Shr, shr, $type);
        impl_binary_op_with_type!(<$( $param ),*>, Sub, sub, $type);
    };
}

impl_all_binary_op_with_type!(<>, bool);
impl_all_binary_op_with_type!(<>, usize);
impl_all_binary_op_with_type!(<>, u8);
impl_all_binary_op_with_type!(<>, u16);
impl_all_binary_op_with_type!(<>, u32);
impl_all_binary_op_with_type!(<>, u64);
#[cfg(has_i128)]
impl_all_binary_op_with_type!(<>, u128);
impl_all_binary_op_with_type!(<>, isize);
impl_all_binary_op_with_type!(<>, i8);
impl_all_binary_op_with_type!(<>, i16);
impl_all_binary_op_with_type!(<>, i32);
impl_all_binary_op_with_type!(<>, i64);
#[cfg(has_i128)]
impl_all_binary_op_with_type!(<>, i128);
impl_all_binary_op_with_type!(<>, f32);
impl_all_binary_op_with_type!(<>, f64);
impl_all_binary_op_with_type!(<T>, Complex<T>);
impl_all_binary_op_with_type!(<T>, Wrapping<T>);

macro_rules! impl_binary_op_for_type {
    ($trait:ident, $op:ident, $type:ty) => {
        impl<D, O, S> $trait<ArrayBase<S, D, O>> for $type
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut<Elem = $type> + StorageOwned<Elem = $type>,
        {
            type Output = ArrayBase<S, D, O>;

            fn $op(self, mut rhs: ArrayBase<S, D, O>) -> Self::Output {
                for elem in rhs.iter_mut() {
                    *elem = self.$op(elem.clone())
                }
                rhs
            }
        }

        impl<D, O, S> $trait<&ArrayBase<S, D, O>> for $type
        where
            D: Dimensionality,
            <D as Dimensionality>::Shape: Shape<Dimensionality = D>,
            O: Order,
            S: Storage<Elem = $type>,
        {
            type Output = ArrayBase<<S as Storage>::Owned, D, O>;

            fn $op(self, rhs: &ArrayBase<S, D, O>) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&rhs.shape);
                for (dst, src) in out.iter_mut().zip(rhs.iter()) {
                    *dst = self.$op(src)
                }
                out
            }
        }
    };
}

macro_rules! impl_all_binary_op_for_type {
    ($type:ty) => {
        impl_binary_op_for_type!(Add, add, $type);
        impl_binary_op_for_type!(BitAnd, bitand, $type);
        impl_binary_op_for_type!(BitOr, bitor, $type);
        impl_binary_op_for_type!(BitXor, bitxor, $type);
        impl_binary_op_for_type!(Div, div, $type);
        impl_binary_op_for_type!(Mul, mul, $type);
        impl_binary_op_for_type!(Rem, rem, $type);
        impl_binary_op_for_type!(Shl, shl, $type);
        impl_binary_op_for_type!(Shr, shr, $type);
        impl_binary_op_for_type!(Sub, sub, $type);
    };
}

impl_binary_op_for_type!(BitAnd, bitand, bool);
impl_binary_op_for_type!(BitOr, bitor, bool);
impl_binary_op_for_type!(BitXor, bitxor, bool);

impl_all_binary_op_for_type!(usize);
impl_all_binary_op_for_type!(u8);
impl_all_binary_op_for_type!(u16);
impl_all_binary_op_for_type!(u32);
impl_all_binary_op_for_type!(u64);
#[cfg(has_i128)]
impl_all_binary_op_for_type!(u128);
impl_all_binary_op_for_type!(isize);
impl_all_binary_op_for_type!(i8);
impl_all_binary_op_for_type!(i16);
impl_all_binary_op_for_type!(i32);
impl_all_binary_op_for_type!(i64);
#[cfg(has_i128)]
impl_all_binary_op_for_type!(i128);

impl_binary_op_for_type!(Add, add, f32);
impl_binary_op_for_type!(Div, div, f32);
impl_binary_op_for_type!(Mul, mul, f32);
impl_binary_op_for_type!(Rem, rem, f32);
impl_binary_op_for_type!(Sub, sub, f32);

impl_binary_op_for_type!(Add, add, f64);
impl_binary_op_for_type!(Div, div, f64);
impl_binary_op_for_type!(Mul, mul, f64);
impl_binary_op_for_type!(Rem, rem, f64);
impl_binary_op_for_type!(Sub, sub, f64);

macro_rules! impl_binary_op_for_wrapper {
    ($trait:ident, $op:ident, $wrapper:ident) => {
        impl<D, O, S, T> $trait<ArrayBase<S, D, O>> for $wrapper<T>
        where
            Self: $trait<<S as Storage>::Elem, Output = <S as Storage>::Elem>,
            D: Dimensionality,
            O: Order,
            S: StorageMut + StorageOwned,
            T: Copy,
        {
            type Output = ArrayBase<S, D, O>;

            fn $op(self, mut rhs: ArrayBase<S, D, O>) -> Self::Output {
                for elem in rhs.iter_mut() {
                    *elem = self.$op(elem.clone())
                }
                rhs
            }
        }

        impl<D, O, S, T> $trait<&ArrayBase<S, D, O>> for $wrapper<T>
        where
            Self: $trait<<S as Storage>::Elem, Output = <S as Storage>::Elem>,
            D: Dimensionality,
            <D as Dimensionality>::Shape: Shape<Dimensionality = D>,
            O: Order,
            S: Storage,
            T: Copy,
        {
            type Output = ArrayBase<<S as Storage>::Owned, D, O>;

            fn $op(self, rhs: &ArrayBase<S, D, O>) -> Self::Output {
                let mut out = Self::Output::allocate_uninitialized(&rhs.shape);
                for (dst, src) in out.iter_mut().zip(rhs.iter()) {
                    *dst = self.$op(src.clone())
                }
                out
            }
        }
    };
}

macro_rules! impl_all_binary_op_for_wrapper {
    ($wrapper:ident) => {
        impl_binary_op_for_wrapper!(Add, add, $wrapper);
        impl_binary_op_for_wrapper!(BitAnd, bitand, $wrapper);
        impl_binary_op_for_wrapper!(BitOr, bitor, $wrapper);
        impl_binary_op_for_wrapper!(BitXor, bitxor, $wrapper);
        impl_binary_op_for_wrapper!(Div, div, $wrapper);
        impl_binary_op_for_wrapper!(Mul, mul, $wrapper);
        impl_binary_op_for_wrapper!(Rem, rem, $wrapper);
        impl_binary_op_for_wrapper!(Shl, shl, $wrapper);
        impl_binary_op_for_wrapper!(Shr, shr, $wrapper);
        impl_binary_op_for_wrapper!(Sub, sub, $wrapper);
    };
}

impl_all_binary_op_for_wrapper!(Complex);
impl_all_binary_op_for_wrapper!(Wrapping);

macro_rules! impl_binary_assign_op {
    ($trait:ident, $op:ident) => {
        impl<'a, D, D1, O, S, S1, T> $trait<T> for ArrayBase<S, D, O>
        where
            D: Dimensionality + DimensionalityMax<D1>,
            D1: Dimensionality,
            O: Order,
            S: StorageMut,
            <S as Storage>::Elem: $trait<&'a <S1 as Storage>::Elem>,
            S1: Storage,
            <S1 as Storage>::Elem: 'a,
            T: NDArray<D = D1, O = O, S = S1>,
        {
            fn $op(&mut self, rhs: T) {
                if self.shape.as_ref() == rhs.shape().as_ref() {
                    for (dst, src) in self.iter_mut().zip(rhs.iter()) {
                        dst.$op(src);
                    }
                } else {
                    let out_shape =
                        routine::broadcast_shape::<D, D1>(&self.shape, &rhs.shape()).unwrap();
                    if self.shape.as_ref() == out_shape.as_ref() {
                        for (dst, src) in self.iter_mut().zip(
                            rhs.broadcast_to::<<D as DimensionalityMax<D1>>::Output>(&out_shape)
                                .unwrap()
                                .iter(),
                        ) {
                            dst.$op(src);
                        }
                    } else {
                        panic!(
                            "cannot broadcast array from shape {:?} to {:?}",
                            rhs.shape(),
                            self.shape
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

macro_rules! impl_binary_assign_op_with_type {
    (<$( $param:ident ),*>, $trait:ident, $op:ident, $type:ty) => {
        impl<D, O, S, $( $param ),*> $trait<$type> for ArrayBase<S, D, O>
        where
            D: Dimensionality,
            O: Order,
            S: StorageMut,
            <S as Storage>::Elem: $trait<$type>,
            $( $param: Copy ),*
        {
            fn $op(&mut self, rhs: $type) {
                for elem in self.iter_mut() {
                    elem.$op(rhs)
                }
            }
        }
    };
}

macro_rules! impl_all_binary_assign_op_with_type {
    (<$( $param:ident ),*>, $type:ty) => {
        impl_binary_assign_op_with_type!(<$( $param ),*>, AddAssign, add_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, BitAndAssign, bitand_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, BitOrAssign, bitor_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, BitXorAssign, bitxor_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, DivAssign, div_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, MulAssign, mul_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, RemAssign, rem_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, ShlAssign, shl_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, ShrAssign, shr_assign, $type);
        impl_binary_assign_op_with_type!(<$( $param ),*>, SubAssign, sub_assign, $type);
    };
}

impl_all_binary_assign_op_with_type!(<>, bool);
impl_all_binary_assign_op_with_type!(<>, usize);
impl_all_binary_assign_op_with_type!(<>, u8);
impl_all_binary_assign_op_with_type!(<>, u16);
impl_all_binary_assign_op_with_type!(<>, u32);
impl_all_binary_assign_op_with_type!(<>, u64);
#[cfg(has_i128)]
impl_all_binary_assign_op_with_type!(<>, u128);
impl_all_binary_assign_op_with_type!(<>, isize);
impl_all_binary_assign_op_with_type!(<>, i8);
impl_all_binary_assign_op_with_type!(<>, i16);
impl_all_binary_assign_op_with_type!(<>, i32);
impl_all_binary_assign_op_with_type!(<>, i64);
#[cfg(has_i128)]
impl_all_binary_assign_op_with_type!(<>, i128);
impl_all_binary_assign_op_with_type!(<>, f32);
impl_all_binary_assign_op_with_type!(<>, f64);
impl_all_binary_assign_op_with_type!(<T>, Complex<T>);
impl_all_binary_assign_op_with_type!(<T>, Wrapping<T>);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use crate::{ArrayBase, NDArray, NDArrayOwned};

    #[test]
    fn unary_ops() {
        let a3 = (0_isize..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
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
            .collect::<ArrayBase<_, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let a3 = a3_.slice(crate::s!(.., .., 2..));
        let b3_ = (0_usize..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
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
            .collect::<ArrayBase<_, _>>()
            .into_shape(vec![2, 3, 4])
            .unwrap();
        let a3 = a3_.slice(crate::s!(.., .., 2..));
        let b3_ = (0_usize..)
            .take(24)
            .collect::<ArrayBase<_, _>>()
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

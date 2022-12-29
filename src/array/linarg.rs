use core::ops::{AddAssign, Mul};

use num_traits::Zero;

use crate::{
    array::iter::SequenceIter, storage::Storage, ArrayBase, Dimensionality, DimensionalityAfterDot,
    Dot, NDArray, NDArrayMut, NDArrayOwned, Order, Shape,
};

macro_rules! impl_dot {
    ($type:ty) => {
        impl<'a, 'b, D, D1, O, S, S1, T> Dot<T> for $type
        where
            D: Dimensionality + DimensionalityAfterDot<D1>,
            <<D as DimensionalityAfterDot<D1>>::Output as Dimensionality>::Shape:
                Shape<Dimensionality = <D as DimensionalityAfterDot<D1>>::Output>,
            D1: Dimensionality,
            O: Order,
            S: Storage,
            <S as Storage>::Elem: AddAssign<<S as Storage>::Elem> + Zero + 'a,
            &'a <S as Storage>::Elem: Mul<&'b <S1 as Storage>::Elem, Output = <S as Storage>::Elem>,
            S1: Storage,
            <S1 as Storage>::Elem: 'b,
            T: NDArray<D = D1, O = O, S = S1>,
        {
            type Output =
                ArrayBase<<S as Storage>::Owned, <D as DimensionalityAfterDot<D1>>::Output, O>;

            fn dot(&self, rhs: T) -> Self::Output {
                let in_n_dims = self.ndims();
                let rhs_n_dims = rhs.ndims();

                if in_n_dims == 0 || rhs_n_dims == 0 {
                    panic!("dot products for 0-dimensional arrays are not supported");
                }

                let match_axis = if rhs_n_dims > 1 { rhs_n_dims - 2 } else { 0 };
                if self.shape()[in_n_dims - 1] != rhs.shape()[match_axis] {
                    panic!(
                        "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim {})",
                        self.shape(),
                        rhs.shape(),
                        self.shape()[in_n_dims - 1],
                        in_n_dims - 1,
                        rhs.shape()[match_axis],
                        match_axis
                    );
                }

                let out_n_dims = in_n_dims + rhs_n_dims - 2;
                let mut out_shape =
                    <D as DimensionalityAfterDot<D1>>::Output::shape_zeroed(out_n_dims);
                for (out_dim, dim) in out_shape.as_mut().iter_mut().zip(
                    self.shape()
                        .as_ref()
                        .iter()
                        .take(in_n_dims - 1)
                        .chain(rhs.shape().as_ref().iter().take(match_axis)),
                ) {
                    *out_dim = *dim
                }
                if rhs_n_dims > 1 {
                    out_shape[out_n_dims - 1] = rhs.shape()[rhs_n_dims - 1];
                }

                let mut out = Self::Output::zeros(&out_shape);
                let mut out_iter = out.iter_mut();
                let in_iters = SequenceIter::new(self, in_n_dims - 1);
                for in_iter in in_iters {
                    let rhs_iters = SequenceIter::new(&rhs, match_axis);
                    for rhs_iter in rhs_iters {
                        if let Some(out_elem) = out_iter.next() {
                            let mut it = in_iter.clone().zip(rhs_iter);
                            if let Some((in_elem, rhs_elem)) = it.next() {
                                *out_elem = in_elem * rhs_elem;
                                for (in_elem, rhs_elem) in it {
                                    *out_elem += in_elem * rhs_elem;
                                }
                            }
                        }
                    }
                }
                out
            }
        }
    };
}

impl_dot!(ArrayBase<S, D, O>);
impl_dot!(&'a ArrayBase<S, D, O>);
impl_dot!(&'a mut ArrayBase<S, D, O>);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use crate::{array, s, storage::StorageBase, ArrayBase, Dot, NDArray, NDArrayOwned};

    #[test]
    fn dot_1d() {
        let a = array!([1, 2, 3]);
        let actual = a.dot(&a.transpose());
        let expected = ArrayBase::from(vec![14]).slice(s![0]).to_owned_array();

        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_2d() {
        let a = array!([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let actual = a.dot(&a.transpose());
        let expected = ArrayBase::from(vec![14, 32, 50, 32, 77, 122, 50, 122, 194])
            .into_shape([3, 3])
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_4d() {
        let a = (1_usize..)
            .take(16)
            .collect::<ArrayBase<_, _>>()
            .into_shape([2, 2, 2, 2])
            .unwrap();
        let actual = a.dot(&a.transpose());
        let expected = ArrayBase::from(vec![
            11, 35, 17, 41, 14, 38, 20, 44, 23, 79, 37, 93, 30, 86, 44, 100, 35, 123, 57, 145, 46,
            134, 68, 156, 47, 167, 77, 197, 62, 182, 92, 212, 59, 211, 97, 249, 78, 230, 116, 268,
            71, 255, 117, 301, 94, 278, 140, 324, 83, 299, 137, 353, 110, 326, 164, 380, 95, 343,
            157, 405, 126, 374, 188, 436,
        ])
        .into_shape([2, 2, 2, 2, 2, 2])
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_of_view() {
        let a = array!([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let view = a.slice(s![..;2, ..;2]);
        let actual = view.dot(&view.transpose());
        let expected = ArrayBase::from(vec![10, 34, 34, 130])
            .into_shape([2, 2])
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_of_zeros() {
        let a = ArrayBase::<StorageBase<Vec<usize>>, _>::zeros(&[3, 3]);

        assert!(a.dot(&a).iter().all(|&x| x == 0));
    }

    #[test]
    fn dot_of_zero_length_arrays() {
        let a: ArrayBase<StorageBase<Vec<usize>>, _> = array!([]).into_shape([2, 0]).unwrap();
        let b: ArrayBase<StorageBase<Vec<usize>>, _> = array!([]).into_shape([0, 2]).unwrap();

        assert_eq!(a.dot(&b), array!([[0, 0], [0, 0]]));
    }
}

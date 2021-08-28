use crate::{Dimensionality, Shape};

pub trait Order: 'static {
    fn compute_new_strides<S>(old_strides: &[isize], new_shape: &S, new_strides: &mut S::Strides)
    where
        S: Shape;
    fn convert_shape_to_strides<S>(shape: &S, strides: &mut S::Strides)
    where
        S: Shape;
    fn name<'a>() -> &'a str;
    fn is_data_contiguous<D>(
        shape: &<D as Dimensionality>::Shape,
        strides: &<<D as Dimensionality>::Shape as Shape>::Strides,
    ) -> bool
    where
        D: Dimensionality;
    fn is_data_aligned_monotonically(shape: &[usize], strides: &[isize]) -> bool;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RowMajor;

impl Order for RowMajor {
    fn compute_new_strides<S>(old_strides: &[isize], new_shape: &S, new_strides: &mut S::Strides)
    where
        S: Shape,
    {
        debug_assert!(!old_strides.is_empty());
        debug_assert!(new_shape.n_dims() > 0);
        debug_assert_eq!(new_strides.as_ref().len(), new_shape.n_dims());

        let mut last_stride = old_strides[old_strides.len() - 1];
        for (stride, dim) in new_strides
            .as_mut()
            .iter_mut()
            .rev()
            .zip(new_shape.as_ref().iter().rev())
        {
            *stride = last_stride;
            last_stride *= *dim as isize;
        }
    }

    fn convert_shape_to_strides<S>(shape: &S, strides: &mut S::Strides)
    where
        S: Shape,
    {
        strides
            .as_mut()
            .iter_mut()
            .zip(shape.as_ref().iter())
            .rfold(1, |acc, (stride, &dim)| {
                *stride = acc;
                acc * (dim as isize).max(1)
            });
    }

    fn name<'a>() -> &'a str {
        r#""row major""#
    }

    fn is_data_contiguous<D>(
        shape: &<D as Dimensionality>::Shape,
        strides: &<<D as Dimensionality>::Shape as Shape>::Strides,
    ) -> bool
    where
        D: Dimensionality,
    {
        let len = shape.array_len();
        if len == 0 || len == 1 {
            return true;
        }

        let mut stride_expected = 1_usize;
        for (&dim, &stride) in shape
            .as_ref()
            .iter()
            .rev()
            .zip(strides.as_ref().iter().rev())
        {
            if dim == 1 {
                continue;
            }
            if stride != stride_expected as isize {
                return false;
            }
            stride_expected *= dim;
        }

        true
    }

    fn is_data_aligned_monotonically(shape: &[usize], strides: &[isize]) -> bool {
        let len = shape.len();
        debug_assert_eq!(len, strides.len());

        let mut stride_expected = shape[len - 1] as isize * strides[len - 1];
        for (&dim, &stride) in shape[..len - 1]
            .iter()
            .rev()
            .zip(strides[..len - 1].iter().rev())
        {
            if stride != stride_expected as isize {
                return false;
            }
            stride_expected *= dim as isize;
        }

        true
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColumnMajor;

impl Order for ColumnMajor {
    fn compute_new_strides<S>(old_strides: &[isize], new_shape: &S, new_strides: &mut S::Strides)
    where
        S: Shape,
    {
        debug_assert!(!old_strides.is_empty());
        debug_assert!(new_shape.n_dims() > 0);
        debug_assert_eq!(new_strides.as_ref().len(), new_shape.n_dims());

        let mut last_stride = old_strides[0];
        for (stride, dim) in new_strides.as_mut().iter_mut().zip(new_shape.as_ref()) {
            *stride = last_stride;
            last_stride *= *dim as isize;
        }
    }

    fn convert_shape_to_strides<S>(shape: &S, strides: &mut S::Strides)
    where
        S: Shape,
    {
        strides
            .as_mut()
            .iter_mut()
            .zip(shape.as_ref().iter())
            .fold(1, |acc, (stride, &dim)| {
                *stride = acc;
                acc * (dim as isize).max(1)
            });
    }

    fn name<'a>() -> &'a str {
        r#""column major""#
    }

    fn is_data_contiguous<D>(
        shape: &<D as Dimensionality>::Shape,
        strides: &<<D as Dimensionality>::Shape as Shape>::Strides,
    ) -> bool
    where
        D: Dimensionality,
    {
        let len = shape.array_len();
        if len == 0 || len == 1 {
            return true;
        }

        let mut stride_expected = 1_usize;
        for (&dim, &stride) in shape.as_ref().iter().zip(strides.as_ref()) {
            if dim == 1 {
                continue;
            }
            if stride != stride_expected as isize {
                return false;
            }
            stride_expected *= dim;
        }

        true
    }

    fn is_data_aligned_monotonically(shape: &[usize], strides: &[isize]) -> bool {
        let len = shape.len();
        debug_assert_eq!(len, strides.len());

        let mut stride_expected = shape[0] as isize * strides[0];
        for (&dim, &stride) in shape[1..].iter().zip(strides[1..].iter()) {
            if stride != stride_expected as isize {
                return false;
            }
            stride_expected *= dim as isize;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use crate::NDims;

    use super::{ColumnMajor, Order, RowMajor};

    #[test]
    fn convert_shape_to_stride_with_c_order() {
        let mut strides = [0_isize; 3];
        RowMajor::convert_shape_to_strides(&[2_usize, 3, 4], &mut strides);

        assert_eq!(strides, [12, 4, 1]);
    }

    #[test]
    fn convert_shape_to_stride_with_f_order() {
        let mut strides = [0_isize; 3];
        ColumnMajor::convert_shape_to_strides(&[2_usize, 3, 4], &mut strides);

        assert_eq!(strides, [1, 2, 6]);
    }

    #[test]
    fn check_whether_data_is_contiguous_with_c_order() {
        assert!(RowMajor::is_data_contiguous::<NDims<3>>(
            &[2, 3, 4],
            &[12, 4, 1]
        ));
        assert!(!RowMajor::is_data_contiguous::<NDims<3>>(
            &[2, 3, 4],
            &[12, -4, 1]
        ));
    }

    #[test]
    fn check_whether_data_is_contiguous_with_f_order() {
        assert!(ColumnMajor::is_data_contiguous::<NDims<3>>(
            &[2, 3, 4],
            &[1, 2, 6]
        ));
        assert!(!ColumnMajor::is_data_contiguous::<NDims<3>>(
            &[2, 3, 4],
            &[1, -2, 6]
        ));
    }
}

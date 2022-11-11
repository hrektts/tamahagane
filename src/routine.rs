use crate::{Dimensionality, DimensionalityMax, Result, Shape, ShapeError};

pub fn broadcast_shape<Lhs, Rhs>(
    lhs_shape: &<Lhs as Dimensionality>::Shape,
    rhs_shape: &<Rhs as Dimensionality>::Shape,
) -> Result<<<Lhs as DimensionalityMax<Rhs>>::Output as Dimensionality>::Shape>
where
    Lhs: Dimensionality + DimensionalityMax<Rhs>,
    Rhs: Dimensionality,
{
    let lhs_n_dims = lhs_shape.ndims();
    let rhs_n_dims = rhs_shape.ndims();
    let mut ret = <Lhs as DimensionalityMax<Rhs>>::Output::shape_zeroed(lhs_n_dims.max(rhs_n_dims));

    let mut compose_shape = |long: &[usize], short: &[usize], diff: usize| -> Result<()> {
        for (dst, src) in ret.as_mut().iter_mut().zip(long.as_ref().iter().take(diff)) {
            *dst = *src;
        }
        for (dst, (l, s)) in ret
            .as_mut()
            .iter_mut()
            .skip(diff)
            .zip(long.as_ref().iter().skip(diff).zip(short.as_ref()))
        {
            if *l == *s {
                *dst = *l;
            } else if *l == 1 {
                *dst = *s;
            } else if *s == 1 {
                *dst = *l;
            } else {
                return Err(ShapeError::IncompatibleShape(
                    "operands cannot be broadcast to a single shape".into(),
                )
                .into());
            }
        }
        Ok(())
    };

    if lhs_n_dims > rhs_n_dims {
        compose_shape(
            lhs_shape.as_ref(),
            rhs_shape.as_ref(),
            lhs_n_dims - rhs_n_dims,
        )?;
    } else {
        compose_shape(
            rhs_shape.as_ref(),
            lhs_shape.as_ref(),
            rhs_n_dims - lhs_n_dims,
        )?;
    }

    Ok(ret)
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use crate::{DynDim, Result};

    #[test]
    #[should_panic]
    fn broadcast_incompatible_shapes() {
        super::broadcast_shape::<DynDim, DynDim>(&vec![1, 2], &vec![3, 4, 5]).unwrap();
    }

    #[test]
    fn broadcast_shapes() -> Result<()> {
        let s0 = super::broadcast_shape::<DynDim, DynDim>(&vec![1, 5], &vec![3, 4, 1])?;
        let s1 = super::broadcast_shape::<DynDim, DynDim>(&vec![3, 4, 1], &vec![1, 5])?;

        assert_eq!(s0, vec![3, 4, 5]);
        assert_eq!(s0, s1);

        Ok(())
    }
}

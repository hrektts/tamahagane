use crate::{Result, ShapeError};

pub fn normalize_axis(axis: isize, n_dims: usize) -> Result<usize> {
    if axis < -(n_dims as isize) || axis >= n_dims as isize {
        return Err(ShapeError::IncompatibleAxis(format!(
            "axis {axis} is out of bounds for array of dimension {n_dims}"
        ))
        .into());
    }

    let ret = if axis < 0 {
        (axis + n_dims as isize) as usize
    } else {
        axis as usize
    };
    Ok(ret)
}

use core::{default::Default, fmt};

use super::Array;
use crate::{dyn_s, storage::Storage, Dimensionality, NDArray, Order};

const NUM_EDGE_ELEMENTS: usize = 3;
const TRUNCATION_THRESHOLD: usize = 1_000;

struct FormatOption {
    num_edge_elements: usize,
}

impl Default for FormatOption {
    fn default() -> Self {
        Self {
            num_edge_elements: NUM_EDGE_ELEMENTS,
        }
    }
}

impl FormatOption {
    fn new(array_len: usize, truncate: bool) -> Self {
        Self::default().without_truncation(truncate || array_len < TRUNCATION_THRESHOLD)
    }

    fn without_truncation(mut self, valid: bool) -> Self {
        if valid {
            self.num_edge_elements = usize::MAX / 2;
        }
        self
    }
}

fn format_array<S, D, O, F>(
    array: &Array<S, D, O>,
    indent: usize,
    option: &FormatOption,
    f: &mut fmt::Formatter<'_>,
    mut fmt: F,
) -> fmt::Result
where
    D: Dimensionality,
    F: Clone + FnMut(&<S as Storage>::Elem, &mut fmt::Formatter<'_>) -> fmt::Result,
    O: Order,
    S: Storage,
{
    if array.is_empty() {
        let n = array.n_dims();
        write!(f, "{}{}", "[".repeat(n), "]".repeat(n))?;
        return Ok(());
    }

    match array.n_dims() {
        0 => {
            for elem in array.iter() {
                fmt(elem, f)?;
            }
        }
        n_dims => {
            f.write_str("[")?;
            let len = array.shape[0];
            if len > option.num_edge_elements * 2 {
                for i in 0..option.num_edge_elements {
                    fmt_indent(i, n_dims, indent, f)?;
                    format_array(
                        &array.slice(dyn_s!(i as isize)),
                        indent + 1,
                        option,
                        f,
                        fmt.clone(),
                    )?;
                }
                fmt_indent(1, n_dims, indent, f)?;
                f.write_str("...")?;
                fmt_indent(1, n_dims, indent, f)?;
                for i in 0..option.num_edge_elements {
                    fmt_indent(i, n_dims, indent, f)?;
                    format_array(
                        &array.slice(dyn_s!(i as isize - option.num_edge_elements as isize)),
                        indent + 1,
                        option,
                        f,
                        fmt.clone(),
                    )?;
                }
            } else {
                for i in 0..len {
                    fmt_indent(i, n_dims, indent, f)?;
                    format_array(
                        &array.slice(dyn_s!(i as isize)),
                        indent + 1,
                        option,
                        f,
                        fmt.clone(),
                    )?;
                }
            }
            f.write_str("]")?;
        }
    }

    Ok(())
}

fn fmt_indent(i: usize, n_dims: usize, indent: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if i != 0 {
        f.write_str(",")?;
        if n_dims > 1 {
            write!(f, "{}", "\n".repeat(n_dims - 1))?;
            write!(f, "{}", " ".repeat(indent))?;
        } else {
            f.write_str(" ")?;
        }
    }
    Ok(())
}

impl<D, S> fmt::Binary for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::Binary,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, O, S> fmt::Debug for Array<S, D, O>
where
    D: Dimensionality,
    O: Order,
    S: Storage,
    <S as Storage>::Elem: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        write!(
            f,
            ", shape={:?}, strides={:?}, order={}",
            self.shape,
            self.strides,
            O::name(),
        )?;
        Ok(())
    }
}

impl<D, S> fmt::Display for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, S> fmt::LowerExp for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::LowerExp,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, S> fmt::LowerHex for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::LowerHex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, S> fmt::Octal for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::Octal,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, S> fmt::Pointer for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::Pointer,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, S> fmt::UpperExp for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::UpperExp,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

impl<D, S> fmt::UpperHex for Array<S, D>
where
    D: Dimensionality,
    S: Storage,
    <S as Storage>::Elem: fmt::UpperHex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let option = FormatOption::new(self.len(), f.alternate());
        format_array(self, 1, &option, f, <_>::fmt)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use crate::{dyn_s, Array, NDArray, NDArrayOwned, Result};

    #[test]
    fn format_empty_arrays() -> Result<()> {
        let a1 = Array::from(Vec::<usize>::new());
        let a2 = a1.to_shape([0, 0])?;
        let a3 = a1.to_shape([2, 0, 4])?;

        assert_eq!(format!("{}", a1), "[]");
        assert_eq!(format!("{}", a2), "[[]]");
        assert_eq!(format!("{}", a3), "[[[]]]");

        Ok(())
    }

    #[test]
    fn format_0d_array() {
        let a1 = Array::from(vec![1_usize]);
        let a0 = a1.slice(dyn_s!(0));

        assert_eq!(format!("{}", a0), "1");
    }

    #[test]
    fn format_1d_array() {
        const N: usize = super::TRUNCATION_THRESHOLD + 10;
        let data = vec![1; N];
        let a1 = Array::from(data.clone());

        assert_eq!(format!("{}", a1), "[1, 1, 1, ..., 1, 1, 1]");
        assert_eq!(format!("{:#}", a1), format!("[{}]", ["1"; N].join(", ")));
    }

    #[test]
    fn format_3d_array() -> Result<()> {
        let a3 = Array::from(vec![1; 50 * 50 * 50]).into_shape([50, 50, 50])?;
        let expected = "\
[[[1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  ...,
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1]],

 [[1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  ...,
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1]],

 [[1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  ...,
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1]],

 ...,

 [[1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  ...,
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1]],

 [[1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  ...,
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1]],

 [[1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  ...,
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1],
  [1, 1, 1, ..., 1, 1, 1]]]";

        assert_eq!(format!("{}", a3), expected);

        Ok(())
    }
}

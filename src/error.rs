#[cfg(not(feature = "std"))]
use alloc::string::String;
use core::num::TryFromIntError;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub enum Error {
    Shape(ShapeError),
    TryFromInt(TryFromIntError),
}

impl From<ShapeError> for Error {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}

impl From<TryFromIntError> for Error {
    fn from(e: TryFromIntError) -> Self {
        Self::TryFromInt(e)
    }
}

#[derive(Clone, Debug)]
pub enum ShapeError {
    IncompatibleAxis(String),
    IncompatibleDimension(String),
    IncompatibleShape(String),
}

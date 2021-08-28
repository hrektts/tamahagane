use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::Slice;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NewAxis;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ArrayIndex {
    Index(isize),
    Slice(Slice),
    NewAxis,
}

impl ArrayIndex {
    pub const fn is_index(&self) -> bool {
        matches!(self, Self::Index(_))
    }

    pub const fn is_new_axis(&self) -> bool {
        matches!(self, Self::NewAxis)
    }

    pub const fn is_slice(&self) -> bool {
        matches!(self, Self::Slice(_))
    }
}

impl const From<isize> for ArrayIndex {
    fn from(v: isize) -> Self {
        Self::Index(v)
    }
}

impl const From<NewAxis> for ArrayIndex {
    fn from(_: NewAxis) -> Self {
        Self::NewAxis
    }
}

impl const From<Range<isize>> for ArrayIndex {
    fn from(v: Range<isize>) -> Self {
        Self::from(Slice::from(v))
    }
}

impl const From<RangeFrom<isize>> for ArrayIndex {
    fn from(v: RangeFrom<isize>) -> Self {
        Self::from(Slice::from(v))
    }
}

impl const From<RangeFull> for ArrayIndex {
    fn from(v: RangeFull) -> Self {
        Self::from(Slice::from(v))
    }
}

impl const From<RangeInclusive<isize>> for ArrayIndex {
    fn from(v: RangeInclusive<isize>) -> Self {
        Self::from(Slice::from(v))
    }
}

impl const From<RangeToInclusive<isize>> for ArrayIndex {
    fn from(v: RangeToInclusive<isize>) -> Self {
        Self::from(Slice::from(v))
    }
}

impl const From<RangeTo<isize>> for ArrayIndex {
    fn from(v: RangeTo<isize>) -> Self {
        Self::from(Slice::from(v))
    }
}

impl const From<Slice> for ArrayIndex {
    fn from(v: Slice) -> Self {
        Self::Slice(v)
    }
}

#[cfg(test)]
mod tests {
    use super::{ArrayIndex, NewAxis};
    use crate::slice::Slice;

    #[test]
    fn convert_from_index() {
        const IDX: ArrayIndex = ArrayIndex::from(3);

        assert!(IDX.is_index());
    }

    #[test]
    fn convert_from_new_axis() {
        const IDX: ArrayIndex = ArrayIndex::from(NewAxis);

        assert!(IDX.is_new_axis());
    }

    #[test]
    fn convert_from_range() {
        const IDX: ArrayIndex = ArrayIndex::from(2..3);

        assert!(IDX.is_slice());
    }

    #[test]
    fn convert_from_range_from() {
        const IDX: ArrayIndex = ArrayIndex::from(2..);

        assert!(IDX.is_slice());
    }

    #[test]
    fn convert_from_range_full() {
        const IDX: ArrayIndex = ArrayIndex::from(..);

        assert!(IDX.is_slice());
    }

    #[test]
    fn convert_from_range_inclusive() {
        const IDX: ArrayIndex = ArrayIndex::from(2..=3);

        assert!(IDX.is_slice());
    }

    #[test]
    fn convert_from_range_to_inclusive() {
        const IDX: ArrayIndex = ArrayIndex::from(..=3);

        assert!(IDX.is_slice());
    }

    #[test]
    fn convert_from_range_to() {
        const IDX: ArrayIndex = ArrayIndex::from(..3);

        assert!(IDX.is_slice());
    }

    #[test]
    fn convert_from_slice() {
        const IDX: ArrayIndex = ArrayIndex::from(Slice::from(3..5));

        assert!(IDX.is_slice());
    }
}

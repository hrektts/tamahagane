use core::{
    num::NonZeroIsize,
    ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Slice {
    pub(crate) start: Bound<isize>,
    pub(crate) end: Bound<isize>,
    pub(crate) step: isize,
}

impl const From<Range<isize>> for Slice {
    fn from(v: Range<isize>) -> Self {
        Self {
            start: Bound::Included(v.start),
            end: Bound::Excluded(v.end),
            step: 1,
        }
    }
}

impl const From<RangeFrom<isize>> for Slice {
    fn from(v: RangeFrom<isize>) -> Self {
        Self {
            start: Bound::Included(v.start),
            end: Bound::Unbounded,
            step: 1,
        }
    }
}

impl const From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Self {
            start: Bound::Unbounded,
            end: Bound::Unbounded,
            step: 1,
        }
    }
}

impl const From<RangeInclusive<isize>> for Slice {
    fn from(v: RangeInclusive<isize>) -> Self {
        Self {
            start: Bound::Included(*v.start()),
            end: Bound::Included(*v.end()),
            step: 1,
        }
    }
}

impl const From<RangeToInclusive<isize>> for Slice {
    fn from(v: RangeToInclusive<isize>) -> Self {
        Self {
            start: Bound::Unbounded,
            end: Bound::Included(v.end),
            step: 1,
        }
    }
}

impl const From<RangeTo<isize>> for Slice {
    fn from(v: RangeTo<isize>) -> Self {
        Self {
            start: Bound::Unbounded,
            end: Bound::Excluded(v.end),
            step: 1,
        }
    }
}

impl Slice {
    fn end_with_dim(&self, dim: usize) -> isize {
        match self.end {
            Bound::Excluded(x) => {
                if x < 0 {
                    (-1).max(x + dim as isize)
                } else {
                    x.min(dim as isize)
                }
            }
            Bound::Included(x) => {
                let c = if self.step > 0 { 1 } else { -1 };
                if x < 0 {
                    (-1).max(x + c + dim as isize)
                } else {
                    (x + c).min(dim as isize)
                }
            }
            Bound::Unbounded => {
                if self.step > 0 {
                    dim as isize
                } else {
                    -1
                }
            }
        }
    }

    pub(crate) fn len_with_dim(&self, dim: usize) -> usize {
        0.max(
            (self.end_with_dim(dim) - self.start_with_dim(dim)
                + self.step
                + if self.step > 0 { -1 } else { 1 })
                / self.step,
        ) as _
    }

    pub(crate) fn start_with_dim(&self, dim: usize) -> isize {
        match self.start {
            Bound::Excluded(_) => unreachable!(),
            Bound::Included(x) => {
                let first = if self.step > 0 { 0 } else { -1 };
                if x < 0 {
                    first.max(x + dim as isize)
                } else {
                    x.min(dim as isize + first)
                }
            }
            Bound::Unbounded => {
                if self.step > 0 {
                    0
                } else {
                    dim as isize - 1
                }
            }
        }
    }

    pub const fn step_by(self, step: NonZeroIsize) -> Self {
        Self {
            step: step.get(),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use core::{num::NonZeroIsize, ops::Bound};

    use super::Slice;
    use crate::Result;

    #[test]
    fn new() {
        {
            const S: Slice = Slice::from(-2..2);

            assert_eq!(S.start, Bound::Included(-2));
            assert_eq!(S.end, Bound::Excluded(2));
            assert_eq!(S.step, 1);
        }
        {
            const S: Slice = Slice::from(2..-2);

            assert_eq!(S.start, Bound::Included(2));
            assert_eq!(S.end, Bound::Excluded(-2));
            assert_eq!(S.step, 1);
        }
        {
            const S: Slice = Slice::from(-2..2).step_by(NonZeroIsize::new(2).unwrap());

            assert_eq!(S.start, Bound::Included(-2));
            assert_eq!(S.end, Bound::Excluded(2));
            assert_eq!(S.step, 2);
        }
        {
            const S: Slice = Slice::from(2..-2).step_by(NonZeroIsize::new(-2).unwrap());

            assert_eq!(S.start, Bound::Included(2));
            assert_eq!(S.end, Bound::Excluded(-2));
            assert_eq!(S.step, -2);
        }
    }

    #[test]
    fn exclusive_end() -> Result<()> {
        let dim = 4_usize;
        for end in -(dim as isize * 2)..-(dim as isize) {
            let s1 = Slice::from(..end);
            let s2 = Slice::from(..end).step_by((-1).try_into()?);

            assert_eq!(s1.end_with_dim(dim), -1, "end: {}", end);
            assert_eq!(s2.end_with_dim(dim), -1, "end: {}", end);
        }
        for end in -(dim as isize)..0 {
            let s1 = Slice::from(..end);
            let s2 = Slice::from(..end).step_by((-1).try_into()?);

            assert_eq!(s1.end_with_dim(dim), end + dim as isize);
            assert_eq!(s2.end_with_dim(dim), end + dim as isize);
        }
        for end in 0_isize..(dim as isize) {
            let s1 = Slice::from(..end);
            let s2 = Slice::from(..end).step_by((-1).try_into()?);

            assert_eq!(s1.end_with_dim(dim), end);
            assert_eq!(s2.end_with_dim(dim), end);
        }
        for end in (dim as isize)..(dim as isize * 2) {
            let s1 = Slice::from(..end);
            let s2 = Slice::from(..end).step_by((-1).try_into()?);

            assert_eq!(s1.end_with_dim(dim), dim as isize, "end: {}", end);
            assert_eq!(s2.end_with_dim(dim), dim as isize, "end: {}", end);
        }

        Ok(())
    }

    #[test]
    fn inclusive_end_with_negative_step() -> Result<()> {
        let dim = 4_usize;
        for end in -(dim as isize * 2)..-(dim as isize) {
            let s = Slice::from(..=end).step_by((-1).try_into()?);

            assert_eq!(s.end_with_dim(dim), -1, "end: {}", end);
        }
        for end in -(dim as isize)..0 {
            let s = Slice::from(..=end).step_by((-1).try_into()?);

            assert_eq!(s.end_with_dim(dim), end - 1 + dim as isize);
        }
        for end in 0_isize..=(dim as isize) {
            let s = Slice::from(..=end).step_by((-1).try_into()?);

            assert_eq!(s.end_with_dim(dim), end - 1);
        }
        for end in (dim as isize + 1)..(dim as isize * 2) {
            let s = Slice::from(..=end).step_by((-1).try_into()?);

            assert_eq!(s.end_with_dim(dim), dim as isize, "end: {}", end);
        }

        Ok(())
    }

    #[test]
    fn inclusive_end_with_positive_step() {
        let dim = 4_usize;
        for end in -(dim as isize * 2)..(-(dim as isize) - 1) {
            let s = Slice::from(..=end);

            assert_eq!(s.end_with_dim(dim), -1, "end: {}", end);
        }
        for end in (-(dim as isize) - 1)..0 {
            let s = Slice::from(..=end);

            assert_eq!(s.end_with_dim(dim), end + 1 + dim as isize);
        }
        for end in 0_isize..(dim as isize) {
            let s = Slice::from(..=end);

            assert_eq!(s.end_with_dim(dim), end + 1);
        }
        for end in (dim as isize)..(dim as isize * 2) {
            let s = Slice::from(..=end);

            assert_eq!(s.end_with_dim(dim), dim as isize, "end: {}", end);
        }
    }

    #[test]
    fn len_with_range_from() {
        assert_eq!(Slice::from(-11..).len_with_dim(10), 10);
        assert_eq!(Slice::from(-10..).len_with_dim(10), 10);
        assert_eq!(Slice::from(-5..).len_with_dim(10), 5);
        assert_eq!(Slice::from(-1..).len_with_dim(10), 1);
        assert_eq!(Slice::from(0..).len_with_dim(10), 10);
        assert_eq!(Slice::from(5..).len_with_dim(10), 5);
        assert_eq!(Slice::from(10..).len_with_dim(10), 0);
        assert_eq!(Slice::from(11..).len_with_dim(10), 0);

        {
            let step = (-1).try_into().unwrap();

            assert_eq!(Slice::from(10..).step_by(step).len_with_dim(10), 10);
            assert_eq!(Slice::from(9..).step_by(step).len_with_dim(10), 10);
            assert_eq!(Slice::from(4..).step_by(step).len_with_dim(10), 5);
            assert_eq!(Slice::from(0..).step_by(step).len_with_dim(10), 1);
            assert_eq!(Slice::from(-1..).step_by(step).len_with_dim(10), 10);
            assert_eq!(Slice::from(-6..).step_by(step).len_with_dim(10), 5);
            assert_eq!(Slice::from(-11..).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(-12..).step_by(step).len_with_dim(10), 0);
        }
        {
            let step = 3.try_into().unwrap();

            assert_eq!(Slice::from(-10..).step_by(step).len_with_dim(10), 4);
            assert_eq!(Slice::from(1..).step_by(step).len_with_dim(10), 3);
        }
        {
            let step = (-3).try_into().unwrap();

            assert_eq!(Slice::from(9..).step_by(step).len_with_dim(10), 4);
            assert_eq!(Slice::from(-2..).step_by(step).len_with_dim(10), 3);
        }
    }

    #[test]
    fn len_with_range() {
        assert_eq!(Slice::from(5..12).len_with_dim(10), 5);
        assert_eq!(Slice::from(0..5).len_with_dim(10), 5);
        assert_eq!(Slice::from(-1..1).len_with_dim(10), 0);
        assert_eq!(Slice::from(-6..-1).len_with_dim(10), 5);
        assert_eq!(Slice::from(-12..-5).len_with_dim(10), 5);

        {
            let step = (-1).try_into().unwrap();

            assert_eq!(Slice::from(12..4).step_by(step).len_with_dim(10), 5);
            assert_eq!(Slice::from(5..0).step_by(step).len_with_dim(10), 5);
            assert_eq!(Slice::from(1..-1).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(-1..-6).step_by(step).len_with_dim(10), 5);
            assert_eq!(Slice::from(-6..-12).step_by(step).len_with_dim(10), 5);
        }
        {
            let step = 3.try_into().unwrap();

            assert_eq!(Slice::from(2..10).step_by(step).len_with_dim(10), 3);
            assert_eq!(Slice::from(-8..-1).step_by(step).len_with_dim(10), 3);
        }
        {
            let step = (-3).try_into().unwrap();

            assert_eq!(Slice::from(9..1).step_by(step).len_with_dim(10), 3);
            assert_eq!(Slice::from(-1..-11).step_by(step).len_with_dim(10), 4);
        }
    }

    #[test]
    fn len_with_range_full() {
        assert_eq!(Slice::from(..).len_with_dim(10), 10);
        assert_eq!(
            Slice::from(..)
                .step_by(2.try_into().unwrap())
                .len_with_dim(10),
            5
        );
        assert_eq!(
            Slice::from(..)
                .step_by((-2).try_into().unwrap())
                .len_with_dim(10),
            5
        );
    }

    #[test]
    fn len_with_range_to() {
        assert_eq!(Slice::from(..-11).len_with_dim(10), 0);
        assert_eq!(Slice::from(..-10).len_with_dim(10), 0);
        assert_eq!(Slice::from(..-9).len_with_dim(10), 1);
        assert_eq!(Slice::from(..-1).len_with_dim(10), 9);
        assert_eq!(Slice::from(..0).len_with_dim(10), 0);
        assert_eq!(Slice::from(..5).len_with_dim(10), 5);
        assert_eq!(Slice::from(..10).len_with_dim(10), 10);
        assert_eq!(Slice::from(..11).len_with_dim(10), 10);

        {
            let step = (-1).try_into().unwrap();

            assert_eq!(Slice::from(..10).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(..9).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(..8).step_by(step).len_with_dim(10), 1);
            assert_eq!(Slice::from(..0).step_by(step).len_with_dim(10), 9);
            assert_eq!(Slice::from(..-1).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(..-2).step_by(step).len_with_dim(10), 1);
            assert_eq!(Slice::from(..-10).step_by(step).len_with_dim(10), 9);
            assert_eq!(Slice::from(..-11).step_by(step).len_with_dim(10), 10);
            assert_eq!(Slice::from(..-12).step_by(step).len_with_dim(10), 10);
        }
        {
            let step = 3.try_into().unwrap();

            assert_eq!(Slice::from(..10).step_by(step).len_with_dim(10), 4);
            assert_eq!(Slice::from(..-2).step_by(step).len_with_dim(10), 3);
        }
        {
            let step = (-3).try_into().unwrap();

            assert_eq!(Slice::from(..1).step_by(step).len_with_dim(10), 3);
            assert_eq!(Slice::from(..-11).step_by(step).len_with_dim(10), 4);
        }
    }

    #[test]
    fn len_with_range_to_inclusive() {
        assert_eq!(Slice::from(..=-12).len_with_dim(10), 0);
        assert_eq!(Slice::from(..=-11).len_with_dim(10), 0);
        assert_eq!(Slice::from(..=-10).len_with_dim(10), 1);
        assert_eq!(Slice::from(..=-1).len_with_dim(10), 10);
        assert_eq!(Slice::from(..=0).len_with_dim(10), 1);
        assert_eq!(Slice::from(..=5).len_with_dim(10), 6);
        assert_eq!(Slice::from(..=9).len_with_dim(10), 10);
        assert_eq!(Slice::from(..=10).len_with_dim(10), 10);

        {
            let step = (-1).try_into().unwrap();

            assert_eq!(Slice::from(..=11).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(..=10).step_by(step).len_with_dim(10), 0);
            assert_eq!(Slice::from(..=9).step_by(step).len_with_dim(10), 1);
            assert_eq!(Slice::from(..=0).step_by(step).len_with_dim(10), 10);
            assert_eq!(Slice::from(..=-1).step_by(step).len_with_dim(10), 1);
            assert_eq!(Slice::from(..=-9).step_by(step).len_with_dim(10), 9);
            assert_eq!(Slice::from(..=-10).step_by(step).len_with_dim(10), 10);
            assert_eq!(Slice::from(..=-11).step_by(step).len_with_dim(10), 10);
        }
        {
            let step = 3.try_into().unwrap();

            assert_eq!(Slice::from(..=9).step_by(step).len_with_dim(10), 4);
            assert_eq!(Slice::from(..=-3).step_by(step).len_with_dim(10), 3);
        }
        {
            let step = (-3).try_into().unwrap();

            assert_eq!(Slice::from(..=1).step_by(step).len_with_dim(10), 3);
            assert_eq!(Slice::from(..=-10).step_by(step).len_with_dim(10), 4);
        }
    }

    #[test]
    fn start_with_negative_step() -> Result<()> {
        let dim = 4_usize;
        for start in -(dim as isize * 2)..-(dim as isize) {
            let s = Slice::from(start..).step_by((-1).try_into()?);

            assert_eq!(s.start_with_dim(dim), -1, "start: {}", start);
        }
        for start in -(dim as isize)..0 {
            let s = Slice::from(start..).step_by((-1).try_into()?);
            assert_eq!(s.start_with_dim(dim), start + dim as isize);
        }
        for start in 0_isize..(dim as isize) {
            let s = Slice::from(start..).step_by((-1).try_into()?);
            assert_eq!(s.start_with_dim(dim), start);
        }
        for start in (dim as isize)..(dim as isize * 2) {
            let s = Slice::from(start..).step_by((-1).try_into()?);
            assert_eq!(s.start_with_dim(dim), dim as isize - 1, "start: {}", start);
        }

        Ok(())
    }

    #[test]
    fn start_with_positive_step() {
        let dim = 4_usize;
        for start in -(dim as isize * 2)..=-(dim as isize) {
            let s = Slice::from(start..);
            assert_eq!(s.start_with_dim(dim), 0, "start: {}", start);
        }
        for start in (-(dim as isize) + 1)..0 {
            let s = Slice::from(start..);
            assert_eq!(s.start_with_dim(dim), start + dim as isize);
        }
        for start in 0_isize..(dim as isize) {
            let s = Slice::from(start..);
            assert_eq!(s.start_with_dim(dim), start);
        }
        for start in (dim as isize)..(dim as isize * 2) {
            let s = Slice::from(start..);
            assert_eq!(s.start_with_dim(dim), dim as isize, "start: {}", start);
        }
    }
}

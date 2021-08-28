#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{ArrayIndex, DimensionalityDiff, DynDimDiff};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SliceInfo<T, D>
where
    D: DimensionalityDiff,
{
    pub dim_diff: isize,
    pub indices: T,
    pub phantom: PhantomData<D>,
}

impl<D, T> AsRef<[ArrayIndex]> for SliceInfo<T, D>
where
    D: DimensionalityDiff,
    T: AsRef<[ArrayIndex]>,
{
    fn as_ref(&self) -> &[ArrayIndex] {
        self.indices.as_ref()
    }
}

impl From<Vec<ArrayIndex>> for SliceInfo<Vec<ArrayIndex>, DynDimDiff> {
    fn from(indices: Vec<ArrayIndex>) -> Self {
        let dim_diff = indices.iter().fold(0_isize, |acc, index| match index {
            ArrayIndex::Index(_) => acc - 1,
            ArrayIndex::Slice(_) => acc,
            ArrayIndex::NewAxis => acc + 1,
        });
        Self {
            dim_diff,
            indices,
            phantom: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! s {
    (@fold $diff:expr, [$( $idx:tt )*] $r:expr;$s:expr) => {
        $crate::SliceInfo::<_, $crate::DimDiff<{ $diff }>> {
            dim_diff: $diff,
            indices: [$( $idx )* $crate::ArrayIndex::from(
                $crate::Slice::from($r).step_by(core::num::NonZeroIsize::new($s).unwrap())
            )],
            phantom: core::marker::PhantomData
        }
    };
    (@fold $diff:expr, [$( $idx:tt )*] $r:expr) => {
        $crate::SliceInfo::<_, $crate::DimDiff<{ update_diff($diff, $r) }>> {
            dim_diff: update_diff($diff, $r),
            indices: [$( $idx )* $crate::ArrayIndex::from($r)],
            phantom: core::marker::PhantomData
        }
    };
    (@fold $diff:expr, [$( $idx:tt )*] $r:expr;$s:expr, $( $t:tt )*) => {
        $crate::s!(@fold
            $diff,
            [$( $idx )* $crate::ArrayIndex::from($crate::Slice::from($r).step_by(core::num::NonZeroIsize::new($s).unwrap())),]
            $( $t )*
        )
    };
    (@fold $diff:expr, [$( $idx:tt )*] $r:expr, $( $t:tt )*) => {
        $crate::s!(@fold
            update_diff($diff, $r),
            [$( $idx )* $crate::ArrayIndex::from($r),]
            $( $t )*
        )
    };
    ($( $t:tt )*) => {{
        #[allow(dead_code)]
        const fn update_diff<R>(diff: isize, r: R) -> isize
        where
            $crate::ArrayIndex: ~const From<R>,
        {
            match $crate::ArrayIndex::from(r) {
                $crate::ArrayIndex::Index(_) => diff - 1,
                $crate::ArrayIndex::Slice(_) => diff,
                $crate::ArrayIndex::NewAxis => diff + 1,
            }
        }

        $crate::s!(@fold 0, [] $( $t )*)
    }};
}

#[macro_export]
macro_rules! dyn_s {
    (@fold [$( $idx:tt )*] $r:expr;$s:expr) => {{
        let indices = [$( $idx )* $crate::ArrayIndex::from($crate::Slice::from($r).step_by(
            core::num::NonZeroIsize::new($s).expect("slice step cannot be zero")
        ))];
        let dim_diff = indices.iter().fold(0, |acc, x| match x {
            $crate::ArrayIndex::Index(_) => acc - 1,
            $crate::ArrayIndex::Slice(_) => acc,
            $crate::ArrayIndex::NewAxis => acc + 1,
        });
        $crate::SliceInfo::<_, $crate::DynDimDiff> {
            dim_diff,
            indices,
            phantom: core::marker::PhantomData,
        }
    }};
    (@fold [$( $idx:tt )*] $r:expr) => {{
        let indices = [$( $idx )* $crate::ArrayIndex::from($r)];
        let dim_diff = indices.iter().fold(0, |acc, x| match x {
            $crate::ArrayIndex::Index(_) => acc - 1,
            $crate::ArrayIndex::Slice(_) => acc,
            $crate::ArrayIndex::NewAxis => acc + 1,
        });
        $crate::SliceInfo::<_, $crate::DynDimDiff> {
            dim_diff,
            indices,
            phantom: core::marker::PhantomData,
        }
    }};
    (@fold [$( $idx:tt )*] $r:expr;$s:expr, $( $t:tt )*) => {
        $crate::dyn_s!(@fold
            [$( $idx )* $crate::ArrayIndex::from($crate::Slice::from($r).step_by(
                core::num::NonZeroIsize::new($s).expect("slice step cannot be zero")
            )),]
            $( $t )*
        )
    };
    (@fold [$( $idx:tt )*] $r:expr, $( $t:tt )*) => {
        $crate::dyn_s!(@fold
            [$( $idx )* $crate::ArrayIndex::from($r),]
            $( $t )*
        )
    };
    ($( $t:tt )*) => {
        $crate::dyn_s!(@fold [] $( $t )*)
    };
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroIsize;

    use super::SliceInfo;
    use crate::{ArrayIndex, DimDiff, DynDimDiff, NewAxis, Result, Slice};

    #[test]
    fn from_vec() -> Result<()> {
        let subject = SliceInfo::from(vec![NewAxis.into(), Slice::from(..).into(), 1.into()]);

        assert!(subject.as_ref()[0].is_new_axis());
        assert!(subject.as_ref()[1].is_slice());
        assert!(subject.as_ref()[2].is_index());

        Ok(())
    }

    #[test]
    fn s_with_index() -> Result<()> {
        const INFO: SliceInfo<[ArrayIndex; 1], DimDiff<-1>> = s!(1);

        assert_eq!(INFO.dim_diff, -1);

        Ok(())
    }

    #[test]
    fn dyn_s_with_index() {
        let x = 1;
        let info: SliceInfo<[ArrayIndex; 1], DynDimDiff> = dyn_s!(x);

        assert_eq!(info.dim_diff, -1);
        assert_eq!(info.indices[0], ArrayIndex::Index(1));
    }

    #[test]
    fn s_with_ranges() {
        const INFO: SliceInfo<[ArrayIndex; 8], DimDiff<0>> =
            s!(2..3, 1.., ..2, 2..=3, ..=4, .., 2..7;2, 1..;3);

        assert_eq!(INFO.dim_diff, 0);
        assert_eq!(INFO.indices.len(), 8);
        assert_eq!(INFO.indices[0], ArrayIndex::Slice((2..3).into()));
        assert_eq!(INFO.indices[1], ArrayIndex::Slice((1..).into()));
        assert_eq!(INFO.indices[2], ArrayIndex::Slice((..2).into()));
        assert_eq!(INFO.indices[3], ArrayIndex::Slice((2..=3).into()));
        assert_eq!(INFO.indices[4], ArrayIndex::Slice((..=4).into()));
        assert_eq!(INFO.indices[5], ArrayIndex::Slice((..).into()));
        assert_eq!(
            INFO.indices[6],
            ArrayIndex::Slice(Slice::from(2..7).step_by(NonZeroIsize::new(2).unwrap()))
        );
        assert_eq!(
            INFO.indices[7],
            ArrayIndex::Slice(Slice::from(1..).step_by(NonZeroIsize::new(3).unwrap()))
        );
    }

    #[test]
    fn dyn_s_with_ranges() -> Result<()> {
        let x1 = 2..3;
        let x2 = 1..;
        let x3 = ..2;
        let x4 = 2..=3;
        let x5 = ..=4;
        let x6 = ..;
        let info: SliceInfo<[ArrayIndex; 8], DynDimDiff> =
            dyn_s!(x1, x2, x3, x4, x5, x6, 2..7;2, 1..;3);

        assert_eq!(info.dim_diff, 0);
        assert_eq!(info.indices.len(), 8);
        assert_eq!(info.indices[0], ArrayIndex::Slice((2..3).into()));
        assert_eq!(info.indices[1], ArrayIndex::Slice((1..).into()));
        assert_eq!(info.indices[2], ArrayIndex::Slice((..2).into()));
        assert_eq!(info.indices[3], ArrayIndex::Slice((2..=3).into()));
        assert_eq!(info.indices[4], ArrayIndex::Slice((..=4).into()));
        assert_eq!(info.indices[5], ArrayIndex::Slice((..).into()));
        assert_eq!(
            info.indices[6],
            ArrayIndex::Slice(Slice::from(2..7).step_by(2.try_into()?))
        );
        assert_eq!(
            info.indices[7],
            ArrayIndex::Slice(Slice::from(1..).step_by(3.try_into()?))
        );

        Ok(())
    }

    #[test]
    #[should_panic]
    fn s_with_invalid_range() {
        s!(2..10;0);
    }

    #[test]
    #[should_panic]
    fn s_with_invalid_ranges() {
        s!(2..10;0, 1..);
    }

    #[test]
    #[should_panic]
    fn dyn_s_with_invalid_range() {
        dyn_s!(2..10;0);
    }

    #[test]
    #[should_panic]
    fn dyn_s_with_invalid_ranges() {
        dyn_s!(2..10;0, 1..);
    }

    #[test]
    fn s_with_new_axis() {
        const INFO: SliceInfo<[ArrayIndex; 1], DimDiff<1>> = s!(NewAxis);

        assert_eq!(INFO.dim_diff, 1);
        assert_eq!(INFO.indices.len(), 1);
        assert_eq!(INFO.indices[0], ArrayIndex::NewAxis);
    }

    #[test]
    fn dyn_s_with_new_axis() {
        let x = NewAxis;
        let info: SliceInfo<[ArrayIndex; 1], DynDimDiff> = dyn_s!(x);

        assert_eq!(info.dim_diff, 1);
        assert_eq!(info.indices.len(), 1);
        assert_eq!(info.indices[0], ArrayIndex::NewAxis);
    }
}

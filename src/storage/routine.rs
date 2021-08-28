#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub fn create_buf<T>(len: usize, value: T) -> Vec<T>
where
    T: Clone,
{
    let mut buf = Vec::<T>::new();
    buf.reserve_exact(len);
    buf.resize(len, value);
    buf
}

pub fn create_uninitialized_buf<T>(len: usize) -> Vec<T> {
    let mut buf = Vec::<T>::new();
    buf.reserve_exact(len);
    unsafe { buf.set_len(len) };
    buf
}

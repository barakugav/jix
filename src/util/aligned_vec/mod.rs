#![allow(unused)]

//! https://github.com/sarah-quinones/aligned-vec

use core::fmt::Debug;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use raw::RawAlignedBytes;

mod raw;
extern crate alloc;

/// Type wrapping a runtime alignment value.
#[derive(Copy, Clone)]
pub struct RuntimeAlign {
    align: usize,
}

impl RuntimeAlign {
    #[inline]
    #[track_caller]
    fn new(align: usize) -> Self {
        if align != 0 {
            assert!(
                align.is_power_of_two(),
                "alignment ({align}) is not a power of two.",
            );
        }
        RuntimeAlign { align }
    }

    #[inline]
    fn alignment(self) -> usize {
        self.align
    }
}

/// Aligned vector. See [`Vec`] for more info.
///
/// Note: passing an alignment value of `0` or a power of two that is less than the minimum
/// alignment will cause the vector to use the minimum valid alignment for the type `T` and
/// alignment type `A`.
pub struct AlignedBytes {
    buf: RawAlignedBytes,
    len: usize,
}

impl Deref for AlignedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl DerefMut for AlignedBytes {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for AlignedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &**self
    }
}

impl AsMut<[u8]> for AlignedBytes {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut **self
    }
}

impl Drop for AlignedBytes {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: dropping initialized elements
        unsafe { (self.as_mut_slice() as *mut [u8]).drop_in_place() }
    }
}

impl AlignedBytes {
    /// Returns a new [`AlignedBytes`] with the provided alignment.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn new(align: usize) -> Self {
        unsafe {
            Self {
                buf: RawAlignedBytes::new_unchecked(RuntimeAlign::new(align).alignment()),
                len: 0,
            }
        }
    }

    /// Creates a new empty vector with enough capacity for at least `capacity` elements to
    /// be inserted in the vector. If `capacity` is 0, the vector will not allocate.
    ///
    /// # Panics
    ///
    /// Panics if the capacity exceeds `isize::MAX` bytes.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn with_capacity(align: usize, capacity: usize) -> Self {
        unsafe {
            Self {
                buf: RawAlignedBytes::with_capacity_unchecked(
                    capacity,
                    RuntimeAlign::new(align).alignment(),
                ),
                len: 0,
            }
        }
    }

    /// Returns a new [`AlignedBytes`] from its raw parts.
    ///
    /// # Safety
    ///
    /// The arguments to this function must be acquired from a previous call to
    /// [`Self::into_raw_parts`].
    #[inline]
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: *mut u8, align: usize, len: usize, capacity: usize) -> Self {
        Self {
            buf: unsafe { RawAlignedBytes::from_raw_parts(ptr, capacity, align) },
            len,
        }
    }

    /// Decomposes an [`AlignedBytes`] into its raw parts: `(ptr, alignment, length, capacity)`.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut u8, usize, usize, usize) {
        let mut this = ManuallyDrop::new(self);
        let len = this.len();
        let cap = this.capacity();
        let align = this.alignment();
        let ptr = this.as_mut_ptr();
        (ptr, align, len, cap)
    }

    /// Returns the length of the vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector's length is equal to `0`, and false otherwise.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements the vector can hold without needing to reallocate.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Reserves enough capacity for at least `additional` more elements to be inserted in the
    /// vector. After this call to `reserve`, capacity will be greater than or equal to `self.len()
    /// + additional`. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if additional > self.capacity().wrapping_sub(self.len) {
            unsafe { self.buf.grow_amortized(self.len, additional) };
        }
    }

    /// Reserves enough capacity for exactly `additional` more elements to be inserted in the
    /// vector. After this call to `reserve`, capacity will be greater than or equal to `self.len()
    /// + additional`. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        if additional > self.capacity().wrapping_sub(self.len) {
            unsafe { self.buf.grow_exact(self.len, additional) };
        }
    }

    /// Returns the alignment of the vector.
    #[inline]
    #[must_use]
    pub fn alignment(&self) -> usize {
        self.buf.align()
    }

    /// Returns a pointer to the objects held by the vector.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    /// Returns a mutable pointer to the objects held by the vector.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    /// Returns a reference to a slice over the objects held by the vector.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        let len = self.len();
        let ptr = self.as_ptr();

        // ptr points to `len` initialized elements and is properly aligned since
        // self.align is at least `align_of::<u8>()`
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }

    /// Returns a mutable reference to a slice over the objects held by the vector.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.len();
        let ptr = self.as_mut_ptr();

        // ptr points to `len` initialized elements and is properly aligned since
        // self.align is at least `align_of::<u8>()`
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Push the given value to the end of the vector, reallocating if needed.
    #[inline]
    pub fn push(&mut self, value: u8) {
        if self.len == self.capacity() {
            unsafe { self.buf.grow_amortized(self.len, 1) };
        }

        // SAFETY: self.capacity is greater than self.len so the write is valid
        unsafe {
            let past_the_end = self.as_mut_ptr().add(self.len);
            past_the_end.write(value);
            self.len += 1;
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    /// The capacity will remain at least as large as both the length and the supplied value.
    /// If the current capacity is less than the lower limit, this is a no-op.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let min_capacity = min_capacity.max(self.len());
        if self.capacity() > min_capacity {
            unsafe { self.buf.shrink_to(min_capacity) };
        }
    }

    /// Shrinks the capacity of the vector as much as possible without dropping any elements.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if self.capacity() > self.len {
            unsafe { self.buf.shrink_to(self.len) };
        }
    }

    /// Drops the last elements of the vector until its length is equal to `len`.
    /// If `len` is greater than or equal to `self.len()`, this is a no-op.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            let old_len = self.len;
            self.len = len;
            unsafe {
                let ptr = self.as_mut_ptr();
                core::ptr::slice_from_raw_parts_mut(ptr.add(len), old_len - len).drop_in_place()
            }
        }
    }

    /// Drops the all the elements of the vector, setting its length to `0`.
    #[inline]
    pub fn clear(&mut self) {
        let old_len = self.len;
        self.len = 0;
        unsafe {
            let ptr = self.as_mut_ptr();
            core::ptr::slice_from_raw_parts_mut(ptr, old_len).drop_in_place()
        }
    }

    /// Collects an iterator into an [`AlignedBytes`] with the provided alignment.
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = u8>>(align: usize, iter: I) -> Self {
        Self::from_iter_impl(iter.into_iter(), align)
    }

    /// Collects a slice into an [`AlignedBytes`] with the provided alignment.
    #[inline]
    pub fn from_slice(align: usize, slice: &[u8]) -> Self
    where
        u8: Clone,
    {
        let len = slice.len();
        let mut vec = AlignedBytes::with_capacity(align, len);
        {
            let len = &mut vec.len;
            let ptr: *mut u8 = vec.buf.ptr.as_ptr();

            for (i, item) in slice.iter().enumerate() {
                unsafe { ptr.add(i).write(item.clone()) };
                *len += 1;
            }
        }
        vec
    }

    fn from_iter_impl<I: Iterator<Item = u8>>(mut iter: I, align: usize) -> Self {
        let (lower_bound, upper_bound) = iter.size_hint();
        let mut this = Self::with_capacity(align, lower_bound);

        if upper_bound == Some(lower_bound) {
            let len = &mut this.len;
            let ptr = this.buf.ptr.as_ptr();

            let first_chunk = iter.take(lower_bound);
            first_chunk.enumerate().for_each(|(i, item)| {
                unsafe { ptr.add(i).write(item) };
                *len += 1;
            });
        } else {
            let len = &mut this.len;
            let ptr = this.buf.ptr.as_ptr();

            let first_chunk = (&mut iter).take(lower_bound);
            first_chunk.enumerate().for_each(|(i, item)| {
                unsafe { ptr.add(i).write(item) };
                *len += 1;
            });
            iter.for_each(|item| {
                this.push(item);
            });
        }

        this
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len;
    }

    pub fn append(&mut self, other: &mut AlignedBytes) {
        unsafe {
            let len = self.len();
            let count = other.len();
            self.reserve(count);
            core::ptr::copy_nonoverlapping(other.as_ptr(), self.as_mut_ptr().add(len), count);
            self.len += count;
            other.len = 0;
        }
    }

    /// Resizes the `Vec` in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    pub fn resize(&mut self, new_len: usize, value: u8) {
        // Copied somewhat from the standard library
        let len = self.len();

        if new_len > len {
            self.extend_with(new_len - len, value)
        } else {
            self.truncate(new_len);
        }
    }

    /// Extend the vector by `n` clones of value.
    fn extend_with(&mut self, n: usize, value: u8) {
        // Copied somewhat from the standard library
        self.reserve(n);

        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len());

            // Write all elements except the last one
            for _ in 1..n {
                core::ptr::write(ptr, value.clone());
                ptr = ptr.add(1);
                // Increment the length in every step in case clone() panics
                self.len += 1;
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                core::ptr::write(ptr, value);
                self.len += 1;
            }
        }
    }

    /// Clones and appends all elements in a slice to the `Vec`.
    pub fn extend_from_slice(&mut self, other: &[u8]) {
        // Copied somewhat from the standard library
        let count = other.len();
        self.reserve(count);
        let len = self.len();
        unsafe {
            core::ptr::copy_nonoverlapping(other.as_ptr(), self.as_mut_ptr().add(len), count)
        };
        self.len += count;
    }
}

impl Debug for AlignedBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        Self::from_slice(self.alignment(), self.deref())
    }
}

unsafe impl Sync for AlignedBytes {}
unsafe impl Send for AlignedBytes {}

#[cfg(test)]
mod tests {
    use super::*;
    use core::iter::repeat;

    #[test]
    fn new() {
        let v = AlignedBytes::new(32);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert_eq!(v.alignment(), 32);
        assert_eq!(v.as_ptr().align_offset(32), 0);

        let v = AlignedBytes::new(4096);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert_eq!(v.alignment(), 4096);
        assert_eq!(v.as_ptr().align_offset(4096), 0);
        assert_eq!(v.as_ptr().align_offset(4096), 0);
    }

    #[test]
    fn collect() {
        let v = AlignedBytes::from_iter(64, 0..4);
        assert_eq!(&*v, &[0, 1, 2, 3]);
        let v = AlignedBytes::from_iter(64, repeat(77).take(4));
        assert_eq!(&*v, &[77, 77, 77, 77]);
    }

    #[test]
    fn push() {
        let mut v = AlignedBytes::new(16);
        v.push(0);
        v.push(1);
        v.push(2);
        v.push(3);
        assert_eq!(&*v, &[0, 1, 2, 3]);

        let mut v = AlignedBytes::from_iter(64, 0..4);
        v.push(4);
        v.push(5);
        v.push(6);
        v.push(7);
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn shrink() {
        let mut v = AlignedBytes::with_capacity(16, 10);
        v.push(0);
        v.push(1);
        v.push(2);

        assert_eq!(v.capacity(), 10);
        v.shrink_to_fit();
        assert_eq!(v.len(), 3);
        assert_eq!(v.capacity(), 3);

        let mut v = AlignedBytes::with_capacity(16, 10);
        v.push(0);
        v.push(1);
        v.push(2);

        assert_eq!(v.capacity(), 10);
        v.shrink_to(0);
        assert_eq!(v.len(), 3);
        assert_eq!(v.capacity(), 3);
    }

    #[test]
    fn truncate() {
        let mut v = AlignedBytes::new(16);
        v.push(0);
        v.push(1);
        v.push(2);

        v.truncate(1);
        assert_eq!(v.len(), 1);
        assert_eq!(&*v, &[0]);

        v.clear();
        assert_eq!(v.len(), 0);
        assert_eq!(&*v, &[]);
    }

    #[test]
    fn extend_from_slice() {
        let mut v = AlignedBytes::new(16);
        v.extend_from_slice(&[0, 1, 2, 3]);
        v.extend_from_slice(&[4, 5, 6, 7, 8]);
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn resize() {
        let mut v = AlignedBytes::new(16);
        v.push(0);
        v.push(1);
        v.push(2);

        v.resize(1, 10);
        assert_eq!(v.len(), 1);
        assert_eq!(&*v, &[0]);

        v.resize(3, 20);
        assert_eq!(v.len(), 3);
        assert_eq!(&*v, &[0, 20, 20]);
    }
}

use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::{cmp, fmt, iter, mem, ptr, slice};

type LenUint = u8;

/// A vector with a fixed capacity.
///
/// A simplified version of `arrayvec::ArrayVec`.
/// See https://github.com/bluss/arrayvec
#[repr(C)]
pub struct ArrayVec<T, const CAP: usize> {
    len: LenUint,
    data: [MaybeUninit<T>; CAP],
}

impl<T, const CAP: usize> Drop for ArrayVec<T, CAP> {
    #[inline]
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T, const CAP: usize> ArrayVec<T, CAP> {
    const CAPACITY: usize = CAP;

    #[inline]
    #[track_caller]
    pub const fn new() -> ArrayVec<T, CAP> {
        if size_of::<usize>() > size_of::<LenUint>() {
            assert!(
                CAP <= LenUint::MAX as usize,
                "ArrayVec: capacity exceeds maximum supported capacity ",
            );
        }
        ArrayVec {
            data: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Option<Self>
    where
        T: Clone,
    {
        (slice.len() <= Self::CAPACITY).then(|| {
            let mut array = Self::new();
            array.extend_from_slice(slice);
            array
        })
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        let len = self.len as usize;
        unsafe { std::hint::assert_unchecked(len <= CAP) };
        len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        Self::CAPACITY
    }

    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    #[inline]
    pub const fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Push `element` to the end of the vector.
    ///
    /// ***Panics*** if the vector is already full.
    #[inline]
    #[track_caller]
    pub fn push(&mut self, element: T) {
        assert!(self.len() < Self::CAPACITY);
        unsafe { self.push_unchecked(element) };
    }

    /// Push `element` to the end of the vector without checking for capacity.
    ///
    /// The caller must ensure the vector is not full.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, element: T) {
        let len = self.len();
        debug_assert!(len < Self::CAPACITY);
        unsafe { std::hint::assert_unchecked(len < Self::CAPACITY) };
        unsafe { ptr::write(self.as_mut_ptr().add(len), element) };
        unsafe { self.set_len(len + 1) };
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater than the vector's current length this has no
    /// effect.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        let len = self.len();
        if new_len < len {
            unsafe {
                self.set_len(new_len);
                let tail = slice::from_raw_parts_mut(self.as_mut_ptr().add(new_len), len - new_len);
                ptr::drop_in_place(tail);
            }
        }
    }

    /// Remove all elements in the vector.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0)
    }

    /// Get pointer to where element at `index` would be
    #[inline(always)]
    unsafe fn get_unchecked_ptr(&mut self, index: usize) -> *mut T {
        unsafe { self.as_mut_ptr().add(index) }
    }

    /// Insert `element` at position `index`.
    ///
    /// Shift up all elements after `index`.
    ///
    /// It is an error if the index is greater than the length or if the
    /// arrayvec is full.
    ///
    /// ***Panics*** if the array is full or the `index` is out of bounds.
    #[inline]
    #[track_caller]
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(
            index <= self.len(),
            "ArrayVec::insert: index {} is out of bounds in vector of length {}",
            index,
            self.len()
        );
        let len = self.len();
        assert!(len < self.capacity());

        unsafe {
            {
                let p: *mut _ = self.get_unchecked_ptr(index);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy(p, p.offset(1), len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.set_len(len + 1);
        }
    }

    #[inline]
    pub const fn insert_first_const(&mut self, value: T)
    where
        T: Copy,
    {
        assert!(self.len() < self.capacity(), "ArrayVec capacity exceeded");
        let mut new_data: [MaybeUninit<T>; CAP] = unsafe { MaybeUninit::uninit().assume_init() };
        let (first, new_data_tail) = new_data.split_first_mut().unwrap();
        first.write(value);
        new_data_tail.copy_from_slice(self.data.split_last().unwrap().1);

        self.len = self.len + 1;
        self.data = new_data;
    }

    /// Set the vector's length without dropping or moving out elements
    #[inline]
    pub unsafe fn set_len(&mut self, length: usize) {
        debug_assert!(length <= self.capacity());
        self.len = length as LenUint;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        unsafe { slice::from_raw_parts(self.as_ptr(), len) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len();
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as _
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as _
    }
}

impl<T, const CAP: usize> Deref for ArrayVec<T, CAP> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const CAP: usize> DerefMut for ArrayVec<T, CAP> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, const CAP: usize> AsRef<[T]> for ArrayVec<T, CAP> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, const CAP: usize> AsMut<[T]> for ArrayVec<T, CAP> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<'a, T: 'a, const CAP: usize> IntoIterator for &'a ArrayVec<T, CAP> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: 'a, const CAP: usize> IntoIterator for &'a mut ArrayVec<T, CAP> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Iterate the `ArrayVec` with each element by value.
impl<T, const CAP: usize> IntoIterator for ArrayVec<T, CAP> {
    type Item = T;
    type IntoIter = IntoIter<T, CAP>;
    #[inline]
    fn into_iter(self) -> IntoIter<T, CAP> {
        IntoIter { index: 0, v: self }
    }
}

/// By-value iterator for `ArrayVec`.
pub struct IntoIter<T, const CAP: usize> {
    index: usize,
    v: ArrayVec<T, CAP>,
}
impl<T, const CAP: usize> Iterator for IntoIter<T, CAP> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.v.len() {
            None
        } else {
            unsafe {
                let index = self.index;
                self.index = index + 1;
                Some(ptr::read(self.v.get_unchecked_ptr(index)))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.v.len() - self.index;
        (len, Some(len))
    }
}

impl<T, const CAP: usize> DoubleEndedIterator for IntoIter<T, CAP> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.v.len() {
            None
        } else {
            unsafe {
                let new_len = self.v.len() - 1;
                self.v.set_len(new_len);
                Some(ptr::read(self.v.get_unchecked_ptr(new_len)))
            }
        }
    }
}

impl<T, const CAP: usize> ExactSizeIterator for IntoIter<T, CAP> {}

impl<T, const CAP: usize> Drop for IntoIter<T, CAP> {
    #[inline]
    fn drop(&mut self) {
        // panic safety: Set length to 0 before dropping elements.
        let index = self.index;
        let len = self.v.len();
        unsafe {
            self.v.set_len(0);
            let elements = slice::from_raw_parts_mut(self.v.get_unchecked_ptr(index), len - index);
            ptr::drop_in_place(elements);
        }
    }
}

impl<T, const CAP: usize> Clone for IntoIter<T, CAP>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> IntoIter<T, CAP> {
        let mut v = ArrayVec::new();
        v.extend_from_slice(&self.v[self.index..]);
        v.into_iter()
    }
}

struct ScopeExitGuard<T, Data, F>
where
    F: FnMut(&Data, &mut T),
{
    value: T,
    data: Data,
    f: F,
}

impl<T, Data, F> Drop for ScopeExitGuard<T, Data, F>
where
    F: FnMut(&Data, &mut T),
{
    #[inline]
    fn drop(&mut self) {
        (self.f)(&self.data, &mut self.value)
    }
}

/// Extend the `ArrayVec` with an iterator.
///
/// ***Panics*** if extending the vector exceeds its capacity.
impl<T, const CAP: usize> Extend<T> for ArrayVec<T, CAP> {
    /// Extend the `ArrayVec` with an iterator.
    ///
    /// ***Panics*** if extending the vector exceeds its capacity.
    #[inline]
    #[track_caller]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe { self.extend_from_iter::<_, true>(iter) }
    }
}

#[inline(never)]
#[cold]
#[track_caller]
fn extend_panic() -> ! {
    panic!("ArrayVec: capacity exceeded in extend/from_iter");
}

impl<T, const CAP: usize> ArrayVec<T, CAP> {
    /// Extend the arrayvec from the iterable.
    ///
    /// ## Safety
    ///
    /// Unsafe because if CHECK is false, the length of the input is not checked.
    /// The caller must ensure the length of the input fits in the capacity.
    #[inline]
    #[track_caller]
    pub(super) unsafe fn extend_from_iter<I, const CHECK: bool>(&mut self, iterable: I)
    where
        I: IntoIterator<Item = T>,
    {
        unsafe {
            let take = self.capacity() - self.len();
            let len = self.len();
            let mut ptr = raw_ptr_add(self.as_mut_ptr(), len);
            let end_ptr = raw_ptr_add(ptr, take);
            // Keep the length in a separate variable, write it back on scope
            // exit. To help the compiler with alias analysis and stuff.
            // We update the length to handle panic in the iteration of the
            // user's iterator, without dropping any elements on the floor.
            let mut guard = ScopeExitGuard {
                value: &mut self.len,
                data: len,
                f: move |&len, self_len| {
                    let self_len: &mut LenUint = self_len;
                    *self_len = len as LenUint;
                },
            };
            let mut iter = iterable.into_iter();
            loop {
                if let Some(elt) = iter.next() {
                    if ptr == end_ptr && CHECK {
                        extend_panic();
                    }
                    debug_assert_ne!(ptr, end_ptr);
                    if mem::size_of::<T>() != 0 {
                        ptr.write(elt);
                    }
                    ptr = raw_ptr_add(ptr, 1);
                    guard.data += 1;
                } else {
                    return; // success
                }
            }
        }
    }

    /// Extend the ArrayVec with clones of elements from the slice;
    /// the length of the slice must be <= the remaining capacity in the arrayvec.
    #[inline]
    pub(super) fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Clone,
    {
        let take = self.capacity() - self.len();
        debug_assert!(slice.len() <= take);
        unsafe {
            let slice = if take < slice.len() {
                &slice[..take]
            } else {
                slice
            };
            self.extend_from_iter::<_, false>(slice.iter().cloned());
        }
    }
}

/// Rawptr add but uses arithmetic distance for ZST
#[inline]
unsafe fn raw_ptr_add<T>(ptr: *mut T, offset: usize) -> *mut T {
    unsafe {
        if mem::size_of::<T>() == 0 {
            // Special case for ZST
            ptr.cast::<u8>().wrapping_add(offset).cast::<T>()
        } else {
            ptr.add(offset)
        }
    }
}

/// Create an `ArrayVec` from an iterator.
///
/// ***Panics*** if the number of elements in the iterator exceeds the arrayvec's capacity.
impl<T, const CAP: usize> iter::FromIterator<T> for ArrayVec<T, CAP> {
    /// Create an `ArrayVec` from an iterator.
    ///
    /// ***Panics*** if the number of elements in the iterator exceeds the arrayvec's capacity.
    #[inline]
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut array = ArrayVec::new();
        array.extend(iter);
        array
    }
}

impl<T, const CAP: usize, I> Index<I> for ArrayVec<T, CAP>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}
impl<T, const CAP: usize, I> IndexMut<I> for ArrayVec<T, CAP>
where
    [T]: IndexMut<I>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T, const CAP: usize> Clone for ArrayVec<T, CAP>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        let mut array: ArrayVec<T, CAP> = ArrayVec::new();
        {
            let mut guard = ScopeExitGuard {
                value: &mut array.len,
                data: 0 as LenUint,
                f: move |&len, self_len| {
                    **self_len = len;
                },
            };

            for i in 0..CAP {
                if i < self.len() {
                    let val = unsafe { &*self.data[i].as_ptr() }.clone();
                    if mem::size_of::<T>() != 0 {
                        unsafe { array.data[i].as_mut_ptr().write(val) };
                    } else {
                        // The ZST element has logically been moved into the vector.
                        // There is no memory to write, but dropping `elt` here would
                        // drop it once now and once again when the vector is dropped.
                        mem::forget(val);
                    }
                    guard.data += 1;
                } else {
                    // we are done copying all the elements.
                    // we continue to copy uninitialized elements past len up to CAP. This seems
                    // weird and counter intuitive, but when T is trivial (like i8/i32), copying the
                    // whole array is faster as the compiler doesnt have to loop up to len, and the
                    // generated code is branchless copy of the whole struct (like Copy).
                    //
                    // We only do it if;
                    // - T does not requires drop, which we take as an indication that T::clone()
                    //   is not trivial and therefore the loop will not be optimized away by the
                    //   compiler.
                    // - size_of > 0, to avoid reading self.xs[i].as_ptr()
                    // - the total array is less or equal to 128 bytes. An arbitrary threshold to
                    //   avoid unnecessary copying of large arrays.
                    if mem::needs_drop::<T>()
                        || mem::size_of::<T>() == 0
                        || CAP > 128 / mem::size_of::<T>()
                    {
                        break;
                    }
                    // Safety: copy of MaybeUninit to MaybeUninit
                    unsafe { ptr::copy_nonoverlapping(&self.data[i], &mut array.data[i], 1) };
                }
            }
        }

        // This assignment seems redundant as guard.data is already equal to len and will set the
        // array len on drop, but setting it here explicitly helps the compiler understand it
        // can just copy the len instead of accumulating it in the guard. This is especially
        // useful for T::clone() that can not panic.
        debug_assert_eq!(array.len, self.len);
        array.len = self.len;

        array
    }

    fn clone_from(&mut self, rhs: &Self) {
        // recursive case for the common prefix
        let prefix = cmp::min(self.len(), rhs.len());
        self[..prefix].clone_from_slice(&rhs[..prefix]);

        if prefix < self.len() {
            // rhs was shorter
            self.truncate(prefix);
        } else {
            let rhs_elems = &rhs[self.len()..];
            self.extend_from_slice(rhs_elems);
        }
    }
}

impl<T, const CAP: usize> PartialEq for ArrayVec<T, CAP>
where
    T: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T, const CAP: usize> PartialEq<[T]> for ArrayVec<T, CAP>
where
    T: PartialEq,
{
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        **self == *other
    }
}

impl<T, const CAP: usize> Eq for ArrayVec<T, CAP> where T: Eq {}

impl<T, const CAP: usize> fmt::Debug for ArrayVec<T, CAP>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T, const CAP: usize> Default for ArrayVec<T, CAP> {
    /// Return an empty array
    #[inline]
    fn default() -> ArrayVec<T, CAP> {
        ArrayVec::new()
    }
}

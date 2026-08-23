use crate::util::iter::{impl_merge_extension, NdIterExtension};
use crate::util::Idx;
use crate::Dimension;
use crate::{default_logical_strides_slice, DimVec};

/// An nd-iterator extension that tracks a `*const u8` pointer into a strided buffer.
///
/// On each dimension change the pointer is adjusted by the difference in byte offsets:
/// `ptr += (after - before) * stride[dim]`.
pub(crate) struct NdIterExtStridesPtr<D: Dimension, T, S: Copy>(NdIterExtStridesPtrMut<D, T, S>);

impl<D: Dimension, T, S: Copy> Clone for NdIterExtStridesPtr<D, T, S>
where
    D::Vec<S>: Clone,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: Dimension, T, S: Copy> NdIterExtStridesPtr<D, T, S> {
    /// Creates the extension starting at `initial_ptr` with the given per-dimension byte strides.
    #[inline]
    pub fn new<V>(strides: V, initial_ptr: *const T) -> Self
    where
        D: Dimension<Vec<S> = V>,
        V: DimVec<S, Dimension = D>,
    {
        Self(NdIterExtStridesPtrMut::new(strides, initial_ptr.cast_mut()))
    }
}
impl<D, T, S> NdIterExtension for NdIterExtStridesPtr<D, T, S>
where
    D: Dimension,
    S: Idx,
{
    type Item<'a>
        = *const T
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_decrease(dim, before, after, diff);
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        <NdIterExtStridesPtrMut<D, T, S> as NdIterExtension>::value(&self.0).cast_const()
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        <NdIterExtStridesPtrMut<D, T, S> as NdIterExtension>::check_ndim(&self.0, ndim)
    }

    impl_merge_extension!();
}

/// An nd-iterator extension that tracks a `*mut u8` pointer into a strided buffer.
///
/// On each dimension change the pointer is adjusted by the difference in byte offsets:
/// `ptr += (after - before) * stride[dim]`.
pub(crate) struct NdIterExtStridesPtrMut<D: Dimension, T, S: Copy> {
    strides: D::Vec<S>,
    current_ptr: *mut T,
}

impl<D: Dimension, T, S: Copy> Clone for NdIterExtStridesPtrMut<D, T, S>
where
    D::Vec<S>: Clone,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            strides: <_ as DimVec<_>>::clone(&self.strides),
            current_ptr: self.current_ptr,
        }
    }
}

impl<D: Dimension, T, S: Copy> NdIterExtStridesPtrMut<D, T, S> {
    /// Creates the extension starting at `initial_ptr` with the given per-dimension byte strides.
    #[inline]
    pub fn new<V>(strides: V, initial_ptr: *mut T) -> Self
    where
        D: Dimension<Vec<S> = V>,
        V: DimVec<S, Dimension = D>,
    {
        Self {
            strides,
            current_ptr: initial_ptr,
        }
    }
}
impl<D, T, S> NdIterExtension for NdIterExtStridesPtrMut<D, T, S>
where
    D: Dimension,
    S: Idx,
{
    type Item<'a>
        = *mut T
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = diff as usize;
        let stride = self.strides[dim].u64() as usize;
        self.current_ptr = unsafe { self.current_ptr.add(diff * stride) };
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = diff as usize;
        let stride = self.strides[dim].u64() as usize;
        self.current_ptr = unsafe { self.current_ptr.sub(diff * stride) };
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        self.current_ptr
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.as_ref().len() == ndim
    }

    impl_merge_extension!();
}

/// An nd-iterator extension that tracks an offset into a strided array.
///
/// On each dimension change the offset is adjusted by the difference in element counts:
/// `offset += (after - before) * stride[dim]`.
pub(crate) struct NdIterExtStridesOffset<D: Dimension, S> {
    strides: D::Vec<S>,
    offset: S,
}
impl<D: Dimension, S: Idx> NdIterExtStridesOffset<D, S> {
    #[inline]
    pub fn new<V>(strides: V, initial_offset: S) -> Self
    where
        D: Dimension<Vec<S> = V>,
        V: DimVec<S, Dimension = D>,
    {
        Self {
            strides,
            offset: initial_offset,
        }
    }
}
impl<D: Dimension, S: Idx> NdIterExtension for NdIterExtStridesOffset<D, S> {
    type Item<'a>
        = S
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        self.offset += S::from_u64(diff) * self.strides[dim];
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        self.offset -= S::from_u64(diff) * self.strides[dim];
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        self.offset
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.as_ref().len() == ndim
    }

    impl_merge_extension!();
}

/// An nd-iterator extension that tracks `N` offsets into `N` strided arrays simultaneously.
///
/// Like [`NdIterExtStridesOffset`] but for a fixed number of operands sharing one index walk: on
/// each dimension change every offset is advanced by `(after - before) * strides[operand][dim]`, and
/// [`value`](NdIterExtension::value) yields `[S; N]`. Since that item does not borrow, an `NdIter`
/// carrying this extension is a plain [`Iterator`].
pub(crate) struct NdIterExtStridesOffsetMulti<D: Dimension, S, const N: usize> {
    strides: [D::Vec<S>; N],
    offsets: [S; N],
}
impl<D: Dimension, S: Idx, const N: usize> NdIterExtStridesOffsetMulti<D, S, N> {
    #[inline]
    pub fn new(strides: [D::Vec<S>; N], initial_offsets: [S; N]) -> Self {
        Self {
            strides,
            offsets: initial_offsets,
        }
    }
}
impl<D: Dimension, S: Idx, const N: usize> NdIterExtension
    for NdIterExtStridesOffsetMulti<D, S, N>
{
    type Item<'a>
        = [S; N]
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for i in 0..N {
            self.offsets[i] += diff * self.strides[i][dim];
        }
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for i in 0..N {
            self.offsets[i] -= diff * self.strides[i][dim];
        }
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        self.offsets
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.iter().all(|s| s.as_ref().len() == ndim)
    }

    impl_merge_extension!();
}

/// The runtime-length sibling of [`NdIterExtStridesOffsetMulti`], for callers whose operand count is
/// not known at compile time.
///
/// The count lives in a [`Vec`] rather than a const generic, so one instantiation serves every
/// caller and there is no bound on the number of operands. [`value`](NdIterExtension::value) hands
/// back a borrowed `&[S]` - the reason [`NdIterExtension::Item`] carries a lifetime - which also
/// means an `NdIter` carrying this extension is *not* an [`Iterator`] and must be driven with
/// [`NdIter::next`].
pub(crate) struct NdIterExtStridesOffsetMultiDyn<D: Dimension, S> {
    strides: Vec<D::Vec<S>>,
    offsets: Vec<S>,
}
impl<D: Dimension, S: Idx> NdIterExtStridesOffsetMultiDyn<D, S> {
    #[inline]
    pub fn new(strides: Vec<D::Vec<S>>, initial_offsets: Vec<S>) -> Self {
        assert_eq!(strides.len(), initial_offsets.len());
        Self {
            strides,
            offsets: initial_offsets,
        }
    }
}
impl<D: Dimension, S: Idx> NdIterExtension for NdIterExtStridesOffsetMultiDyn<D, S> {
    /// The current offset of each operand, in the order the strides were given.
    type Item<'a>
        = &'a [S]
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for (offset, strides) in self.offsets.iter_mut().zip(self.strides.iter()) {
            *offset += diff * strides[dim];
        }
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for (offset, strides) in self.offsets.iter_mut().zip(self.strides.iter()) {
            *offset -= diff * strides[dim];
        }
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        &self.offsets
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.iter().all(|s| s.as_ref().len() == ndim)
    }

    impl_merge_extension!();
}

#[inline]
pub(crate) fn nd_iter_ext_logical_global_index<D: Dimension>(
    shape: &[u64],
    begin: &[u64],
) -> NdIterExtStridesOffset<D, u64> {
    let logical_strides = default_logical_strides_slice(shape);
    let initial_offset = (0..shape.len())
        .map(|dim| begin[dim] * logical_strides[dim])
        .sum();
    NdIterExtStridesOffset::new(
        D::vec(logical_strides.len(), |i| logical_strides[i]),
        initial_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::iter::NdIter;
    use crate::util::DimArray;
    use crate::{Dim, DimDyn, SliceExt};

    /// Build a [`DimArray`] (the `DimDyn` vec container) from a slice, for constructing test
    /// extensions whose `new` takes an owned `D::Vec<S>`.
    fn dv<T: Copy>(s: &[T]) -> DimArray<T> {
        s.to_dim_vec::<DimDyn>()
    }

    #[test]
    fn strides_ptr_mut_1d_stride_1() {
        let mut data = [0u8; 8];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(dv(&[1usize]), base);
        let iter = NdIter::new(dv(&[8u64]), ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_ptr_mut_1d_larger_stride() {
        let mut data = [0u8; 64];
        let base = data.as_mut_ptr();
        let stride = 8usize;
        let ext = NdIterExtStridesPtrMut::new(dv(&[stride]), base);
        let iter = NdIter::new(dv(&[8u64]), ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i * stride) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_ptr_mut_2d_row_major_contiguous() {
        let rows = 3usize;
        let cols = 4usize;
        let elem = 4usize; // e.g. f32
        let mut data = vec![0u8; rows * cols * elem];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(dv(&[cols * elem, elem]), base);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        let mut flat = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(flat * elem) }, "flat index {flat}");
            flat += 1;
        }
        assert_eq!(flat, rows * cols);
    }

    #[test]
    fn strides_ptr_mut_2d_non_contiguous_column_major() {
        // Simulate a column-major (Fortran-order) 2*3 layout.
        // Row stride = 1, column stride = nrows = 2.
        let rows = 2usize;
        let cols = 3usize;
        let mut data = vec![0u8; rows * cols];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(dv(&[1, rows]), base);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        // Iteration order is row-major by *index*, but pointer jumps follow column-major layout:
        // [0,0]=0, [0,1]=2, [0,2]=4, [1,0]=1, [1,1]=3, [1,2]=5
        let expected_offsets: &[usize] = &[0, 2, 4, 1, 3, 5];
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(expected_offsets[i]) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, rows * cols);
    }

    #[test]
    fn strides_ptr_mut_with_begin_offset() {
        // begin=[1,1], end=[3,3], strides=[4,1].
        // Initial pointer = base + 1*4 + 1*1 = base+5.
        // Expected traversal offsets: [1,1]=5, [1,2]=6, [2,1]=9, [2,2]=10.
        let mut data = vec![0u8; 16];
        let base = data.as_mut_ptr();
        let start = unsafe { base.add(4 + 1) };
        let ext = NdIterExtStridesPtrMut::new(dv(&[4usize, 1]), start);
        let iter = NdIter::new_with_begin(dv(&[1u64, 1]), dv(&[3u64, 3]), ext);
        let expected: &[usize] = &[5, 6, 9, 10];
        let mut i = 0;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(expected[i]) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, 4);
    }

    #[test]
    fn strides_ptr_mut_empty_range_yields_no_pointers() {
        let mut data = [0u8; 16];
        let ext = NdIterExtStridesPtrMut::new(dv(&[4usize, 1]), data.as_mut_ptr());
        let mut iter = NdIter::new_with_begin(dv(&[2u64, 2]), dv(&[2u64, 5]), ext);
        assert!(iter.next().is_none());
    }

    #[test]
    fn strides_ptr_const_1d_matches_mut() {
        let data = [0u8; 8];
        let base = data.as_ptr();
        let ext = NdIterExtStridesPtr::new(dv(&[1usize]), base);
        let iter = NdIter::new(dv(&[8u64]), ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i) });
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_ptr_const_2d_contiguous() {
        let rows = 3usize;
        let cols = 4usize;
        let data = vec![0u8; rows * cols];
        let base = data.as_ptr();
        let ext = NdIterExtStridesPtr::new(dv(&[cols, 1]), base);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        let mut flat = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(flat) });
            flat += 1;
        }
        assert_eq!(flat, rows * cols);
    }

    // --- Non-u8 pointer types ---------------------------------------------------------------
    // With `T != u8`, `ptr.add(n)` advances `n * size_of::<T>()` bytes, so strides are in units
    // of `T` elements (not bytes). The tests below pin down that element-size scaling for the
    // generic `*const T` / `*mut T` pointers.

    #[test]
    fn strides_ptr_const_scales_by_element_size() {
        // Stride 1 over a `u64` buffer must advance one element (8 bytes) per step, not one byte.
        let data = [0u64; 4];
        let base = data.as_ptr();
        let base_byte = base as usize;
        let ext = NdIterExtStridesPtr::new(dv(&[1usize]), base);
        let iter = NdIter::new(dv(&[4u64]), ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i) }, "step {i}");
            assert_eq!(
                ptr as usize,
                base_byte + i * size_of::<u64>(),
                "byte offset step {i}"
            );
            i += 1;
        }
        assert_eq!(i, 4);
    }

    #[test]
    fn strides_ptr_mut_2d_u32_writes_to_correct_elements() {
        // Functional check: walk a contiguous row-major u32 buffer and write through each pointer.
        // Element-unit strides are [cols, 1]; if the scaling were wrong the data would not match.
        let rows = 3usize;
        let cols = 4usize;
        let mut data = vec![0u32; rows * cols];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(dv(&[cols, 1]), base);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        let mut flat = 0u32;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(flat as usize) }, "flat index {flat}");
            unsafe { *ptr = flat * 10 + 7 };
            flat += 1;
        }
        assert_eq!(flat as usize, rows * cols);
        let expected: Vec<u32> = (0..(rows * cols) as u32).map(|f| f * 10 + 7).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn strides_ptr_mut_non_contiguous_column_major_u32() {
        // Column-major (Fortran-order) 2*3 layout of u32, mirroring the u8 test but with a
        // non-byte element size: strides and offsets are counted in elements.
        let rows = 2usize;
        let cols = 3usize;
        let mut data = vec![0u32; rows * cols];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(dv(&[1usize, rows]), base);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        let expected_offsets: &[usize] = &[0, 2, 4, 1, 3, 5];
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(expected_offsets[i]) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, rows * cols);
    }

    #[test]
    fn strides_ptr_mut_with_begin_offset_u32() {
        // begin=[1,1], end=[3,3], strides=[4,1] in *element* units over a u32 buffer.
        // Initial pointer = base + (1*4 + 1*1) elements = base+5.
        let mut data = vec![0u32; 16];
        let base = data.as_mut_ptr();
        let start = unsafe { base.add(4 + 1) };
        let ext = NdIterExtStridesPtrMut::new(dv(&[4usize, 1]), start);
        let iter = NdIter::new_with_begin(dv(&[1u64, 1]), dv(&[3u64, 3]), ext);
        let expected: &[usize] = &[5, 6, 9, 10];
        let mut i = 0;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(expected[i]) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, 4);
    }

    #[test]
    fn strides_ptr_const_struct_element_scaling() {
        // A struct element with a non-power-of-two size (3 * u16 = 6 bytes) exercises the scaling
        // for arbitrary `Copy` element types, not just primitives.
        #[derive(Clone, Copy)]
        struct Px {
            _r: u16,
            _g: u16,
            _b: u16,
        }
        let data = [Px {
            _r: 0,
            _g: 0,
            _b: 0,
        }; 5];
        let base = data.as_ptr();
        let base_byte = base as usize;
        let ext = NdIterExtStridesPtr::new(dv(&[1usize]), base);
        let iter = NdIter::new(dv(&[5u64]), ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i) }, "step {i}");
            assert_eq!(
                ptr as usize,
                base_byte + i * size_of::<Px>(),
                "byte offset step {i}"
            );
            i += 1;
        }
        assert_eq!(i, 5);
    }

    // --- Static Dim<N> variants -----------------------------------------------------------
    // These mirror key DimDyn tests using fixed-size Dim<N> to confirm both paths compile and work.

    #[test]
    fn strides_ptr_mut_dim1_static() {
        let mut data = [0u8; 8];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new([1usize], base);
        let iter = NdIter::<Dim<1>, _>::new([8u64], ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_ptr_mut_dim2_static() {
        let rows = 3usize;
        let cols = 4usize;
        let elem = 4usize;
        let mut data = vec![0u8; rows * cols * elem];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new([cols * elem, elem], base);
        let iter = NdIter::<Dim<2>, _>::new([rows as u64, cols as u64], ext);
        let mut flat = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(flat * elem) }, "flat index {flat}");
            flat += 1;
        }
        assert_eq!(flat, rows * cols);
    }
}

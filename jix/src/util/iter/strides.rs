use crate::util::iter::NdIterExtension;
use crate::util::{default_logical_strides, DimArray, Idx};

/// An nd-iterator extension that tracks a `*const u8` pointer into a strided buffer.
///
/// On each dimension change the pointer is adjusted by the difference in byte offsets:
/// `ptr += (after - before) * stride[dim]`.
#[derive(Clone)]
pub(crate) struct NdIterExtStridesPtr<T, S>(NdIterExtStridesPtrMut<T, S>);

impl<T, S> NdIterExtStridesPtr<T, S> {
    /// Creates the extension starting at `initial_ptr` with the given per-dimension byte strides.
    #[inline(always)]
    pub fn new(strides: &[S], initial_ptr: *const T) -> Self
    where
        S: Copy,
    {
        Self(NdIterExtStridesPtrMut::new(strides, initial_ptr.cast_mut()))
    }
}
impl<T, S> NdIterExtension for NdIterExtStridesPtr<T, S>
where
    S: Idx + 'static,
{
    type Item = *const T;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_decrease(dim, before, after, diff);
    }

    #[inline(always)]
    fn next(&self) -> *const T {
        <NdIterExtStridesPtrMut<T, S> as NdIterExtension>::next(&self.0).cast_const()
    }

    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        <NdIterExtStridesPtrMut<T, S> as NdIterExtension>::assert_ndim(&self.0, ndim);
    }
}

/// An nd-iterator extension that tracks a `*mut u8` pointer into a strided buffer.
///
/// On each dimension change the pointer is adjusted by the difference in byte offsets:
/// `ptr += (after - before) * stride[dim]`.
#[derive(Clone)]
pub(crate) struct NdIterExtStridesPtrMut<T, S> {
    strides: DimArray<S>,
    current_ptr: *mut T,
}

impl<T, S> NdIterExtStridesPtrMut<T, S> {
    /// Creates the extension starting at `initial_ptr` with the given per-dimension byte strides.
    #[inline(always)]
    pub fn new(strides: &[S], initial_ptr: *mut T) -> Self
    where
        S: Copy,
    {
        Self {
            strides: DimArray::from_slice(strides).unwrap(),
            current_ptr: initial_ptr,
        }
    }
}
impl<T, S> NdIterExtension for NdIterExtStridesPtrMut<T, S>
where
    S: Idx + 'static,
{
    type Item = *mut T;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = diff as usize;
        let stride = self.strides[dim].usize();
        self.current_ptr = unsafe { self.current_ptr.add(diff * stride) };
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = diff as usize;
        let stride = self.strides[dim].usize();
        self.current_ptr = unsafe { self.current_ptr.sub(diff * stride) };
    }

    #[inline(always)]
    fn next(&self) -> *mut T {
        self.current_ptr
    }

    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        assert_eq!(self.strides.len(), ndim);
    }
}

/// An nd-iterator extension that tracks an offset into a strided array.
///
/// On each dimension change the offset is adjusted by the difference in element counts:
/// `offset += (after - before) * stride[dim]`.
pub(crate) struct NdIterExtStridesOffset {
    strides: DimArray<u64>,
    offset: u64,
}
impl NdIterExtStridesOffset {
    #[inline(always)]
    pub fn new(strides: &[u64], initial_offset: u64) -> Self {
        Self {
            strides: DimArray::from_slice(strides).unwrap(),
            offset: initial_offset,
        }
    }
}
impl NdIterExtension for NdIterExtStridesOffset {
    type Item = u64;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        self.offset += diff * self.strides[dim];
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        self.offset -= diff * self.strides[dim];
    }

    #[inline(always)]
    fn next(&self) -> u64 {
        self.offset
    }

    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        assert_eq!(self.strides.len(), ndim);
    }
}

#[inline(always)]
pub(crate) fn nd_iter_ext_logical_global_index(
    shape: &[u64],
    begin: &[u64],
) -> NdIterExtStridesOffset {
    let logical_strides = default_logical_strides(shape);
    let initial_offset = (0..shape.len())
        .map(|dim| begin[dim] * logical_strides[dim])
        .sum();
    NdIterExtStridesOffset::new(&logical_strides, initial_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::iter::NdIter;

    #[test]
    fn strides_ptr_mut_1d_stride_1() {
        let mut data = [0u8; 8];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(&[1usize], base);
        let iter = NdIter::new(&[8u64], ext);
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
        let ext = NdIterExtStridesPtrMut::new(&[stride], base);
        let iter = NdIter::new(&[8u64], ext);
        let mut i = 0usize;
        for (_, ptr) in iter {
            assert_eq!(ptr, unsafe { base.add(i * stride) }, "step {i}");
            i += 1;
        }
    }

    #[test]
    fn strides_ptr_mut_2d_row_major_contiguous() {
        let rows = 3usize;
        let cols = 4usize;
        let elem = 4usize; // e.g. f32
        let mut data = vec![0u8; rows * cols * elem];
        let base = data.as_mut_ptr();
        let ext = NdIterExtStridesPtrMut::new(&[cols * elem, elem], base);
        let iter = NdIter::new(&[rows as u64, cols as u64], ext);
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
        // Strides: dim 0 (row) = 1, dim 1 (col) = rows = 2
        let ext = NdIterExtStridesPtrMut::new(&[1, rows], base);
        let iter = NdIter::new(&[rows as u64, cols as u64], ext);
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
        let start = unsafe { base.add(1 * 4 + 1 * 1) };
        let ext = NdIterExtStridesPtrMut::new(&[4usize, 1], start);
        let iter = NdIter::new_with_begin(&[1u64, 1], &[3, 3], ext);
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
        let ext = NdIterExtStridesPtrMut::new(&[4usize, 1], data.as_mut_ptr());
        let mut iter = NdIter::new_with_begin(&[2u64, 2], &[2, 5], ext);
        assert!(iter.next().is_none());
    }

    #[test]
    fn strides_ptr_const_1d_matches_mut() {
        let data = [0u8; 8];
        let base = data.as_ptr();
        let ext = NdIterExtStridesPtr::new(&[1usize], base);
        let iter = NdIter::new(&[8u64], ext);
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
        let ext = NdIterExtStridesPtr::new(&[cols, 1], base);
        let iter = NdIter::new(&[rows as u64, cols as u64], ext);
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
        let ext = NdIterExtStridesPtr::new(&[1usize], base);
        let iter = NdIter::new(&[4u64], ext);
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
        let ext = NdIterExtStridesPtrMut::new(&[cols, 1], base);
        let iter = NdIter::new(&[rows as u64, cols as u64], ext);
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
        // dim 0 (row) stride = 1 element, dim 1 (col) stride = rows = 2 elements.
        let ext = NdIterExtStridesPtrMut::new(&[1usize, rows], base);
        let iter = NdIter::new(&[rows as u64, cols as u64], ext);
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
        let start = unsafe { base.add(1 * 4 + 1 * 1) };
        let ext = NdIterExtStridesPtrMut::new(&[4usize, 1], start);
        let iter = NdIter::new_with_begin(&[1u64, 1], &[3, 3], ext);
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
        let ext = NdIterExtStridesPtr::new(&[1usize], base);
        let iter = NdIter::new(&[5u64], ext);
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
}

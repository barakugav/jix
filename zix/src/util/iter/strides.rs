use crate::util::iter::NdIterExtension;
use crate::util::{default_strides, DimArray, Idx};

/// An nd-iterator extension that tracks a `*const u8` pointer into a strided buffer.
///
/// On each dimension change the pointer is adjusted by the difference in byte offsets:
/// `ptr += (after - before) * stride[dim]`.
#[derive(Clone)]
pub(crate) struct NdIterExtStridesPtr<S>(NdIterExtStridesPtrMut<S>);

impl<S> NdIterExtStridesPtr<S> {
    /// Creates the extension starting at `initial_ptr` with the given per-dimension byte strides.
    pub fn new(strides: &[S], initial_ptr: *const u8) -> Self
    where
        S: Copy,
    {
        Self(NdIterExtStridesPtrMut::new(strides, initial_ptr.cast_mut()))
    }
}
impl<Ix, S> NdIterExtension<Ix> for NdIterExtStridesPtr<S>
where
    Ix: Idx,
    S: Idx + 'static,
{
    type Item<'a> = *const u8;

    fn on_increase(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_increase(dim, before, after, diff);
    }
    fn on_decrease(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_decrease(dim, before, after, diff);
    }

    fn next(&self) -> *const u8 {
        <NdIterExtStridesPtrMut<S> as NdIterExtension<Ix>>::next(&self.0).cast_const()
    }

    fn assert_ndim(&self, ndim: usize) {
        <NdIterExtStridesPtrMut<S> as NdIterExtension<Ix>>::assert_ndim(&self.0, ndim);
    }
}

/// An nd-iterator extension that tracks a `*mut u8` pointer into a strided buffer.
///
/// On each dimension change the pointer is adjusted by the difference in byte offsets:
/// `ptr += (after - before) * stride[dim]`.
#[derive(Clone)]
pub(crate) struct NdIterExtStridesPtrMut<S> {
    strides: DimArray<S>,
    current_ptr: *mut u8,
}

impl<S> NdIterExtStridesPtrMut<S> {
    /// Creates the extension starting at `initial_ptr` with the given per-dimension byte strides.
    pub fn new(strides: &[S], initial_ptr: *mut u8) -> Self
    where
        S: Copy,
    {
        Self {
            strides: strides.try_into().unwrap(),
            current_ptr: initial_ptr,
        }
    }
}
impl<Ix, S> NdIterExtension<Ix> for NdIterExtStridesPtrMut<S>
where
    Ix: Idx,
    S: Idx + 'static,
{
    type Item<'a> = *mut u8;

    fn on_increase(&mut self, dim: usize, _before: Ix, _after: Ix, diff: Ix) {
        let diff: usize = diff.try_into().unwrap();
        let stride: usize = self.strides[dim].try_into().unwrap();
        self.current_ptr = unsafe { self.current_ptr.add(diff * stride) };
    }
    fn on_decrease(&mut self, dim: usize, _before: Ix, _after: Ix, diff: Ix) {
        let diff: usize = diff.try_into().unwrap();
        let stride: usize = self.strides[dim].try_into().unwrap();
        self.current_ptr = unsafe { self.current_ptr.sub(diff * stride) };
    }

    fn next(&self) -> *mut u8 {
        self.current_ptr
    }

    fn assert_ndim(&self, ndim: usize) {
        assert_eq!(self.strides.len(), ndim);
    }
}

/// An nd-iterator extension that tracks an offset into a strided array.
///
/// On each dimension change the offset is adjusted by the difference in element counts:
/// `offset += (after - before) * stride[dim]`.
pub(crate) struct NdIterExtStridesOffset<Ix> {
    strides: DimArray<Ix>,
    offset: Ix,
}
impl<Ix> NdIterExtStridesOffset<Ix>
where
    Ix: Idx,
{
    pub fn new(strides: &[Ix], initial_offset: Ix) -> Self {
        Self {
            strides: strides.try_into().unwrap(),
            offset: initial_offset,
        }
    }
}
impl<Ix> NdIterExtension<Ix> for NdIterExtStridesOffset<Ix>
where
    Ix: Idx,
{
    type Item<'a>
        = Ix
    where
        Self: 'a;

    fn on_increase(&mut self, dim: usize, _before: Ix, _after: Ix, diff: Ix) {
        self.offset += diff * self.strides[dim];
    }
    fn on_decrease(&mut self, dim: usize, _before: Ix, _after: Ix, diff: Ix) {
        self.offset -= diff * self.strides[dim];
    }

    fn next(&self) -> Ix {
        self.offset
    }

    fn assert_ndim(&self, ndim: usize) {
        assert_eq!(self.strides.len(), ndim);
    }
}

pub(crate) fn nd_iter_ext_logical_global_index<Ix: Idx>(
    shape: &[Ix],
    begin: &[Ix],
) -> NdIterExtStridesOffset<Ix> {
    let logical_strides = default_strides(shape, Ix::ONE);
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
        let mut iter = NdIter::new(&[8usize], ext);
        let mut i = 0usize;
        while let Some((_, ptr)) = iter.next() {
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
        let mut iter = NdIter::new(&[8usize], ext);
        let mut i = 0usize;
        while let Some((_, ptr)) = iter.next() {
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
        let mut iter = NdIter::new(&[rows, cols], ext);
        let mut flat = 0usize;
        while let Some((_, ptr)) = iter.next() {
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
        let mut iter = NdIter::new(&[rows, cols], ext);
        // Iteration order is row-major by *index*, but pointer jumps follow column-major layout:
        // [0,0]=0, [0,1]=2, [0,2]=4, [1,0]=1, [1,1]=3, [1,2]=5
        let expected_offsets: &[usize] = &[0, 2, 4, 1, 3, 5];
        let mut i = 0usize;
        while let Some((_, ptr)) = iter.next() {
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
        let mut iter = NdIter::new_with_begin(&[1usize, 1], &[3, 3], ext);
        let expected: &[usize] = &[5, 6, 9, 10];
        let mut i = 0;
        while let Some((_, ptr)) = iter.next() {
            assert_eq!(ptr, unsafe { base.add(expected[i]) }, "step {i}");
            i += 1;
        }
        assert_eq!(i, 4);
    }

    #[test]
    fn strides_ptr_mut_empty_range_yields_no_pointers() {
        let mut data = [0u8; 16];
        let ext = NdIterExtStridesPtrMut::new(&[4usize, 1], data.as_mut_ptr());
        let mut iter = NdIter::new_with_begin(&[2usize, 2], &[2, 5], ext);
        assert!(iter.next().is_none());
    }

    #[test]
    fn strides_ptr_const_1d_matches_mut() {
        let data = [0u8; 8];
        let base = data.as_ptr();
        let ext = NdIterExtStridesPtr::new(&[1usize], base);
        let mut iter = NdIter::new(&[8usize], ext);
        let mut i = 0usize;
        while let Some((_, ptr)) = iter.next() {
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
        let mut iter = NdIter::new(&[rows, cols], ext);
        let mut flat = 0usize;
        while let Some((_, ptr)) = iter.next() {
            assert_eq!(ptr, unsafe { base.add(flat) });
            flat += 1;
        }
        assert_eq!(flat, rows * cols);
    }
}

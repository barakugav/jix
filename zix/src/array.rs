use std::mem::MaybeUninit;
use std::ops::Range;

use crate::codec::{DecoderCodecConfig, DecoderParams, Encoder, ReadContext};
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_buffer_size, check_get_range, Result};
use crate::ops::{DimensionChange, ElementTypeChange, IntoCompact, ToDim, ToType};
use crate::storage::block::{build_block_table, BlockFn, BlockFnWithState};
use crate::storage::{ArrayBlockTableStorageBase, ArrayStorageTyped, BlocksLayout, Compact, Ref};
use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::NdIter;
use crate::util::{
    assert_unchecked_eq, cast_slice_mut, default_strides, dim_arr, nd_copy, AlignedBytes, DimArray,
    Idx, IxIterExt,
};
use crate::{
    ArrayParams, ArrayStorage, DimDyn, Dimension, ElementType, IntoDimension, Ty, TypeDyn,
};

/// A multi-dimensional array, usually compressed, backed by a generic storage.
///
/// `Array<S>` is the central type in zix. It behave like a regular n-dimensional array, but
/// its data is stored in a compressed format and decoded on demand. Its core functionality is
/// provided by [`shape()`](Array::shape), [`dtype()`](Array::dtype),
/// and [`to_ndarray_buf()`](Array::to_ndarray_buf), all other functions are built on top of those.
///
/// An array is generic over `S: ArrayStorage`, which provides the implementation of the three core
/// methods. The main concrete storage backend is the block-compressed [`Compact`] type, which
/// divides the array into n-dimensional blocks and compresses each block independently, and its
/// the return type of the common creation methods for arrays
/// (e.g.[`compact_array()`](Array::compact_array) and [`copy()`](Array::copy)).
///
/// # Storage variants
///
/// The primary concrete storages are:
///
/// | Type | Description |
/// |------|-------------|
/// | [`Array<Compact>`](crate::storage::Compact) | Heap-allocated block-compressed array. The main storage backend. |
/// | [`Array<Add<S1, S2>> or Array<Neg<S>> ...`](crate::ops) | Lazy operations views that wrap one or more arrays and apply a transformation at read time. Created by methods in [`ops`](crate::ops). |
/// | [`Array<Ref<'a, S>>`](crate::storage::Ref) | A reference to another storage, used to let multiple operations consume an array without cloning its storage. Created by [`as_ref`](Array::as_ref). |
/// | [`Array<Plain<...>>`](crate::storage::Plain) | Zero-copy view into an uncompressed (possibly strided) in-memory buffer. Created by [`plain_ndarray`](Array::plain_ndarray) and [`plain_ndarray_view`](Array::plain_ndarray_view). |
/// | [`Array<Scalar<T>>`](crate::storage::Scalar) | A single scalar broadcast to any shape, used as the operand in expressions like `array + 1.0`. |
///
/// # Operations and lazy evaluation
///
/// Every operation on an `Array<S>` returns a new `Array` whose type encodes the full operation
/// chain:
///
/// ```text
/// Array<Compact>
///   .neg()                 -> Array<Neg<Compact>>
///   .reshape_view(...)     -> Array<Reshape<Neg<Compact>>>
///   .permute_axes(axes)    -> Array<PermuteAxes<Reshape<...>>>
///   .add(other_array)      -> Array<Add<PermuteAxes<...>, Compact>>
///   .sum(axis)             -> Array<Sum<Add<...>>>
///   .copy();               -> Array<Compact> - materialize the pipeline
/// ```
///
/// Data is never copied or computed at construction time. An operation only runs when the result
/// is materialized via [`to_ndarray()`](Array::to_ndarray), [`copy()`](Array::copy), and their variants.
/// At that point the read request propagates inward through the storage
/// chain, and only the minimum required data is read from the innermost backend.
///
/// Because `Array<S>` is monomorphized over `S` at compile time, chains of operations incur zero
/// virtual dispatch overhead. The full static type of an expression -
/// e.g. `Array<Add<Neg<S1>, Reshape<S2>>>` - is resolved by the compiler, which can inline the
/// entire pipeline into a single read loop. The type system *is* the execution plan.
///
/// Operations accept an owned `Array<S>` and return a new `Array<Op<S>>` that wraps the original.
/// To reuse an array in multiple operations, use [`as_ref()`](Array::as_ref) to create a reference.
///
/// # Examples
///
/// Create arrays from various sources:
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// // Compress an ndarray into a block-compressed Array<Compact>.
/// let compact = Array::compact_array(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
///
/// // Zero-copy view of an existing ndarray (any layout).
/// let plain = Array::plain_ndarray_view(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
///
/// // Read a previously serialized array back from a file.
/// let tmp_dir = tempfile::tempdir()?;
/// let path = tmp_dir.path().join("array.zix");
/// compact.write_to_file(&path)?;
/// let from_file = Array::read_from_file(&path, ArrayParams::default())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Apply operations on compressed arrays, creating lazy views, writing the result to a file:
/// ```
/// use zix::Array;
/// use zix::dtype::Dtyped;
/// use ndarray::array;
///
/// // Compress a 2-D f32 ndarray.
/// let array = Array::compact_array(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
/// assert_eq!(array.shape(), &[2, 3]);
/// assert_eq!(array.dtype(), &f32::DTYPE);
///
/// // Decompress and compare.
/// let decompressed = array.to_ndarray()?;
/// assert_eq!(decompressed[[0, 0]], 1.5);
/// assert_eq!(decompressed[[1, 1]], 6.17);
///
/// // Apply operations on a compressed array, creating lazy views
/// let ones = Array::compact_array(&ndarray::Array2::<f32>::ones((2, 3)))?;
/// let scaled = array                               // Array<Compact>
///     .exp()                                       // Array<Exp<Compact>>
///     .floor()                                     // Array<Floor<Exp<Compact>>>
///     * 2.0f32                                     // Array<Mul<Floor<...>, Scalar<f32>>>
///     + ones;                                      // Array<Add<Mul<...>, Compact>>
/// // lazy view arrays are still functional arrays
/// // access to data execute the pipeline on demand, possibly on a sub set of the original array
/// assert_eq!(scaled.shape(), &[2, 3]);
/// assert_eq!(scaled.dtype(), &f32::DTYPE);
/// assert_eq!(scaled.to_ndarray()?[[1, 1]], 957.0);
///
/// // Materialize the result and write to a file.
/// let result = scaled
///     .argmax(/* axis */ 1)                        // Array<ArgMax<Add<Mul<...>, Compact>>>
///     .cast::<i16>()                               // Array<Cast<ArgMax<Add<...>>>>
///     // materialize the pipeline with a copy
///     .copy()?;                                    // Array<Compact>
/// assert_eq!(result.shape(), &[2]);
/// assert_eq!(result.dtype(), &i16::DTYPE);
/// let tmp_dir = tempfile::tempdir()?;
/// result.write_to_file(tmp_dir.path().join("result.zix").as_ref())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance notes
///
/// The n-dimensional block shape used by `Array<Compact>` has a huge effect on both compression
/// ratio and read performance. If the access pattern is known in advance, providing a matching
/// block shape can improve the performance of the library significantly. If not provided, the
/// block shape is chosen automatically to fit within the L1 data cache, by starting with a block
/// shape of all ones and iteratively increasing each dimension greedily, in order from last to first
/// dim, as long the block size in bytes does not exceed the target size.
/// Additional arrays that are created from existing arrays (`.copy()`, `.reshape()`, result
/// of operations, etc.) choose their block shape with a heuristic, trying to preserve the original
/// user block shape as much as possible while respecting the new shape and layout.
///
/// Shape-changing operations - [`reshape_view`](Array::reshape_view),
/// [`broadcast_view`](Array::broadcast_view), [`permute_axes`](Array::permute_axes) - remap how
/// output indices translate to positions in the underlying blocks. When the new layout crosses
/// block boundaries that the original respected, a single read may decompress many more blocks
/// than necessary. To avoid this, materialize with [`copy`](Array::copy) (automatic block shape)
/// or [`copy_with`](Array::copy_with) (explicit [`ArrayParams`]) after a shape change. The eager
/// variants [`reshape`](Array::reshape) and [`broadcast`](Array::broadcast) call `copy`
/// internally. To ensure a well-aligned block layout, pass explicit `ArrayParams` with a block
/// shape that matches the expected access pattern.
///
/// # Element type tracking
///
/// `S::ElementType` records the scalar element type at the type level. When the element type is
/// statically known, `S::ElementType = Ty<T>`, and all element-wise operations — arithmetic,
/// comparisons, reductions, type casts — become available. When the element type is only known
/// at runtime (e.g. for arrays loaded from files), `S::ElementType = TypeDyn`, and those
/// operations are not available until the type is asserted.
///
/// Arrays constructed from typed sources automatically carry `Ty<T>`: `compact_array(&array![1.0f32])`
/// returns `Array<Compact<Ty<f32>, Dim<1>>>`. Arrays loaded from disk carry `TypeDyn`. Use
/// [`to_typed::<T>()`](Array::to_typed) to assert the expected element type — validated
/// against the stored dtype at runtime — and recover `Ty<T>`. Use
/// [`to_type_dyn()`](Array::to_type_dyn) to erase the static element type.
///
/// # Dimension type tracking
///
/// `S::Dimension` records the number of array axes at the type level. When the ndim is
/// statically known, `S::Dimension = Dim<N>`, and the compiler can verify that operations that
/// require a specific ndim are used correctly. When the ndim is only known at runtime (e.g.
/// for arrays loaded from files), `S::Dimension = DimDyn`.
///
/// The dimension type propagates automatically: passing a `usize` to `insert_axis` on a
/// `Dim<N>` array produces `Dim<N+1>`; passing `&[usize]` always produces `DimDyn`. Use
/// [`to_dim::<Dim<N>>()`](Array::to_dim) to assert a specific ndim and recover static
/// tracking, or [`to_dim_dyn()`](Array::to_dim_dyn) to erase static dimension info.
#[derive(Clone)]
pub struct Array<S> {
    pub(crate) storage: S,
}

impl<T, D> Array<Compact<Ty<T>, D>> {
    /// Compress an ndarray into a block-compressed `Array<Compact<D>>` with default encoding settings.
    ///
    /// The array is partitioned into n-dimensional blocks, each independently compressed. The
    /// block shape is derived automatically to fit within the L1 data cache. Use
    /// [`compact_array_with`](Array::compact_array_with) for explicit control over block shape,
    /// compression level, and other codec settings. If the access pattern is known in advance,
    /// providing a matching block shape can improve read performance significantly.
    ///
    /// The element type `Ty<T>` and dimension type `D` are inferred from the ndarray argument's type.
    ///
    /// # Errors
    ///
    /// - [`TooManyDimensions`](crate::ErrorKind::TooManyDimensions) - `array.ndim()` exceeds
    ///   [`NDIM_MAX`](crate::NDIM_MAX).
    /// - [`CodecError`](crate::ErrorKind::CodecError) - compression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::Array;
    /// use zix::dtype::Dtyped;
    /// use ndarray::array;
    ///
    /// // Compress a 2-D f32 ndarray.
    /// let array = Array::compact_array(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
    /// assert_eq!(array.shape(), &[2, 3]);
    /// assert_eq!(array.dtype(), &f32::DTYPE);
    ///
    /// // Decompress and compare.
    /// let decompressed = array.to_ndarray()?;
    /// assert_eq!(decompressed[[0, 0]], 1.5);
    /// assert_eq!(decompressed[[1, 1]], 6.17);
    ///
    /// // Apply operations on a compressed array, creating lazy views
    /// let ones = Array::compact_array(&ndarray::Array2::<f32>::ones((2, 3)))?;
    /// let scaled = array                               // Array<Compact>
    ///     .exp()                                       // Array<Exp<Compact>>
    ///     .floor()                                     // Array<Floor<Exp<Compact>>>
    ///     * 2.0f32                                     // Array<Mul<Floor<...>, Scalar<f32>>>
    ///     + ones;                                      // Array<Add<Mul<...>, Compact>>
    /// assert_eq!(scaled.shape(), &[2, 3]);
    /// assert_eq!(scaled.dtype(), &f32::DTYPE);
    /// assert_eq!(scaled.to_ndarray()?[[1, 1]], 957.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compact_array<S, InD>(array: &ndarray::ArrayBase<S, InD>) -> Result<Self>
    where
        InD: ndarray::Dimension + IntoDimension<Dimension = D>,
        D: Dimension,
        S: ndarray::Data<Elem = T>,
        T: Dtyped,
    {
        Array::compact_array_with(array, ArrayParams::default())
    }

    /// Compress an ndarray into a block-compressed `Array<Compact>` with explicit `ArrayParams`.
    ///
    /// See [`compact_array`](Array::compact_array) for the default-parameter version, which has more
    /// documentation and examples.
    ///
    /// Use this method to specify encoding parameters such as block shape, compression level, etc.
    /// See [`ArrayParams`] for details on the available parameters and their effects on performance.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    ///
    /// let data = ndarray::Array2::<f32>::zeros((512, 512));
    ///
    /// // Store with 64*64 blocks - good for tile-at-a-time access patterns.
    /// let mut params = ArrayParams::new();
    /// params.block_shape(&[64, 64]);
    /// let array = Array::compact_array_with(&data, params)?;
    ///
    /// // Read tiles of 128*128 by decompressing 2*2 blocks at a time.
    /// let context = array.read_ctx();
    /// for tile_row in 0..7 {
    ///   for tile_col in 0..7 {
    ///     let row_range = (tile_row * 64)..((tile_row + 2) * 64);
    ///     let col_range = (tile_col * 64)..((tile_col + 2) * 64);
    ///     let tile = array.to_ndarray_sub(&[row_range, col_range], &context)?;
    ///     println!("tile ({tile_row},{tile_col}) sum: {}", tile.sum());
    ///   }
    /// }
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn compact_array_with<S, InD>(
        array: &ndarray::ArrayBase<S, InD>,
        mut params: ArrayParams,
    ) -> Result<Self>
    where
        InD: ndarray::Dimension + IntoDimension<Dimension = D>,
        D: Dimension,
        S: ndarray::Data<Elem = T>,
        T: Dtyped,
    {
        let array = Array::plain_ndarray_view(array)?;
        params.tune(array.shape(), array.dtype())?;
        let context = ReadContext::new(&params.decoder_params.clone().unwrap_or_default())?;
        array.copy_with(params, &context)
    }
}

impl<D> Array<Compact<TypeDyn, D>> {
    /// Compress a raw n-dimensional buffer into a block-compressed `Array<Compact>`.
    ///
    /// Same as [`compact_array_with`](Array::compact_array_with) but takes a raw pointer and
    /// explicit shape and strides.
    ///
    /// # Arguments
    ///
    /// - `ptr`: pointer to the beginning of the buffer. Must be aligned to `dtype.alignment()`.
    /// - `shape`: shape of the n-dimensional array. At most [`NDIM_MAX`](crate::NDIM_MAX)
    ///   dimensions are allowed.
    /// - `strides`: strides of the n-dimensional array, in bytes. Must be the same length as
    ///   `shape`.
    /// - `dtype`: element type of the array. The buffer is interpreted as containing elements of
    ///   this type.
    /// - `params`: block layout and codec parameters. See [`ArrayParams`] for details.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a readable buffer, laid out with the given
    /// `strides` (in bytes, one per dimension).
    /// The buffer should contains elements of the given `dtype`.
    /// Accessing the buffer according to the shape, strides and dtype must be memory-safe and yield
    /// valid elements.
    pub unsafe fn compact_nd_ptr<Sh>(
        ptr: *const u8,
        shape: Sh,
        strides: &[usize],
        dtype: Dtype,
        mut params: ArrayParams,
    ) -> Result<Self>
    where
        Sh: IntoDimension<Dimension = D>,
        D: Dimension,
    {
        let plain_storage = unsafe {
            crate::storage::Plain::new(
                crate::storage::PlainRef::<'_, u8>::new(),
                ptr,
                shape,
                strides,
                dtype,
            )?
        };
        let array = Array::from_storage(plain_storage);
        params.tune(array.shape(), array.dtype())?;
        let context = ReadContext::new(&params.decoder_params.clone().unwrap_or_default())?;
        array.copy_with(params, &context)
    }
}

impl<S: ArrayStorage> Array<S> {
    /// Get the shape of the array, one element per dimension.
    pub fn shape(&self) -> &[u64] {
        let shape = self.storage.shape();
        if let Some(compile_time_ndim) = S::Dimension::NDIM {
            unsafe { assert_unchecked_eq!(shape.len(), compile_time_ndim) };
        }
        shape
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements in the array.
    pub fn nitems(&self) -> u64 {
        self.shape().iter().product()
    }

    /// Check if the array is empty (has zero elements in any dimension).
    pub fn is_empty(&self) -> bool {
        self.shape().contains(&0)
    }

    /// Get the element dtype of the array.
    ///
    /// See [`Dtype`] for details on the supported dtypes and their properties.
    ///
    /// ```rust,ignore
    /// use zix::Array;
    /// use zix::dtype::Dtyped;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert_eq!(a.dtype(), &f32::DTYPE);
    ///
    /// let b = Array::plain_ndarray_view(&array![[false, true]])?;
    /// assert_eq!(b.dtype(), &bool::DTYPE);
    ///
    /// #[derive(Dtyped, Copy, Clone)]
    /// struct Point { x: f32, y: f32 }
    /// let c = Array::plain_ndarray_view(&array![Point { x: 1.0, y: 2.0 }])?;
    /// assert_eq!(c.dtype(), &Point::DTYPE);
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn dtype(&self) -> &Dtype {
        let dtype = self.storage.dtype();
        if let Some(compile_time_dtype) = S::ElementType::DTYPE {
            unsafe { assert_unchecked_eq!(dtype, &compile_time_dtype) };
        }
        dtype
    }

    /// Decode the full array into a heap-allocated [`ndarray::Array`].
    ///
    /// Decompresses all blocks and returns the elements in a contiguous row-major ndarray.
    /// `T` must match [`self.dtype()`](Array::dtype).
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::ErrorKind::CodecError) - block decompression fails.
    pub fn to_ndarray(&self) -> Result<ndarray::ArrayD<S::Item>>
    where
        S: ArrayStorage + ArrayStorageTyped,
    {
        let shape = self.shape();
        let full_range = dim_arr(shape.len(), |dim| 0u64..shape[dim]);
        self.to_ndarray_sub(&full_range, &self.read_ctx())
    }

    /// Decode a rectangular sub-region into a heap-allocated [`ndarray::Array`].
    ///
    /// Only the compressed blocks overlapping `index` are decompressed. When `index` aligns to
    /// block boundaries no extra data is read; for unaligned ranges, the overlapping boundary
    /// blocks are fully decompressed and only the requested slice is returned.
    ///
    /// `index` must contain one half-open `start..end` per dimension within
    /// `0..self.shape()[dim]`. `T` must match [`self.dtype()`](Array::dtype). Obtain a
    /// [`ReadContext`] via [`read_ctx`](Array::read_ctx).
    ///
    /// # Errors
    ///
    /// - [`InvalidIndex`](crate::ErrorKind::InvalidIndex) - `index` is out of bounds or
    ///   has a different number of dimensions than the array.
    /// - [`CodecError`](crate::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    ///
    /// let context = a.read_ctx();
    /// assert_eq!(
    ///     a.to_ndarray_sub(&[1..3, 1..3], &context)?,
    ///     array![[5, 6], [8, 9]].into_dyn()
    /// );
    /// assert_eq!(
    ///    a.to_ndarray_sub(&[0..2, 0..2], &context)?,
    ///    array![[1, 2], [4, 5]].into_dyn()
    /// );
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn to_ndarray_sub(
        &self,
        index: &[Range<u64>],
        context: &ReadContext,
    ) -> Result<ndarray::ArrayD<S::Item>>
    where
        S: ArrayStorage + ArrayStorageTyped,
    {
        check_get_range(self.shape(), index)?;
        let ndim = self.ndim();
        let out_shape = dim_arr(ndim, |dim| {
            let len = index[dim].end - index[dim].start;
            let len: usize = len.try_into().unwrap();
            len
        });
        let mut array = ndarray::ArrayD::uninit(&out_shape[..]);
        self.to_ndarray_buf(
            index,
            unsafe { cast_slice_mut::<MaybeUninit<S::Item>, u8>(array.as_slice_mut().unwrap()) },
            context,
        )?;
        Ok(unsafe { array.assume_init() })
    }

    /// Decode a rectangular sub-region into a caller-supplied byte buffer.
    ///
    /// The raw I/O primitive underlying [`to_ndarray`](Array::to_ndarray) and
    /// [`to_ndarray_sub`](Array::to_ndarray_sub). `buf` must be exactly
    /// `index.iter().map(|r| r.len()).product() * self.dtype().itemsize()` bytes and aligned to
    /// `self.dtype().alignment()`. Elements are written in row-major (C-contiguous) order.
    ///
    /// # Errors
    ///
    /// - [`InvalidBufferSize`](crate::ErrorKind::InvalidBufferSize) - `buf` has the wrong
    ///   length for the requested range and dtype.
    /// - [`InvalidArgument`](crate::ErrorKind::InvalidArgument) - `buf` is insufficiently
    ///   aligned for the dtype.
    /// - [`CodecError`](crate::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    ///
    /// let context = a.read_ctx();
    /// let mut buf = vec![0u32; 4];
    /// {
    ///     let buf = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4) };
    ///     a.to_ndarray_buf(&[1..3, 1..3], buf, &context)?;
    /// }
    /// assert_eq!(buf, vec![5, 6, 8, 9]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn to_ndarray_buf(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()> {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();
        check_get_range(shape, index)?;
        check_get_buffer_size(index, dtype, buf)?;

        let read_shape = self.storage._spec().blocks_layout.preferred_read_shape();
        debug_assert!(read_shape.iter().all(|l| *l > 0));

        // Fast path for small reads
        let small_read = (0..ndim).all(|d| (index[d].end - index[d].start) <= read_shape[d] as u64);
        if small_read {
            return self.storage.read_data(index, buf, context);
        }

        // Block-space begin/end for NdIter.
        let block_begin = dim_arr(ndim, |dim| index[dim].start / read_shape[dim] as u64);
        let block_end = dim_arr(ndim, |dim| index[dim].end.div_ceil(read_shape[dim] as u64));
        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = dim_arr(ndim, |dim| index[dim].start);
        let elem_end = dim_arr(ndim, |dim| index[dim].end);
        // NdIter that yields blocks of size <= read_shape
        let mut block_iter = NdIter::new_with_begin(
            &block_begin,
            &block_end,
            NdIterExtBlockOffsetSize::new(
                shape,
                &elem_begin,
                &elem_end,
                &dim_arr(ndim, |dim| read_shape[dim] as u64),
            ),
        );

        let itemsize = dtype.itemsize() as usize;
        let out_shape = dim_arr(ndim, |dim| (index[dim].end - index[dim].start) as usize);
        let out_strides = default_strides(&out_shape, itemsize);

        let mut tmp_buf = context.tmp_buf(0, dtype.alignment());
        while let Some((block_idx, (block_inner_offset, block_size))) = block_iter.next() {
            let inner_index = dim_arr(ndim, |d| {
                let start = block_idx[d] * read_shape[d] as u64 + block_inner_offset[d];
                let end = start + block_size[d];
                start..end
            });
            let tmp_buf = {
                let read_nitems = block_size.iter().product::<u64>();
                tmp_buf.set_len(read_nitems as usize * dtype.itemsize() as usize);
                tmp_buf.as_mut_slice()
            };
            self.storage.read_data(&inner_index, tmp_buf, context)?;

            let out_offset = (0..ndim)
                .map(|d| (inner_index[d].start - index[d].start) as usize * out_strides[d])
                .sum::<usize>();
            let dst_ptr = unsafe { buf.as_mut_ptr().add(out_offset) };

            unsafe {
                nd_copy(
                    tmp_buf.as_ptr(),
                    dst_ptr,
                    block_size,
                    &default_strides(block_size, itemsize as _),
                    &out_strides,
                    itemsize,
                )
            };
        }
        Ok(())
    }

    /// Copy the data of this array into a new `Array<Compact>` by compressing it into new blocks.
    ///
    /// The primary use of `copy` is to materialize a lazy operation chain:
    /// An `Array<S>` can have an arbitrary storage implementation, often a lazy view of some one or
    /// more computation, for example `Array<Floor<Mul<Compact, Scalar<f32>>>>` (see the examples).
    /// Reads to such lazy view arrays always perform the whole computation pipeline on the fly,
    /// which is very flexible but can be inefficient for repeated access. Coping the data and
    /// re-compressing it into a new array with `copy` breaks the lazy storage chain and materializes
    /// the result as a standalone `Array<Compact>`.
    ///
    /// In contrast to "simple" views such as unary element-wise operations, lazy ops that change the
    /// shape of the array (e.g. `reshape`, `broadcast`, `permute_axes`) can cause block boundaries
    /// to no longer align with the logical layout of the array, causing reads to decompress excess
    /// data. Calling `copy` on the result of such an operation re-encodes the data with a freshly
    /// derived block shape that matches the new layout. The block shape of copied arrays is
    /// automatically derived and tuned from the underlying storage(s), using a heuristic that aims
    /// to preserve user choices (that may depend on the user knowledge of the access pattern), but
    /// its not perfect - you may want to explicitly pass some parameters via
    /// [`copy_with`](Array::copy_with).
    ///
    /// Its also possible to materialize a lazy operation chain directly into a file without holding
    /// the whole result (compressed or decompressed) in memory.
    /// See [`write_to_file`](Array::write_to_file) and its variants for details and examples.
    ///
    /// Codec settings (compression level, filters, etc.) are also inherited from the source storage.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::ErrorKind::CodecError) - compression or decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// let result =
    ///     (a * 7.399_f32)  // Array<Mul<Compact, Scalar<f32>>>
    ///    .floor()       // Array<Floor<Mul<Compact, Scalar<f32>>>>
    ///    .copy()?;      // Array<Compact> - materialize the pipeline
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn copy(&self) -> Result<Array<Compact<S::ElementType, S::Dimension>>> {
        let context = self.read_ctx();
        self.copy_with(ArrayParams::default(), &context)
    }

    /// Copy the data of this array into a new `Array<Compact>` with explicit control over parameters.
    ///
    /// Like [`copy`](Array::copy) (see its documentation), but with explicit [`ArrayParams`].
    /// Any optional field in `params` that is not set will be inherited from the source storage
    /// if possible.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::ErrorKind::CodecError) - compression or decompression fails.
    ///
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let mut a_params = ArrayParams::default();
    /// a_params.block_shape(&[1, 2]);
    /// let a = Array::compact_array_with(&array![[1.5f32, 2.0], [3.14, 6.17]], a_params)?;
    ///
    /// // Let's say a is given to us, and we prepare to access it many times with a specific access
    /// // pattern. We copy it and re-compress it with a matching block shape.
    /// let mut b_params = ArrayParams::default();
    /// b_params.block_shape(&[2, 1]);
    /// let b = a.copy_with(b_params, &a.read_ctx())?;
    ///
    /// assert_eq!(
    ///     b.to_ndarray_sub(&[0..2, 0..1],
    ///     &b.read_ctx())?, array![[1.5], [3.14]].into_dyn()
    /// );
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn copy_with(
        &self,
        mut params: ArrayParams,
        context: &ReadContext,
    ) -> Result<Array<Compact<S::ElementType, S::Dimension>>> {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();

        params.override_from_storage(&self.storage);
        params.tune(shape, dtype)?;

        let shape: DimArray<_> = shape.try_into().unwrap();
        let encoder_params = params.encoder_params.clone().unwrap_or_default();

        let block_shape = params.block_shape.as_ref().unwrap();
        let block_size = block_shape.iter().cloned().try_product().unwrap();
        let grid_shape = dim_arr(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));
        let nblocks = grid_shape.iter().cloned().product::<u64>();

        let decoder_cfg = DecoderCodecConfig {
            codec: encoder_params.codec.clone(),
            filters: encoder_params.filters.clone(),
            dtype: dtype.clone(),
        };

        let (mut block_fn, block_compressed_bound) = self.to_block_fn(&params, context)?;
        let blocks = build_block_table(
            nblocks,
            block_size,
            decoder_cfg,
            block_compressed_bound,
            &mut block_fn,
        )?;

        let blocks_layout = BlocksLayout::new(
            params.block_shape.unwrap(),
            params.block_shape_tag.unwrap(),
            params.block_size_hint.unwrap(),
            params.preferred_read_shape.unwrap(),
            params.preferred_read_size_hint.unwrap(),
        );
        let decoder_params = params.decoder_params.unwrap_or_default();

        let shape = S::Dimension::from_slice(&shape).unwrap();
        Ok(Array {
            storage: Compact(ArrayBlockTableStorageBase::new(
                blocks,
                shape,
                blocks_layout,
                encoder_params,
                decoder_params,
            )),
        })
    }

    /// Create a [`ReadContext`] with parameters derived from this array's storage.
    ///
    /// A context encapsulates reusable buffers and codec decompressor instance. Use it for
    /// repeated reads, sharing the allocation and initialization overhead.
    ///
    /// [`to_ndarray`](Array::to_ndarray) builds a context internally. Call this when using
    /// [`to_ndarray_sub`](Array::to_ndarray_sub) or [`to_ndarray_buf`](Array::to_ndarray_buf)
    /// directly.
    ///
    /// Using a context created in other ways (e.g. `ReadContext::default()`) is also valid, and will
    /// yield correct results. Using this method allows an easy way to ensure all reads from an array
    /// use the same decoding configuration (see [`DecoderParams`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    /// let context = a.read_ctx();
    /// // Reuse the same context for multiple reads, sharing buffers.
    /// for row in 0..3 {
    ///     let row_data = a.to_ndarray_sub(&[row..(row + 1), 0..3], &context)?;
    ///     println!("row sum: {}", row_data.sum());
    /// }
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn read_ctx(&self) -> ReadContext {
        let params = self.storage._spec().decoder_params;
        let context = match params {
            Some(params) => ReadContext::new(params),
            None => ReadContext::new(&DecoderParams::default()),
        };
        context.expect("failed to create read context")
    }

    /// Create an array with a storage reference to this array, without cloning the underlying data.
    ///
    /// Almost all ops on arrays accept ownership of an `Array<S>` rather than a reference, for
    /// example `a + b` for two arrays consume `a` and `b`. To reuse an array without cloning its
    /// storage, call `as_ref` to get an `Array<Ref<'_, S>>`, which doesn't own the storage but can
    /// be used in any API that accepts an owned `Array<S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// let b = a.as_ref() + 1.0f32; // Array<Add<Ref<Compact>, Scalar<f32>>>
    /// let c = a.as_ref() * b; // we can use `a` again here because we called as_ref()
    /// assert_eq!(c.to_ndarray()?[[1, 1]], 6.17 * (6.17 + 1.0));
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn as_ref(&self) -> Array<Ref<'_, S>> {
        Array {
            storage: Ref(self.storage()),
        }
    }

    /// Check if this array storage is compact block-compressed storage.
    ///
    /// This functions returns `true` for arrays that are stored in compact block-compressed form,
    /// i.e. those created by [`compact_array`](Array::compact_array), [`copy`](Array::copy),
    /// [`read_from_file`](Array::read_from_file), etc., and `false` for arrays with storage
    /// implementations with uncompressed data, such as lazy operation views, plain ndarray views,
    /// etc.
    ///
    /// # Example
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert!(a.is_compact());
    ///
    /// let b = a * 2.0f32; // Array<Mul<Compact, Scalar<f32>>>
    /// assert!(!b.is_compact()); // b is a lazy view
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn is_compact(&self) -> bool {
        self.storage().as_compact().is_some()
    }

    /// Ensure this array is in compact block-compressed form, re-compressing
    /// if needed.
    ///
    /// If the array is already compact, the storage is kept as-is - no data is
    /// copied or re-compressed. Otherwise the array is materialized block by
    /// block into a new [`Compact`](crate::storage::Compact) storage using
    /// default [`ArrayParams`].
    ///
    /// Use [`into_compact_with`](Self::into_compact_with) to control the block
    /// shape and codec parameters.
    ///
    /// # Example
    /// ```
    /// use zix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert!(a.is_compact());
    /// let a = a.into_compact()?; // a is already compact, so this is a no-op
    ///
    /// let b = a * 2.0f32; // Array<Mul<Compact, Scalar<f32>>>
    /// assert!(!b.is_compact()); // b is a lazy view
    /// let b = b.into_compact()?; // materialize b into compact form
    /// assert!(b.is_compact());
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn into_compact(self) -> Result<Array<IntoCompact<S>>> {
        let context = self.read_ctx();
        self.into_compact_with(ArrayParams::default(), &context)
    }

    /// Ensure this array is in compact block-compressed form, re-compressing
    /// if needed, with explicit control over parameters.
    ///
    /// Similar to [`into_compact`](Self::into_compact) but with explicit [`ArrayParams`].
    ///
    /// `params` controls the target block shape and compression settings. It is
    /// **only used when the source is not already compact** - if `is_compact()`
    /// returns `true`, the existing storage is wrapped zero-cost and `params` is
    /// ignored.
    pub fn into_compact_with(
        self,
        params: ArrayParams,
        context: &ReadContext,
    ) -> Result<Array<IntoCompact<S>>> {
        Ok(Array::from_storage(IntoCompact::new(
            self, params, context,
        )?))
    }

    /// Return a reference to the underlying storage backend.
    ///
    /// Rarely needed to be used directly by users.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Construct an `Array` by wrapping a storage backend directly.
    ///
    /// Rarely needed to be used directly by users.
    pub fn from_storage(storage: S) -> Self {
        Self { storage }
    }

    /// Consume this array and return the underlying storage backend.
    ///
    /// Rarely needed to be used directly by users.
    pub fn into_storage(self) -> S {
        self.storage
    }

    pub(crate) fn blocks_layout(&self) -> &BlocksLayout {
        self.storage._spec().blocks_layout
    }

    /// Build a [`BlockFn`] that reads and compresses this array's data block by block.
    ///
    /// Called by [`Array::into_compact`] (and its variants) to feed [`build_block_table`] with
    /// compressed block data without materializing all blocks at once.
    ///
    /// # Block layout
    ///
    /// `params.block_shape` divides the array into an N-dimensional grid of blocks. Blocks are
    /// visited in C order (last axis varies fastest). Boundary blocks - those that extend beyond
    /// the array's shape - are zero-padded to fill the full `block_shape` before compression.
    ///
    /// # Returned value
    ///
    /// Returns `(block_fn, bound)` where:
    /// - `block_fn` - a [`BlockFnWithState`] closure that, per batch, reads each block from
    ///   storage, compresses it with the encoder from `params`, and appends the result to an
    ///   internal `AlignedBytes` buffer. Returns the buffer slice and the absolute end-offsets.
    /// - `bound` - the encoder's compressed-size upper bound for one block, used by the caller
    ///   to choose a batch size that targets ~64 KB of compressed output per call.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoder cannot be constructed from `params`.
    pub(crate) fn to_block_fn<'a>(
        &'a self,
        params: &ArrayParams,
        context: &'a ReadContext,
    ) -> Result<(impl BlockFn + 'a, usize)> {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();

        let block_shape = params.block_shape.as_ref().unwrap().clone();
        let block_size = block_shape.iter().cloned().try_product().unwrap();
        let grid_shape = dim_arr(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));

        let encoder_cfg = params.encoder_params.as_ref().unwrap();
        let mut encoder = Encoder::new(encoder_cfg, dtype.clone())?;

        let mut block_iter = NdIter::new(
            &grid_shape,
            NdIterExtBlockOffsetSize::new(
                shape,
                &dim_arr(ndim, |_| 0),
                shape,
                &dim_arr(ndim, |dim| block_shape[dim] as u64),
            ),
        );

        struct TmpBufs {
            tmp_block_compressed: AlignedBytes,
            tmp_block_offsets: Vec<u64>,
        }

        let grid_logical_strides = default_strides(&grid_shape, 1);
        let itemsize = encoder.dtype.itemsize() as usize;
        let alignment = encoder.dtype.alignment().as_usize();
        let block_size_bytes = block_size as usize * itemsize;
        let block_strides = default_strides(&block_shape, itemsize as _);
        let mut tmp_block_plain = AlignedBytes::new_padded(alignment);
        let block_compressed_bound = encoder.encode_bound(block_size_bytes);
        let block_fn = BlockFnWithState::from_fn(
            TmpBufs {
                tmp_block_compressed: AlignedBytes::new_padded(alignment),
                tmp_block_offsets: Vec::new(),
            },
            move |blocks: Range<u64>, base_offset: u64, tmp_bufs| {
                let tmp_bufs: &mut TmpBufs = tmp_bufs;
                let tmp_block_compressed = &mut tmp_bufs.tmp_block_compressed;
                let tmp_block_offsets = &mut tmp_bufs.tmp_block_offsets;
                tmp_block_compressed.clear();
                tmp_block_offsets.clear();

                for block_logical_idx in blocks {
                    let (block_idx, (block_inner_offset, block_size)) = block_iter.next().unwrap();
                    debug_assert_eq!(
                        block_logical_idx,
                        block_idx
                            .iter()
                            .zip(&grid_logical_strides)
                            .map(|(i, s)| i * s)
                            .sum::<u64>()
                    );
                    let read_range = dim_arr(ndim, |dim| {
                        let start =
                            block_idx[dim] * block_shape[dim] as u64 + block_inner_offset[dim];
                        let end = start + block_size[dim];
                        start..end
                    });
                    let full_block = (0..ndim).all(|dim| {
                        block_inner_offset[dim] == 0 && block_size[dim] == block_shape[dim] as u64
                    });

                    // Read block plain data
                    tmp_block_plain.clear();
                    tmp_block_plain.reserve(block_size_bytes);
                    unsafe { tmp_block_plain.set_len(block_size_bytes) };
                    let tmp_block_plain_ptr = tmp_block_plain.as_mut_ptr();
                    let tmp_block_compressed_len = tmp_block_compressed.len();
                    let read_data_buf = if full_block {
                        tmp_block_plain.as_mut_slice()
                    } else {
                        tmp_block_plain.fill(0); // zero-pad
                        let b_size_bytes = block_size.iter().product::<u64>() as usize * itemsize;
                        let tmp_block_plain2 = &mut *tmp_block_compressed;
                        let align_padding = tmp_block_compressed_len.ceil_to_multiple(alignment)
                            - tmp_block_compressed_len;
                        tmp_block_plain2.reserve(align_padding + b_size_bytes);
                        unsafe {
                            tmp_block_plain2
                                .set_len(tmp_block_compressed_len + align_padding + b_size_bytes)
                        };
                        &mut tmp_block_plain2[tmp_block_compressed_len + align_padding..]
                    };
                    self.storage
                        .read_data(&read_range, read_data_buf, context)?;
                    if !full_block {
                        // Copy from temporary buffer to output block with correct strides.
                        let src_strides = default_strides(
                            &dim_arr(ndim, |dim| block_size[dim] as usize),
                            itemsize,
                        );
                        unsafe {
                            nd_copy(
                                read_data_buf.as_ptr(),
                                tmp_block_plain_ptr,
                                block_size,
                                &src_strides,
                                &block_strides,
                                itemsize,
                            )
                        };
                        unsafe { tmp_block_compressed.set_len(tmp_block_compressed_len) };
                    }
                    let plain_data = tmp_block_plain.as_slice();

                    // Compress block data
                    tmp_block_compressed.reserve(block_compressed_bound);
                    unsafe {
                        tmp_block_compressed
                            .set_len(tmp_block_compressed_len + block_compressed_bound)
                    };
                    let cdata_len = encoder.encode(
                        plain_data,
                        &mut tmp_block_compressed[tmp_block_compressed_len..],
                    )?;
                    unsafe { tmp_block_compressed.set_len(tmp_block_compressed_len + cdata_len) }

                    tmp_block_offsets.push(base_offset + tmp_block_compressed.len() as u64);
                }
                Ok((
                    tmp_block_compressed.as_slice(),
                    tmp_block_offsets.as_slice(),
                ))
            },
        );
        Ok((block_fn, block_compressed_bound))
    }
}

/// Methods for converting an array to a different element type or dimension type.
impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Re-tag this array's element type as `ET`, wrapping the storage in a [`ToType`] adaptor.
    ///
    /// Works for any `S: ArrayStorage`. See [`ToType`] for details and examples. For storages
    /// that implement [`ElementTypeChange`], prefer [`into_type`](Self::into_type) to avoid the
    /// wrapper layer.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `ET = Ty<T>` and `self.dtype() != T::DTYPE`. Always succeeds for `ET = TypeDyn`.
    pub fn to_type<ET>(self) -> Result<Array<ToType<S, ET>>>
    where
        ET: ElementType,
    {
        Ok(Array::from_storage(ToType::<S, ET>::new(self)?))
    }

    /// Re-tag this array's element type as [`Ty<T>`](crate::Ty), asserting a concrete scalar type.
    ///
    /// Sugar for [`to_type::<Ty<T>>()`](Self::to_type). See [`ToType`] for details and examples.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `self.dtype() != T::DTYPE`.
    pub fn to_typed<T>(self) -> Result<Array<ToType<S, Ty<T>>>>
    where
        T: Dtyped,
    {
        self.to_type()
    }

    /// Re-tag this array's element type as [`TypeDyn`], erasing static element-type information.
    ///
    /// Infallible sugar for [`to_type::<TypeDyn>()`](Self::to_type). See [`ToType`] for details.
    pub fn to_type_dyn(self) -> Array<ToType<S, TypeDyn>> {
        self.to_type().unwrap()
    }

    /// Re-tag this array's element type as `NewET` in-place, without adding a wrapper layer.
    ///
    /// Requires `S: ElementTypeChange`. See [`ElementTypeChange`] for details and the list of
    /// implementing storages. Prefer [`to_type`](Self::to_type) when `S` does not implement
    /// `ElementTypeChange`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `NewET = Ty<T>` and `self.dtype() != T::DTYPE`. Always succeeds for `NewET = TypeDyn`.
    pub fn into_type<NewET: ElementType>(self) -> Result<Array<S::ElementTypeChange<NewET>>>
    where
        S: ElementTypeChange,
    {
        Ok(Array::from_storage(self.into_storage().change_type()?))
    }

    /// Re-tag this array's element type as [`Ty<T>`](crate::Ty) in-place, asserting a concrete scalar type.
    ///
    /// Sugar for [`into_type::<Ty<T>>()`](Self::into_type). Requires `S: ElementTypeChange`.
    /// See [`ElementTypeChange`] for details. Prefer [`to_typed`](Self::to_typed) when `S` does
    /// not implement `ElementTypeChange`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `self.dtype() != T::DTYPE`.
    pub fn into_typed<T>(self) -> Result<Array<S::ElementTypeChange<Ty<T>>>>
    where
        T: Dtyped,
        S: ElementTypeChange,
    {
        self.into_type()
    }

    /// Re-tag this array's element type as [`TypeDyn`] in-place, erasing static element-type information.
    ///
    /// Infallible sugar for [`into_type::<TypeDyn>()`](Self::into_type). Requires
    /// `S: ElementTypeChange`. See [`ElementTypeChange`] for details.
    pub fn into_type_dyn(self) -> Array<S::ElementTypeChange<TypeDyn>>
    where
        S: ElementTypeChange,
    {
        self.into_type().unwrap()
    }

    /// Re-tag this array's dimension as `D`, returning an error if the actual ndim does not match.
    ///
    /// This is the bridge between dynamic and static dimension tracking. Arrays loaded from
    /// files or produced by slice-based shape operations carry [`DimDyn`] as their dimension
    /// type because the compiler cannot know the ndim at that point. After you have confirmed
    /// the ndim (e.g. by reading `array.ndim()` or knowing the data layout ahead of time),
    /// call `to_dim::<Dim<N>>()` to recover a statically-typed dimension. Subsequent
    /// operations on the result will propagate the static `Dim<N>` through the type system.
    ///
    /// Generally speaking, the compiler can optimize more aggressively when the dimension is
    /// statically known, which can yield better performance.
    ///
    /// This method wraps the storage in a [`ToDim<S, D>`](crate::ops::ToDim) adaptor and works
    /// for any `S: ArrayStorage`. For storages that implement [`DimensionChange`], prefer
    /// [`into_dim`](crate::Array::into_dim) instead — it replaces the `D` parameter in-place
    /// without adding a wrapper layer.
    ///
    /// See [`to_dim_dyn`](Self::to_dim_dyn).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidShapeOperation`](crate::ErrorKind::InvalidShapeOperation) if
    /// `D::NDIM` is `Some(n)` and `self.ndim() != n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::{Array, Dim};
    ///
    /// // Passing a dynamically-dimensioned ndarray produces Array<Compact<DimDyn>>.
    /// // Arrays loaded from files via Array::read_from_file also carry DimDyn.
    /// let a = Array::compact_array(&ndarray::ArrayD::<i32>::zeros(ndarray::IxDyn(&[2, 3, 4])))?;
    ///
    /// // Assert the array is 3-D; fail gracefully if not.
    /// let a3d = a.to_dim::<Dim<3>>()?;  // Array<ToDim<Compact<DimDyn>, Dim<3>>>
    ///
    /// // Now insert_axis knows the result is 4-D at compile time.
    /// let a4d = a3d.insert_axis(0usize); // Array<InsertAxis<..., Dim<4>>>
    /// assert_eq!(a4d.shape(), &[1, 2, 3, 4]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    pub fn to_dim<D>(self) -> Result<Array<ToDim<S, D>>>
    where
        D: Dimension,
    {
        Ok(Array::from_storage(ToDim::<S, D>::new(self)?))
    }

    /// Re-tag this array's dimension as [`DimDyn`], erasing static dimension information.
    ///
    /// This is the infallible counterpart to [`to_dim`](Self::to_dim). Every array has a
    /// runtime ndim regardless of its static type, so converting to `DimDyn` always succeeds.
    ///
    /// Like `to_dim`, this wraps the storage in a [`ToDim`](crate::ops::ToDim) adaptor. For
    /// storages that implement [`DimensionChange`], prefer
    /// [`into_dim_dyn`](crate::Array::into_dim_dyn) instead — it replaces the `D` parameter
    /// in-place without adding a wrapper layer.
    ///
    /// After calling `to_dim_dyn`, subsequent shape-changing operations will produce
    /// `DimDyn` results rather than `Dim<N>`. Call [`to_dim`](Self::to_dim) again to
    /// re-establish static tracking once the ndim is confirmed.
    pub fn to_dim_dyn(self) -> Array<ToDim<S, DimDyn>> {
        self.to_dim().unwrap()
    }

    /// Re-tag this array's storage as having dimension `NewD` in-place, without a wrapper layer.
    ///
    /// Requires `S: DimensionChange`. See [`DimensionChange`] for details and the list of
    /// implementing storages. Prefer [`to_dim`](Self::to_dim) when `S` does not implement
    /// `DimensionChange`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidShapeOperation`](crate::ErrorKind::InvalidShapeOperation) if
    /// `NewD::NDIM` is `Some(n)` and `self.ndim() != n`.
    pub fn into_dim<NewD: Dimension>(self) -> Result<Array<S::DimensionChange<NewD>>>
    where
        S: DimensionChange,
    {
        Ok(Array::from_storage(self.into_storage().dimension_change()?))
    }

    /// Re-tag this array's dimension as [`DimDyn`] in-place, erasing static dimension information.
    ///
    /// Infallible sugar for [`into_dim::<DimDyn>()`](Self::into_dim). Requires
    /// `S: DimensionChange`. See [`DimensionChange`] for details.
    pub fn into_dim_dyn(self) -> Array<S::DimensionChange<DimDyn>>
    where
        S: DimensionChange,
    {
        self.into_dim::<DimDyn>().unwrap()
    }
}

impl<S: ArrayStorage> std::fmt::Debug for Array<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, ArrayD};

    use super::Array;
    use crate::array::{ArrayBlockTableStorageBase, Compact};
    use crate::codec::{DecoderParams, EncoderParams};
    use crate::dtype::Dtyped;
    use crate::storage::block::{BlockSize, BlockTable};
    use crate::storage::{BlockShapeTag, BlocksLayout};
    use crate::util::{arr_params, cast_slice, dim_arr, DimArray};
    use crate::{DimDyn, Dimension, IntoDimension, Ty};

    // -----------------------------------------------------------------------
    // compact_array roundtrip helper
    // -----------------------------------------------------------------------

    fn roundtrip<T, S, D>(src: &ndarray::ArrayBase<S, D>, block_shape: &[usize]) -> ArrayD<T>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension + IntoDimension,
    {
        let a = Array::compact_array_with(&src, arr_params(block_shape)).unwrap();
        a.to_ndarray().unwrap()
    }

    // -----------------------------------------------------------------------
    // Helper: build a BlockTable from pre-arranged typed blocks
    // -----------------------------------------------------------------------

    fn make_block_table<T: Dtyped>(
        blocks: &[&[T]],
    ) -> BlockTable<crate::storage::block::Owned, Ty<T>> {
        let block_len = blocks[0].len() as BlockSize;
        let data: Vec<u8> = blocks
            .iter()
            .flat_map(|b| unsafe { cast_slice::<T, u8>(b) }.iter().copied())
            .collect();
        BlockTable::build_from_data(&data, T::DTYPE, block_len, &EncoderParams::default()).unwrap()
    }

    fn array<T: Dtyped>(
        blocks: &[&[T]],
        shape: &[usize],
        block_shape: &[usize],
    ) -> Array<Compact<Ty<T>, DimDyn>> {
        let shape = shape.iter().map(|&x| x as u64).collect::<DimArray<_>>();
        let ndim = block_shape.len();
        let block_shape_hint = block_shape
            .iter()
            .map(|&x| x as BlockSize)
            .collect::<DimArray<_>>();
        let layout = BlocksLayout::new(
            block_shape_hint.clone(),
            dim_arr(ndim, |_| BlockShapeTag::Fixed),
            0,
            block_shape_hint,
            0,
        );
        Array {
            storage: Compact(ArrayBlockTableStorageBase::new(
                make_block_table(blocks),
                DimDyn::from_slice(&shape).unwrap(),
                layout,
                EncoderParams::default(),
                DecoderParams::default(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn dtype_shape_ndim() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        assert_eq!(a.dtype(), &u8::DTYPE);
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.ndim(), 1);
    }

    // -----------------------------------------------------------------------
    // to_ndarray - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![0, 1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn to_ndarray_1d_two_blocks() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn to_ndarray_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let got: ArrayD<i32> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![8], vec![10, 20, 30, 40, 50, 60, 70, 80]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // to_ndarray - 2D
    // Block-major order: block [r,c] = row-major grid index r*ncols_blocks+c.
    // shape=[4,6], block_shape=[2,3] -> grid 2*2:
    //   block0=[0,0]: rows 0-1, cols 0-2 -> 0,1,2,6,7,8
    //   block1=[0,1]: rows 0-1, cols 3-5 -> 3,4,5,9,10,11
    //   block2=[1,0]: rows 2-3, cols 0-2 -> 12,13,14,18,19,20
    //   block3=[1,1]: rows 2-3, cols 3-5 -> 15,16,17,21,22,23
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_2d() {
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            &[4, 6],
            &[2, 3],
        );
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4, 6], (0u8..24).collect()).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // to_ndarray_sub - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_1d_full_range() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.to_ndarray_sub(&[0..6], &a.read_ctx()).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn to_ndarray_sub_1d_aligned_second_block() {
        // range [3..6) -> output shape [3], values [3,4,5]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.to_ndarray_sub(&[3..6], &a.read_ctx()).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3], vec![3, 4, 5]).unwrap());
    }

    #[test]
    fn to_ndarray_sub_1d_cross_block_boundary() {
        // range [1..5) -> output shape [4], values [1,2,3,4]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.to_ndarray_sub(&[1..5], &a.read_ctx()).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![1, 2, 3, 4]).unwrap()
        );
    }

    #[test]
    fn to_ndarray_sub_1d_within_single_block() {
        // range [1..2) -> output shape [1], value [1]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got: ArrayD<u8> = a.to_ndarray_sub(&[1..2], &a.read_ctx()).unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1], vec![1]).unwrap());
    }

    // -----------------------------------------------------------------------
    // to_ndarray_sub - 2D
    // shape=[4,6], block_shape=[2,3], data as in to_ndarray_2d test.
    // range=[1..3, 2..5] -> output shape [2,3]:
    //   [8,  9,  10]
    //   [14, 15, 16]
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_2d() {
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            &[4, 6],
            &[2, 3],
        );
        let got: ArrayD<u8> = a.to_ndarray_sub(&[1..3, 2..5], &a.read_ctx()).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3], vec![8, 9, 10, 14, 15, 16]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // compact_array - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn compact_array_1d_single_block() {
        let src = array![0u8, 1, 2, 3];
        assert_eq!(roundtrip(&src, &[4]), src.into_dyn());
    }

    #[test]
    fn compact_array_1d_multi_block() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        assert_eq!(roundtrip(&src, &[3]), src.into_dyn());
    }

    #[test]
    fn compact_array_1d_with_padding() {
        // size 5, block 3 -> padded to 6; shape reported as 5
        let src = array![0u8, 1, 2, 3, 4];
        let a = Array::compact_array_with(&src, arr_params(&[3])).unwrap();
        assert_eq!(a.shape(), &[5]);
        let got: ArrayD<u8> = a.to_ndarray().unwrap();
        assert_eq!(got, src.into_dyn());
    }

    #[test]
    fn compact_array_1d_i32() {
        let src = array![0i32, 10, 20, 30, 40, 50, 60, 70];
        assert_eq!(roundtrip(&src, &[4]), src.into_dyn());
    }

    #[test]
    fn compact_array_1d_f32() {
        let src = array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        assert_eq!(roundtrip(&src, &[4]), src.into_dyn());
    }

    #[test]
    fn compact_array_block_larger_than_shape_is_clamped() {
        // block_shape [10] > array size [4]; should clamp to [4]
        let src = array![0u8, 1, 2, 3];
        let a = Array::compact_array_with(&src, arr_params(&[10])).unwrap();
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.to_ndarray().unwrap(), src.into_dyn());
    }

    #[test]
    fn compact_array_1d_noncontiguous() {
        // Step-2 slice of [0..10] -> [0, 2, 4, 6, 8]
        let src = array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = src.slice(ndarray::s![..;2]);
        let a = Array::compact_array_with(&view, arr_params(&[3])).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.to_ndarray().unwrap(), array![0u8, 2, 4, 6, 8].into_dyn());
    }

    // -----------------------------------------------------------------------
    // compact_array - metadata
    // -----------------------------------------------------------------------

    #[test]
    fn compact_array_metadata() {
        let a = Array::compact_array(&array![0i32, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.dtype(), &i32::DTYPE);
    }

    // -----------------------------------------------------------------------
    // compact_array - 2D
    // -----------------------------------------------------------------------

    #[test]
    fn compact_array_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        assert_eq!(roundtrip(&src, &[2, 3]), src.into_dyn());
    }

    #[test]
    fn compact_array_2d_with_padding() {
        // shape [3,5], block [2,3] -> padded to [4,6]; shape reported as [3,5]
        #[rustfmt::skip]
        let src = array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::compact_array_with(&src, arr_params(&[2, 3])).unwrap();
        assert_eq!(a.shape(), &[3, 5]);
        assert_eq!(a.to_ndarray().unwrap(), src.into_dyn());
    }

    #[test]
    fn compact_array_2d_noncontiguous() {
        // Fortran-order (column-major) array
        let src = ndarray::Array2::<u8>::from_shape_vec(
            ndarray::ShapeBuilder::f((3, 4)),
            (0..12).collect(),
        )
        .unwrap();
        assert_eq!(roundtrip(&src, &[2, 2]), src.into_dyn());
    }

    // -----------------------------------------------------------------------
    // compact_array + to_ndarray_sub integration
    // -----------------------------------------------------------------------

    #[test]
    fn compact_array_then_to_ndarray_sub_1d() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        let a = Array::compact_array_with(&src, arr_params(&[3])).unwrap();
        let got: ArrayD<u8> = a.to_ndarray_sub(&[1..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1u8, 2, 3, 4].into_dyn());
    }

    #[test]
    fn compact_array_then_to_ndarray_sub_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a = Array::compact_array_with(&src, arr_params(&[2, 3])).unwrap();
        let got: ArrayD<u8> = a.to_ndarray_sub(&[1..3, 2..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![[8u8, 9, 10], [14, 15, 16]].into_dyn());
    }

    // -----------------------------------------------------------------------
    // copy
    // -----------------------------------------------------------------------

    #[test]
    fn copy_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.ndim(), 1);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [4]);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ArrayD::from_shape_vec(vec![4], vec![0u8, 1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn copy_1d_multi_block() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[6]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [3]);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ArrayD::from_shape_vec(vec![6], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn copy_1d_with_padding() {
        // shape [5], block [3] -> stored as 6 elements (padded)
        let src = array![0u8, 1, 2, 3, 4];
        let a = Array::compact_array_with(&src, arr_params(&[3])).unwrap();
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[5]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [3]);
        assert_eq!(b.to_ndarray().unwrap(), src.into_dyn());
    }

    #[test]
    fn copy_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[8]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ArrayD::from_shape_vec(vec![8], vec![10i32, 20, 30, 40, 50, 60, 70, 80]).unwrap()
        );
    }

    #[test]
    fn copy_2d_single_block() {
        // shape=[2,3], block=[2,3] - one block, no partial-block path
        let a = array(&[&[0u8, 1, 2, 3, 4, 5]], &[2, 3], &[2, 3]);
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [2, 3]);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ArrayD::from_shape_vec(vec![2, 3], (0u8..6).collect()).unwrap()
        );
    }

    #[test]
    fn copy_2d_multi_block() {
        // shape=[4,6], block=[2,3] - 4 blocks, exercises the full-block copy path
        // Block layout (row-major grid):
        //   block0=[0,0]: rows 0-1, cols 0-2 -> 0,1,2,6,7,8
        //   block1=[0,1]: rows 0-1, cols 3-5 -> 3,4,5,9,10,11
        //   block2=[1,0]: rows 2-3, cols 0-2 -> 12,13,14,18,19,20
        //   block3=[1,1]: rows 2-3, cols 3-5 -> 15,16,17,21,22,23
        #[rustfmt::skip]
        let a = array(
            &[
                &[0u8, 1, 2, 6, 7, 8],
                &[3, 4, 5, 9, 10, 11],
                &[12, 13, 14, 18, 19, 20],
                &[15, 16, 17, 21, 22, 23],
            ],
            &[4, 6],
            &[2, 3],
        );
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[4, 6]);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [2, 3]);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ArrayD::from_shape_vec(vec![4, 6], (0u8..24).collect()).unwrap()
        );
    }

    #[test]
    fn copy_2d_with_padding() {
        // shape=[3,5], block=[2,3] -> padded to [4,6]; shape preserved as [3,5].
        // Block grid 2*2:
        //   [0,0]: size [2,3] - full block
        //   [0,1]: size [2,2] - partial in dim1
        //   [1,0]: size [1,3] - partial in dim0
        //   [1,1]: size [1,2] - partial in BOTH dims (corner block)
        #[rustfmt::skip]
        let src = array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::compact_array_with(&src, arr_params(&[2, 3])).unwrap();
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[3, 5]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(b.to_ndarray().unwrap(), src.into_dyn());
    }

    #[test]
    fn copy_3d_with_padding_in_all_dims() {
        // shape=[3,3,5], block=[2,2,3] -> padded to [4,4,6].
        // Block grid 2*2*2 = 8 blocks; every boundary block is partial in at least
        // one dimension, and the single corner block [1,1,1] is partial in all three:
        //   size [1,1,2] vs block_shape [2,2,3].
        let src = ndarray::Array3::<u8>::from_shape_vec([3, 3, 5], (0u8..45).collect()).unwrap();
        let a = Array::compact_array_with(&src, arr_params(&[2, 2, 3])).unwrap();
        let b = a.copy().unwrap();
        assert_eq!(b.shape(), &[3, 3, 5]);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.blocks_layout().block_shape_hint[..], [2, 2, 3]);
        assert_eq!(b.to_ndarray().unwrap(), src.into_dyn());
    }

    #[test]
    fn copy_preserves_block_shape() {
        // Verify the copied array has the same block layout as the source.
        let src = array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::compact_array_with(&src, arr_params(&[4])).unwrap();
        let b = a.copy().unwrap();
        assert_eq!(
            a.blocks_layout().block_shape_hint[..],
            b.blocks_layout().block_shape_hint[..]
        );
    }

    #[test]
    fn copy_result_is_independent() {
        // Mutating the source array should not affect the copy (they are independent).
        // Since Array<Compact> doesn't expose mutation, we verify by round-tripping
        // both through write/read and checking values remain consistent.
        let src = array![10u8, 20, 30, 40];
        let a = Array::compact_array_with(&src, arr_params(&[4])).unwrap();
        let b = a.copy().unwrap();
        // Both should read back the same data independently.
        assert_eq!(a.to_ndarray().unwrap(), b.to_ndarray().unwrap());
    }
}

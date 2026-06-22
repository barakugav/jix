use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

use crate::codec::{DecoderCodecConfig, Encoder, ReadContext};
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_buffer_size, check_get_range, Result};
use crate::ops::MaybeCompact;
use crate::storage::block::{BlockFn, BlockFnWithState, BlockTableBuilder};
use crate::storage::params::ArraySpecOwned;
use crate::storage::{
    ArrayBlockTableStorageBase, ArrayStorageAny, ArrayStorageTyped, Compact, Ref,
};
use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::NdIterExtStridesPtrMut;
use crate::util::iter::NdIter;
use crate::util::{
    assert_unchecked_eq, cast_slice_mut, default_strides, dim_arr, nd_copy, AlignedBytes, DimArray,
    IterExt,
};
use crate::{
    ArrayAny, ArrayParams, ArrayStorage, DimDyn, Dimension, ElementType, IntoDimension, Ty, TypeDyn,
};

/// A multi-dimensional array, usually compressed, backed by a generic storage.
///
/// `Array<S>` is the central type in jix. It behave like a regular n-dimensional array, but
/// its data is stored in a compressed format and decoded on demand. Its core functionality is
/// provided by [`shape()`](Array::shape), [`dtype()`](Array::dtype),
/// and [`to_ndarray_buf()`](Array::to_ndarray_buf), all other functions are built on top of those.
///
/// An array is generic over `S: ArrayStorage`, which provides the implementation of the three core
/// methods. The main concrete storage backend is the block-compressed [`Compact`] type, which
/// divides the array into n-dimensional blocks and compresses each block independently, and its
/// the return type of the common creation methods for arrays
/// (e.g.[`compact_ndarray()`](Array::compact_ndarray) and [`compact()`](Array::compact)).
///
/// # Storage variants
///
/// The primary concrete storages are:
///
/// | Type | Description |
/// |------|-------------|
/// | [`Array<Compact>`](crate::storage::Compact) | Heap-allocated block-compressed array. The main storage backend. |
/// | [`Array<Add<S1, S2>> or Array<Neg<S>> ...`](crate::ops) | Lazy operations views that wrap one or more arrays and apply a transformation at read time. Created by methods in [`ops`](crate::ops). |
/// | [`Array<Plain<...>>`](crate::storage::Plain) | Zero-copy view into an uncompressed (possibly strided) in-memory buffer. Created by [`plain_ndarray`](Array::plain_ndarray) and [`plain_ndarray_ref`](Array::plain_ndarray_ref). |
///
/// # Operations and lazy evaluation
///
/// Every operation on an `Array<S>` returns a new `Array` whose type encodes the full operation
/// chain:
///
/// ```text
/// Array<Compact>
///   .neg()                 -> Array<Neg<Compact>>
///   .reshape(...)          -> Array<Reshape<Neg<Compact>>>
///   .permute_axes(axes)    -> Array<PermuteAxes<Reshape<...>>>
///   .add(other_array)      -> Array<Add<PermuteAxes<...>, Compact>>
///   .sum(axis)             -> Array<Sum<Add<...>>>
///   .compact();            -> Array<Compact> - materialize the pipeline
/// ```
///
/// Data is never copied or computed at construction time. An operation only runs when the result
/// is materialized via [`to_ndarray()`](Array::to_ndarray), [`compact()`](Array::compact), and their variants.
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
/// use jix::{Array, ArrayParams};
/// use ndarray::array;
///
/// // Compress an ndarray into a block-compressed Array<Compact>.
/// let compact = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
///
/// // Zero-copy view of an existing ndarray (any layout).
/// let plain = Array::plain_ndarray_ref(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
///
/// // Read a previously serialized array back from a file.
/// let tmp_dir = tempfile::tempdir()?;
/// let path = tmp_dir.path().join("array.jix");
/// compact.write_to_file(&path)?;
/// let from_file = Array::read_from_file(&path, ArrayParams::default())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Apply operations on compressed arrays, creating lazy views, writing the result to a file:
/// ```
/// use jix::Array;
/// use jix::dtype::Dtyped;
/// use ndarray::array;
///
/// // Compress a 2-D f32 ndarray.
/// let array = Array::compact_ndarray(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
/// assert_eq!(array.shape(), &[2, 3]);
/// assert_eq!(array.dtype(), &f32::DTYPE);
///
/// // Decompress and compare.
/// let decompressed = array.to_ndarray()?;
/// assert_eq!(decompressed[[0, 0]], 1.5);
/// assert_eq!(decompressed[[1, 1]], 6.17);
///
/// // Apply operations on a compressed array, creating lazy views
/// let ones = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
/// let scaled = array                               // Array<Compact>
///     .exp()                                       // Array<Exp<Compact>>
///     .floor()                                     // Array<Floor<Exp<Compact>>>
///     .map(|x| x * 2.0f32)                         // Array<Map<Floor<...>>>
///     + ones;                                      // Array<Add<Map<...>, Compact>>
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
///     // materialize the pipeline bt recompressing
///     .compact()?;                                 // Array<Compact>
/// assert_eq!(result.shape(), &[2]);
/// assert_eq!(result.dtype(), &i16::DTYPE);
/// let tmp_dir = tempfile::tempdir()?;
/// result.write_to_file(tmp_dir.path().join("result.jix").as_ref())?;
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
/// Additional arrays that are created from existing arrays (`.compact()`, `.reshape()`, result
/// of operations, etc.) choose their block shape with a heuristic, trying to preserve the original
/// user block shape as much as possible while respecting the new shape and layout.
///
/// Shape-changing operations - [`reshape`](Array::reshape),
/// [`broadcast`](Array::broadcast), [`permute_axes`](Array::permute_axes) - remap how
/// output indices translate to positions in the underlying blocks. When the new layout crosses
/// block boundaries that the original respected, a single read may decompress many more blocks
/// than necessary. To avoid this, materialize with [`compact`](Array::compact) (automatic block shape)
/// or [`compact_with`](Array::compact_with) (explicit [`ArrayParams`]) after a shape change.
/// To ensure a well-aligned block layout, pass explicit `ArrayParams` with a block
/// shape that matches the expected access pattern.
///
/// # Element type tracking
///
/// `S::ElementType` records the scalar element type at the type level. When the element type is
/// statically known, `S::ElementType = Ty<T>`, and all element-wise operations - arithmetic,
/// comparisons, reductions, type casts - become available. When the element type is only known
/// at runtime (e.g. for arrays loaded from files), `S::ElementType = TypeDyn`, and those
/// operations are not available until the type is asserted.
///
/// Arrays constructed from typed sources automatically carry `Ty<T>`: `compact_ndarray(&array![1.0f32])`
/// returns `Array<Compact<Ty<f32>, Dim<1>>>`. Arrays loaded from disk carry `TypeDyn`. Use
/// [`into_typed::<T>()`](Array::into_typed) to assert the expected element type - validated
/// against the stored dtype at runtime - and recover `Ty<T>`. Use
/// [`into_type_dyn()`](Array::into_type_dyn) to erase the static element type.
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
/// [`into_dim::<Dim<N>>()`](Array::into_dim) to assert a specific ndim and recover static
/// tracking, or [`into_dim_dyn()`](Array::into_dim_dyn) to erase static dimension info.
#[derive(Clone)]
pub struct Array<S> {
    pub(crate) storage: S,
}

impl<T, D> Array<Compact<Ty<T>, D>> {
    /// Compress an ndarray into a block-compressed `Array<Compact<D>>` with default encoding settings.
    ///
    /// The array is partitioned into n-dimensional blocks, each independently compressed. The
    /// block shape is derived automatically to fit within the L1 data cache. Use
    /// [`compact_ndarray_with`](Array::compact_ndarray_with) for explicit control over block shape,
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
    /// use jix::dtype::Dtyped;
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// // Compress a 2-D f32 ndarray.
    /// let array = Array::compact_ndarray(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
    /// assert_eq!(array.shape(), &[2, 3]);
    /// assert_eq!(array.dtype(), &f32::DTYPE);
    ///
    /// // Decompress and compare.
    /// let decompressed = array.to_ndarray()?;
    /// assert_eq!(decompressed[[0, 0]], 1.5);
    /// assert_eq!(decompressed[[1, 1]], 6.17);
    ///
    /// // Apply operations on a compressed array, creating lazy views
    /// let ones = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
    /// let scaled = array                               // Array<Compact>
    ///     .exp()                                       // Array<Exp<Compact>>
    ///     .floor()                                     // Array<Floor<Exp<Compact>>>
    ///     .map(|x| x * 2.0f32)                         // Array<Map<Floor<...>>>
    ///     + ones;                                      // Array<Add<Map<...>, Compact>>
    /// assert_eq!(scaled.shape(), &[2, 3]);
    /// assert_eq!(scaled.dtype(), &f32::DTYPE);
    /// assert_eq!(scaled.to_ndarray()?[[1, 1]], 957.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compact_ndarray<S, InD>(array: &ndarray::ArrayBase<S, InD>) -> Result<Self>
    where
        InD: ndarray::Dimension + IntoDimension<Dimension = D>,
        D: Dimension,
        S: ndarray::Data<Elem = T>,
        T: Dtyped,
    {
        Array::compact_ndarray_with(array, ArrayParams::default())
    }

    /// Compress an ndarray into a block-compressed `Array<Compact>` with explicit `ArrayParams`.
    ///
    /// See [`compact_ndarray`](Array::compact_ndarray) for the default-parameter version, which has more
    /// documentation and examples.
    ///
    /// Use this method to specify encoding parameters such as block shape, compression level, etc.
    /// See [`ArrayParams`] for details on the available parameters and their effects on performance.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    ///
    /// let data = ndarray::Array2::<f32>::zeros((512, 512));
    ///
    /// // Store with 64*64 blocks - good for tile-at-a-time access patterns.
    /// let mut params = ArrayParams::new();
    /// params.block_shape(&[64, 64]);
    /// let array = Array::compact_ndarray_with(&data, params)?;
    ///
    /// // Read tiles of 128*128 by decompressing 2*2 blocks at a time.
    /// let context = array.read_ctx();
    /// for tile_row in 0..7 {
    ///     for tile_col in 0..7 {
    ///         let row_range = (tile_row * 64)..((tile_row + 2) * 64);
    ///         let col_range = (tile_col * 64)..((tile_col + 2) * 64);
    ///         let tile = array.to_ndarray_sub(&[row_range, col_range], &context)?;
    ///         println!("tile ({tile_row},{tile_col}) sum: {}", tile.sum());
    ///     }
    /// }
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_ndarray_with<S, InD>(
        array: &ndarray::ArrayBase<S, InD>,
        mut params: ArrayParams,
    ) -> Result<Self>
    where
        InD: ndarray::Dimension + IntoDimension<Dimension = D>,
        D: Dimension,
        S: ndarray::Data<Elem = T>,
        T: Dtyped,
    {
        let array = Array::plain_ndarray_ref(array)?;
        params.tune(array.shape(), array.dtype())?;
        array.compact_with(params, &array.try_read_ctx()?)
    }
}

impl<D> Array<Compact<TypeDyn, D>> {
    /// Compress a raw n-dimensional buffer into a block-compressed `Array<Compact>`.
    ///
    /// Same as [`compact_ndarray_with`](Array::compact_ndarray_with) but takes a raw pointer and
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
        let array = unsafe { Array::plain_ndarray_ptr(ptr, shape, strides, dtype)? };
        params.tune(array.shape(), array.dtype())?;
        array.compact_with(params, &array.try_read_ctx()?)
    }
}

impl<T, D> Array<Compact<Ty<T>, D>> {
    /// Create a block-compressed array by evaluating a function `f` at each index.
    ///
    /// The function `f` is called with the index of each element in the output array, and should
    /// return the value for that element.
    /// Elements are visited in an arbitrary order - this is not some theoretical use case, most of
    /// the times elements will NOT be visited in row-major order, and the function should not rely
    /// on any specific order for correctness or performance.
    ///
    /// # Arguments
    ///
    /// - `shape`: shape of the output array.
    /// - `f`: function that produces the value of each element given its index.
    ///
    /// # Examples
    ///
    /// 1D - the index is a `u64`:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_fn(5, |i: u64| i * 2)?;
    /// assert_eq!(a.to_ndarray()?, array![0, 2, 4, 6, 8]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// 2D - the index is a `(u64, u64)` tuple:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_fn((3, 3), |(x, y)| x * 10 + y)?;
    /// assert_eq!(
    ///     a.to_ndarray()?,
    ///     array![[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    /// );
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// Dynamic rank - passing a slice produces a dynamic-dimensional array whose
    /// callback receives `&[u64]`:
    ///
    /// ```
    /// use jix::Array;
    ///
    /// let a = Array::compact_fn([2, 3].as_slice(), |i: &[u64]| i[0] + i[1])?;
    /// assert_eq!(a.shape(), &[2, 3]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_fn<Sh, F>(shape: Sh, f: F) -> Result<Self>
    where
        Sh: IntoDimension<Dimension = D>,
        D: Dimension,
        F: Fn(D::Index<'_>) -> T,
        T: Dtyped,
    {
        Self::compact_fn_with(shape, ArrayParams::default(), f)
    }

    /// Create a block-compressed array by evaluating a function `f` at each index, with explicit `ArrayParams`.
    ///
    /// The function `f` is called with the index of each element in the output array, and should
    /// return the value for that element.
    /// Elements are visited in an arbitrary order - this is not some theoretical use case, most of
    /// the times elements will NOT be visited in row-major order, and the function should not rely
    /// on any specific order for correctness or performance.
    ///
    /// # Arguments
    ///
    /// - `shape`: shape of the output array.
    /// - `params`: block layout and codec parameters. See [`ArrayParams`] for details.
    /// - `f`: function that produces the value of each element given its index.
    ///
    /// # Examples
    ///
    /// Materialize a large 2D array with an explicit block shape tuned to the
    /// access pattern (here, square 64x64 tiles):
    ///
    /// ```
    /// # use jix::{Array, ArrayParams};
    /// let mut params = ArrayParams::new();
    /// params.block_shape(&[64, 64]);
    ///
    /// let a = Array::compact_fn_with((256, 256), params, |(x, y)| (x as f32) * 0.5 + (y as f32))?;
    /// assert_eq!(a.shape(), &[256, 256]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_fn_with<Sh, F>(shape: Sh, mut params: ArrayParams, f: F) -> Result<Self>
    where
        Sh: IntoDimension<Dimension = D>,
        D: Dimension,
        F: Fn(D::Index<'_>) -> T,
        T: Dtyped,
    {
        struct FnStorage<T, D, F> {
            dtype: Ty<T>,
            shape: D,
            f: F,
            spec: ArraySpecOwned,
        }
        impl<T, D, F> ArrayStorage for FnStorage<T, D, F>
        where
            T: Dtyped,
            D: Dimension,
            F: Fn(D::Index<'_>) -> T,
        {
            type ElementType = Ty<T>;
            type Dimension = D;

            #[inline]
            fn read_data(
                &self,
                index: &[Range<u64>],
                buf: &mut [u8],
                _context: &ReadContext,
            ) -> Result<()> {
                let ndim = self.shape().len();
                let read_shape = D::from_fn(ndim, |dim| index[dim].end - index[dim].start);
                let read_strides = default_strides(read_shape.as_slice(), size_of::<T>() as u64);
                let iter = NdIter::new(
                    read_shape,
                    NdIterExtStridesPtrMut::new(&read_strides, buf.as_mut_ptr()),
                );
                for (idx, out) in iter {
                    let value = (self.f)(idx.to_index());
                    unsafe { out.cast::<T>().write(value) };
                }
                Ok(())
            }

            #[inline(always)]
            fn shape(&self) -> &[u64] {
                self.shape.as_slice()
            }

            #[inline(always)]
            fn dtype(&self) -> &Dtype {
                self.dtype.dtype()
            }

            #[inline]
            fn spec(&self) -> crate::storage::ArraySpec<'_> {
                self.spec.as_ref()
            }

            crate::ops::impl_dimension_change_default!();
            crate::ops::impl_element_type_change_default!();
        }

        let shape = shape.into_dimension()?;
        let dtype = Ty::<T>::new();

        params.tune(shape.as_slice(), dtype.dtype())?;
        let spec = params.clone().into_spec(shape.as_slice(), dtype.dtype())?;
        let array = Array::from_storage(FnStorage {
            dtype,
            shape,
            f,
            spec,
        });

        array.compact_with(params, &array.try_read_ctx()?)
    }
}

impl<S: ArrayStorage> Array<S> {
    /// Get the shape of the array, one element per dimension.
    #[inline(always)]
    pub fn shape(&self) -> &[u64] {
        let shape = self.storage.shape();
        if let Some(compile_time_ndim) = S::Dimension::NDIM {
            unsafe { assert_unchecked_eq!(shape.len(), compile_time_ndim) };
        }
        shape
    }

    /// Get the number of dimensions.
    #[inline(always)]
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
    /// use jix::Array;
    /// use jix::dtype::Dtyped;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert_eq!(a.dtype(), &f32::DTYPE);
    ///
    /// let b = Array::plain_ndarray_ref(&array![[false, true]])?;
    /// assert_eq!(b.dtype(), &bool::DTYPE);
    ///
    /// #[derive(Dtyped, Copy, Clone)]
    /// struct Point { x: f32, y: f32 }
    /// let c = Array::plain_ndarray_ref(&array![Point { x: 1.0, y: 2.0 }])?;
    /// assert_eq!(c.dtype(), &Point::DTYPE);
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline(always)]
    pub fn dtype(&self) -> &Dtype {
        let dtype = self.storage.dtype();
        if let Some(compile_time_dtype) = S::ElementType::DTYPE {
            unsafe { assert_unchecked_eq!(dtype, &compile_time_dtype) };
        }
        dtype
    }

    /// Decode the entire array into a fresh heap-allocated [`ndarray::Array`].
    ///
    /// This is the simplest way to materialize a jix array into a standard in-memory ndarray.
    /// All blocks of the underlying compact storage are decompressed and the elements are
    /// returned in a contiguous row-major (C-order) ndarray with the same shape and element
    /// type as the source.
    ///
    /// For a lazy view (e.g. `Array<Add<Mul<Compact, _>, _>>`), `to_ndarray` walks the entire
    /// operation pipeline: each composed op is evaluated on the fly as data flows out of the
    /// innermost storage. Materializing the same chain repeatedly will redo the work; if you
    /// plan to read the result more than once, call [`compact`](Array::compact) first to re-compress
    /// the result into a fresh `Array<Compact>` (or [`compact_with`](Array::compact_with) to control
    /// the block shape).
    ///
    /// `to_ndarray` allocates a fresh [`ReadContext`] internally, so callers don't need to
    /// manage one. For repeated reads (e.g. iterating tiles over a large array) prefer
    /// [`to_ndarray_sub`](Array::to_ndarray_sub) with an explicit context obtained from
    /// [`read_ctx`](Array::read_ctx) - that way the codec scratch buffers and decompressor
    /// instance are shared across calls.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
    /// let nd = a.to_ndarray()?;
    /// assert_eq!(nd, array![[1.0f32, 2.0], [3.0, 4.0]]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// Materialize a lazy pipeline. None of the arithmetic runs at construction time - it
    /// only executes when `to_ndarray` walks the chain:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    /// let b = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
    ///
    /// // Build a lazy view: (a + b) * 2 - 1
    /// let lazy = (a + b).map(|x| x * 2.0f32 - 1.0f32);
    /// assert_eq!(lazy.shape(), &[2, 3]);
    ///
    /// let nd = lazy.to_ndarray()?; // executes the pipeline
    /// assert_eq!(nd[[0, 0]], (1.0 + 1.0) * 2.0 - 1.0);
    /// assert_eq!(nd[[1, 2]], (6.0 + 1.0) * 2.0 - 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn to_ndarray(
        &self,
    ) -> Result<ndarray::Array<S::Item, <S::Dimension as ndarray::IntoDimension>::Dim>>
    where
        S: ArrayStorageTyped,
    {
        let shape = self.shape();
        let full_range = dim_arr(shape.len(), |dim| 0u64..shape[dim]);
        self.to_ndarray_sub(&full_range, &self.read_ctx())
    }

    /// Decode a rectangular sub-region of the array into a fresh heap-allocated [`ndarray::Array`].
    ///
    /// `index` contains one half-open `start..end` range per dimension within
    /// `0..self.shape()[dim]`. The result has shape `[index[0].len(), index[1].len(), ...]`
    /// and contains the corresponding elements from the source in row-major order.
    ///
    /// # When to use this over [`to_ndarray`](Array::to_ndarray)
    ///
    /// Use `to_ndarray_sub` when you only need a window of a large array, or when you want
    /// to stream over an array tile-by-tile without ever materializing the whole thing in
    /// memory. Use `to_ndarray` when you need the entire array at once and don't plan to
    /// issue further reads.
    ///
    /// # What gets decompressed
    ///
    /// For [`Array<Compact>`](crate::storage::Compact) (and views layered on top of it),
    /// `to_ndarray_sub` only touches the blocks that overlap `index`:
    ///
    /// - **Block-aligned ranges**: if `index` aligns to block boundaries on every axis,
    ///   exactly the covered blocks are decompressed - no wasted work.
    /// - **Unaligned ranges**: the overlapping boundary blocks are decompressed in full and
    ///   the requested slice is copied out. Elements outside `index` within those boundary
    ///   blocks are decoded but then discarded.
    ///
    /// To keep sub-region reads cheap, choose a `block_shape` (via
    /// [`ArrayParams::block_shape`](crate::ArrayParams::block_shape) when creating the array)
    /// that matches your access pattern.
    ///
    /// For lazy views (e.g. after `reshape`, `permute_axes`, `broadcast`, or element-wise
    /// ops), the index range is propagated inward through the chain and only the
    /// corresponding region of the innermost storage is read. A shape-changing op can
    /// scramble the mapping such that a small output range still touches many input blocks -
    /// if this matters for performance, materialize with [`compact`](Array::compact) before
    /// iterating sub-regions.
    ///
    /// # `ReadContext` reuse
    ///
    /// `context` carries the decoder instance and scratch buffers used during decompression.
    /// Allocating these on every read is expensive, especially for small windows. Obtain a
    /// single context via [`read_ctx`](Array::read_ctx) once and pass it to every call.
    /// [`read_ctx`](Array::read_ctx) inherits the array's stored decoder configuration;
    /// `ReadContext::default()` works too but uses default decoder parameters regardless of
    /// the array.
    ///
    /// # Errors
    ///
    /// - [`InvalidIndex`](crate::ErrorKind::InvalidIndex) - `index` is out of bounds or
    ///   has a different number of dimensions than the array.
    /// - [`CodecError`](crate::ErrorKind::CodecError) - block decompression fails.
    ///
    /// # Examples
    ///
    /// Stream a large array tile-by-tile, reusing a single `ReadContext`. The block shape
    /// is chosen to divide the tile shape, so every tile read decompresses exactly its
    /// covering blocks with no boundary waste:
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    ///
    /// // 256x256 f32 array stored as 32x32 blocks.
    /// let data = ndarray::Array2::<f32>::from_shape_fn((256, 256), |(i, j)| (i + j) as f32);
    /// let mut params = ArrayParams::new();
    /// params.block_shape(&[32, 32]);
    /// let a = Array::compact_ndarray_with(&data, params)?;
    ///
    /// // Walk the array as 64x64 tiles. 32 divides 64, so each call decompresses exactly
    /// // 2*2 = 4 blocks - no boundary waste. The same ReadContext is reused across all
    /// // 16 reads, sharing decompressor and scratch buffers.
    /// let ctx = a.read_ctx();
    /// let mut total = 0.0f32;
    /// for tr in 0..4 {
    ///     for tc in 0..4 {
    ///         let tile = a.to_ndarray_sub(
    ///             &[(tr * 64)..((tr + 1) * 64), (tc * 64)..((tc + 1) * 64)],
    ///             &ctx,
    ///         )?;
    ///         total += tile.sum();
    ///     }
    /// }
    /// assert!(total > 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// Sub-region reads also work on lazy views - the requested range is propagated through
    /// the pipeline so the inner storage only sees the corresponding window:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    /// let scaled = a.map(|x| x * 10i32);
    ///
    /// let ctx = scaled.read_ctx();
    /// assert_eq!(
    ///     scaled.to_ndarray_sub(&[1..3, 1..3], &ctx)?,
    ///     array![[50, 60], [80, 90]],
    /// );
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn to_ndarray_sub(
        &self,
        index: &[Range<u64>],
        context: &ReadContext,
    ) -> Result<ndarray::Array<S::Item, <S::Dimension as ndarray::IntoDimension>::Dim>>
    where
        S: ArrayStorageTyped,
    {
        check_get_range(self.shape(), index)?;
        let ndim = self.ndim();
        let out_shape = dim_arr(ndim, |dim| {
            let len = index[dim].end - index[dim].start;
            let len: usize = len.try_into().unwrap();
            len
        });
        let array = ndarray::ArrayD::uninit(&out_shape[..]);
        let mut array = array
            .into_dimensionality::<<S::Dimension as ndarray::IntoDimension>::Dim>()
            .unwrap();
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    ///
    /// let context = a.read_ctx();
    /// let mut buf = vec![0u32; 4];
    /// {
    ///     let buf =
    ///         unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4) };
    ///     a.to_ndarray_buf(&[1..3, 1..3], buf, &context)?;
    /// }
    /// assert_eq!(buf, vec![5, 6, 8, 9]);
    /// # Ok::<(), jix::Error>(())
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
        let nitems = check_get_buffer_size(index, dtype, buf)?;

        // Fast path for small reads
        let spec = self.storage.spec();
        let small_read = nitems as u64 <= spec.read_size() * dtype.itemsize() as u64;
        if small_read {
            return self.storage.read_data(index, buf, context);
        }

        let read_shape: S::Dimension = spec.read_shape_heuristic(index, shape, dtype.itemsize());
        // Block-space begin/end for NdIter.
        let block_begin = S::Dimension::from_fn(ndim, |dim| index[dim].start / read_shape[dim]);
        let block_end = S::Dimension::from_fn(ndim, |dim| index[dim].end.div_ceil(read_shape[dim]));
        // Element-space begin/end for NdIterExtBlockOffsetSize.
        let elem_begin = S::Dimension::from_fn(ndim, |dim| index[dim].start);
        let elem_end = S::Dimension::from_fn(ndim, |dim| index[dim].end);
        // NdIter that yields blocks of size <= read_shape
        let block_iter = NdIter::new_with_begin(
            block_begin,
            block_end,
            NdIterExtBlockOffsetSize::new(
                elem_begin,
                elem_end,
                S::Dimension::from_fn(ndim, |dim| read_shape[dim]),
            ),
        );

        let itemsize = dtype.itemsize() as usize;
        let out_shape = dim_arr(ndim, |dim| (index[dim].end - index[dim].start) as usize);
        let out_strides = default_strides(&out_shape, itemsize);

        let mut tmp_buf = context.tmp_buf(0, dtype.alignment());
        for (block_idx, (block_inner_offset, block_size)) in block_iter {
            let inner_index = dim_arr(ndim, |dim| {
                let start = block_idx[dim] * read_shape[dim] + block_inner_offset[dim];
                let end = start + block_size[dim];
                start..end
            });
            let tmp_buf = {
                let read_nitems = block_size.as_slice().iter().product::<u64>();
                tmp_buf.set_len(read_nitems as usize * itemsize);
                tmp_buf.as_mut_slice()
            };
            self.storage.read_data(&inner_index, tmp_buf, context)?;

            let out_offset = (0..ndim)
                .map(|dim| (inner_index[dim].start - index[dim].start) as usize * out_strides[dim])
                .sum::<usize>();
            let dst_ptr = unsafe { buf.as_mut_ptr().add(out_offset) };

            unsafe {
                nd_copy(
                    tmp_buf.as_ptr(),
                    dst_ptr,
                    block_size.clone(),
                    &default_strides(block_size.as_slice(), itemsize as _),
                    &out_strides,
                    itemsize,
                )
            };
        }
        Ok(())
    }

    /// Compress the data of this array into a new `Array<Compact>` with new blocks.
    ///
    /// The primary use of `compact` is to materialize a lazy operation chain:
    /// An `Array<S>` can have an arbitrary storage implementation, often a lazy view of some one or
    /// more computation, for example `Array<Floor<Map<Compact>>>` (see the examples).
    /// Reads to such lazy view arrays always perform the whole computation pipeline on the fly,
    /// which is very flexible but can be inefficient for repeated access. Coping the data and
    /// re-compressing it into a new array with `compact` breaks the lazy storage chain and materializes
    /// the result as a standalone `Array<Compact>`.
    ///
    /// In contrast to "simple" views such as unary element-wise operations, lazy ops that change the
    /// shape of the array (e.g. `reshape`, `broadcast`, `permute_axes`) can cause block boundaries
    /// to no longer align with the logical layout of the array, causing reads to decompress excess
    /// data. Calling `compact` on the result of such an operation re-encodes the data with a freshly
    /// derived block shape that matches the new layout. The block shape of copied arrays is
    /// automatically derived and tuned from the underlying storage(s), using a heuristic that aims
    /// to preserve user choices (that may depend on the user knowledge of the access pattern), but
    /// its not perfect - you may want to explicitly pass some parameters via
    /// [`compact_with`](Array::compact_with).
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// let result =                  // Array<Compact>
    ///     a.map(|x| x * 7.399_f32)  // Array<Map<Compact>>
    ///    .floor()                   // Array<Floor<Map<Compact>>>
    ///    .compact()?;               // Array<Compact> - materialize the pipeline
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact(&self) -> Result<Array<Compact<S::ElementType, S::Dimension>>> {
        let context = self.read_ctx();
        self.compact_with(ArrayParams::default(), &context)
    }

    /// Compress the data of this array into a new `Array<Compact>` with explicit control over parameters.
    ///
    /// Like [`compact`](Array::compact) (see its documentation), but with explicit [`ArrayParams`].
    /// Any optional field in `params` that is not set will be inherited from the source storage
    /// if possible.
    ///
    /// # Errors
    ///
    /// - [`CodecError`](crate::ErrorKind::CodecError) - compression or decompression fails.
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let mut a_params = ArrayParams::default();
    /// a_params.block_shape(&[1, 2]);
    /// let a = Array::compact_ndarray_with(&array![[1.5f32, 2.0], [3.14, 6.17]], a_params)?;
    ///
    /// // Let's say a is given to us, and we prepare to access it many times with a specific access
    /// // pattern. We re-compress it with a matching block shape.
    /// let mut b_params = ArrayParams::default();
    /// b_params.block_shape(&[2, 1]);
    /// let b = a.compact_with(b_params, &a.read_ctx())?;
    ///
    /// assert_eq!(
    ///     b.to_ndarray_sub(&[0..2, 0..1], &b.read_ctx())?,
    ///     array![[1.5], [3.14]]
    /// );
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_with(
        &self,
        mut params: ArrayParams,
        context: &ReadContext,
    ) -> Result<Array<Compact<S::ElementType, S::Dimension>>> {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();

        params.override_from_storage(&self.storage);
        params.tune(shape, dtype)?;

        let shape = DimArray::from_slice(shape).unwrap();
        let encoder_params = params.encoder_params.clone().unwrap_or_default();

        let block_shape = params.block_shape.as_ref().unwrap();
        let block_size = block_shape.iter().cloned().try_product().unwrap();
        let grid_shape = dim_arr(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));
        let nblocks = grid_shape.iter().cloned().try_product().unwrap();

        let decoder_cfg = DecoderCodecConfig {
            codec: encoder_params.codec.clone(),
            filters: encoder_params.filters.clone(),
            dtype: dtype.clone(),
        };

        let mut block_fn = self.to_block_fn(&params, context)?;
        let mut builder = BlockTableBuilder::start(nblocks, block_size, decoder_cfg)?;
        for block_index in 0..nblocks {
            let data = block_fn.get_compressed_block(block_index)?;
            builder.write_compressed_block(data)?;
        }
        let blocks = builder.finalize()?;

        let shape = S::Dimension::from_slice(&shape);
        Ok(Array {
            storage: Compact(ArrayBlockTableStorageBase::new(blocks, shape, params)?),
        })
    }

    /// Create a [`ReadContext`] with parameters derived from this array.
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
    /// use the same decoding configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]])?;
    /// let context = a.read_ctx();
    /// // Reuse the same context for multiple reads, sharing buffers.
    /// for row in 0..3 {
    ///     let row_data = a.to_ndarray_sub(&[row..(row + 1), 0..3], &context)?;
    ///     println!("row sum: {}", row_data.sum());
    /// }
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn read_ctx(&self) -> ReadContext {
        self.try_read_ctx().expect("failed to create read context")
    }
    fn try_read_ctx(&self) -> Result<ReadContext> {
        ReadContext::new(self.storage.spec().decoder_params())
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
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// let b = a.as_ref().map(|x| x + 1.0f32); // Array<Map<Ref<Compact>>>
    /// let c = a.as_ref() * b; // we can use `a` again here because we called as_ref()
    /// assert_eq!(c.to_ndarray()?[[1, 1]], 6.17 * (6.17 + 1.0));
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline(always)]
    pub fn as_ref(&self) -> Array<Ref<'_, S>> {
        Array {
            storage: Ref(self.storage()),
        }
    }

    /// Convert this array into a type-erased [`ArrayAny`](crate::ArrayAny).
    ///
    /// The storage is wrapped in an `Arc` and hidden behind [`ArrayStorageAny`], so the
    /// resulting array can be stored alongside arrays of other concrete storage types.
    ///
    /// Only arrays that are already dynamically typed (`TypeDyn`, `DimDyn`) can be erased this
    /// way. Call [`Array::into_type_dyn`] and [`Array::into_dim_dyn`] first if needed.
    pub fn into_any(self) -> ArrayAny
    where
        S: ArrayStorage<ElementType = TypeDyn, Dimension = DimDyn> + Send + Sync + 'static,
    {
        Array::from_storage(ArrayStorageAny::new(Arc::new(self.into_storage())))
    }

    /// Check if this array storage is compact block-compressed storage.
    ///
    /// This functions returns `true` for arrays that are stored in compact block-compressed form,
    /// i.e. those created by [`compact_ndarray`](Array::compact_ndarray), [`compact`](Array::compact),
    /// [`read_from_file`](Array::read_from_file), etc., and `false` for arrays with storage
    /// implementations with uncompressed data, such as lazy operation views, plain ndarray views,
    /// etc.
    ///
    /// # Example
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert!(a.is_compact());
    ///
    /// let b = a.map(|x| x * 2.0f32); // Array<Map<Compact>>
    /// assert!(!b.is_compact()); // b is a lazy view
    /// # Ok::<(), jix::Error>(())
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
    /// Use [`maybe_compact_with`](Self::maybe_compact_with) to control the block
    /// shape and codec parameters.
    ///
    /// # Example
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert!(a.is_compact());
    /// let a = a.maybe_compact()?; // a is already compact, so this is a no-op
    ///
    /// let b = a.map(|x| x * 2.0f32); // Array<Map<Compact>>
    /// assert!(!b.is_compact()); // b is a lazy view
    /// let b = b.maybe_compact()?; // materialize b into compact form
    /// assert!(b.is_compact());
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn maybe_compact(self) -> Result<Array<MaybeCompact<S>>> {
        let context = self.read_ctx();
        self.maybe_compact_with(ArrayParams::default(), &context)
    }

    /// Ensure this array is in compact block-compressed form, re-compressing
    /// if needed, with explicit control over parameters.
    ///
    /// Similar to [`maybe_compact`](Self::maybe_compact) but with explicit [`ArrayParams`].
    ///
    /// `params` controls the target block shape and compression settings. It is
    /// **only used when the source is not already compact** - if `is_compact()`
    /// returns `true`, the existing storage is wrapped zero-cost and `params` is
    /// ignored.
    pub fn maybe_compact_with(
        self,
        params: ArrayParams,
        context: &ReadContext,
    ) -> Result<Array<MaybeCompact<S>>> {
        MaybeCompact::new_array(self, params, context)
    }

    /// Return a reference to the underlying storage backend.
    ///
    /// Rarely needed to be used directly by users.
    #[inline(always)]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Construct an `Array` by wrapping a storage backend directly.
    ///
    /// Rarely needed to be used directly by users.
    #[inline(always)]
    pub fn from_storage(storage: S) -> Self {
        Self { storage }
    }

    /// Consume this array and return the underlying storage backend.
    ///
    /// Rarely needed to be used directly by users.
    #[inline(always)]
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Build a [`BlockFn`] that reads and compresses this array's data block by block.
    ///
    /// Called by [`Array::maybe_compact`] (and its variants) to feed [`BlockTableBuilder`] with
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
    /// Returns a [`BlockFnWithState`] closure that, for each requested block (in order), reads the
    /// block from storage, compresses it with the encoder from `params` into an internal
    /// `AlignedBytes` buffer, and returns that buffer's slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoder cannot be constructed from `params`.
    pub(crate) fn to_block_fn<'a>(
        &'a self,
        params: &ArrayParams,
        context: &'a ReadContext,
    ) -> Result<impl BlockFn + 'a> {
        let shape = self.shape();
        let ndim = shape.len();
        let dtype = self.dtype();

        let block_shape = params.block_shape.as_ref().unwrap().clone();
        let block_size = block_shape.iter().cloned().try_product().unwrap();
        let grid_shape =
            S::Dimension::from_fn(ndim, |dim| shape[dim].div_ceil(block_shape[dim] as u64));

        let encoder_cfg = params.encoder_params.as_ref().unwrap();
        let mut encoder = Encoder::new(encoder_cfg, dtype.clone())?;

        let mut block_iter = NdIter::new(
            grid_shape.clone(),
            NdIterExtBlockOffsetSize::new(
                S::Dimension::from_fn(ndim, |_| 0),
                S::Dimension::from_slice(shape),
                S::Dimension::from_fn(ndim, |dim| block_shape[dim] as u64),
            ),
        );

        let grid_logical_strides = default_strides(grid_shape.as_slice(), 1);
        let itemsize = encoder.dtype.itemsize() as usize;
        let alignment = encoder.dtype.alignment().as_usize();
        let block_size_bytes = block_size as usize * itemsize;
        let block_strides = default_strides(&block_shape, itemsize as _);
        let mut tmp_block_plain = AlignedBytes::new_padded(alignment);
        let block_compressed_bound = encoder.encode_bound(block_size_bytes);
        let block_fn = BlockFnWithState::from_fn(
            AlignedBytes::new_padded(alignment),
            move |block_logical_idx: u64, tmp_block_compressed: &mut AlignedBytes| {
                tmp_block_compressed.clear();

                let (block_idx, (block_inner_offset, block_size)) = block_iter.next().unwrap();
                debug_assert_eq!(
                    block_logical_idx,
                    block_idx
                        .as_slice()
                        .iter()
                        .zip(&grid_logical_strides)
                        .map(|(i, s)| i * s)
                        .sum::<u64>()
                );
                let read_range = dim_arr(ndim, |dim| {
                    let start = block_idx[dim] * block_shape[dim] as u64 + block_inner_offset[dim];
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
                let read_data_buf = if full_block {
                    tmp_block_plain.as_mut_slice()
                } else {
                    tmp_block_plain.fill(0); // zero-pad
                    let b_size_bytes =
                        block_size.as_slice().iter().product::<u64>() as usize * itemsize;
                    // Borrow the (cleared, aligned) compressed buffer as scratch for the strided
                    // read, then release it before compressing into it below.
                    tmp_block_compressed.reserve(b_size_bytes);
                    unsafe { tmp_block_compressed.set_len(b_size_bytes) };
                    tmp_block_compressed.as_mut_slice()
                };
                self.storage
                    .read_data(&read_range, read_data_buf, context)?;
                if !full_block {
                    // Copy from temporary buffer to output block with correct strides.
                    let src_strides =
                        default_strides(&dim_arr(ndim, |dim| block_size[dim] as usize), itemsize);
                    unsafe {
                        nd_copy(
                            read_data_buf.as_ptr(),
                            tmp_block_plain_ptr,
                            block_size.clone(),
                            &src_strides,
                            &block_strides,
                            itemsize,
                        )
                    };
                    unsafe { tmp_block_compressed.set_len(0) };
                }
                let plain_data = tmp_block_plain.as_slice();

                // Compress block data
                tmp_block_compressed.reserve(block_compressed_bound);
                unsafe { tmp_block_compressed.set_len(block_compressed_bound) };
                let cdata_len = encoder.encode(plain_data, tmp_block_compressed.as_mut_slice())?;
                unsafe { tmp_block_compressed.set_len(cdata_len) };

                Ok(tmp_block_compressed.as_slice())
            },
        );
        Ok(block_fn)
    }
}

/// Methods for converting an array to a different element type or dimension type.
impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Re-tag this array's element type as `NewET`, returning an error if the runtime dtype does
    /// not match.
    ///
    /// This is the bridge between dynamic and static element-type tracking. Arrays loaded from
    /// files carry [`TypeDyn`] as their element type because the compiler cannot know the dtype
    /// at that point. After you have confirmed the dtype (e.g. by reading `array.dtype()` or
    /// knowing the data ahead of time), call `into_type::<Ty<T>>()` (or the
    /// [`into_typed::<T>`](Self::into_typed) sugar) to recover a statically-typed element type.
    /// Subsequent operations on the result will propagate the static `Ty<T>` through the type
    /// system.
    ///
    /// Most element-wise operations require a static element type (`ArrayStorageTyped`).
    ///
    /// This method replaces the inner storage with `S::ElementTypeChange<NewET>`, which is
    /// implemented by the simpler [`IntoType<S, NewET>`](crate::ops::IntoType) adaptor for some
    /// storages, but may be implemented as an in-place replacement for others.
    ///
    /// See [`into_type_dyn`](Self::into_type_dyn).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `NewET = Ty<T>` and `self.dtype() != T::DTYPE`. Always succeeds for `NewET = TypeDyn`.
    #[inline(always)]
    pub fn into_type<NewET>(self) -> Result<Array<S::ElementTypeChange<NewET>>>
    where
        NewET: ElementType,
    {
        Ok(Array::from_storage(
            self.into_storage().element_type_change()?,
        ))
    }

    /// Re-tag this array's element type as [`Ty<T>`](crate::Ty), asserting a concrete scalar type.
    ///
    /// Sugar for [`into_type::<Ty<T>>()`](Self::into_type).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `self.dtype() != T::DTYPE`.
    #[inline(always)]
    pub fn into_typed<T>(self) -> Result<Array<S::ElementTypeChange<Ty<T>>>>
    where
        T: Dtyped,
    {
        self.into_type()
    }

    /// Re-tag this array's element type as [`TypeDyn`], erasing static element-type information.
    ///
    /// Infallible sugar for [`into_type::<TypeDyn>()`](Self::into_type).
    #[inline(always)]
    pub fn into_type_dyn(self) -> Array<S::ElementTypeChange<TypeDyn>> {
        self.into_type().unwrap()
    }

    /// Re-tag this array's dimension as `D`, returning an error if the actual ndim does not match.
    ///
    /// This is the bridge between dynamic and static dimension tracking. Arrays loaded from
    /// files or produced by slice-based shape operations carry [`DimDyn`] as their dimension
    /// type because the compiler cannot know the ndim at that point. After you have confirmed
    /// the ndim (e.g. by reading `array.ndim()` or knowing the data layout ahead of time),
    /// call `into_dim::<Dim<N>>()` to recover a statically-typed dimension. Subsequent
    /// operations on the result will propagate the static `Dim<N>` through the type system.
    ///
    /// Generally speaking, the compiler can optimize more aggressively when the dimension is
    /// statically known, which can yield better performance.
    ///
    /// This method replace the inner storage with `S::DimensionChange<D>`, which is implemented by
    /// simpler [`IntoDim<S, D>`](crate::ops::IntoDim) adaptor for some storages, but may be implemented
    /// as an in-place replacement for others.
    ///
    /// See [`into_dim_dyn`](Self::into_dim_dyn).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidShapeOperation`](crate::ErrorKind::InvalidShapeOperation) if
    /// `D::NDIM` is `Some(n)` and `self.ndim() != n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::{Array, Dim};
    ///
    /// // Passing a dynamically-dimensioned ndarray produces Array<Compact<DimDyn>>.
    /// // Arrays loaded from files via Array::read_from_file also carry DimDyn.
    /// let a = Array::compact_ndarray(&ndarray::ArrayD::<i32>::zeros(vec![2, 3, 4]))?;
    ///
    /// // Assert the array is 3-D; fail gracefully if not.
    /// let a3d = a.into_dim::<Dim<3>>()?; // Array<Compact<Dim<3>>>
    ///
    /// // Now insert_axis knows the result is 4-D at compile time.
    /// let a4d = a3d.insert_axis(0); // Array<InsertAxis<..., Dim<4>>>
    /// assert_eq!(a4d.shape(), &[1, 2, 3, 4]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline(always)]
    pub fn into_dim<D>(self) -> Result<Array<S::DimensionChange<D>>>
    where
        D: Dimension,
    {
        Ok(Array::from_storage(self.into_storage().dimension_change()?))
    }

    /// Re-tag this array's dimension as [`DimDyn`], erasing static dimension information.
    ///
    /// This is the infallible counterpart to [`into_dim`](Self::into_dim). Every array has a
    /// runtime ndim regardless of its static type, so converting to `DimDyn` always succeeds.
    ///
    /// After calling `into_dim_dyn`, subsequent shape-changing operations will produce
    /// `DimDyn` results rather than `Dim<N>`. Call [`into_dim`](Self::into_dim) again to
    /// re-establish static tracking once the ndim is confirmed.
    #[inline(always)]
    pub fn into_dim_dyn(self) -> Array<S::DimensionChange<DimDyn>> {
        self.into_dim().unwrap()
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
    use ndarray::array;

    use super::Array;
    use crate::array::{ArrayBlockTableStorageBase, Compact};
    use crate::codec::EncoderParams;
    use crate::dtype::Dtyped;
    use crate::storage::block::{BlockSize, BlockTable};
    use crate::util::{arr_params, cast_slice, DimArray};
    use crate::{ArrayParams, ArrayStorage, Dimension, ErrorKind, IntoDimension, Ty};

    // -----------------------------------------------------------------------
    // compact_ndarray roundtrip helper
    // -----------------------------------------------------------------------

    fn roundtrip<T, S, D>(
        src: &ndarray::ArrayBase<S, D>,
        block_shape: &[usize],
    ) -> ndarray::Array<T, D>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension + IntoDimension,
    {
        let a = Array::compact_ndarray_with(&src, arr_params(block_shape)).unwrap();
        a.to_ndarray().unwrap().into_dimensionality().unwrap()
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

    fn array<T, Sh>(
        blocks: &[&[T]],
        shape: Sh,
        block_shape: &[usize],
    ) -> Array<Compact<Ty<T>, Sh::Dimension>>
    where
        T: Dtyped,
        Sh: IntoDimension,
    {
        let shape = shape.into_dimension().unwrap();
        let shape = shape
            .as_slice()
            .iter()
            .map(|&x| x as u64)
            .collect::<DimArray<_>>();
        let block_shape = block_shape
            .iter()
            .map(|&x| x as BlockSize)
            .collect::<DimArray<_>>();
        let params = ArrayParams::default().block_shape(&block_shape).clone();
        Array {
            storage: Compact(
                ArrayBlockTableStorageBase::new(
                    make_block_table(blocks),
                    Sh::Dimension::from_slice(&shape),
                    params,
                )
                .unwrap(),
            ),
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
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![0, 1, 2, 3]);
    }

    #[test]
    fn to_ndarray_1d_two_blocks() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn to_ndarray_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![10, 20, 30, 40, 50, 60, 70, 80]);
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
        let got = a.to_ndarray().unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([4, 6], (0u8..24).collect()).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // to_ndarray_sub - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn to_ndarray_sub_1d_full_range() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got = a.to_ndarray_sub(&[0..6], &a.read_ctx()).unwrap();
        assert_eq!(got, array![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn to_ndarray_sub_1d_aligned_second_block() {
        // range [3..6) -> output shape [3], values [3,4,5]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got = a.to_ndarray_sub(&[3..6], &a.read_ctx()).unwrap();
        assert_eq!(got, array![3, 4, 5]);
    }

    #[test]
    fn to_ndarray_sub_1d_cross_block_boundary() {
        // range [1..5) -> output shape [4], values [1,2,3,4]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got = a.to_ndarray_sub(&[1..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1, 2, 3, 4]);
    }

    #[test]
    fn to_ndarray_sub_1d_within_single_block() {
        // range [1..2) -> output shape [1], value [1]
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let got = a.to_ndarray_sub(&[1..2], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1]);
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
        let got = a.to_ndarray_sub(&[1..3, 2..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![[8u8, 9, 10], [14, 15, 16]]);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_1d_single_block() {
        let src = array![0u8, 1, 2, 3];
        assert_eq!(roundtrip(&src, &[4]), src);
    }

    #[test]
    fn compact_ndarray_1d_multi_block() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        assert_eq!(roundtrip(&src, &[3]), src);
    }

    #[test]
    fn compact_ndarray_1d_with_padding() {
        // size 5, block 3 -> padded to 6; shape reported as 5
        let src = array![0u8, 1, 2, 3, 4];
        let a = Array::compact_ndarray_with(&src, arr_params(&[3])).unwrap();
        assert_eq!(a.shape(), &[5]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, src);
    }

    #[test]
    fn compact_ndarray_1d_i32() {
        let src = array![0i32, 10, 20, 30, 40, 50, 60, 70];
        assert_eq!(roundtrip(&src, &[4]), src);
    }

    #[test]
    fn compact_ndarray_1d_f32() {
        let src = array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        assert_eq!(roundtrip(&src, &[4]), src);
    }

    #[test]
    fn compact_ndarray_block_larger_than_shape_is_rejected() {
        // block_shape [10] > array shape [4]; must be rejected per the
        // `b <= s.max(1)` invariant enforced by `ArrayParams::tune`.
        let src = array![0u8, 1, 2, 3];
        let err = Array::compact_ndarray_with(&src, arr_params(&[10])).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    }

    #[test]
    fn compact_ndarray_1d_noncontiguous() {
        // Step-2 slice of [0..10] -> [0, 2, 4, 6, 8]
        let src = array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = src.slice(ndarray::s![..;2]);
        let a = Array::compact_ndarray_with(&view, arr_params(&[3])).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.to_ndarray().unwrap(), array![0u8, 2, 4, 6, 8]);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray - metadata
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_metadata() {
        let a = Array::compact_ndarray(&array![0i32, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.dtype(), &i32::DTYPE);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray - 2D
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        assert_eq!(roundtrip(&src, &[2, 3]), src);
    }

    #[test]
    fn compact_ndarray_2d_with_padding() {
        // shape [3,5], block [2,3] -> padded to [4,6]; shape reported as [3,5]
        #[rustfmt::skip]
        let src = array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 3])).unwrap();
        assert_eq!(a.shape(), &[3, 5]);
        assert_eq!(a.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_ndarray_2d_noncontiguous() {
        // Fortran-order (column-major) array
        let src = ndarray::Array2::<u8>::from_shape_vec(
            ndarray::ShapeBuilder::f((3, 4)),
            (0..12).collect(),
        )
        .unwrap();
        assert_eq!(roundtrip(&src, &[2, 2]), src);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray + to_ndarray_sub integration
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_then_to_ndarray_sub_1d() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        let a = Array::compact_ndarray_with(&src, arr_params(&[3])).unwrap();
        let got = a.to_ndarray_sub(&[1..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1u8, 2, 3, 4]);
    }

    #[test]
    fn compact_ndarray_then_to_ndarray_sub_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 3])).unwrap();
        let got = a.to_ndarray_sub(&[1..3, 2..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![[8u8, 9, 10], [14, 15, 16]]);
    }

    // -----------------------------------------------------------------------
    // compact_fn
    // -----------------------------------------------------------------------

    #[test]
    fn compact_fn_1d() {
        let a = Array::compact_fn(47, |i| i * 4 + 6).unwrap();
        assert_eq!(a.shape(), &[47]);
        assert_eq!(
            a.to_ndarray().unwrap().as_slice().unwrap()[..4],
            [6, 10, 14, 18]
        );
    }

    #[test]
    fn compact_fn_2d() {
        let a = Array::compact_fn((3, 3), |(x, y)| x * 10 + y).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(
            a.to_ndarray().unwrap(),
            array![[0, 1, 2], [10, 11, 12], [20, 21, 22]]
        );

        let a = Array::compact_fn([3, 3].as_slice(), |i| i[0] * 7 + i[1]).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(
            a.to_ndarray().unwrap(),
            array![[0, 1, 2], [7, 8, 9], [14, 15, 16]].into_dyn()
        );
    }

    #[test]
    fn compact_fn_0d() {
        let a = Array::compact_fn((), |()| 42).unwrap();
        assert_eq!(a.shape(), &[]);
        assert_eq!(a.to_ndarray().unwrap().as_slice().unwrap(), &[42]);
    }

    // -----------------------------------------------------------------------
    // compact
    // -----------------------------------------------------------------------

    #[test]
    fn compact_1d_single_block() {
        let a = array(&[&[0u8, 1, 2, 3]], &[4], &[4]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.ndim(), 1);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.storage.spec().block_shape()[..], [4]);
        assert_eq!(b.to_ndarray().unwrap(), array![0, 1, 2, 3]);
    }

    #[test]
    fn compact_1d_multi_block() {
        let a = array(&[&[0u8, 1, 2], &[3, 4, 5]], &[6], &[3]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[6]);
        assert_eq!(b.storage.spec().block_shape()[..], [3]);
        assert_eq!(b.to_ndarray().unwrap(), array![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn compact_1d_with_padding() {
        // shape [5], block [3] -> stored as 6 elements (padded)
        let src = array![0u8, 1, 2, 3, 4];
        let a = Array::compact_ndarray_with(&src, arr_params(&[3])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[5]);
        assert_eq!(b.storage.spec().block_shape()[..], [3]);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_1d_i32() {
        let a = array(&[&[10i32, 20, 30, 40], &[50, 60, 70, 80]], &[8], &[4]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[8]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(
            b.to_ndarray().unwrap(),
            array![10, 20, 30, 40, 50, 60, 70, 80]
        );
    }

    #[test]
    fn compact_2d_single_block() {
        // shape=[2,3], block=[2,3] - one block, no partial-block path
        let a = array(&[&[0u8, 1, 2, 3, 4, 5]], &[2, 3], &[2, 3]);
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.storage.spec().block_shape()[..], [2, 3]);
        assert_eq!(b.to_ndarray().unwrap(), array![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn compact_2d_multi_block() {
        // shape=[4,6], block=[2,3] - 4 blocks, exercises the full-block compact path
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
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[4, 6]);
        assert_eq!(b.storage.spec().block_shape()[..], [2, 3]);
        assert_eq!(
            b.to_ndarray().unwrap(),
            ndarray::Array::from_shape_vec([4, 6], (0u8..24).collect()).unwrap()
        );
    }

    #[test]
    fn compact_2d_with_padding() {
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
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 3])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[3, 5]);
        assert_eq!(b.dtype(), &i32::DTYPE);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_3d_with_padding_in_all_dims() {
        // shape=[3,3,5], block=[2,2,3] -> padded to [4,4,6].
        // Block grid 2*2*2 = 8 blocks; every boundary block is partial in at least
        // one dimension, and the single corner block [1,1,1] is partial in all three:
        //   size [1,1,2] vs block_shape [2,2,3].
        let src = ndarray::Array3::<u8>::from_shape_vec([3, 3, 5], (0u8..45).collect()).unwrap();
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 2, 3])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(b.shape(), &[3, 3, 5]);
        assert_eq!(b.dtype(), &u8::DTYPE);
        assert_eq!(b.storage.spec().block_shape()[..], [2, 2, 3]);
        assert_eq!(b.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_preserves_block_shape() {
        // Verify the copied array has the same block layout as the source.
        let src = array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let a = Array::compact_ndarray_with(&src, arr_params(&[4])).unwrap();
        let b = a.compact().unwrap();
        assert_eq!(
            a.storage.spec().block_shape()[..],
            b.storage.spec().block_shape()[..]
        );
    }

    #[test]
    fn compact_result_is_independent() {
        // Mutating the source array should not affect the compact (they are independent).
        // Since Array<Compact> doesn't expose mutation, we verify by round-tripping
        // both through write/read and checking values remain consistent.
        let src = array![10u8, 20, 30, 40];
        let a = Array::compact_ndarray_with(&src, arr_params(&[4])).unwrap();
        let b = a.compact().unwrap();
        // Both should read back the same data independently.
        assert_eq!(a.to_ndarray().unwrap(), b.to_ndarray().unwrap());
    }
}

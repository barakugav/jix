use crate::dtype::Dtype;
use crate::storage::{ReadParams, StorageAny};
use crate::{ArrayAny, ElementType, StorageBase};

mod construct;

mod read;

mod compact;

/// A multi-dimensional array, usually compressed, backed by a generic storage.
///
/// `Array<S>` is the central type in jix. It behaves like a regular n-dimensional array, but its
/// data is stored in a compressed format and decoded on demand. Its core functionality is provided
/// by [`shape()`](Array::shape), [`dtype()`](Array::dtype) and
/// [`to_buf_strided(sub_range)`](Array::to_buf_strided); all other functions are built on top of
/// those.
///
/// An array is generic over `S: StorageBase`, which provides the implementation of the three core
/// methods. The main concrete storage backend is the block-compressed
/// [`Compact`](crate::storage::Compact) type, which divides the array into n-dimensional blocks
/// and compresses each block independently, and it's the return type of the common creation
/// methods for arrays (e.g. [`compact_ndarray()`](Array::compact_ndarray) and
/// [`compact()`](Array::compact)).
///
/// # Storage variants
///
/// The primary concrete storages are:
///
/// | Type | Description |
/// |------|-------------|
/// | [`Array<Compact>`](crate::storage::Compact) | Heap-allocated block-compressed array. The main storage backend. |
/// | [`Array<Add<S1, S2>> or Array<Neg<S>> ...`](crate::ops) | Lazy operation views that wrap one or more arrays and apply a transformation at read time. Created by methods in [`ops`](crate::ops). |
/// | [`Array<Plain<T>>`](crate::storage::Plain) | Zero-copy view into an uncompressed (possibly strided) in-memory buffer. Created by [`plain_ndarray`](Array::plain_ndarray) and [`plain_ndarray_ref`](Array::plain_ndarray_ref). |
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
///   .permute_dims(dims)    -> Array<PermuteDims<Reshape<...>>>
///   .add(other_array)      -> Array<Add<PermuteDims<...>, Compact>>
///   .sum(dim)              -> Array<Sum<Add<...>>>
///   .compact()?            -> Array<Compact> - materialize the pipeline
/// ```
///
/// Data is never copied or computed at construction time. An operation only runs when the result
/// is materialized via [`to_ndarray()`](Array::to_ndarray), [`compact()`](Array::compact), and
/// their variants. At that point the read request propagates inward through the storage chain, and
/// only the minimum required data is read from the innermost backend.
///
/// Because `Array<S>` is monomorphized over `S` at compile time, chains of operations incur zero
/// virtual dispatch overhead. The full static type of an expression -
/// e.g. `Array<Add<Neg<S1>, Reshape<S2>>>` - is resolved by the compiler, which can inline the
/// entire pipeline into a single read loop. The type system *is* the execution plan.
///
/// Operations accept an owned `Array<S>` and return a new `Array<Op<S>>` that wraps the original.
/// To reuse an array in multiple operations, use [`view()`](Array::view) to create a reference.
///
/// # Examples
///
/// Create arrays from various sources:
/// ```
/// use jix::Array;
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
/// let from_file = Array::read_from_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Apply operations on compressed arrays, creating lazy views, writing the result to a file:
/// ```
/// use jix::Array;
/// use jix::dtype::Dtyped;
/// use ndarray::{Ix2, array};
///
/// // Compress a 2-D f32 ndarray.
/// let array = Array::compact_ndarray(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
/// assert_eq!(array.shape(), &[2, 3]);
/// assert_eq!(array.dtype(), &f32::DTYPE);
///
/// // Decompress and compare.
/// let decompressed = array.to_ndarray::<Ix2>()?;
/// assert_eq!(decompressed[[0, 0]], 1.5);
/// assert_eq!(decompressed[[1, 1]], 6.17);
///
/// // Read a single element and a sub-region. Only the blocks they touch are
/// // decompressed, not the whole array, so random access reads are relatively fast.
/// assert_eq!(array.get(&[1, 1])?, 6.17);
/// assert_eq!(
///     array.to_ndarray_sub(&[0..2, 1..3])?,
///     array![[2.0, -9.0], [6.17, 0.0]]
/// );
///
/// // Apply operations on a compressed array, creating lazy views
/// let ones = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
/// let scaled = array                               // Array<Compact>
///     .exp()                                       // Array<Exp<Compact>>
///     .floor()                                     // Array<Floor<Exp<Compact>>>
///     * 2.0                                        // Array<MulScalar<Floor<...>, f32>>
///     + ones;                                      // Array<Add<MulScalar<...>, Compact>>
/// // Lazy views are still arrays: reading data runs the pipeline on demand, possibly on a
/// // subset of the original array.
/// assert_eq!(scaled.shape(), &[2, 3]);
/// assert_eq!(scaled.dtype(), &f32::DTYPE);
/// assert_eq!(scaled.to_ndarray::<Ix2>()?[[1, 1]], 957.0);
///
/// // Materialize the result and write to a file.
/// let result = scaled
///     .argmax(/* dim */ 1)                        // Array<ArgMax<Add<MulScalar<...>, Compact>>>
///     .cast::<i16>()                               // Array<Cast<ArgMax<Add<...>>>>
///     // materialize the pipeline by recompressing
///     .compact()?;                                 // Array<Compact>
/// assert_eq!(result.shape(), &[2]);
/// assert_eq!(result.dtype(), &i16::DTYPE);
/// let tmp_dir = tempfile::tempdir()?;
/// result.write_to_file(&tmp_dir.path().join("result.jix"))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance notes
///
/// The block shape of an `Array<Compact>` has a large effect on both compression ratio and read
/// performance; a block shape matching a known access pattern can help significantly. If none is
/// given, it is chosen according to the CPU cache sizes. Arrays created from existing arrays
/// (`.compact()`, `.reshape()`, results of operations, etc.) choose their block shape with a
/// heuristic that tries to preserve the user's block shape while respecting the new shape and
/// layout.
///
/// Shape-changing operations - [`reshape`](Array::reshape), [`broadcast`](Array::broadcast),
/// [`permute_dims`](Array::permute_dims) - remap how output indices translate to positions in the
/// underlying blocks. When the new layout crosses block boundaries that the original respected, a
/// single read may decompress many more blocks than necessary. To avoid this, consider
/// materializing with [`compact`](Array::compact) (automatic block shape),
/// [`compact_with`](Array::compact_with) (explicit
/// [`CompactParams`](crate::storage::CompactParams), e.g. a block shape matching the expected
/// access pattern) or [`to_plain`](Array::to_plain) after a shape change.
///
/// # Element type tracking
///
/// `S::ElementType` records the scalar element type at the type level. When the element type is
/// statically known, `S::ElementType = T`, and all element-wise operations - arithmetic,
/// comparisons, reductions, type casts - become available. When the element type is only known
/// at runtime (e.g. for arrays loaded from files), `S::ElementType = TypeDyn`, and those
/// operations are not available until the type is asserted.
///
/// Arrays built from typed sources are typed: `compact_ndarray(&array![1.0f32])` returns
/// `Array<Compact<f32>>`. Use [`into_typed::<T>()`](Array::into_typed) to assert the element
/// type - validated against the stored dtype at runtime - and recover `S::ElementType = T`. Use
/// [`into_typed_dyn()`](Array::into_typed_dyn) to erase the static element type.
///
/// The number of dims is a runtime property, read from [`shape`](Array::shape).
#[derive(Clone)]
pub struct Array<S> {
    pub(crate) storage: S,
}

impl<S: StorageBase> Array<S> {
    /// Get the shape of the array, one element per dimension.
    #[inline(always)]
    pub fn shape(&self) -> &[u64] {
        self.storage.shape()
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

    /// Check if the array is empty (a dimension has size 0).
    pub fn is_empty(&self) -> bool {
        self.shape().contains(&0)
    }

    /// Get the element dtype of the array.
    ///
    /// See [`Dtype`] for details on the supported dtypes and their properties.
    ///
    /// ```rust
    /// use jix::Array;
    /// use jix::dtype::Dtyped;
    /// use jix::scalar::bool8;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// assert_eq!(a.dtype(), &f32::DTYPE);
    ///
    /// let bools = array![[bool8::FALSE, bool8::TRUE]];
    /// let b = Array::plain_ndarray_ref(&bools)?;
    /// assert_eq!(b.dtype(), &bool8::DTYPE);
    ///
    /// #[derive(Copy, Clone, Dtyped)]
    /// #[repr(C)]
    /// struct Point {
    ///     x: f32,
    ///     y: f32,
    /// }
    /// let points = array![Point { x: 1.0, y: 2.0 }];
    /// let c = Array::plain_ndarray_ref(&points)?;
    /// assert_eq!(c.dtype(), &Point::DTYPE);
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline(always)]
    pub fn dtype(&self) -> &Dtype {
        const { &S::ElementType::STATIC_DTYPE }
            .as_ref()
            .unwrap_or_else(|| self.storage.dtype())
    }

    /// The read parameters of this array, e.g. its read size.
    ///
    /// Reads without a [`ReadContext`](crate::ReadContext) use them, and [`compact`](Array::compact)
    /// keeps them. A lazy view takes them from its input (a view over several arrays takes them
    /// from one of them, usually the first). Change them on a compact or plain array with
    /// `read_params_mut`, e.g.
    /// [`CompactArrayMethods::read_params_mut`](crate::storage::CompactArrayMethods::read_params_mut).
    #[inline]
    pub fn read_params(&self) -> &ReadParams {
        self.storage.spec().read_params()
    }

    /// Borrow this array as an `Array<&S>`, without cloning the storage.
    ///
    /// Almost all ops take their arrays by value: `a + b` consumes `a` and `b`. An `Array<&S>` can
    /// be passed to any API that takes an owned `Array<S>`, so `view` lets an array be reused.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::{Ix2, array};
    ///
    /// let a = Array::compact_ndarray(&array![[1.5f32, 2.0], [3.14, 6.17]])?;
    /// let b = a.view().map(|x| x + 1.0f32); // Array<Map<&Compact>>
    /// let c = a.view() * b; // we can use `a` again here because we called view()
    /// assert_eq!(c.to_ndarray::<Ix2>()?[[1, 1]], 6.17 * (6.17 + 1.0));
    /// # Ok::<(), jix::Error>(())
    /// ```
    #[inline]
    pub fn view(&self) -> Array<&S> {
        Array::from_storage(self.storage())
    }

    /// Convert this array into a type-erased [`ArrayAny`](crate::ArrayAny).
    ///
    /// The storage is wrapped in an `Arc` and hidden behind [`StorageAny`], so the
    /// resulting array can be stored alongside arrays of other concrete storage types. The
    /// element type is kept; erase it with [`into_typed_dyn`](Array::into_typed_dyn).
    pub fn into_any(self) -> ArrayAny<S::ElementType>
    where
        S: StorageBase + std::any::Any + Send + Sync + 'static,
    {
        Array::from_storage(StorageAny::new(self.into_storage()))
    }

    /// Check if this array's storage is block-compressed.
    ///
    /// `true` for arrays created by [`compact_ndarray`](Array::compact_ndarray),
    /// [`compact`](Array::compact), [`read_from_file`](Array::read_from_file), etc., also through
    /// [`view`](Array::view) or [`into_any`](Array::into_any); `false` for lazy operation views and
    /// plain ndarray views.
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
        self.storage().spec().flags().is_compact()
    }

    /// Return a reference to the underlying storage backend.
    ///
    /// Rarely needed directly.
    #[inline(always)]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Construct an `Array` by wrapping a storage backend directly.
    ///
    /// Rarely needed directly.
    #[inline(always)]
    pub fn from_storage(storage: S) -> Self {
        #[cfg(debug_assertions)]
        {
            use crate::util::IterExt;
            use crate::{DimIdx, NDIM_MAX};

            let shape = storage.shape();
            let ndim = shape.len();
            debug_assert!(ndim <= NDIM_MAX);
            debug_assert!(shape.iter().cloned().try_shape_product().is_some());
            let spec = storage.spec();
            debug_assert_eq!(spec.block_shape().len(), ndim);
            debug_assert!(spec.block_shape().iter().all(|&s| s > 0));
            debug_assert!(
                spec.block_shape()
                    .iter()
                    .zip(shape.iter())
                    .all(|(&b, &s)| b as u64 <= s.max(1))
            );
            debug_assert!(spec.block_size() > 0);
            debug_assert_eq!(spec.block_shape_fixed_dims().len(), ndim);
            let read_size = spec.read_params().read_size;
            debug_assert!(0 < read_size.min && read_size.min <= read_size.max);
            debug_assert!(
                spec.read_shape_scale_order().len() == ndim
                    && (0..ndim).all(|d| spec.read_shape_scale_order().contains(&(d as DimIdx)))
            );
            debug_assert!(
                spec.read_layout_order().len() == ndim
                    && (0..ndim).all(|d| spec.read_layout_order().contains(&(d as DimIdx)))
            );
            debug_assert!(spec.element_cost().is_finite() && spec.element_cost() >= 0.0);
        }

        Self { storage }
    }

    /// Consume this array and return the underlying storage backend.
    ///
    /// Rarely needed directly.
    #[inline(always)]
    pub fn into_storage(self) -> S {
        self.storage
    }
}

impl<S: StorageBase> std::fmt::Debug for Array<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(crate) mod tests {
    use ndarray::array;

    use super::Array;
    use crate::codec::EncoderParams;
    use crate::dtype::Dtyped;
    use crate::storage::block::{BlockSize, BlockTable};
    use crate::storage::{ArrayBlockTableStorageBase, Compact, CompactParams};
    use crate::util::{DimArray, arr_params, cast_slice};
    use crate::{CompactArray, Dimension, IntoDimension};

    // -----------------------------------------------------------------------
    // compact_ndarray roundtrip helper
    // -----------------------------------------------------------------------

    pub(crate) fn roundtrip<T, S, D>(
        src: &ndarray::ArrayBase<S, D>,
        block_shape: &[usize],
    ) -> ndarray::Array<T, D>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let a = Array::compact_ndarray_with(src, arr_params(block_shape)).unwrap();
        a.to_ndarray().unwrap()
    }

    // -----------------------------------------------------------------------
    // Helper: build a BlockTable from pre-arranged typed blocks
    // -----------------------------------------------------------------------

    pub(crate) fn make_block_table<T: Dtyped>(
        blocks: &[&[T]],
    ) -> BlockTable<crate::storage::block::Owned, T> {
        let block_len = blocks[0].len() as BlockSize;
        let data: Vec<u8> = blocks
            .iter()
            .flat_map(|b| unsafe { cast_slice::<T, u8>(b) }.iter().copied())
            .collect();
        BlockTable::build_from_data(&data, T::DTYPE, block_len, &EncoderParams::default()).unwrap()
    }

    pub(crate) fn array<T>(
        blocks: &[&[T]],
        shape: impl IntoDimension,
        block_shape: &[usize],
    ) -> CompactArray<T>
    where
        T: Dtyped,
    {
        let shape = shape.into_dimension().unwrap();
        let shape = shape.as_slice().iter().copied().collect::<DimArray<_>>();
        let block_shape = block_shape
            .iter()
            .map(|&x| x as BlockSize)
            .collect::<DimArray<_>>();
        let params = CompactParams::default().block_shape(&block_shape).unwrap();
        Array::from_storage(Compact(
            ArrayBlockTableStorageBase::new(
                make_block_table(blocks),
                crate::DimDyn::from_slice(&shape),
                params,
            )
            .unwrap(),
        ))
    }

    // -----------------------------------------------------------------------
    // Accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn dtype_shape_ndim() {
        let a = array(&[&[0u8, 1, 2, 3]], [4], &[4]);
        assert_eq!(a.dtype(), &u8::DTYPE);
        assert_eq!(a.shape(), &[4]);
        assert_eq!(a.ndim(), 1);
    }

    #[test]
    fn is_compact_through_wrappers() {
        let make = || Array::compact_ndarray(&array![[1i32, 2], [3, 4]]).unwrap();

        // Wrappers that do not change the data keep reporting the source as compact.
        assert!(make().is_compact());
        assert!(make().view().is_compact());
        assert!(make().view().into_typed_dyn().is_compact());
        assert!(make().into_any().is_compact());
        assert!(make().into_any().into_any().is_compact());
        assert!(make().into_any().into_typed_dyn().is_compact());

        // Lazy ops are not compact, also when type-erased.
        let a = make();
        assert!(!(-a.view()).is_compact());
        assert!(!a.view().reshape(&[4]).is_compact());
        assert!(!(-make()).into_any().is_compact());
    }
}

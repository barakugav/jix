use std::cell::UnsafeCell;
use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::{
    n_operands_mul, n_operands_sum, ArraySpec, ArrayStorageTyped, ElementwisePipelineImpl, Operand,
    StridedBuf,
};
use crate::{array_from_fn_inline, ArrayExt, ArrayStorage, Dimension, ElementType};

/// A sequence of arrays passed to multi-array operations such as [`stack`](crate::ops::stack)
/// and [`concatenate`](crate::ops::concatenate).
///
/// Arrays in the sequence may have different element types or dimensions, both runtime and compile-time.
/// The sub traits [`ArraySequenceElementType`] and [`ArraySequenceDimension`] are used to
/// constrain operations to sequences where all arrays share the same element type and dimension.
///
/// This is a sealed trait - it cannot be implemented outside this crate. It is implemented for the
/// following collection types, covering both homogeneous and heterogeneous cases:
///
/// | Type | Notes |
/// |------|-------|
/// | `[Array<S>; N]` | Fixed-length array; all elements share the same storage type. |
/// | `Vec<Array<S>>` | Dynamic-length list; all elements share the same storage type. |
/// | `&[Array<S>]` | Borrowed slice; all elements share the same storage type. |
/// | `(Array<S0>, Array<S1>, ...)` | Tuple of up to 10 arrays; each element may have a different storage type. |
///
/// # Examples
///
/// Stack with a fixed-length array (homogeneous):
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![4i32, 5, 6])?;
/// let stacked = jix::ops::stack([a, b], 0);
/// assert_eq!(stacked.shape(), &[2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Stack a tuple of arrays with different storage types (heterogeneous):
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![4i32, 5, 6])?;
/// // Lazy view has a different storage type from the original Compact arrays,
/// // but a tuple still implements ArraySequence.
/// let c = a.map(|x| x + 8);
/// let stacked = jix::ops::stack((b, c), 0);
/// assert_eq!(stacked.shape(), &[2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
#[allow(private_bounds)]
pub trait ArraySequence: ArraySequenceImpl {}

/// Private implementation trait for [`ArraySequence`]. Not part of the public API.
pub(crate) trait ArraySequenceImpl {
    fn narrays(&self) -> usize;
    fn read_data<'a>(
        &'a self,
        arr: usize,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>>;
    fn shape(&self, arr: usize) -> &[u64];
    fn dtype(&self, arr: usize) -> &Dtype;
    fn spec(&self, arr: usize) -> ArraySpec<'_>;

    fn as_array_storage(&self, arr: usize) -> &dyn ArrayStorage;
}

/// Subtrait of [`ArraySequence`] for sequences whose arrays all share the same element type.
///
/// Operations such as [`stack`](crate::ops::stack) and [`concatenate`](crate::ops::concatenate)
/// require this bound to guarantee every array has an identical element type, which also determines
/// the element type of the result.
pub trait ArraySequenceElementType: ArraySequence {
    /// The element type of all arrays in the sequence.
    ///
    /// Used in operations like `stack` and `concatenate` to ensure all arrays have the same
    /// element type, and to determine the output element type of the result.
    type ElementType: ElementType;
}

/// Subtrait of [`ArraySequence`] for sequences whose arrays all share the same dimension.
///
/// Operations such as [`stack`](crate::ops::stack) and [`concatenate`](crate::ops::concatenate)
/// require this bound to guarantee every array has the same number of axes, which also determines
/// the dimension of the result.
pub trait ArraySequenceDimension: ArraySequence {
    /// The dimension of all arrays in the sequence.
    ///
    /// Used in operations like `stack` and `concatenate` to ensure all arrays have the same number
    /// of axes, and to determine the output dimension of the result.
    type Dimension: Dimension;
}

/// Subtrait of [`ArraySequence`] for sequences whose arrays all have a statically-known element
/// type (`Ty<T>`), so their elements can be read back as concrete Rust values.
///
/// This is the bound required by element-wise multi-array operations such as
/// [`map_multiple`](crate::ops::map_multiple). Unlike [`ArraySequenceElementType`] it does *not*
/// require every array to share the same element type, which is what allows heterogeneous tuples
/// like `(Array<..i32..>, Array<..f32..>)` to be combined.
#[allow(private_bounds)]
pub trait ArraySequenceTyped: ArraySequence + ArraySequenceTypedImpl<Self> {
    /// The value handed to a per-element closure: one element drawn from each array in the
    /// sequence, grouped by position.
    ///
    /// The concrete type depends on the sequence: `[T; N]` for fixed-length arrays `[Array<S>; N]`
    /// (and references to them), `&[T]` for `Vec<Array<S>>` and `&[Array<S>]` slices, and a tuple
    /// `(S0::Item, S1::Item, ...)` for tuples of arrays.
    type ItemSequence<'a>;
}
pub(crate) trait ArraySequenceTypedImpl<ArraysT: ArraySequenceTyped + ?Sized = Self> {
    /// Read every array of the sequence as one element-wise pipeline.
    ///
    /// Each array contributes its own pipeline, so all their operands stay separate and the whole
    /// sequence is walked in one pass.
    fn read_as_elementwise_pipeline<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipelineTuple<ArraysT> + use<'a, ArraysT, Self>>;
}

/// One element drawn from each array of the sequence, grouped by position.
pub(crate) trait ElementwisePipelineTuple<ArraysT: ArraySequenceTyped + ?Sized> {
    /// How many operands [`operands`](Self::operands) yields across the whole sequence, if that is
    /// known at compile time.
    ///
    /// `None` for a sequence whose length is only known at runtime (`Vec<Array<_>>`,
    /// `&[Array<_>]`), or one whose arrays are themselves pipelines with an unknown operand count.
    const N_OPERANDS: Option<usize>;

    /// The leaf operands the pipeline reads from.
    fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's;

    /// Read the next `N` positions through the operand cursors and advance them, yielding one
    /// `ItemSequence` per position.
    ///
    /// # Safety
    ///
    /// Same contract as [`ElementwisePipelineImpl::read_bulk`]. In addition, the returned
    /// iterator (and anything it yielded) must be dropped before the next call: a runtime-length
    /// sequence groups each position through a scratch buffer that the next call overwrites.
    unsafe fn read_bulk_as_iter<'s, const N: usize, const CONTIGUOUS: bool>(
        &'s self,
    ) -> impl Iterator<Item = ArraysT::ItemSequence<'s>> + 's;
}

impl<S: ArrayStorage, const N: usize> ArraySequence for [Array<S>; N] {}
impl<S, const N: usize> ArraySequenceImpl for [Array<S>; N]
where
    S: ArrayStorage,
{
    #[inline(always)]
    fn narrays(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        arr: usize,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self[arr].storage.read_data(index, context, out)
    }

    #[inline(always)]
    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    #[inline(always)]
    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    #[inline]
    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }

    #[inline]
    fn as_array_storage(&self, arr: usize) -> &dyn ArrayStorage {
        &self[arr].storage
    }
}

impl<S: ArrayStorage, const N: usize> ArraySequenceElementType for [Array<S>; N] {
    type ElementType = S::ElementType;
}
impl<S: ArrayStorage, const N: usize> ArraySequenceDimension for [Array<S>; N] {
    type Dimension = S::Dimension;
}
impl<S: ArrayStorageTyped, const N: usize> ArraySequenceTyped for [Array<S>; N] {
    type ItemSequence<'a> = [S::Item; N];
}
impl<S: ArrayStorageTyped, const N: usize> ArraySequenceTypedImpl for [Array<S>; N] {
    #[inline]
    fn read_as_elementwise_pipeline<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipelineTuple<Self> + use<'a, S, N>> {
        let inners = self.each_ref().try_map_inline(|arr| {
            arr.storage
                .read_as_elementwise_pipeline::<S::Item>(index, context)
        })?;
        struct PipelineTupleImpl<D, const N: usize> {
            inners: [D; N],
        }
        impl<S, D, const N: usize> ElementwisePipelineTuple<[Array<S>; N]> for PipelineTupleImpl<D, N>
        where
            S: ArrayStorageTyped,
            D: ElementwisePipelineImpl<S::Item>,
        {
            const N_OPERANDS: Option<usize> = n_operands_mul(D::N_OPERANDS, N);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inners.iter().flat_map(|inner| inner.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk_as_iter<'s, const M: usize, const CONTIGUOUS: bool>(
                &'s self,
            ) -> impl Iterator<Item = [S::Item; N]> + 's {
                let items = self
                    .inners
                    .each_ref()
                    .map_inline(|inner| unsafe { inner.read_bulk::<M, CONTIGUOUS>() });
                (0..M).map(move |item_idx| {
                    array_from_fn_inline::<_, N>(|arr_idx| items[arr_idx][item_idx])
                })
            }
        }
        Ok(PipelineTupleImpl { inners })
    }
}

impl<S: ArrayStorage, const N: usize> ArraySequence for &[Array<S>; N] {}
impl<S, const N: usize> ArraySequenceImpl for &[Array<S>; N]
where
    S: ArrayStorage,
{
    #[inline(always)]
    fn narrays(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        arr: usize,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self[arr].storage.read_data(index, context, out)
    }

    #[inline(always)]
    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    #[inline(always)]
    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    #[inline]
    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }

    #[inline]
    fn as_array_storage(&self, arr: usize) -> &dyn ArrayStorage {
        &self[arr].storage
    }
}
impl<S: ArrayStorage, const N: usize> ArraySequenceElementType for &[Array<S>; N] {
    type ElementType = S::ElementType;
}
impl<S: ArrayStorage, const N: usize> ArraySequenceDimension for &[Array<S>; N] {
    type Dimension = S::Dimension;
}
impl<S: ArrayStorageTyped, const N: usize> ArraySequenceTyped for &[Array<S>; N] {
    type ItemSequence<'a> = [S::Item; N];
}
impl<'b, S: ArrayStorageTyped, const N: usize> ArraySequenceTypedImpl for &'b [Array<S>; N] {
    #[inline]
    fn read_as_elementwise_pipeline<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipelineTuple<Self> + use<'a, 'b, S, N>> {
        let inners = self.each_ref().try_map_inline(|arr| {
            arr.storage
                .read_as_elementwise_pipeline::<S::Item>(index, context)
        })?;
        struct PipelineTupleImpl<D, const N: usize> {
            inners: [D; N],
        }
        impl<S, D, const N: usize> ElementwisePipelineTuple<&[Array<S>; N]> for PipelineTupleImpl<D, N>
        where
            S: ArrayStorageTyped,
            D: ElementwisePipelineImpl<S::Item>,
        {
            const N_OPERANDS: Option<usize> = n_operands_mul(D::N_OPERANDS, N);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inners.iter().flat_map(|inner| inner.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk_as_iter<'s, const M: usize, const CONTIGUOUS: bool>(
                &'s self,
            ) -> impl Iterator<Item = [S::Item; N]> + 's {
                let items = self
                    .inners
                    .each_ref()
                    .map_inline(|inner| unsafe { inner.read_bulk::<M, CONTIGUOUS>() });
                (0..M).map(move |item_idx| {
                    array_from_fn_inline::<_, N>(|arr_idx| items[arr_idx][item_idx])
                })
            }
        }
        Ok(PipelineTupleImpl { inners })
    }
}

impl<S: ArrayStorage> ArraySequence for Vec<Array<S>> {}
impl<S> ArraySequenceImpl for Vec<Array<S>>
where
    S: ArrayStorage,
{
    #[inline(always)]
    fn narrays(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        arr: usize,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self[arr].storage.read_data(index, context, out)
    }

    #[inline(always)]
    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    #[inline(always)]
    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    #[inline]
    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }
    #[inline]
    fn as_array_storage(&self, arr: usize) -> &dyn ArrayStorage {
        &self[arr].storage
    }
}
impl<S: ArrayStorage> ArraySequenceElementType for Vec<Array<S>> {
    type ElementType = S::ElementType;
}
impl<S: ArrayStorage> ArraySequenceDimension for Vec<Array<S>> {
    type Dimension = S::Dimension;
}
impl<S: ArrayStorageTyped> ArraySequenceTyped for Vec<Array<S>> {
    type ItemSequence<'a> = &'a [S::Item];
}
impl<S: ArrayStorageTyped> ArraySequenceTypedImpl for Vec<Array<S>> {
    #[inline]
    fn read_as_elementwise_pipeline<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipelineTuple<Self> + use<'a, S>> {
        let inners = self
            .iter()
            .map(|arr| {
                arr.storage
                    .read_as_elementwise_pipeline::<S::Item>(index, context)
            })
            .collect::<Result<Vec<_>>>()?;
        struct PipelineTupleImpl<D, T> {
            inners: Vec<D>,
            /// One position's worth of elements per bulk slot, so each `ItemSequence` can be handed
            /// out as a slice. `read_bulk_as_iter` takes `&self` - the operand cursors live in
            /// `Cell`s - so the scratch needs interior mutability.
            tmp_buf: UnsafeCell<Vec<T>>,
        }
        impl<S, D> ElementwisePipelineTuple<Vec<Array<S>>> for PipelineTupleImpl<D, S::Item>
        where
            S: ArrayStorageTyped,
            D: ElementwisePipelineImpl<S::Item>,
        {
            // The number of arrays in the sequence is only known at runtime.
            const N_OPERANDS: Option<usize> = None;

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inners.iter().flat_map(|inner| inner.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk_as_iter<'s, const M: usize, const CONTIGUOUS: bool>(
                &'s self,
            ) -> impl Iterator<Item = &'s [S::Item]> + 's {
                let narrays = self.inners.len();
                // SAFETY: the caller must have dropped everything the previous call yielded, so
                // this is the only live reference to the scratch. It is downgraded to a shared
                // slice below and never re-borrowed mutably within this call.
                let tmp_buf = unsafe { &mut *self.tmp_buf.get() };
                tmp_buf.clear();
                tmp_buf.reserve(narrays * M);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    tmp_buf.set_len(narrays * M)
                };

                for (arr, inner) in self.inners.iter().enumerate() {
                    let items = unsafe { inner.read_bulk::<M, CONTIGUOUS>() };
                    for (item_idx, item) in items.into_iter().enumerate() {
                        tmp_buf[item_idx * narrays + arr] = item;
                    }
                }

                let tmp_buf: &'s [S::Item] = tmp_buf;
                array_from_fn_inline::<_, M>(|item_idx| &tmp_buf[item_idx * narrays..][..narrays])
                    .into_iter()
            }
        }
        Ok(PipelineTupleImpl {
            inners,
            tmp_buf: UnsafeCell::new(Vec::new()),
        })
    }
}

impl<S: ArrayStorage> ArraySequence for &[Array<S>] {}
impl<S> ArraySequenceImpl for &[Array<S>]
where
    S: ArrayStorage,
{
    #[inline(always)]
    fn narrays(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        arr: usize,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self[arr].storage.read_data(index, context, out)
    }

    #[inline(always)]
    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    #[inline(always)]
    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    #[inline]
    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }
    #[inline]
    fn as_array_storage(&self, arr: usize) -> &dyn ArrayStorage {
        &self[arr].storage
    }
}
impl<S: ArrayStorage> ArraySequenceElementType for &[Array<S>] {
    type ElementType = S::ElementType;
}
impl<S: ArrayStorage> ArraySequenceDimension for &[Array<S>] {
    type Dimension = S::Dimension;
}
impl<S: ArrayStorageTyped> ArraySequenceTyped for &[Array<S>] {
    type ItemSequence<'a> = &'a [S::Item];
}
impl<'b, S: ArrayStorageTyped> ArraySequenceTypedImpl for &'b [Array<S>] {
    #[inline]
    fn read_as_elementwise_pipeline<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipelineTuple<Self> + use<'a, 'b, S>> {
        let inners = self
            .iter()
            .map(|arr| {
                arr.storage
                    .read_as_elementwise_pipeline::<S::Item>(index, context)
            })
            .collect::<Result<Vec<_>>>()?;
        struct PipelineTupleImpl<D, T> {
            inners: Vec<D>,
            /// One position's worth of elements per bulk slot, so each `ItemSequence` can be handed
            /// out as a slice. `read_bulk_as_iter` takes `&self` - the operand cursors live in
            /// `Cell`s - so the scratch needs interior mutability.
            tmp_buf: UnsafeCell<Vec<T>>,
        }
        impl<S, D> ElementwisePipelineTuple<&[Array<S>]> for PipelineTupleImpl<D, S::Item>
        where
            S: ArrayStorageTyped,
            D: ElementwisePipelineImpl<S::Item>,
        {
            // The number of arrays in the sequence is only known at runtime.
            const N_OPERANDS: Option<usize> = None;

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.inners.iter().flat_map(|inner| inner.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk_as_iter<'s, const M: usize, const CONTIGUOUS: bool>(
                &'s self,
            ) -> impl Iterator<Item = &'s [S::Item]> + 's {
                let narrays = self.inners.len();
                // SAFETY: the caller must have dropped everything the previous call yielded, so
                // this is the only live reference to the scratch. It is downgraded to a shared
                // slice below and never re-borrowed mutably within this call.
                let tmp_buf = unsafe { &mut *self.tmp_buf.get() };
                tmp_buf.clear();
                tmp_buf.reserve(narrays * M);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    tmp_buf.set_len(narrays * M)
                };

                for (arr, inner) in self.inners.iter().enumerate() {
                    let items = unsafe { inner.read_bulk::<M, CONTIGUOUS>() };
                    for (item_idx, item) in items.into_iter().enumerate() {
                        tmp_buf[item_idx * narrays + arr] = item;
                    }
                }

                let tmp_buf: &'s [S::Item] = tmp_buf;
                array_from_fn_inline::<_, M>(|item_idx| &tmp_buf[item_idx * narrays..][..narrays])
                    .into_iter()
            }
        }
        Ok(PipelineTupleImpl {
            inners,
            tmp_buf: UnsafeCell::new(Vec::new()),
        })
    }
}

macro_rules! impl_array_sequence_for_tuple {
    ($($idx:tt : $S:ident, $D:ident),+ $(,)?) => {
        impl<$($S),+> ArraySequence for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage,)+
        {
        }
        impl<$($S),+> ArraySequenceImpl for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage,)+
        {
            #[inline(always)]
            fn narrays(&self) -> usize {
                impl_array_sequence_for_tuple!(@count $($idx)+)
            }

            #[inline(always)]
            fn read_data<'read>(
                &'read self,
                arr: usize,
                index: &[Range<u64>],
                context: &'read ReadContext,
                out: Option<&'read mut StridedBuf<'_>>,
            ) -> Result<StridedBuf<'read>> {
                match arr {
                    $($idx => self.$idx.storage.read_data(index, context, out),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            #[inline(always)]
            fn shape(&self, arr: usize) -> &[u64] {
                match arr {
                    $($idx => self.$idx.shape(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            #[inline(always)]
            fn dtype(&self, arr: usize) -> &Dtype {
                match arr {
                    $($idx => self.$idx.dtype(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            #[inline]
            fn spec(&self, arr: usize) -> ArraySpec<'_> {
                match arr {
                    $($idx => self.$idx.storage.spec(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            #[inline]
            fn as_array_storage(&self, arr: usize) -> &dyn ArrayStorage {
                match arr {
                    $($idx => &self.$idx.storage,)+
                    _ => out_of_bounds_array_index(arr),
                }
            }
        }
        impl<$($S),+, ET> ArraySequenceElementType for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage<ElementType = ET>,)+
            ET: ElementType,
        {
            type ElementType = ET;
        }
        impl<$($S),+, D> ArraySequenceDimension for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage<Dimension = D>,)+
            D: Dimension,
        {
            type Dimension = D;
        }
        impl<$($S),+> ArraySequenceTyped for ($(Array<$S>,)+)
        where
            $($S: ArrayStorageTyped,)+
        {
            type ItemSequence<'a> = ($($S::Item,)+);
        }
        impl<$($S),+> ArraySequenceTypedImpl for ($(Array<$S>,)+)
        where
            $($S: ArrayStorageTyped,)+
        {
            #[inline]
            fn read_as_elementwise_pipeline<'a>(
                &'a self,
                index: &[Range<u64>],
                context: &'a ReadContext,
            ) -> Result<impl ElementwisePipelineTuple<Self> + use<'a, $($S),+>> {
                struct PipelineTupleImpl<$($D),+>($($D),+);
                impl<$($S),+, $($D),+> ElementwisePipelineTuple<($(Array<$S>,)+)> for PipelineTupleImpl<$($D),+>
                where
                    $($S: ArrayStorageTyped,)+
                    $($D: ElementwisePipelineImpl<$S::Item>,)+
                {
                    const N_OPERANDS: Option<usize> = n_operands_sum(&[$($D::N_OPERANDS),+]);

                    #[inline(always)]
                    fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                        std::iter::empty()
                            $(.chain(self.$idx.operands()))+
                    }

                    #[inline(always)]
                    unsafe fn read_bulk_as_iter<'s, const N: usize, const CONTIGUOUS: bool>(
                        &'s self,
                    ) -> impl Iterator<Item = ($($S::Item,)+)> + 's {
                        let items = ($(
                            unsafe { self.$idx.read_bulk::<N, CONTIGUOUS>() },
                        )+);
                        (0..N).map(move |item_idx| {
                            ($(items.$idx[item_idx],)+)
                        })
                    }
                }
                Ok(PipelineTupleImpl (
                    $(
                        self.$idx.storage.read_as_elementwise_pipeline::<$S::Item>(index, context)?
                    ),+
                ))
            }
        }
    };

    (@count $($t:tt)+) => {
        0 $(+ impl_array_sequence_for_tuple!(@replace $t 1))+
    };
    (@replace $_t:tt $sub:expr) => { $sub };
}

impl_array_sequence_for_tuple!(0: S0, D0);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3, 4: S4, D4);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3, 4: S4, D4, 5: S5, D5);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3, 4: S4, D4, 5: S5, D5, 6: S6, D6);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3, 4: S4, D4, 5: S5, D5, 6: S6, D6, 7: S7, D7);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3, 4: S4, D4, 5: S5, D5, 6: S6, D6, 7: S7, D7, 8: S8, D8);
impl_array_sequence_for_tuple!(0: S0, D0, 1: S1, D1, 2: S2, D2, 3: S3, D3, 4: S4, D4, 5: S5, D5, 6: S6, D6, 7: S7, D7, 8: S8, D8, 9: S9, D9);

#[cold]
#[inline(never)]
fn out_of_bounds_array_index(arr: usize) -> ! {
    panic!("array index out of bounds: {}", arr);
}

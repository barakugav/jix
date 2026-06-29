use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::{ArraySpec, ArrayStorageTyped, ReadData};
use crate::{ArrayExt, ArrayStorage, Dimension, ElementType, OutBuf};

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
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()>;
    fn shape(&self, arr: usize) -> &[u64];
    fn dtype(&self, arr: usize) -> &Dtype;
    fn spec(&self, arr: usize) -> ArraySpec<'_>;
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
    fn read_data_typed<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadDataTuple<ArraysT> + use<'a, ArraysT, Self>>;
}

pub(crate) trait ReadDataTuple<ArraysT: ArraySequenceTyped + ?Sized> {
    fn len(&self) -> usize;

    #[allow(unused)]
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn read_bulk_as_iter<'a, const N: usize>(
        &'a mut self,
        offset: usize,
    ) -> impl Iterator<Item = ArraysT::ItemSequence<'a>> + 'a
    where
        Self: Sized;
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
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
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
    fn read_data_typed<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadDataTuple<Self> + use<'a, S, N>> {
        let data = self
            .each_ref()
            .try_map_(|arr| arr.storage.read_data_typed::<S::Item>(index, context))?;
        struct ReadDataTupleImpl<D, const N: usize> {
            data: [D; N],
        }
        impl<S, D, const N: usize> ReadDataTuple<[Array<S>; N]> for ReadDataTupleImpl<D, N>
        where
            S: ArrayStorageTyped,
            D: ReadData<S::Item>,
        {
            fn len(&self) -> usize {
                self.data.first().map_or(0, |d| d.len())
            }

            fn read_bulk_as_iter<'a, const M: usize>(
                &'a mut self,
                offset: usize,
            ) -> impl Iterator<Item = [S::Item; N]> + 'a
            where
                Self: Sized,
            {
                let items = self.data.each_mut().map(|data| data.read_bulk::<M>(offset));
                (0..M).map(move |item_idx| {
                    std::array::from_fn::<_, N, _>(|arr_idx| items[arr_idx][item_idx])
                })
            }
        }
        Ok(ReadDataTupleImpl { data })
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
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
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
    fn read_data_typed<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadDataTuple<Self> + use<'a, 'b, S, N>> {
        let data = self
            .each_ref()
            .try_map_(|arr| arr.storage.read_data_typed::<S::Item>(index, context))?;
        struct ReadDataTupleImpl<D, const N: usize> {
            data: [D; N],
        }
        impl<S, D, const N: usize> ReadDataTuple<&[Array<S>; N]> for ReadDataTupleImpl<D, N>
        where
            S: ArrayStorageTyped,
            D: ReadData<S::Item>,
        {
            fn len(&self) -> usize {
                self.data.first().map_or(0, |d| d.len())
            }

            fn read_bulk_as_iter<'a, const M: usize>(
                &'a mut self,
                offset: usize,
            ) -> impl Iterator<Item = [S::Item; N]> + 'a
            where
                Self: Sized,
            {
                let items = self.data.each_mut().map(|data| data.read_bulk::<M>(offset));
                (0..M).map(move |item_idx| {
                    std::array::from_fn::<_, N, _>(|arr_idx| items[arr_idx][item_idx])
                })
            }
        }
        Ok(ReadDataTupleImpl { data })
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
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
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
    fn read_data_typed<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadDataTuple<Self> + use<'a, S>> {
        let data = self
            .iter()
            .map(|arr| arr.storage.read_data_typed::<S::Item>(index, context))
            .collect::<Result<Vec<_>>>()?;
        struct ReadDataTupleImpl<D, T> {
            data: Vec<D>,
            tmp_buf: Vec<T>,
        }
        impl<S, D> ReadDataTuple<Vec<Array<S>>> for ReadDataTupleImpl<D, S::Item>
        where
            S: ArrayStorageTyped,
            D: ReadData<S::Item>,
        {
            fn len(&self) -> usize {
                self.data.first().map_or(0, |d| d.len())
            }

            fn read_bulk_as_iter<'a, const M: usize>(
                &'a mut self,
                offset: usize,
            ) -> impl Iterator<Item = &'a [S::Item]> + 'a
            where
                Self: Sized,
            {
                let narrays = self.data.len();
                self.tmp_buf.clear();
                self.tmp_buf.reserve(narrays * M);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    self.tmp_buf.set_len(narrays * M)
                };
                let tmp_buf = self.tmp_buf.as_mut_slice();

                for (arr, data) in self.data.iter_mut().enumerate() {
                    for (item_idx, item) in data.read_bulk::<M>(offset).into_iter().enumerate() {
                        tmp_buf[item_idx * narrays + arr] = item;
                    }
                }

                std::array::from_fn::<_, M, _>(|item_idx| &tmp_buf[item_idx * narrays..][..narrays])
                    .into_iter()
            }
        }
        Ok(ReadDataTupleImpl {
            data,
            tmp_buf: Vec::new(),
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
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
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
    fn read_data_typed<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadDataTuple<Self> + use<'a, 'b, S>> {
        let data = self
            .iter()
            .map(|arr| arr.storage.read_data_typed::<S::Item>(index, context))
            .collect::<Result<Vec<_>>>()?;
        struct ReadDataTupleImpl<D, T> {
            data: Vec<D>,
            tmp_buf: Vec<T>,
        }
        impl<S, D> ReadDataTuple<&[Array<S>]> for ReadDataTupleImpl<D, S::Item>
        where
            S: ArrayStorageTyped,
            D: ReadData<S::Item>,
        {
            fn len(&self) -> usize {
                self.data.first().map_or(0, |d| d.len())
            }

            fn read_bulk_as_iter<'a, const M: usize>(
                &'a mut self,
                offset: usize,
            ) -> impl Iterator<Item = &'a [S::Item]> + 'a
            where
                Self: Sized,
            {
                let narrays = self.data.len();
                self.tmp_buf.clear();
                self.tmp_buf.reserve(narrays * M);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    self.tmp_buf.set_len(narrays * M)
                };
                let tmp_buf = self.tmp_buf.as_mut_slice();

                for (arr, data) in self.data.iter_mut().enumerate() {
                    for (item_idx, item) in data.read_bulk::<M>(offset).into_iter().enumerate() {
                        tmp_buf[item_idx * narrays + arr] = item;
                    }
                }

                std::array::from_fn::<_, M, _>(|item_idx| &tmp_buf[item_idx * narrays..][..narrays])
                    .into_iter()
            }
        }
        Ok(ReadDataTupleImpl {
            data,
            tmp_buf: Vec::new(),
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
            fn read_data(
                &self,
                arr: usize,
                index: &[Range<u64>],
                buf: &mut OutBuf,
                context: &ReadContext,
            ) -> Result<()> {
                match arr {
                    $($idx => self.$idx.storage.read_data(index, buf, context),)+
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
            fn read_data_typed<'a>(
                &'a self,
                index: &[Range<u64>],
                context: &'a ReadContext,
            ) -> Result<impl ReadDataTuple<Self> + use<'a, $($S),+>> {
                struct ReadDataTupleImpl<$($D),+>($($D),+);
                impl<$($S),+, $($D),+> ReadDataTuple<($(Array<$S>,)+)> for ReadDataTupleImpl<$($D),+>
                where
                    $($S: ArrayStorageTyped,)+
                    $($D: ReadData<$S::Item>,)+
                {
                    fn len(&self) -> usize {
                        self.0.len()
                    }

                    fn read_bulk_as_iter<'a, const N: usize>(&'a mut self, offset: usize) -> impl Iterator<Item = ($($S::Item,)+)> + 'a
                    where
                        Self: Sized,
                    {
                        let items = ($(
                            self.$idx.read_bulk::<N>(offset),
                        )+);
                        (0..N).map(move |item_idx| {
                            ($(items.$idx[item_idx],)+)
                        })
                    }
                }
                Ok(ReadDataTupleImpl (
                    $(
                        self.$idx.storage.read_data_typed::<$S::Item>(index, context)?
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

use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::ArraySpec;
use crate::{ArrayStorage, Dimension, ElementType};

/// Private implementation trait for [`ArraySequence`]. Not part of the public API.
pub(crate) trait ArraySequenceImpl {
    fn narrays(&self) -> usize;
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()>;
    fn shape(&self, arr: usize) -> &[u64];
    fn dtype(&self, arr: usize) -> &Dtype;
    fn spec(&self, arr: usize) -> ArraySpec<'_>;
}

/// A sequence of arrays passed to multi-array operations such as [`stack`](crate::ops::stack)
/// and [`concatenate`](crate::ops::concatenate).
///
/// All arrays in the sequence must have the same element type and dimension, but may have
/// different storage types.
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
pub trait ArraySequence: ArraySequenceImpl {
    /// The element type of all arrays in the sequence.
    ///
    /// Used in operations like `stack` and `concatenate` to ensure all arrays have the same
    /// element type, and to determine the output element type of the result.
    type ElementType: ElementType;

    /// The dimension of all arrays in the sequence.
    ///
    /// Used in operations like `stack` and `concatenate` to ensure all arrays have the same number
    /// of axes, and to determine the output dimension of the result.
    type Dimension: Dimension;
}

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
        buf: &mut [u8],
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

    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }
}
impl<S: ArrayStorage, const N: usize> ArraySequence for [Array<S>; N] {
    type Dimension = S::Dimension;
    type ElementType = S::ElementType;
}

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
        buf: &mut [u8],
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

    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }
}
impl<S: ArrayStorage> ArraySequence for Vec<Array<S>> {
    type Dimension = S::Dimension;
    type ElementType = S::ElementType;
}

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
        buf: &mut [u8],
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

    fn spec(&self, arr: usize) -> ArraySpec<'_> {
        self[arr].storage.spec()
    }
}
impl<S: ArrayStorage> ArraySequence for &[Array<S>] {
    type Dimension = S::Dimension;
    type ElementType = S::ElementType;
}

macro_rules! impl_array_sequence_for_tuple {
    ($($idx:tt : $S:ident),+ $(,)?) => {
        impl<$($S),+, ET, D> ArraySequence for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage<ElementType = ET, Dimension = D>,)+
            ET: ElementType,
            D: Dimension,
        {
            type ElementType = ET;
            type Dimension = D;
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
                buf: &mut [u8],
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

            fn spec(&self, arr: usize) -> ArraySpec<'_> {
                match arr {
                    $($idx => self.$idx.storage.spec(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }
        }
    };

    (@count $($t:tt)+) => {
        0 $(+ impl_array_sequence_for_tuple!(@replace $t 1))+
    };
    (@replace $_t:tt $sub:expr) => { $sub };
}

impl_array_sequence_for_tuple!(0: S0);
impl_array_sequence_for_tuple!(0: S0, 1: S1);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3, 4: S4);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5, 6: S6);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5, 6: S6, 7: S7);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5, 6: S6, 7: S7, 8: S8);
impl_array_sequence_for_tuple!(0: S0, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5, 6: S6, 7: S7, 8: S8, 9: S9);

#[cold]
#[inline(never)]
fn out_of_bounds_array_index(arr: usize) -> ! {
    panic!("array index out of bounds: {}", arr);
}

use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::storage::{ArrayStorage, ArrayStorageSpec};
use crate::{Dimension, ElementType};

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
    fn _spec(&self, arr: usize) -> ArrayStorageSpec<'_>;
}

/// A sequence of arrays passed to multi-array operations such as [`stack`](crate::ops::stack)
/// and [`concatenate`](crate::ops::concatenate).
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
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![1i32, 2, 3])?;
/// let b = Array::compact_array(&array![4i32, 5, 6])?;
/// let stacked = zix::ops::stack([a, b], 0);
/// assert_eq!(stacked.shape(), &[2, 3]);
/// # Ok::<(), zix::Error>(())
/// ```
///
/// Stack a tuple of arrays with different storage types (heterogeneous):
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![1i32, 2, 3])?;
/// let b = Array::compact_array(&array![4i32, 5, 6])?;
/// // Lazy view has a different storage type from the original Compact arrays,
/// // but a tuple still implements ArraySequence.
/// let c = a + 8;
/// let stacked = zix::ops::stack((b, c), 0);
/// assert_eq!(stacked.shape(), &[2, 3]);
/// # Ok::<(), zix::Error>(())
/// ```
#[allow(private_bounds)]
pub trait ArraySequence: ArraySequenceImpl {
    /// The compile-time dimension of the first array in the sequence.
    ///
    /// This is used by operations like `stack` and `concatenate` to determine the output dimension
    /// of the result, which is always derived from the first array's dimension.
    type FirstArrayDimension: Dimension;

    ///
    type FirstArrayElementType: ElementType;
}

impl<S, const N: usize> ArraySequenceImpl for [Array<S>; N]
where
    S: ArrayStorage,
{
    fn narrays(&self) -> usize {
        self.len()
    }

    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
    }

    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    fn _spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
        self[arr].storage._spec()
    }
}
impl<S, const N: usize> ArraySequence for [Array<S>; N]
where
    S: ArrayStorage,
{
    type FirstArrayDimension = S::Dimension;
    type FirstArrayElementType = S::ElementType;
}

impl<S> ArraySequenceImpl for Vec<Array<S>>
where
    S: ArrayStorage,
{
    fn narrays(&self) -> usize {
        self.len()
    }

    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
    }

    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    fn _spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
        self[arr].storage._spec()
    }
}
impl<S: ArrayStorage> ArraySequence for Vec<Array<S>> {
    type FirstArrayDimension = S::Dimension;
    type FirstArrayElementType = S::ElementType;
}

impl<S> ArraySequenceImpl for &[Array<S>]
where
    S: ArrayStorage,
{
    fn narrays(&self) -> usize {
        self.len()
    }

    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> Result<()> {
        self[arr].storage.read_data(index, buf, context)
    }

    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].shape()
    }

    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].dtype()
    }

    fn _spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
        self[arr].storage._spec()
    }
}
impl<S: ArrayStorage> ArraySequence for &[Array<S>] {
    type FirstArrayDimension = S::Dimension;
    type FirstArrayElementType = S::ElementType;
}

macro_rules! impl_array_sequence_for_tuple {
    ($($idx:tt : $S:ident),+ $(,)?) => {
        impl<$($S),+> ArraySequence for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage,)+
        {
            type FirstArrayDimension = S0::Dimension;
            type FirstArrayElementType = S0::ElementType;
        }
        impl<$($S),+> ArraySequenceImpl for ($(Array<$S>,)+)
        where
            $($S: ArrayStorage,)+
        {
            fn narrays(&self) -> usize {
                impl_array_sequence_for_tuple!(@count $($idx)+)
            }

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

            fn shape(&self, arr: usize) -> &[u64] {
                match arr {
                    $($idx => self.$idx.shape(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            fn dtype(&self, arr: usize) -> &Dtype {
                match arr {
                    $($idx => self.$idx.dtype(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            fn _spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
                match arr {
                    $($idx => self.$idx.storage._spec(),)+
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

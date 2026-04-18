use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::storage::{ArrayStorage, ArrayStorageSpec};

pub(crate) trait ArraySequenceItemImpl {
    type __Storage: ArrayStorage;
    fn __storage(&self) -> &Self::__Storage;
}
#[allow(private_bounds)]
pub trait ArraySequenceItem: ArraySequenceItemImpl {}

impl<S: ArrayStorage> ArraySequenceItemImpl for Array<S> {
    type __Storage = S;
    fn __storage(&self) -> &Self::__Storage {
        &self.storage
    }
}
impl<S: ArrayStorage> ArraySequenceItem for Array<S> {}

impl<S: ArrayStorage> ArraySequenceItemImpl for &Array<S> {
    type __Storage = S;
    fn __storage(&self) -> &Self::__Storage {
        &self.storage
    }
}
impl<S: ArrayStorage> ArraySequenceItem for &Array<S> {}

pub(crate) trait ArraySequenceImpl {
    fn narrays(&self) -> usize;
    fn read_data(
        &self,
        arr: usize,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()>;
    fn shape(&self, arr: usize) -> &[u64];
    fn dtype(&self, arr: usize) -> &Dtype;
    fn spec(&self, arr: usize) -> ArrayStorageSpec<'_>;
}

#[allow(private_bounds)]
pub trait ArraySequence: ArraySequenceImpl {}

impl<A, const N: usize> ArraySequenceImpl for [A; N]
where
    A: ArraySequenceItem,
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
    ) -> io::Result<()> {
        self[arr].__storage().read_data(index, buf, context)
    }

    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].__storage().shape()
    }

    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].__storage().dtype()
    }

    fn spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
        self[arr].__storage().spec()
    }
}
impl<A, const N: usize> ArraySequence for [A; N] where A: ArraySequenceItem {}

impl<A> ArraySequenceImpl for Vec<A>
where
    A: ArraySequenceItem,
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
    ) -> io::Result<()> {
        self[arr].__storage().read_data(index, buf, context)
    }

    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].__storage().shape()
    }

    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].__storage().dtype()
    }

    fn spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
        self[arr].__storage().spec()
    }
}
impl<A: ArraySequenceItem> ArraySequence for Vec<A> {}

impl<A> ArraySequenceImpl for &[A]
where
    A: ArraySequenceItem,
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
    ) -> io::Result<()> {
        self[arr].__storage().read_data(index, buf, context)
    }

    fn shape(&self, arr: usize) -> &[u64] {
        self[arr].__storage().shape()
    }

    fn dtype(&self, arr: usize) -> &Dtype {
        self[arr].__storage().dtype()
    }

    fn spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
        self[arr].__storage().spec()
    }
}
impl<A: ArraySequenceItem> ArraySequence for &[A] {}

macro_rules! impl_array_sequence_for_tuple {
    ($($idx:tt : $A:ident),+ $(,)?) => {
        impl<$($A),+> ArraySequence for ($($A,)+)
        where
            $($A: ArraySequenceItem,)+
        {}
        impl<$($A),+> ArraySequenceImpl for ($($A,)+)
        where
            $($A: ArraySequenceItem,)+
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
            ) -> io::Result<()> {
                match arr {
                    $($idx => ArraySequenceItemImpl::__storage(&self.$idx).read_data(index, buf, context),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            fn shape(&self, arr: usize) -> &[u64] {
                match arr {
                    $($idx => ArraySequenceItemImpl::__storage(&self.$idx).shape(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            fn dtype(&self, arr: usize) -> &Dtype {
                match arr {
                    $($idx => ArraySequenceItemImpl::__storage(&self.$idx).dtype(),)+
                    _ => out_of_bounds_array_index(arr),
                }
            }

            fn spec(&self, arr: usize) -> ArrayStorageSpec<'_> {
                match arr {
                    $($idx => ArraySequenceItemImpl::__storage(&self.$idx).spec(),)+
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

impl_array_sequence_for_tuple!(0: A0);
impl_array_sequence_for_tuple!(0: A0, 1: A1);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3, 4: A4);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5, 6: A6);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5, 6: A6, 7: A7);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5, 6: A6, 7: A7, 8: A8);
impl_array_sequence_for_tuple!(0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5, 6: A6, 7: A7, 8: A8, 9: A9);

#[cold]
#[inline(never)]
fn out_of_bounds_array_index(arr: usize) -> ! {
    panic!("array index out of bounds: {}", arr);
}

use crate::dtype::{DtypeScalarKind, Dtyped};
use crate::scalar::Complex;
use crate::{DimDyn, Dimension};

macro_rules! define_array_op1_method {
    ($method:ident : $Op:ident, $($trait:ident)::+) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self) -> crate::Array<$Op<S>>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
            <S::Item as $($trait)::+>::Output: crate::dtype::Dtyped,
        {
            $Op::new_array(self).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+, fixed_output_type = true) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self) -> crate::Array<$Op<S>>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+,
        {
            $Op::new_array(self).unwrap()
        }
    };
}
macro_rules! define_array_op2_method {
    ($method:ident : $Op:ident, $($trait:ident)::+) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S2: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<S2::Item>,
            <S::Item as $($trait)::+<S2::Item>>::Output: crate::dtype::Dtyped,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+, fixed_output_type = true) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S2: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<S2::Item>,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+, fixed_lhs_type = $lhs_type:ty) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::ArrayStorage + crate::storage::ArrayStorageTyped,
            S2: crate::ArrayStorage + crate::storage::ArrayStorageTyped<Item = $lhs_type>,
            S::Item: $($trait)::+,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
}

// macro_rules! or_else {
//     ($( { $($optional:tt)+ } )? or { $($else:tt)+ }) => {
//         crate::ops::common::or_else!(@impl_ $( { $($optional)+ } )? or { $($else)* });
//     };
//     (@impl_ { $($optional:tt)+ } or { $($else:tt)* }) => {
//         $($optional)*
//     };
//     (@impl_ or { $($else:tt)* }) => {
//         $($else)*
//     };
// }
// macro_rules! if_none {
//     ($( { $($optional:tt)+ } )? than { $($else:tt)+ }) => {
//         crate::ops::common::if_none!(@impl_ $( { $($optional)+ } )? than { $($else)* });
//     };
//     (@impl_ { $($optional:tt)+ } than { $($else:tt)* }) => {
//     };
//     (@impl_ than { $($else:tt)* }) => {
//         $($else)*
//     };
// }

pub(crate) use {define_array_op1_method, define_array_op2_method};

// TODO: remove
pub(crate) trait BulkInfo {
    const BULK: usize;
}
macro_rules! impl_bulk_info {
    ($ty:ty, $bulk:expr) => {
        impl BulkInfo for $ty {
            const BULK: usize = $bulk;
        }
    };
}
impl_bulk_info!(i8, 128 / size_of::<i8>());
impl_bulk_info!(i16, 128 / size_of::<i16>());
impl_bulk_info!(i32, 128 / size_of::<i32>());
impl_bulk_info!(i64, 128 / size_of::<i64>());
impl_bulk_info!(u8, 128 / size_of::<u8>());
impl_bulk_info!(u16, 128 / size_of::<u16>());
impl_bulk_info!(u32, 128 / size_of::<u32>());
impl_bulk_info!(u64, 128 / size_of::<u64>());
impl_bulk_info!(crate::scalar::f16, 128 / size_of::<crate::scalar::f16>());
impl_bulk_info!(f32, 128 / size_of::<f32>());
impl_bulk_info!(f64, 128 / size_of::<f64>());
impl_bulk_info!(Complex<f32>, 128 / size_of::<Complex<f32>>());
impl_bulk_info!(Complex<f64>, 128 / size_of::<Complex<f64>>());
impl_bulk_info!(bool, 128 / size_of::<bool>());

#[inline(always)]
pub(crate) fn bulk_size<T: Dtyped>() -> usize {
    // this is a compile time check, the compiler knows the value of `T::DTYPE.try_to_scalar()`
    match T::DTYPE.try_to_scalar() {
        Some(scalar) => match scalar {
            DtypeScalarKind::I8 => i8::BULK,
            DtypeScalarKind::I16 => i16::BULK,
            DtypeScalarKind::I32 => i32::BULK,
            DtypeScalarKind::I64 => i64::BULK,
            DtypeScalarKind::U8 => u8::BULK,
            DtypeScalarKind::U16 => u16::BULK,
            DtypeScalarKind::U32 => u32::BULK,
            DtypeScalarKind::U64 => u64::BULK,
            DtypeScalarKind::F16 => crate::scalar::f16::BULK,
            DtypeScalarKind::F32 => f32::BULK,
            DtypeScalarKind::F64 => f64::BULK,
            DtypeScalarKind::ComplexF32 => Complex::<f32>::BULK,
            DtypeScalarKind::ComplexF64 => Complex::<f64>::BULK,
            DtypeScalarKind::Bool => bool::BULK,
        },
        None => {
            if size_of::<T>() == 0 {
                return 1024;
            }
            (128 / size_of::<T>()).next_power_of_two()
        }
    }
}
// #[inline(always)]
// pub(crate) fn bulk_size2<T1: Dtyped, T2: Dtyped>() -> usize {
//     let (bs1, bs2) = (bulk_size::<T1>(), bulk_size::<T2>());
//     if bs1 < bs2 {
//         bs1
//     } else {
//         bs2
//     }
// }

/// An argument that specifies a set of axis indices, encoding the dimension change in the type.
///
/// Operations that add or remove axes — `insert_axis`, `remove_axis`, `sum`, `max`, etc. — are
/// generic over `Ax: AxesArg`. The associated types `ReducedDimension` and `ExpandedDimension`
/// compute the output dimension purely from the input dimension `D` and the concrete type of the
/// axis argument. This means the compiler knows the output ndim without any runtime information.
///
/// # Dimension rules by argument type
///
/// | `Ax` type | `ReducedDimension<D>` | `ExpandedDimension<D>` |
/// |---|---|---|
/// | `usize` | `D::Smaller` | `D::Larger` |
/// | `[usize; 0]` / `()` | `D` | `D` |
/// | `[usize; N]` / `(usize, ...)` (N ≥ 1) | `D::Smaller` repeated N times | `D::Larger` repeated N times |
/// | `&[usize; N]` (any N) | same as fixed array | same as fixed array |
/// | `&[usize]` / `&Vec<usize>` | `DimDyn` | `DimDyn` |
///
/// When the argument is a slice reference (`&[usize]`), the ndim of the result is only known at
/// runtime, so both associated types resolve to [`DimDyn`](crate::DimDyn). Using a typed array
/// argument (`[usize; N]` or a tuple) preserves static dimension information.
///
/// # Examples
///
/// ```rust
/// use zix::{Array, Dim, ArrayParams};
/// use ndarray::array;
///
/// let a = Array::compact_array(&array![[1i32, 2], [3, 4]])?;
///
/// // usize: one axis removed/added — statically one smaller/larger
/// let b = a.as_ref().sum(0);               // D::Smaller when a: Dim<N> → Dim<N-1>
/// let c = a.as_ref().insert_axis(0);       // D::Larger  when a: Dim<N> → Dim<N+1>
///
/// // [usize; 2]: two axes removed — statically two smaller
/// let b = a.as_ref().sum([0, 1]);          // Dim<N-2>
/// let c = a.as_ref().insert_axis([0, 1]);  // Dim<N+2>
///
/// // &[usize]: dynamic count → DimDyn regardless of input dimension
/// let axes = vec![0, 1];
/// let b = a.as_ref().sum(axes.as_slice()); // DimDyn
/// # Ok::<(), zix::Error>(())
/// ```
#[allow(clippy::len_without_is_empty)]
pub trait AxesArg {
    /// The dimension type of an array produced by a *reducing* operation (e.g. `sum`, `max`,
    /// `remove_axis`) that removes the axes described by `self` from an input of dimension `D`.
    ///
    /// For a single-axis arg (`usize`), this is `D::Smaller`.
    /// For an N-element fixed arg (`[usize; N]` or N-tuple), this is `D::Smaller` applied N times.
    /// For a slice arg (`&[usize]`), this is `DimDyn`.
    type ReducedDimension<D: Dimension>: Dimension;

    /// The dimension type of an array produced by an *expanding* operation (e.g. `insert_axis`)
    /// that inserts the axes described by `self` into an input of dimension `D`.
    ///
    /// For a single-axis arg (`usize`), this is `D::Larger`.
    /// For an N-element fixed arg (`[usize; N]` or N-tuple), this is `D::Larger` applied N times.
    /// For a slice arg (`&[usize]`), this is `DimDyn`.
    type ExpandedDimension<D: Dimension>: Dimension;

    /// The number of axes described by this argument.
    fn len(&self) -> usize;

    /// The axis at position `idx` in this argument.
    ///
    /// `idx` must be less than `self.len()`.
    fn get(&self, idx: usize) -> usize;
}

impl AxesArg for usize {
    type ReducedDimension<D: Dimension> = D::Smaller;
    type ExpandedDimension<D: Dimension> = D::Larger;

    fn len(&self) -> usize {
        1
    }
    fn get(&self, idx: usize) -> usize {
        match idx {
            0 => *self,
            _ => panic!("Axis index out of bounds"),
        }
    }
}
impl AxesArg for &[usize] {
    type ReducedDimension<D: Dimension> = DimDyn;
    type ExpandedDimension<D: Dimension> = DimDyn;

    fn len(&self) -> usize {
        <[_]>::len(self)
    }
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl AxesArg for &Vec<usize> {
    type ReducedDimension<D: Dimension> = DimDyn;
    type ExpandedDimension<D: Dimension> = DimDyn;

    fn len(&self) -> usize {
        <Vec<_>>::len(self)
    }
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}

macro_rules! impl_axes_array {
    ($($idx:tt),+ $(,)?) => {
        impl AxesArg for [usize; impl_axes_array!(@count $($idx)*)] {
            type ReducedDimension<D: Dimension> = impl_axes_array!(@shrink D, $($idx),+);
            type ExpandedDimension<D: Dimension> = impl_axes_array!(@expand D, $($idx),+);

            fn len(&self) -> usize {
                impl_axes_array!(@count $($idx)*)
            }
            fn get(&self, idx: usize) -> usize {
                self[idx]
            }
        }
        impl AxesArg for &[usize; impl_axes_array!(@count $($idx)*)] {
            type ReducedDimension<D: Dimension> = impl_axes_array!(@shrink D, $($idx),+);
            type ExpandedDimension<D: Dimension> = impl_axes_array!(@expand D, $($idx),+);

            fn len(&self) -> usize {
                impl_axes_array!(@count $($idx)*)
            }
            fn get(&self, idx: usize) -> usize {
                self[idx]
            }
        }
    };

    (@shrink $D:ty, $head:tt $(, $tail:tt)*) => {
        <impl_axes_array!(@shrink $D, $($tail),*) as Dimension>::Smaller
    };
    (@shrink $D:ty,) => { $D };

    (@expand $D:ty, $head:tt $(, $tail:tt)*) => {
        <impl_axes_array!(@expand $D, $($tail),*) as Dimension>::Larger
    };
    (@expand $D:ty,) => { $D };

    (@count ) => { 0 };
    (@count $head:tt $($tail:tt)*) => { 1 + impl_axes_array!(@count $($tail)*) };
}

impl AxesArg for [usize; 0] {
    type ReducedDimension<D: Dimension> = D;
    type ExpandedDimension<D: Dimension> = D;

    fn len(&self) -> usize {
        0
    }
    fn get(&self, _idx: usize) -> usize {
        unreachable!()
    }
}
impl AxesArg for &[usize; 0] {
    type ReducedDimension<D: Dimension> = D;
    type ExpandedDimension<D: Dimension> = D;

    fn len(&self) -> usize {
        0
    }
    fn get(&self, _idx: usize) -> usize {
        unreachable!()
    }
}
impl_axes_array!(0);
impl_axes_array!(0, 1);
impl_axes_array!(0, 1, 2);
impl_axes_array!(0, 1, 2, 3);
impl_axes_array!(0, 1, 2, 3, 4);
impl_axes_array!(0, 1, 2, 3, 4, 5);
impl_axes_array!(0, 1, 2, 3, 4, 5, 6);
impl_axes_array!(0, 1, 2, 3, 4, 5, 6, 7);

macro_rules! impl_axes_tuple {
    ($($idx:tt),+ $(,)?) => {
        impl AxesArg for ($(impl_axes_tuple!(@replace $idx usize),)+) {
            type ReducedDimension<D: Dimension> = impl_axes_tuple!(@shrink D, $($idx),+);
            type ExpandedDimension<D: Dimension> = impl_axes_tuple!(@expand D, $($idx),+);

            fn len(&self) -> usize {
                impl_axes_tuple!(@count $($idx)*)
            }

            fn get(&self, idx: usize) -> usize {
                match idx {
                    $($idx => self.$idx,)+
                    _ => panic!("Axis index out of bounds"),
                }
            }
        }
    };

    (@shrink $D:ty, $head:tt $(, $tail:tt)*) => {
        <impl_axes_tuple!(@shrink $D, $($tail),*) as Dimension>::Smaller
    };
    (@shrink $D:ty,) => { $D };

    (@expand $D:ty, $head:tt $(, $tail:tt)*) => {
        <impl_axes_tuple!(@expand $D, $($tail),*) as Dimension>::Larger
    };
    (@expand $D:ty,) => { $D };

    (@replace $_t:tt $sub:ty) => { $sub };
    (@count ) => { 0 };
    (@count $head:tt $($tail:tt)*) => { 1 + impl_axes_tuple!(@count $($tail)*) };
}
impl AxesArg for () {
    type ReducedDimension<D: Dimension> = D;
    type ExpandedDimension<D: Dimension> = D;

    fn len(&self) -> usize {
        0
    }
    fn get(&self, _idx: usize) -> usize {
        unreachable!()
    }
}
impl_axes_tuple!(0);
impl_axes_tuple!(0, 1);
impl_axes_tuple!(0, 1, 2);
impl_axes_tuple!(0, 1, 2, 3);
impl_axes_tuple!(0, 1, 2, 3, 4);
impl_axes_tuple!(0, 1, 2, 3, 4, 5);
impl_axes_tuple!(0, 1, 2, 3, 4, 5, 6);
impl_axes_tuple!(0, 1, 2, 3, 4, 5, 6, 7);

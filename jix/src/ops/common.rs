use crate::dtype::Dtyped;
use crate::{DimDyn, Dimension};

macro_rules! define_array_op1_method {
    ($method:ident : $Op:ident, $($trait:ident)::+) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self) -> crate::Array<$Op<S>>
        where
            S: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
        {
            $Op::new_array(self).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+, fixed_output_type = true) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method(self) -> crate::Array<$Op<S>>
        where
            S: crate::storage::ArrayStorageTyped,
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
            S: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S::Dimension>,
            S::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
    ($method:ident : $Op:ident, $($trait:ident)::+, fixed_output_type = true) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Dimension = S::Dimension>,
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
            S: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped<Item = $lhs_type, Dimension = S::Dimension>,
            S::Item: $($trait)::+,
        {
            $Op::new_array(self, other).unwrap()
        }
    };
}

pub(crate) use {define_array_op1_method, define_array_op2_method};

pub(crate) trait LanesInfo {
    const LANES: usize;
}
impl<T: Dtyped> LanesInfo for T {
    const LANES: usize = {
        if size_of::<T>() == 0 {
            1024
        } else {
            (128 / size_of::<T>()).next_power_of_two()
        }
    };
}

/// An argument that specifies a set of axis indices, encoding the dimension change in the type.
///
/// Operations that add or remove axes - `insert_axis`, `remove_axis`, `sum`, `max`, etc. - are
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
/// | `[usize; N]` / `(usize, ...)` (N >= 1) | `D::Smaller` repeated N times | `D::Larger` repeated N times |
/// | `&[usize]` / `&Vec<usize>` | `DimDyn` | `DimDyn` |
///
/// When the argument is a slice reference (`&[usize]`), the ndim of the result is only known at
/// runtime, so both associated types resolve to [`DimDyn`](crate::DimDyn). Using a typed array
/// argument (`[usize; N]` or a tuple) preserves static dimension information.
///
/// # Examples
///
/// ```rust
/// use jix::{Array, Dim};
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![[1i32, 2], [3, 4]])?;
///
/// // usize: one axis removed/added - statically one smaller/larger
/// let b = a.as_ref().sum(0);               // D::Smaller when a: Dim<N> -> Dim<N-1>
/// let c = a.as_ref().insert_axis(0);       // D::Larger  when a: Dim<N> -> Dim<N+1>
///
/// // [usize; 2]: two axes removed - statically two smaller
/// let b = a.as_ref().sum([0, 1]);          // Dim<N-2>
/// let c = a.as_ref().insert_axis([0, 1]);  // Dim<N+2>
///
/// // &[usize]: dynamic count -> DimDyn regardless of input dimension
/// let axes = vec![0, 1];
/// let b = a.as_ref().sum(axes.as_slice()); // DimDyn
/// # Ok::<(), jix::Error>(())
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

    #[inline(always)]
    fn len(&self) -> usize {
        1
    }

    #[inline(always)]
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

    #[inline(always)]
    fn len(&self) -> usize {
        <[_]>::len(self)
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl AxesArg for Vec<usize> {
    type ReducedDimension<D: Dimension> = DimDyn;
    type ExpandedDimension<D: Dimension> = DimDyn;

    #[inline(always)]
    fn len(&self) -> usize {
        <Vec<_>>::len(self)
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl AxesArg for &Vec<usize> {
    type ReducedDimension<D: Dimension> = DimDyn;
    type ExpandedDimension<D: Dimension> = DimDyn;

    #[inline(always)]
    fn len(&self) -> usize {
        <Vec<_>>::len(self)
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}

macro_rules! impl_axes_array {
    ($($idx:tt),+ $(,)?) => {
        impl AxesArg for [usize; impl_axes_array!(@count $($idx)*)] {
            type ReducedDimension<D: Dimension> = impl_axes_array!(@shrink D, $($idx),+);
            type ExpandedDimension<D: Dimension> = impl_axes_array!(@expand D, $($idx),+);

            #[inline(always)]
            fn len(&self) -> usize {
                impl_axes_array!(@count $($idx)*)
            }

            #[inline(always)]
            fn get(&self, idx: usize) -> usize {
                self[idx]
            }
        }
        impl AxesArg for &[usize; impl_axes_array!(@count $($idx)*)] {
            type ReducedDimension<D: Dimension> = impl_axes_array!(@shrink D, $($idx),+);
            type ExpandedDimension<D: Dimension> = impl_axes_array!(@expand D, $($idx),+);

            #[inline(always)]
            fn len(&self) -> usize {
                impl_axes_array!(@count $($idx)*)
            }

            #[inline(always)]
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

    #[inline(always)]
    fn len(&self) -> usize {
        0
    }

    #[inline(always)]
    fn get(&self, _idx: usize) -> usize {
        unreachable!()
    }
}
impl AxesArg for &[usize; 0] {
    type ReducedDimension<D: Dimension> = D;
    type ExpandedDimension<D: Dimension> = D;

    #[inline(always)]
    fn len(&self) -> usize {
        0
    }

    #[inline(always)]
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

            #[inline(always)]
            fn len(&self) -> usize {
                impl_axes_tuple!(@count $($idx)*)
            }

            #[inline(always)]
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

    #[inline(always)]
    fn len(&self) -> usize {
        0
    }

    #[inline(always)]
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

use crate::dtype::Dtyped;

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
}
macro_rules! define_array_op2_method {
    ($method:ident : $Op:ident, $($trait:ident)::+) => {
        #[doc = concat!("Applies the [`", stringify!($Op), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $method<S2>(self, other: crate::Array<S2>) -> crate::Array<$Op<S, S2>>
        where
            S: crate::storage::ArrayStorageTyped,
            S2: crate::storage::ArrayStorageTyped,
            S::Item: $($trait)::+<S2::Item, Output: crate::dtype::Dtyped>,
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
            S2: crate::storage::ArrayStorageTyped<Item = $lhs_type>,
            S::Item: $($trait)::+<Output: crate::dtype::Dtyped>,
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

/// An argument that specifies a set of dim indices.
///
/// Operations that add or remove dims - `insert_dim`, `remove_dim`, `sum`, `max`, etc. - are
/// generic over `Dims: DimsArg`, so a single dim (`usize`), a fixed set (`[usize; N]` or a tuple of
/// `usize`) or a runtime list (`&[usize]`, `Vec<usize>`) can be passed.
///
/// # Examples
///
/// ```rust
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![[1i32, 2], [3, 4]])?;
///
/// let b = a.view().sum(0);
/// let c = a.view().insert_dim(0);
/// let d = a.view().sum([0, 1]);
/// let e = a.view().insert_dim((0, 1));
/// let dims = vec![0, 1];
/// let f = a.view().sum(dims.as_slice());
/// # Ok::<(), jix::Error>(())
/// ```
#[allow(clippy::len_without_is_empty)]
pub trait DimsArg {
    /// The number of dims described by this argument.
    fn len(&self) -> usize;

    /// The dim at position `idx` in this argument.
    ///
    /// `idx` must be less than `self.len()`.
    fn get(&self, idx: usize) -> usize;
}

impl DimsArg for usize {
    #[inline(always)]
    fn len(&self) -> usize {
        1
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        match idx {
            0 => *self,
            _ => panic!("Dim index out of bounds"),
        }
    }
}
impl DimsArg for &[usize] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[_]>::len(self)
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl DimsArg for Vec<usize> {
    #[inline(always)]
    fn len(&self) -> usize {
        <Vec<_>>::len(self)
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl DimsArg for &Vec<usize> {
    #[inline(always)]
    fn len(&self) -> usize {
        <Vec<_>>::len(self)
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl<const N: usize> DimsArg for [usize; N] {
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}
impl<const N: usize> DimsArg for &[usize; N] {
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> usize {
        self[idx]
    }
}

macro_rules! impl_dims_tuple {
    ($($idx:tt),+ $(,)?) => {
        impl DimsArg for ($(impl_dims_tuple!(@replace $idx usize),)+) {
            #[inline(always)]
            fn len(&self) -> usize {
                impl_dims_tuple!(@count $($idx)*)
            }

            #[inline(always)]
            fn get(&self, idx: usize) -> usize {
                match idx {
                    $($idx => self.$idx,)+
                    _ => panic!("Dim index out of bounds"),
                }
            }
        }
    };

    (@replace $_t:tt $sub:ty) => { $sub };
    (@count ) => { 0 };
    (@count $head:tt $($tail:tt)*) => { 1 + impl_dims_tuple!(@count $($tail)*) };
}
impl DimsArg for () {
    #[inline(always)]
    fn len(&self) -> usize {
        0
    }

    #[inline(always)]
    fn get(&self, _idx: usize) -> usize {
        unreachable!()
    }
}
impl_dims_tuple!(0);
impl_dims_tuple!(0, 1);
impl_dims_tuple!(0, 1, 2);
impl_dims_tuple!(0, 1, 2, 3);
impl_dims_tuple!(0, 1, 2, 3, 4);
impl_dims_tuple!(0, 1, 2, 3, 4, 5);
impl_dims_tuple!(0, 1, 2, 3, 4, 5, 6);
impl_dims_tuple!(0, 1, 2, 3, 4, 5, 6, 7);

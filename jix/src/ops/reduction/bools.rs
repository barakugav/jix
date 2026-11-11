use crate::ops::prelude::*;
use crate::ops::reduction::common::{
    define_array_reduction_method, define_reduction_op, ReductionOpKernel,
};

define_reduction_op!(
    /// Reduces one or more dims by testing whether all elements are `true`.
    ///
    /// Input and output dtype are `bool8`. An empty reduction returns `true`.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::all()`](crate::Array::all).
    ///
    /// # Examples
    /// ```
    /// use jix::scalar::bool8;
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[bool8::TRUE, bool8::FALSE, bool8::TRUE], [bool8::TRUE, bool8::TRUE, bool8::TRUE]];
    ///
    /// // All elements true? -> false (contains false)
    /// let all_true = Array::compact_ndarray(&nd)?
    ///     .all((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(all_true[[]], bool8::FALSE);
    ///
    /// // All true along dim 0 (per column) -> shape [3]
    /// let col_all = Array::compact_ndarray(&nd)?
    ///     .all(0).to_ndarray()?;
    /// assert_eq!(col_all, array![bool8::TRUE, bool8::FALSE, bool8::TRUE]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    All,
    AllKernel,
    where {
        S: ArrayStorageTyped<Item = bool8>,
    }
    output = bool8,
);
impl ReductionOpKernel<bool8> for AllKernel {
    type Output = bool8;
    type State = bool8;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(bool8, Option<u64>)>) -> Self::State {
        let mut state = crate::scalar::bool8::TRUE;
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, None);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: bool8, _idx: Option<u64>) -> Self::State {
        state & item
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a & b
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = true;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const NEEDS_INDICES: bool = false;
}

define_reduction_op!(
    /// Reduces one or more dims by testing whether any element is `true`.
    ///
    /// Input and output dtype are `bool8`. An empty reduction returns `false`.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::any()`](crate::Array::any).
    ///
    /// # Examples
    /// ```
    /// use jix::scalar::bool8;
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[bool8::FALSE, bool8::FALSE, bool8::FALSE], [bool8::TRUE, bool8::TRUE, bool8::TRUE]];
    ///
    /// // Any element true? -> true
    /// let any_true = Array::compact_ndarray(&nd)?
    ///     .any((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(any_true[[]], bool8::TRUE);
    ///
    /// // Any true along dim 0 (per column) -> shape [3]
    /// let col_any = Array::compact_ndarray(&nd)?
    ///     .any(0).to_ndarray()?;
    /// assert_eq!(col_any, array![bool8::TRUE, bool8::TRUE, bool8::TRUE]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Any,
    AnyKernel,
    where {
        S: ArrayStorageTyped<Item = bool8>,
    }
    output = bool8,
);
impl ReductionOpKernel<bool8> for AnyKernel {
    type Output = bool8;
    type State = bool8;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(bool8, Option<u64>)>) -> Self::State {
        let mut state = crate::scalar::bool8::FALSE;
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, None);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: bool8, _idx: Option<u64>) -> Self::State {
        state | item
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a | b
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = true;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const NEEDS_INDICES: bool = false;
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(
        all: All,
        where {
            S: ArrayStorageTyped<Item = bool8>,
        }
    );
    define_array_reduction_method!(
        any: Any,
        where {
            S: ArrayStorageTyped<Item = bool8>,
        }
    );
}

#[cfg(test)]
mod tests {
    // test_reduction!(
    //     all,
    //     |items| { items.fold(true, |m, x| m && <_ as crate::scalar::Cast<bool>>::cast(x)) },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool8],
    //     logical_op_strategy,
    //     #[cfg(feature = "half")]
    //     [f16],
    //     #[cfg(feature = "num-complex")]
    //     [complex_f32, complex_f64]
    // );
    // test_reduction!(
    //     any,
    //     |items| { items.fold(false, |m, x| m || <_ as crate::scalar::Cast<bool>>::cast(x)) },
    //     [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool8],
    //     logical_op_strategy,
    //     #[cfg(feature = "half")]
    //     [f16],
    //     #[cfg(feature = "num-complex")]
    //     [complex_f32, complex_f64]
    // );
}

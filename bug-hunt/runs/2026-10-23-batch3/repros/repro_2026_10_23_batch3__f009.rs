//! F009 repro: `ArrayStorage` is a safe, unsealed public trait, yet jix's unsafe code trusts the
//! buffers a storage's `read_data` returns (pull) or claims to have written (push). A user storage
//! written WITHOUT any `unsafe` that returns a too-small buffer drives safe jix calls into an
//! out-of-bounds read (pull, default `read_as_elementwise_pipeline`) or an uninitialized read
//! (push, `to_ndarray_sub`'s `assume_init`).
//!
//! The tests assert the sound behavior: the read must fail (an `Err` or a panic). Returning `Ok`
//! is impossible without UB here, because the storage only ever hands back 1 element (pull) or
//! writes nothing (push).
//!
//! WARNING: `zz_pull_short_buffer_large_region_is_rejected` crashes the whole test process with
//! SIGSEGV while the bug exists. Run the tests one at a time by name.
#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};

use jix::dtype::Dtype;
use jix::ops::IntoType;
use jix::storage::{ArraySpec, ArrayStorageInfo, StridedBuf};
use jix::{Array, ArrayStorage, ElementType, ReadContext};
use ndarray::{array, Ix1};

/// Claims shape `shape`, but every read returns the pull buffer of a 1-element inner array and
/// ignores `out`. Everything else is forwarded to the inner storage. No `unsafe`.
struct Short<S> {
    inner: S,
    shape: Vec<u64>,
}

impl<S: ArrayStorage> ArrayStorage for Short<S> {
    type ElementType = S::ElementType;

    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        _out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>, jix::Error> {
        self.inner.read_data(&vec![0..1; index.len()], context, None)
    }
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        self.inner.dtype()
    }
    fn spec(&self) -> ArraySpec<'_> {
        self.inner.spec()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        self.inner.info()
    }
    type ElementTypeChange<NewET: ElementType> = IntoType<Self, NewET>;
    fn element_type_change<NewET: ElementType>(
        self,
    ) -> Result<Self::ElementTypeChange<NewET>, jix::Error> {
        IntoType::new(self)
    }
}

fn expect_rejected<T: std::fmt::Debug>(what: &str, f: impl FnOnce() -> Result<T, jix::Error>) {
    match catch_unwind(AssertUnwindSafe(f)) {
        Err(_) => println!("{what}: panicked (sound)"),
        Ok(Err(e)) => println!("{what}: Err({e}) (sound)"),
        Ok(Ok(v)) => panic!("{what}: returned Ok although the storage gave too little data: {v:?}"),
    }
}

/// Pull: `-arr` reads through the default `read_as_elementwise_pipeline` (core_trait.rs:132-134),
/// which wraps the 4-byte pull buffer in `OperandTyped::new_input`; the pipeline then reads 3
/// f32 values from it (8 bytes past the allocation). Miri-sized.
#[test]
fn pull_short_buffer_is_rejected() {
    let inner = Array::plain_ndarray(array![1.0f32]).unwrap().into_storage();
    let arr = Array::from_storage(Short {
        inner,
        shape: vec![3],
    });
    expect_rejected("pull n=3", || (-arr).to_ndarray::<Ix1>());
}

/// Push: `to_ndarray` -> `to_ndarray_sub` hands an `ndarray::Array::uninit` buffer as `out`
/// (small_read path), the storage writes nothing, and `assume_init` (array.rs:806) returns
/// uninitialized bytes. Miri-sized.
#[test]
fn push_ignored_out_is_rejected() {
    let inner = Array::plain_ndarray(array![7u8]).unwrap().into_storage();
    let arr = Array::from_storage(Short {
        inner,
        shape: vec![3],
    });
    expect_rejected("push n=3", || arr.to_ndarray::<Ix1>());
}

/// Same as the pull test with 2^22 elements: the pipeline reads 16 MiB past a 4-byte heap
/// allocation. Crashes the process (SIGSEGV) while the bug exists.
#[test]
fn zz_pull_short_buffer_large_region_is_rejected() {
    let inner = Array::plain_ndarray(array![1.0f32]).unwrap().into_storage();
    let arr = Array::from_storage(Short {
        inner,
        shape: vec![1 << 22],
    });
    expect_rejected("pull n=2^22", || {
        (-arr)
            .to_ndarray::<Ix1>()
            .map(|v| v.iter().filter(|&&x| x != -1.0).count())
    });
}

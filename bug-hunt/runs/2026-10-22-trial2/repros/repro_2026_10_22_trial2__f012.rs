// F012 repro (bug-hunt run 2026-10-22-trial2).
//
// `Op2::shape` (jix/src/ops/op2.rs:182) runs `assert_unchecked_eq!(a.shape(), b.shape())`, i.e.
// `debug_assert_eq!` + `std::hint::assert_unchecked`. The child shapes are compared only once, in
// `Op2::new`. `ArrayStorage` is a safe public trait and `Array::from_storage` is safe, so a safe
// user storage whose `shape()` changes after construction makes `.shape()` on any Op2-based array
// (Add/Sub/.../Equal/Less/Greater/ApproxEq/Map2) execute `assert_unchecked(false)`: UB from safe
// code. Debug builds panic in the debug_assert; release builds may miscompile.
//
// The first two tests assert the CORRECT behavior (safe code gets one of the children's shapes, no
// panic, no UB), so they fail while the bug exists:
//   debug:   cargo test --manifest-path jix/Cargo.toml --all-features \
//                --test repro_2026_10_22_trial2__f012 -- --nocapture
//            -> both panic at src/ops/op2.rs:182 (`assertion left == right failed`).
//   release: same command plus `--release`. `release_miscompile_reads_out_of_bounds` CRASHES THE
//            PROCESS (SIGSEGV, observed on x86_64): LLVM takes the false assumption `len == 8`
//            for an empty slice whose pointer is dangling (0x8) and reads 8 u64 from it.
// The third test is a control: the same lying storage without Op2 on top; it always passes.
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use jix::dtype::Dtype;
use jix::storage::{ArraySpec, ArrayStorageInfo, StridedBuf};
use jix::{Array, ArrayStorage, ElementType, ReadContext, Ty};

type Result<T> = std::result::Result<T, jix::Error>;

/// Forwards everything to `inner`, but `shape()` returns `alt` once `lie` is set.
struct Shifty<S> {
    inner: S,
    lie: Arc<AtomicBool>,
    alt: Vec<u64>,
}

impl<S: ArrayStorage<ElementType = Ty<i32>>> ArrayStorage for Shifty<S> {
    type ElementType = Ty<i32>;
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self.inner.read_data(index, context, out)
    }
    fn shape(&self) -> &[u64] {
        if self.lie.load(Ordering::Relaxed) {
            &self.alt
        } else {
            self.inner.shape()
        }
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
    type ElementTypeChange<NewET: ElementType> = jix::ops::IntoType<Self, NewET>;
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        jix::ops::IntoType::new(self)
    }
}

/// Forwards everything to `inner`, but reports a compile-time constant shape.
static FIXED_SHAPE: [u64; 8] = [1; 8];
struct Fixed<S> {
    inner: S,
}

impl<S: ArrayStorage<ElementType = Ty<i32>>> ArrayStorage for Fixed<S> {
    type ElementType = Ty<i32>;
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        self.inner.read_data(index, context, out)
    }
    fn shape(&self) -> &[u64] {
        &FIXED_SHAPE
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
    type ElementTypeChange<NewET: ElementType> = jix::ops::IntoType<Self, NewET>;
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        jix::ops::IntoType::new(self)
    }
}

/// Smallest trigger: `Add` (a plain Op2) and `greater` (Op2 with swapped operands, cmp.rs:435).
#[test]
fn op2_shape_with_unstable_child_shape_is_not_ub() {
    let lie = Arc::new(AtomicBool::new(false));
    let shifty = || {
        let inner = Array::compact_ndarray(&ndarray::array![3i32, 2, 1]).unwrap();
        Array::from_storage(Shifty {
            inner: inner.into_storage(),
            lie: lie.clone(),
            alt: vec![7],
        })
    };
    let a = || Array::compact_ndarray(&ndarray::array![1i32, 2, 3]).unwrap();

    let sum = jix::ops::Add::new_array(shifty(), a()).unwrap();
    let gt = a().greater(shifty());
    lie.store(true, Ordering::Relaxed);

    // Safe code. Any sound answer is one of the children's current shapes ([7] or [3]).
    for shape in [sum.shape(), gt.shape()] {
        assert!(shape == [7] || shape == [3], "unexpected shape {shape:?}");
    }
}

/// Same defect, shaped so that a release build turns the false `assert_unchecked` into an
/// out-of-bounds read: the lying child returns an empty slice (dangling pointer), the other child
/// a constant 8-element slice, so LLVM may take `len == 8` for the empty slice.
#[test]
fn release_miscompile_reads_out_of_bounds() {
    let lie = Arc::new(AtomicBool::new(false));
    let inner = || {
        Array::compact_ndarray(&ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1; 8]), 5i32)).unwrap()
    };
    let shifty = Array::from_storage(Shifty {
        inner: inner().into_storage(),
        lie: lie.clone(),
        alt: vec![],
    });
    let fixed = Array::from_storage(Fixed {
        inner: inner().into_storage(),
    });
    let arr = jix::ops::Add::new_array(shifty, fixed).unwrap();
    lie.store(std::hint::black_box(true), Ordering::Relaxed);

    let s = arr.shape();
    let ndim = s.len();
    let nelem: u64 = s.iter().product();
    println!("ndim = {ndim}, nelem = {nelem}, ptr = {:p}", s.as_ptr());
    // Sound answers: Shifty's current shape [] or Fixed's [1; 8] (both have 1 element).
    assert!(ndim == 0 || ndim == 8, "ndim = {ndim}");
    assert_eq!(nelem, 1);
    if ndim == 8 {
        assert_eq!(s.as_ptr(), FIXED_SHAPE.as_ptr(), "8-dim shape not backed by FIXED_SHAPE");
    }
}

/// Control (passes before and after a fix): the same lying storage and the same safe code, but
/// with no Op2 on top, reads fine in release. So the crash above comes from Op2's assumption.
#[test]
fn control_same_lying_storage_without_op2() {
    let lie = Arc::new(AtomicBool::new(false));
    let inner = Array::compact_ndarray(&ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1; 8]), 5i32))
        .unwrap();
    let arr = Array::from_storage(Shifty {
        inner: inner.into_storage(),
        lie: lie.clone(),
        alt: vec![],
    });
    lie.store(std::hint::black_box(true), Ordering::Relaxed);

    let s = arr.shape();
    let ndim = s.len();
    let nelem: u64 = s.iter().product();
    println!("control: ndim = {ndim}, nelem = {nelem}, ptr = {:p}", s.as_ptr());
    assert_eq!((ndim, nelem), (0, 1));
}

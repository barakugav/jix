// K198 repro (run 2026-11-17-scan-106-107-b).
//
// An identity Slice / Flip / Roll and a `&dyn ArrayStorage` forward their inner's spec unchanged,
// so an element-wise inner keeps ReadTiling::Fusable (#106), but none of them forwards
// `read_as_elementwise_pipeline`. A parent element-wise op then also claims Fusable, its
// top-level read is not tiled (array/read.rs to_buf_strided_checked), and its pipeline falls back
// to the default (core_trait.rs) pull of the WHOLE region into a pooled temporary.
//
// Measured: peak bytes allocated (beyond the caller's preallocated destination) during one
// to_buf_slice of the full region. The test asserts the CORRECT behavior - a no-op view must not
// add a region-sized temporary - so it fails while the bug exists.
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//        --test repro_k198 -- --nocapture
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use jix::{Array, ArrayStorage};
use ndarray::{Array2, Ix2};

struct Tracking;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let c = CUR.fetch_add(n, Ordering::SeqCst) + n;
    PEAK.fetch_max(c, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            CUR.fetch_sub(l.size(), Ordering::SeqCst);
            add(n);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

const C: usize = 1 << 21; // [1, 2^21] f64 = 16 MiB, far above one read tile
const REGION: usize = C * 8;

/// Peak temporary bytes of a full-region push read into a preallocated buffer, plus the result.
fn peak_temp<S: ArrayStorage>(a: &Array<S>) -> (usize, Vec<f64>) {
    let mut out = vec![0f64; C];
    let base = CUR.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    {
        let ctx = a.read_ctx();
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), REGION) };
        a.to_buf_slice(&[0..1, 0..C as u64], bytes, &ctx).unwrap();
    }
    (PEAK.load(Ordering::SeqCst) - base, out)
}

fn tiling<S: ArrayStorage>(a: &Array<S>) -> jix::__private::ReadTiling {
    a.storage().spec().flags().read_tiling()
}

#[test]
fn identity_views_do_not_add_a_region_sized_temporary() {
    let nd = Array2::<f64>::from_shape_fn((1, C), |(_, j)| j as f64);
    let want: Vec<f64> = nd.iter().copied().collect();
    let p = || Array::plain_ndarray_ref(&nd).unwrap();

    // Controls: plain fusion (no temporary) and a non-identity forwarder (tiled, small temporary).
    let fused = -(-p());
    let permuted = -((-p()).permute_dims(&[0, 1]));

    // The four no-op views over an element-wise op.
    let ident_slice = -((-p()).slice((.., ..)));
    let ident_flip = -((-p()).flip(0usize)); // dim 0 has length 1: flip is a no-op
    let ident_roll = -((-p()).roll(0, 1)); // shift 0: roll is a no-op
    let inner = -p();
    let erased: &dyn ArrayStorage = inner.storage();
    let via_dyn = -Array::from_storage(erased).into_typed::<f64>().unwrap();

    let mut failures = Vec::new();
    for (name, (temp, got), kind) in [
        ("fused", peak_temp(&fused), tiling(&fused)),
        ("permute_dims (control)", peak_temp(&permuted), tiling(&permuted)),
        ("identity slice", peak_temp(&ident_slice), tiling(&ident_slice)),
        ("identity flip", peak_temp(&ident_flip), tiling(&ident_flip)),
        ("identity roll", peak_temp(&ident_roll), tiling(&ident_roll)),
        ("&dyn ArrayStorage", peak_temp(&via_dyn), tiling(&via_dyn)),
    ] {
        assert!(got == want, "{name}: wrong values");
        eprintln!("{name:24} tiling={kind:?} peak temporary={temp} bytes (region {REGION})");
        if temp >= REGION / 4 {
            failures.push(format!("{name}: {temp} temporary bytes for a {REGION}-byte region"));
        }
    }
    // Sanity: the values via to_ndarray agree too.
    assert_eq!(ident_slice.to_ndarray::<Ix2>().unwrap(), nd);
    assert!(failures.is_empty(), "region-sized temporaries: {failures:#?}");
}

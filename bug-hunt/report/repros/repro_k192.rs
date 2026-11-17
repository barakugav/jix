// K192: Select is classified ReadTiling::Fusable (#106), but Select::read_data pulls `condition`
// and `y` into region-sized temporaries when they are element-wise ops. A top-level read of
// `cond.select(x, y)` then takes the single-pass path (array/read.rs) and allocates those
// temporaries for the whole region instead of per tile.
//
// The test asserts the correct behavior (a push read into a preallocated destination allocates
// only tile-sized scratch, like the same expression read tiled), so it fails while the bug exists.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use jix::scalar::bool8;
use jix::{Array, ArrayStorage};
use ndarray::Array1;

struct Tracking;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(c, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::SeqCst);
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(c, Ordering::SeqCst);
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            CUR.fetch_sub(l.size(), Ordering::SeqCst);
            let c = CUR.fetch_add(n, Ordering::SeqCst) + n;
            PEAK.fetch_max(c, Ordering::SeqCst);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

const N: usize = 1 << 22; // 4 Mi f32 = 16 MiB output

fn p(x: &Array1<f32>) -> Array<jix::storage::Plain<&(), jix::Ty<f32>>> {
    Array::plain_ndarray_ref(x).unwrap()
}

/// Peak bytes allocated by `f` on top of what was live before it.
fn peak_extra<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let base = CUR.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let r = f();
    (PEAK.load(Ordering::SeqCst) - base, r)
}

#[test]
fn select_push_read_allocates_only_tile_scratch() {
    let a = Array1::from_iter((0..N).map(|i| (i % 7) as f32 - 3.0));
    let b = Array1::from_iter((0..N).map(|i| (i % 5) as f32));
    let zero = Array1::<f32>::zeros(N);
    let want: Vec<f32> = (0..N)
        .map(|i| if a[i] > 0.0 { a[i] } else { b[i] * b[i] })
        .collect();
    let mut dst = vec![0f32; N];
    let out_bytes = N * size_of::<f32>();

    // The expression of the bug: condition and y are element-wise ops over Plain leaves.
    let fused = p(&a).greater(p(&zero)).select(p(&a), p(&b) * p(&b));
    let fused_tiling = fused.storage().spec().flags().read_tiling();
    let ctx = fused.read_ctx();
    let (fused_alloc, r) = peak_extra(|| {
        fused.to_buf_slice(&[0..N as u64], bytemuck_f32(&mut dst), &ctx)
    });
    r.unwrap();
    assert_eq!(dst, want);
    drop(ctx);

    // Baseline: the same values, with condition and y type-erased so Select is read tiled
    // (as before #106, when Select cleared its flags).
    dst.fill(0.0);
    // `into_any` needs 'static storages, so the erased subtrees read owned copies.
    let own = |x: &Array1<f32>| Array::plain_ndarray(x.clone()).unwrap();
    let cond = own(&a).greater(own(&zero)).into_any().into_typed::<bool8>().unwrap();
    let y = (own(&b) * own(&b)).into_any().into_typed::<f32>().unwrap();
    let tiled = cond.select(p(&a), y);
    let tiled_tiling = tiled.storage().spec().flags().read_tiling();
    let ctx = tiled.read_ctx();
    let (tiled_alloc, r) = peak_extra(|| {
        tiled.to_buf_slice(&[0..N as u64], bytemuck_f32(&mut dst), &ctx)
    });
    r.unwrap();
    assert_eq!(dst, want);
    drop(ctx);

    println!(
        "output {} KiB; select(a > 0, a, b*b) [{:?}] push read allocated {} KiB; \
         same with erased cond/y [{:?}] {} KiB",
        out_bytes >> 10,
        fused_tiling,
        fused_alloc >> 10,
        tiled_tiling,
        tiled_alloc >> 10,
    );
    // A push read needs no region-sized temporary: tile-sized scratch only.
    assert!(
        fused_alloc <= out_bytes / 4,
        "push read of a {} KiB region allocated {} KiB (tiled read of the same: {} KiB)",
        out_bytes >> 10,
        fused_alloc >> 10,
        tiled_alloc >> 10,
    );
}

fn bytemuck_f32(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), size_of_val(v)) }
}

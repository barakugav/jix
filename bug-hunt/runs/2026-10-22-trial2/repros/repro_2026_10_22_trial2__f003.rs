// Verifier repro for F003 (run 2026-10-22-trial2): on 32-bit targets (usize = 32 bits) a read
// region of more than usize::MAX elements / bytes is sized with truncating `as usize` casts and
// wrapping usize products, while the read itself walks the full u64 region.
//
// 32-bit only: every test returns early when usize is 64 bits (there, these regions would really
// allocate > 4 GiB). Run natively as i686, one test at a time (several CRASH THE PROCESS while the
// bug exists - SIGSEGV from a heap overflow):
//
//   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//     --test repro_2026_10_22_trial2__f003 -- --nocapture --test-threads 1 --exact <test_name>
//
// Each test asserts the CORRECT behavior: a region that cannot be addressed on this target must be
// rejected with an Err (or, for a pull read, returned as a view whose strides fit the address
// space and describe the region).
use jix::{Array, ArrayStorage, ReadContext};
use ndarray::{array, Array2, Ix1};
use std::ops::Range;

const D0: u64 = 65536;
const D1: u64 = 65536 + 64; // D0 * D1 = 2^32 + 2^22, wraps to 4194304 in usize

fn is_32bit() -> bool {
    usize::BITS == 32
}

// The bytes a pull result claims to cover: sum over dims of (len - 1) * stride, plus itemsize.
// None if it does not fit u64; the caller checks it against isize::MAX.
fn span_bytes(index: &[Range<u64>], strides: &[usize], itemsize: u64) -> Option<u64> {
    let mut span = itemsize;
    for (r, &s) in index.iter().zip(strides) {
        let len = r.end - r.start;
        if len == 0 {
            return Some(0);
        }
        span = span.checked_add((len - 1).checked_mul(s as u64)?)?;
    }
    Some(span)
}

fn assert_err_or_addressable(
    r: Result<jix::storage::StridedBuf<'_>, jix::Error>,
    index: &[Range<u64>],
    itemsize: u64,
) {
    match r {
        Err(e) => println!("Err (correct): {e}"),
        Ok(buf) => {
            let span = span_bytes(index, buf.strides(), itemsize);
            println!("Ok, strides {:?}, span {:?}", buf.strides(), span);
            assert!(
                span.is_some_and(|s| s <= isize::MAX as u64),
                "pull result strides {:?} describe {span:?} bytes, more than a 32-bit address space",
                buf.strides()
            );
        }
    }
}

// Trigger (a): pull read through the slow path. materialize_out_buf (buf.rs:310-313) multiplies the
// usize extents: 65536 * 65600 wraps to 4 MiB, then every tile is written at full strides.
// Debug: 'attempt to multiply with overflow' at buf.rs:312. Release: SIGSEGV (heap overflow).
#[test]
fn pull_read_over_4gib_broadcast() {
    if !is_32bit() {
        return;
    }
    let a = Array::compact_ndarray(&array![[7u8]])
        .unwrap()
        .broadcast(&[D0, D1]);
    let ctx = ReadContext::default();
    let index = [0..D0, 0..D1];
    let r = a.to_ndarray_buf(&index, &ctx, None);
    assert_err_or_addressable(r, &index, 1);
}

// Trigger (b): array.rs:960 truncates nitems = 2^32 + 65536 to 65536, so the "small read" fast
// path calls Map -> Tile::read_data directly; Tile case C sizes its pull buffer with the same
// wrapping product (tile.rs:163 -> buf.rs:312) and copies the full region into it.
#[test]
fn pull_read_over_4gib_tile_map_small_read() {
    if !is_32bit() {
        return;
    }
    let base = Array::compact_ndarray(&Array2::<i8>::zeros((1, 65537))).unwrap();
    let a = base.tile(65536, 0).map(|x: i8| x ^ 1);
    let ctx = ReadContext::default();
    let index = [0..65536, 0..65537];
    let r = a.to_ndarray_buf(&index, &ctx, None);
    assert_err_or_addressable(r, &index, 1);
}

// Same root cause in the push entry point: check_get_buffer_size (error.rs:246-247) truncates the
// u64 element count, so a 4 MiB buffer passes the "exact length" check for a 4.3 GB region and the
// read writes past it. No overflow panic on this path: it is a heap overflow even in debug.
#[test]
fn to_ndarray_slice_rejects_wrapped_buffer_len() {
    if !is_32bit() {
        return;
    }
    let a = Array::compact_ndarray(&array![[7u8]])
        .unwrap()
        .broadcast(&[D0, D1]);
    let ctx = a.read_ctx();
    let wrapped = (D0 * D1) as usize; // 4194304 on 32-bit
    let mut buf = vec![0u8; wrapped];
    let r = a.to_ndarray_slice(&[0..D0, 0..D1], &mut buf, &ctx);
    assert!(
        r.is_err(),
        "accepted a {}-byte buffer for a {}-byte region",
        buf.len(),
        D0 * D1
    );
}

// Same root cause in to_ndarray / to_ndarray_sub (array.rs:764): the ndarray is allocated with the
// truncated dim (2^32 + 8 -> 8), then Tile case C pushes 2^32 + 8 bytes into it.
#[test]
fn to_ndarray_tile_over_4gib() {
    if !is_32bit() {
        return;
    }
    let t = Array::compact_ndarray(&array![7u8])
        .unwrap()
        .broadcast(&[(1u64 << 31) + 4])
        .tile(2, 0); // shape [2^32 + 8]
    assert_eq!(t.shape(), &[(1u64 << 32) + 8]);
    let r = t.to_ndarray::<Ix1>();
    assert!(r.is_err(), "got Ok with len {}", r.unwrap().len());
}

// Same root cause in the element-wise driver (elementwise_pipeline.rs:200, :869): a dim of 2^32
// truncates to 0, the non-empty region is treated as empty and nothing is computed. A correct pull
// result is either an Err or the collapsed broadcast view (stride 0) holding x + 1 = 4.0.
#[test]
fn pull_pipeline_dim_2pow32_not_empty() {
    if !is_32bit() {
        return;
    }
    let m = Array::compact_ndarray(&array![3.0f32])
        .unwrap()
        .broadcast(&[1u64 << 32])
        .map(|x: f32| x + 1.0);
    let ctx = ReadContext::default();
    let index = [0..1u64 << 32];
    for (name, r) in [
        ("to_ndarray_buf", m.to_ndarray_buf(&index, &ctx, None)),
        ("read_data", m.storage().read_data(&index, &ctx, None)),
    ] {
        match r {
            Err(e) => println!("{name}: Err (correct): {e}"),
            Ok(buf) => {
                println!("{name}: Ok, strides {:?}", buf.strides());
                assert_eq!(
                    buf.strides(),
                    &[0],
                    "{name}: pull result does not describe the 2^32-element region"
                );
                let v = unsafe { buf.data_ptr().cast::<f32>().read_unaligned() };
                assert_eq!(v, 4.0, "{name}: element 0 was never computed");
            }
        }
    }
}

// The same truncation through to_ndarray: the 2^32-element (16 GiB f32) array cannot be
// materialized on a 32-bit target, so the correct result is an Err - not an Ok ndarray whose
// shape silently differs from the jix array's shape.
#[test]
fn to_ndarray_pipeline_dim_2pow32() {
    if !is_32bit() {
        return;
    }
    let m = Array::compact_ndarray(&array![3.0f32])
        .unwrap()
        .broadcast(&[1u64 << 32])
        .map(|x: f32| x + 1.0);
    let r = m.to_ndarray::<Ix1>();
    assert!(r.is_err(), "got Ok with shape {:?}", r.unwrap().shape());
}

// Repro for bug-hunt run 2026-10-27-batch5, finding F001.
//
// 32-bit only (i686 / armv7). The reduction driver sizes its temporary state buffer as
// `out_nitems * size_of::<State>()` (jix/src/ops/reduction.rs:516) with an unchecked usize
// multiply. var(f16) has a 2-byte output but a 24-byte Welford state, so a 178_956_971-cell region
// (358 MB of output, accepted by check_region_size) needs 2^32 + 8 bytes of state, which wraps to
// 8. Debug build: panics with 'attempt to multiply with overflow'. Release build: the fold writes
// 24-byte states far past the 8-byte allocation and the PROCESS CRASHES (SIGSEGV).
//
// Run: bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//        --test repro_2026_10_27_batch5__f001 [--release] -- --nocapture --test-threads 1
//
// Correct behavior asserted: Ok with every cell 0.0, or a clean Err.
#![cfg(target_pointer_width = "32")]

use jix::scalar::f16;
use jix::{Array, ArrayParams, ArrayStorage, ReadContext};
use ndarray::{array, Ix1};

// 178_956_971 * 24 = 2^32 + 8.
const N: u64 = (1u64 << 32) / 24 + 1;

#[test]
fn var_f16_read_data_state_buffer_does_not_wrap() {
    let a = Array::compact_ndarray(&array![[f16::from_f32(1.0)]]).unwrap();
    let v = a.broadcast(&[N, 1]).var(1, 0.0);
    let ctx = ReadContext::default();
    match v.storage().read_data(&[0..N], &ctx, None) {
        Ok(buf) => {
            let p = buf.data_ptr().cast::<f16>();
            let s = buf.strides()[0];
            for i in [0usize, 1, N as usize / 2, (N - 1) as usize] {
                let x = unsafe { p.byte_add(i * s).read_unaligned() };
                assert_eq!(x.to_f32(), 0.0, "cell {i}");
            }
        }
        Err(e) => eprintln!("clean error (acceptable): {e}"),
    }
}

#[test]
fn var_f16_compact_with_state_buffer_does_not_wrap() {
    let a = Array::compact_ndarray(&array![[f16::from_f32(1.0)]]).unwrap();
    let v = a.broadcast(&[N, 1]).var(1, 0.0);
    let mut p = ArrayParams::new();
    p.block_shape(&[N as u32]);
    let ctx = ReadContext::default();
    match v.compact_with(p, &ctx) {
        Ok(c) => {
            let got = c.to_ndarray_sub::<Ix1>(&[N - 4..N], &ctx).unwrap();
            assert!(got.iter().all(|x| x.to_f32() == 0.0), "{got:?}");
        }
        Err(e) => eprintln!("clean error (acceptable): {e}"),
    }
}

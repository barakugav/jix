// F013 internal repro, binary form (the inline-test form is F013_internal.rs; this lighter
// driver was used because a full `--lib` test build did not fit the shared disk).
//
// `NdCopier` is pub(crate), so this needs a scratch COPY of the crate with one helper added to
// src/util/bench_util.rs (reachable as jix::__private::bench_util):
//
//   /// F013: NdCopier built from `new_dtype`, copy() called with `copy_dtype`.
//   #[doc(hidden)]
//   pub unsafe fn nd_copy_split_dtype(
//       src: &[u8], dst: &mut [u8], shape: &[usize], src_strides: &[usize],
//       dst_strides: &[usize], new_dtype: &Dtype, copy_dtype: &Dtype,
//   ) {
//       unsafe {
//           NdCopier::new(new_dtype).copy(
//               PtrNoalias::from_slice(src), PtrMutNoalias::from_slice(dst),
//               shape, src_strides, dst_strides, copy_dtype,
//           )
//       }
//   }
//
// and this file saved as src/bin/f013.rs. Run (S = the scratch crate):
//   CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false \
//     CARGO_TARGET_DIR=<scratch>/target-F013 cargo run --manifest-path $S/Cargo.toml --all-features --bin f013
//   (and once more without CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false to see the debug_assert)
//
// It checks the CORRECT behavior: a copy whose byte ranges are sized per copy()'s Safety doc
// (from the itemsize of the dtype passed to copy()) touches no byte outside those ranges. The
// ranges sit at the start of guard-filled 8-aligned buffers so an overrun is visible rather than
// a crash. Exits 1 (or panics) while the bug exists.

use jix::__private::bench_util::nd_copy_split_dtype;
use jix::dtype::{Dtype, Dtyped};

const GUARD: u8 = 0xEE; // dst guard
const SRC_GUARD: u8 = 0xAA; // src guard (different, so a copy of it into dst is visible)

#[repr(C, align(8))]
struct Backing([u8; 64]);

fn span_bytes(shape: &[usize], strides: &[usize], itemsize: usize) -> usize {
    shape.iter().zip(strides).map(|(&l, &s)| s * (l - 1)).sum::<usize>() + itemsize
}

fn check(name: &str, new_dtype: &Dtype, shape: &[usize], strides: &[usize]) -> bool {
    let copy_dtype = u32::DTYPE;
    let span = span_bytes(shape, strides, copy_dtype.itemsize() as usize);
    let mut src = Backing([SRC_GUARD; 64]);
    for (i, b) in src.0[..span].iter_mut().enumerate() {
        *b = i as u8 + 1;
    }
    let mut dst = Backing([GUARD; 64]);
    unsafe {
        nd_copy_split_dtype(
            &src.0[..span],
            &mut dst.0[..span],
            shape,
            strides,
            strides,
            new_dtype,
            &copy_dtype,
        )
    };
    let guard_ok = dst.0[span..].iter().all(|&b| b == GUARD);
    let content_ok = dst.0[..span] == src.0[..span];
    println!(
        "{name}: new=({},{}) copy=(4,4) shape={shape:?} strides={strides:?} span={span} \
         dst[..span+8]={:02x?} -> {}",
        new_dtype.itemsize(),
        new_dtype.alignment().as_usize(),
        &dst.0[..span + 8],
        if guard_ok && content_ok {
            "ok".to_string()
        } else {
            format!(
                "FAIL (wrote past the {span}-byte dst span: {}, span content ok: {content_ok})",
                !guard_ok
            )
        }
    );
    guard_ok && content_ok
}

fn main() {
    let mut ok = true;
    // Control: copier built from the dtype that copy() gets.
    ok &= check("control_1elem", &u32::DTYPE, &[1], &[4]);
    ok &= check("control_1d", &u32::DTYPE, &[4], &[4]);
    ok &= check("control_2d", &u32::DTYPE, &[2, 2], &[8, 4]);
    // Mismatch: copier built for u64.
    ok &= check("single_element", &u64::DTYPE, &[1], &[4]); // nd_copy.rs:97-102
    ok &= check("1d_run", &u64::DTYPE, &[4], &[4]); // strided inner_loop::<u64>
    ok &= check("2d_iter", &u64::DTYPE, &[2, 2], &[8, 4]); // NdIterUnordered path
    if !ok {
        std::process::exit(1);
    }
}

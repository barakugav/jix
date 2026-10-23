// F013 internal repro (inline-test form; NOT RUN - a full `--lib` test build did not fit the shared
// disk; the equivalent binary driver F013_internal_bin.rs was run instead, see findings/F013.md).
// NdCopier::copy trusts that its `dtype` argument matches the one given to
// `NdCopier::new`. `NdCopier` is pub(crate), so this cannot run as an integration test; it is an
// inline test module appended to a scratch COPY of jix/src/util/nd_copy.rs:
//
//   S=bug-hunt/runs/2026-10-23-batch3/scratch/crate-F013   (copy of jix/, jix-macros path made absolute)
//   cat bug-hunt/runs/2026-10-23-batch3/repros/F013_internal.rs >> $S/src/util/nd_copy.rs
//   # 1) default debug build: the debug_assert in scalar_fn fires
//   CARGO_TARGET_DIR=bug-hunt/runs/2026-10-23-batch3/scratch/target-F013 \
//     cargo test --manifest-path $S/Cargo.toml --all-features --lib f013_repro
//   # 2) debug_asserts off (what a release build does): the out-of-span writes happen
//   CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false CARGO_TARGET_DIR=bug-hunt/runs/2026-10-23-batch3/scratch/target-F013-nda \
//     cargo test --manifest-path $S/Cargo.toml --all-features --lib f013_repro
//
// Each test asserts the CORRECT behavior: a copy whose byte ranges are sized per copy()'s Safety
// doc - from the itemsize of the dtype passed to copy() - must not touch any byte outside those
// ranges. The ranges are carved out of larger guard-filled buffers so an overrun is observable
// instead of crashing. The tests fail while the bug exists (a debug_assert panic in debug, guard
// bytes overwritten with debug_asserts off). If the fix removes the `dtype` parameter from copy(),
// these tests stop compiling and should be dropped or rewritten against the new API.

#[cfg(test)]
mod f013_repro {
    use super::*;
    use crate::dtype::Dtyped;
    use crate::util::strided_span_bytes;

    const GUARD: u8 = 0xEE; // dst guard
    const SRC_GUARD: u8 = 0xAA; // src guard (different, so a copy of it into dst is visible)

    #[repr(C, align(8))]
    struct Backing([u8; 64]);

    /// Copies `shape` with a copier built for `new_dtype` but `copy()` called with `copy_dtype`.
    /// The src/dst slices are exactly `strided_span_bytes(shape, strides, copy_dtype.itemsize)`
    /// long, at the start of 8-aligned 64-byte buffers whose remaining bytes are guards.
    /// Returns (src backing, dst backing, span).
    fn run(
        new_dtype: &Dtype,
        copy_dtype: &Dtype,
        shape: &[usize],
        strides: &[usize],
    ) -> (Backing, Backing, usize) {
        let span = strided_span_bytes(shape, strides, copy_dtype.itemsize() as usize);
        let mut src = Backing([SRC_GUARD; 64]);
        for (i, b) in src.0[..span].iter_mut().enumerate() {
            *b = i as u8 + 1;
        }
        let mut dst = Backing([GUARD; 64]);
        let copier = NdCopier::new(new_dtype);
        unsafe {
            copier.copy(
                PtrNoalias::from_slice(&src.0[..span]),
                PtrMutNoalias::from_slice(&mut dst.0[..span]),
                shape,
                strides,
                strides,
                copy_dtype,
            )
        };
        (src, dst, span)
    }

    fn check(new_dtype: &Dtype, shape: &[usize], strides: &[usize]) {
        let (src, dst, span) = run(new_dtype, &u32::DTYPE, shape, strides);
        eprintln!(
            "shape={shape:?} strides={strides:?} span={span} dst[..span+8]={:02x?}",
            &dst.0[..span + 8]
        );
        assert!(
            dst.0[span..].iter().all(|&b| b == GUARD),
            "copy wrote past the {span}-byte dst span: dst[{span}..{}] = {:02x?}",
            span + 8,
            &dst.0[span..span + 8]
        );
        assert_eq!(&dst.0[..span], &src.0[..span], "dst span content");
    }

    // Control: copier built from the same dtype that copy() gets. Passes.
    #[test]
    fn f013_control_same_dtype() {
        check(&u32::DTYPE, &[1], &[4]);
        check(&u32::DTYPE, &[4], &[4]);
        check(&u32::DTYPE, &[2, 2], &[8, 4]);
    }

    // 1-d single-element fast path (nd_copy.rs:97-102): reads/writes 8 bytes of a 4-byte span.
    #[test]
    fn f013_single_element() {
        check(&u64::DTYPE, &[1], &[4]);
    }

    // 1-d run of 4 u32-contiguous elements: with T = u64 the stride 4 is "not contiguous", so the
    // strided inner_loop reads/writes 8 bytes at offsets 0, 4, 8, 12 -> bytes 16..20 out of span.
    #[test]
    fn f013_1d_run() {
        check(&u64::DTYPE, &[4], &[4]);
    }

    // 2-d iterator path: same overrun at the last element.
    #[test]
    fn f013_2d() {
        check(&u64::DTYPE, &[2, 2], &[8, 4]);
    }
}

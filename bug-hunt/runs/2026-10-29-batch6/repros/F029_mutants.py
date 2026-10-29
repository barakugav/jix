"""F029: apply three "bool8 as raw byte" regressions to a SCRATCH COPY of the jix crate, to show
that the existing suite misses them and repro_2026_10_29_batch6__f029.rs catches each one.

Usage (never on the tracked tree):
  python F029_mutants.py <scratch crate dir> [m1] [m2] [m3]     (default: all three)
  CARGO_TARGET_DIR=<dir> cargo test --manifest-path <scratch>/Cargo.toml --all-features --lib -- \
      ops::bitwise ops::where_op ops::reduction scalar::bool ops::cmp ops::cast storage::elementwise_pipeline
  cp jix/tests/repro_2026_10_29_batch6__f029.rs <scratch>/tests/ && \
  CARGO_TARGET_DIR=<dir> cargo test --manifest-path <scratch>/Cargo.toml --all-features \
      --test repro_2026_10_29_batch6__f029

Result in run 2026-10-29-batch6 (x86_64, AVX-512): with m1+m2+m3 applied the 258 existing lib tests
of those modules all pass (scalar::bool::tests::any_non_zero_byte_is_true included); the new test
fails on each mutant applied alone ("bitand on non-canonical bytes", "where -> u8", "sum").
"""

import os
import sys

MUTANTS = {
    # m1: bool8 & | ^ on the raw byte (2 & 1 == 0)
    "m1": ("src/ops/bitwise.rs",
           "                impl $Trait for [bool8] {\n                    fn $f = |a, b| a $op b,",
           "                impl $Trait for [bool8] {\n                    fn $f = |a, b| bool8(a.0 $op b.0),"),
    # m2: where's vectorized LANES body tests only the low bit of the condition byte
    "m2": ("src/ops/where_op.rs",
           "                            out[k] = if bool::from(cond_chunk[k]) {",
           "                            out[k] = if cond_chunk[k].0 & 1 != 0 {"),
    # m3: bool8 sum: the bulk (SIMD-enabled) fold adds the raw byte, the scalar update stays 0/1
    "m3": ("src/ops/reduction.rs",
           '    impl_sum!(bool8, u64, is_precise = true, simd = "enabled");',
           "    impl_sum!(bool8, u64, is_precise = true, {\n"
           "        #[inline(always)]\n"
           "        fn update_bulk<const N: usize, S: fearless_simd::Simd>(\n"
           "            states: [u64; N], items: [Self; N], simd: S,\n"
           "        ) -> [u64; N] {\n"
           "            let _ = simd;\n"
           "            std::array::from_fn(|i| states[i].wrapping_add(items[i].0 as u64))\n"
           "        }\n"
           "    });"),
}

root = sys.argv[1]
assert os.path.abspath(root) != os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../jix")), \
    "refusing to mutate the tracked crate"
for name in sys.argv[2:] or list(MUTANTS):
    path, old, new = MUTANTS[name]
    path = os.path.join(root, path)
    s = open(path).read()
    assert s.count(old) == 1, (name, path)
    open(path, "w").write(s.replace(old, new))
    print("applied", name)

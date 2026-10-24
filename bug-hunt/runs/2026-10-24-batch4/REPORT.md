# jix bug hunt - run 2026-10-24-batch4

- Date: 2026-10-24. The code is at commit `7a1bafd` ("Bug hunt: sync with main after #90"). #90 sealed `ArrayStorage` (K50) and made negative ndarray strides an error (K45).
- Scope: 8 hunter units (listed in section 1).
- Process: Hunt, then Dedup, then Verify, then Report. Each finding got one adversarial verifier, which tried to refute it and then built a repro. Severity follows the table in `bug-hunt/PRIMER.md`.
- What is in this directory:
  - `findings/<ID>.md`: one write-up for each of the 18 findings.
  - `repros/` holds 15 Rust repros (`repro_2026_10_24_batch4__fNNN.rs` for F002-F009 and F012-F018) and 9 Python repros (`F001.py`, `F002.py`, `F004.py`, `F006.py`, `F008.py`, `F009.py`, `F010.py`, `F011.py`, `F015.py`).
  - `scratch/` holds the hunters' Python probes, two Miri logs (`f005_verify_miri_plain.log`, `elementwise_ops_where_transmute_subdtype_miri3.log`), and the pipeline-simd scratch crate `crate-pipeline_simd/`. Its `tests/fallback.rs` holds the SIMD differential harness, run at the Fallback level.
- Paths in the write-ups that are now out of date:
  - Rust repros are cited as `jix/tests/repro_2026_10_24_batch4__fNNN.rs`. They have moved to `repros/` with the same names.
  - The 27 hunter scratch tests (`jix/tests/hunt_2026_10_24_batch4__*.rs`) were deleted. These include the release timing harnesses cited in F004 and F012, F001's Rust control test, and the pipeline-simd harnesses `_1`/`_2`/`_3`.
  - This run's test binaries and fingerprints under `jix/target` and `target-i686`, and `scratch/target-pipeline_simd` (742 MB), were also deleted.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 18 |
| plausible | 0 |
| refuted | 0 |
| intended | 0 |
| unverified (not sent to a verifier) | 0 |

| severity (confirmed + plausible) | count | IDs |
|----------|------:|-----|
| critical | 0 | - |
| high | 1 | F001 |
| medium | 6 | F002, F003, F004, F005, F006, F012 |
| low | 11 | F007, F008, F009, F010, F011, F013, F014, F015, F016, F017, F018 |

By category (confirmed): wrong-result 6, questionable-semantics 4, overflow 3, pathological-performance 2, unsound-api 1, resource 1, panic 1.

Hunters run: 8. Hunters that failed: `[]`.

| hunter | raw reports | findings |
|--------|------------:|----------|
| reduction-op-kernels | 4 | F002, F007, F008, F009 |
| pipeline-simd | 0 | - |
| py-ops-cmp-bitwise-where | 3 | F001, F010, F011 |
| shape-ops-slice-reshape | 5 | F003, F004, F012, F013, F014 |
| storage-core-leaf-backends | 3 | F005, F015 (two reports) |
| storage-core-array-read-paths | 3 | F005, F016, F017 |
| elementwise-ops-bitwise-cast | 1 | F006 |
| elementwise-ops-where-transmute-subdtype | 3 | F001, F005, F018 |

Dedup merged 22 raw reports into 18 findings:
- F005 was reported by three hunters.
- F001 was reported by two hunters.
- F015 merges two reports from one hunter.
- No report was dropped as a `KNOWN.md` duplicate.

Ten findings come from earlier fixes that were incomplete or did not reach this site, or are the same kind of bug as an earlier entry in a new place:
- F006: the K28 fix is incomplete. K28 should be reopened.
- F007: the same bug as K30, in the reduction fold.
- F013: the same kind of bug as K36, in Slice.
- F016: the K16 fix misses empty regions.
- F018: an unchecked add that the K42 fix left behind.
- F005: the same kind of bug as K50; `ElementType` was left unsealed.
- F015: the K45 fix also rejects valid strides on length-1 dims.
- F017: the Rust counterpart of K40.
- F009: the K27 formula at a different call site.
- F014: the same kind of bug as K12 / K32.

Reachability:
- Reachable from Python (9): F001, F002, F004, F006, F008, F009, F010, F011, F015. F008 reaches Python only in the debug extension (`maturin develop`); release wheels are not affected.
- Rust only (9): F003, F005, F007, F012, F013, F014, F016, F017, F018.
- Panics that happen only in debug builds, with correct release behavior: F008, F013, F018, and the second report in F015. F002's `std` panics in debug builds and returns NaN silently in release.
- 32-bit specific: F016 returns a truncated shape on i686 (on 64-bit it panics instead). F013 triggers at much smaller steps on 32-bit.
- The only memory-safety finding is F005, which needs a user `ElementType` impl in safe Rust. No memory-safety issue is reachable from Python.

Headlines:
- **F001 (high): Python `numpy()` gives wrong values for inner-shape dtypes.** For any dtype with an inner shape (a sub-array field read through `dtype_sub_field`, or `read_array` of such a file), `Array.numpy()` returns overlapping elements and uninitialized heap bytes. The Rust core is correct; only the jix-py read-out is wrong.
- **Earlier fixes that are incomplete or have gaps:**
  - F006: f64 -> f16 still mis-rounds on every shipped target except macOS arm64, about 8x as often as before the K28 fix. The only test is gated to aarch64, and the property oracle calls the same buggy function.
  - F016: the K16 check misses empty regions.
  - F018: K42 left one unchecked add behind.
  - F015: the K45 fix rejects valid length-1 strides.
- **Pathological read amplification (F004, F012).** A split or merge Reshape, which includes Python `jix.flatten` and `reshape(-1)`, decompresses a whole block per output element: 400-5000x slower. A stepped Slice does the same: 200-2100x slower.
- **Trust boundary (F005).** `ElementType` is still safe and unsealed. A user impl gives heap over-reads and `assert_unchecked(false)` from safe code.
- **Numerics near the edges of f64:**
  - F002: variance goes negative and `std` panics in debug builds.
  - F009: the complex mean overflows where numpy's result is finite.

## 2. Findings

All findings are confirmed. They are sorted by severity, with ties in ID order.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| F001 | high | confirmed | wrong-result | Array.numpy() gives wrong values, including uninitialized bytes, for arrays whose dtype has an inner shape: numpy's expanded sub-array dims are used as the array's strides (and permuted layouts raise, out= never works) | `jix-py/src/array.rs:570` | [F001](findings/F001.md) |
| F002 | medium | confirmed | wrong-result | Welford variance goes negative/NaN when \|x - mean\| overflows f64 (var([1e308,-1e308]) = -inf, numpy inf); std trips debug_assert (PanicException in dev build) and silently returns NaN in release | `jix/src/ops/reduction.rs:1874` | [F002](findings/F002.md) |
| F003 | medium | confirmed | wrong-result | Rust range -> SliceItem: inclusive end -1 (`..=-1`, `0..=-1`) silently gives an empty slice and `-k..=-1` errors (Array::slice panics); u64/usize bounds >= 2^63 wrap to end-relative indices, `..=i64::MAX` overflows | `jix/src/ops/shape_ops/slice.rs:324` | [F003](findings/F003.md) |
| F004 | medium | confirmed | pathological-performance | Reshape reads every split/merge reshape (flatten, [N]->[a,b], block-aligned splits) one element at a time, decompressing a whole block per element: 400-5000x slower reads | `jix/src/ops/shape_ops/reshape.rs:296` | [F004](findings/F004.md) |
| F005 | medium | confirmed | unsound-api | ElementType is a safe, unsealed trait, but Plain/BlockTable/SubDtype/Transmute take their dtype from a user ET::dtype() without checking it: heap OOB read (Plain::read_data, SubDtype copy) and assert_unchecked(false) (BlockTable::decoder_config via write_to) from safe code | `jix/src/element_type.rs:16` | [F005](findings/F005.md) |
| F006 | medium | confirmed | wrong-result | K28 fix is incomplete: f64 -> f16 via half::f16::from_f64 (no_std software fallback) still mis-rounds on every shipped target except macOS arm64, about 8x as often as the pre-fix f32 double rounding | `jix/src/ops/cast.rs:55` | [F006](findings/F006.md) |
| F012 | medium | confirmed | pathological-performance | A stepped Slice over a non-plain inner makes one inner read per output position of the stepped dims, decompressing whole blocks each time: step-2 slices read 200-1500x slower than the whole array | `jix/src/ops/shape_ops/slice.rs:181` | [F012](findings/F012.md) |
| F007 | low | confirmed | resource | reduce_unordered over wide Dtyped elements overflows a 2 MiB thread stack and aborts: the one-cell fold keeps 16 (strided) / 8 (contiguous) lanes of T whatever the element width (K30 analog in the reduction fold) | `jix/src/ops/reduction.rs:768` | [F007](findings/F007.md) |
| F008 | low | confirmed | overflow | ReductionOp::new adds 4 to the reduced element count unchecked (reduction.rs:177): when 2+ dims are reduced over a valid 1-byte-dtype lazy shape with more than 2^64 - 5 reduced elements, the constructor panics in debug builds instead of returning Ok | `jix/src/ops/reduction.rs:177` | [F008](findings/F008.md) |
| F009 | low | confirmed | wrong-result | Complex mean divides the sum by Complex(n, 0) with num-complex's textbook division, so sum * n overflows: mean([1.7e308+0j, 0j]) = inf+0j and the mean of 1e6 copies of 1e302+1e302j = inf+infj, while numpy and jix's f64 mean are finite | `jix/src/ops/reduction.rs:1766` | [F009](findings/F009.md) |
| F010 | low | confirmed | questionable-semantics | clamp(bool, min, max) returns uint8 while clamp with one bound, maximum and minimum return bool: bool8 is last in the clamp dispatch table, so it is never chosen | `jix-py/src/ops/cmp.rs:447` | [F010](findings/F010.md) |
| F011 | low | confirmed | questionable-semantics | bitwise_rotate_left/right with a Python int value: all-Python-int calls raise TypeError (the shifts work), and a Python int value next to a typed u32 count is rotated as int8 (-128 for 64, OverflowError for 200) | `jix-py/src/ops/bitwise.rs:432` | [F011](findings/F011.md) |
| F013 | low | confirmed | overflow | Slice plain fast path computes inner_stride * step unchecked: a step larger than the dim (output length 1) overflows - debug panic, release returns a wrapped stride on the extent-1 dim | `jix/src/ops/shape_ops/slice.rs:161` | [F013](findings/F013.md) |
| F014 | low | confirmed | panic | SliceSpec::new / From<&[I]> panic (ArrayVec capacity) on more than 8 items, so the fallible Slice::new never gets to return InvalidIndex | `jix/src/ops/shape_ops/slice.rs:252` | [F014](findings/F014.md) |
| F015 | low | confirmed | questionable-semantics | A negative stride on a never-stepped (length 0/1) dim is rejected on input (Rust ndarray constructors and Python asarray/compact/operands), although numpy/ndarray report the array as C-contiguous/standard layout and numpy(out=) accepts the same layout; the same line also has a debug-only multiply overflow for a huge length-1 stride | `jix/src/storage/plain.rs:520` | [F015](findings/F015.md) |
| F016 | low | confirmed | wrong-result | to_ndarray_sub sizes its ndarray with a per-dim `as usize` and bounds the region only on 32-bit, via the element product (0 for an empty region): empty regions with huge dims get a truncated shape on 32-bit (to_ndarray/to_plain return Ok [0, 5] for [0, 2^32+5]) and panic inside ndarray on 64-bit ([0, 2^62, 3]) | `jix/src/array.rs:774` | [F016](findings/F016.md) |
| F017 | low | confirmed | questionable-semantics | Array::to_ndarray docs promise a contiguous row-major (C-order) result, but the result follows read_layout_order (F-order for permuted views and F-order Plain sources) | `jix/src/array.rs:577` | [F017](findings/F017.md) |
| F018 | low | confirmed | overflow | Dtype::from_fields computes the struct itemsize with an unchecked u16 add before check_fields, so debug builds panic instead of returning Err on oversized or malformed field lists | `jix/src/dtype.rs:491` | [F018](findings/F018.md) |

### Bugs

**F001** (high; Python).
`numpy()` allocates with a NumPy sub-array descr, which NumPy expands into trailing dims. It then reads the array's strides from the last NumPy strides, so the outer dims get the inner strides, elements overlap, and the tail of the `np.empty` buffer comes back uninitialized. Permuted layouts raise `ValueError`, and `numpy(out=)` can never succeed.
Fix: allocate with the base descr and shape `out_shape + inner_shape`, take the strides from the first `ndim` NumPy dims, and make `check_output_array` accept such an `out`.

**F002** (medium; Rust and Python).
With finite f64 or complex128 data whose spread exceeds f64::MAX, the Welford update's `x - mean` overflows and `m2` gets `-inf`. So `var([1e308, -1e308]) = -inf` (numpy gives `inf`), other inputs give NaN, and the result depends on which fold path runs.
`std` then fails a `debug_assert` (PanicException in the dev extension). In release it returns NaN silently, because the Sqrt status is dropped.
Fix: keep m2 non-negative (e.g. `delta*delta*(n-1)/n`), and clamp or propagate the status in `std`.

**F003** (medium; Rust only).
`impl_from_range!` turns an inclusive end `e` into `e + 1`. So `..=-1` and `0..=-1` silently select nothing, and `-k..=-1` errors (`Array::slice` panics). ndarray maps an inclusive -1 to `None`.
`u64`/`usize` bounds >= 2^63 wrap to end-relative indices: `0..u64::MAX` silently drops the last element instead of erroring. `..=i64::MAX` overflows the `+ 1`.
Fix: map an inclusive -1 to `None`, use a checked add, and saturate unsigned bounds to i64::MAX so that `resolve` rejects them.

**F004** (medium; Rust and Python).
Reshape's read-time scan runs past the last orig dim on its first miss. Any reshape whose leading new dim splits or merges dims (flatten, `[N]->[a,b]`, `[6,4]->[2,3,4]`) therefore makes one 1-element inner read per output element, and each read decompresses a whole block.
That is 400-5000x slower than reading the array, about 7 us per element in release: `flatten` of a 256 MB array takes about 8 minutes. It happens even when the split is block-aligned, contrary to the code's own comments. The values are correct.
Fix: a scan that only skips orig dims with larger strides, matching on divisor lengths, `flat` computed over all dims, and range splitting for merges. Alternatively, a block cache in `ReadContext`.

**F005** (medium; Rust only, no `unsafe` needed).
`ElementType` is a safe, unsealed trait. In Plain, BlockTable, SubDtype and Transmute, `into_type` stores the user's `from_dtype` result without comparing its `dtype()` with the real dtype.
A user impl whose `dtype()` does not match the stored data gives heap over-reads whose bytes are returned to the caller (Plain, SubDtype; Miri confirms UB). It also gives `assert_unchecked(false)` in `BlockTable::decoder_config` on `write_to`.
Fix: seal the trait as K50 did (or make it `unsafe`), and check `new.dtype() == old` after `from_dtype`.

**F006** (medium; Rust and Python).
`half` is built without `std`, so `f16::from_f64` uses its software fallback on every shipped target except aarch64-apple-darwin. The fallback drops the low 32 mantissa bits before rounding: about 5e-4 of uniform random inputs come out 1 ulp off, 8x the pre-fix rate. The original K28 trigger still fails on x86_64 Linux.
The only correctness test is gated to aarch64, and the property oracle calls the same function.
Fix: a correctly rounded f64 -> f16 in jix (round-to-odd to f32, then `from_f32`). Make the test unconditional and give the oracle an independent reference.

**F012** (medium; Rust only).
A Slice with any step > 1 uses the general path unless it is a pull over a PLAIN_READ inner. That path makes one inner `read_data` per output position of the stepped dims, and on a Compact inner each read decodes whole blocks.
`a[::2]` reads 750-2100x slower than the whole array (about 15 minutes for 1 GiB). The verifier raised the severity from low to match F004.
Fix: read the bounding box once (in chunks) and copy it with `stride * step`, or at least document the cost.

**F007** (low; Rust only).
The one-cell reduction fold keeps 16 (strided) or at least 8 (contiguous) lanes of items and states on the stack, whatever the element width. `reduce_unordered` over 65000-byte elements needs 2.1-5.7 MiB of stack and aborts (SIGABRT) on a 2 MiB thread; the multi-cell control needs 396 KiB.
This is the K30 defect, whose fix did not touch reduction.rs. Debug builds abort with elements of only 4-8 KiB.
Fix: cap the lane count by bytes on both paths.

**F008** (low; Rust, and the Python debug extension).
`ReductionOp::new` computes `(reduced_prod + 4) as f64` without an overflow check. Reducing 2+ dims of a lazy 1-byte-dtype shape with more than 2^64 - 5 elements panics in debug builds. In release only a cost heuristic wraps, and the op stays correct.
Fix: do the add in f64, or use `saturating_add`.

**F009** (low; Rust and Python).
The complex mean computes num-complex `sum / Complex(n, 0)`, which forms `a*n` first. So `mean([1.7e308+0j, 0j])` is `inf+0j` (numpy gives 8.5e307), and the mean of 1e6 copies of 1e302+1e302j is `inf+infj`.
It affects complex128 only. It is the K27 formula at a separate call site, so a K27 fix in op2.rs would not cover it.
Fix: divide each component by the real count, and decide and pin the result of `mean([inf+1j])`.

**F013** (low; Rust only).
Slice's plain pull fast path multiplies `inner_stride * step` without an overflow check. A step larger than the dim (output length 1) overflows: debug builds panic, and release wraps a stride that is never stepped, with correct values.
This is the K36 defect at a site that fix did not cover. On 32-bit, a step of about 1e6 is enough for a 4000-byte row.
Fix: use stride 0 for output dims of extent 1.

**F014** (low; Rust only).
`SliceSpec::new` and `From<&[I]>` collect into an ArrayVec with capacity 8 and panic on 9 or more items. The panic comes before the fallible `Slice::new` can return InvalidIndex.
This is the same kind of bug as K12 / K32.
Fix: a `try_` variant, or carry the item count so `Slice::new` can return the error. At minimum, document the panic.

**F016** (low; Rust only).
`to_ndarray_sub` casts each extent with `as usize`. `check_region_size` runs only on 32-bit, and it bounds the element product, which is 0 for an empty region.
- On i686, empty regions with huge dims return Ok with a truncated shape (`[0, 5]` for `[0, 2^32+5]`), and `to_plain` does the same.
- On x86_64 they panic inside ndarray (`[0, 2^62, 3]`).
- A non-empty 2^63-byte region panics with "capacity overflow" on 64-bit.

Fix: on all targets, convert each extent with `usize::try_from` and require `prod(max(extent, 1)) * itemsize <= isize::MAX`.

**F018** (low; Rust only).
`Dtype::from_fields` computes the itemsize with an unchecked u16 add before `check_fields` runs. Debug builds panic on oversized or malformed field lists; release returns the correct Err.
This add was left over by the K42 fix.
Fix: use `checked_add`, or reuse the end offset that `check_fields` already computes.

### Questionable semantics

**F010** (low; Python).
The two-bound `jix.clamp` table lists `u8` before `bool8`, and a bool operand Safe-casts to anything, so `clamp(bool, min, max)` returns uint8. One-bound `clamp`, `maximum` and `minimum` return bool (since #82). The values are correct.
Fix: move `bool8` to the front of the table, and add bool to `_clamp_dtypes` in the tests.

**F011** (low; Python).
The rotate tables pair each value type with a u32 count.
- If both operands are Python ints, the call raises TypeError, because the default-precision step types the count as int64.
- A Python int value with a typed count is rotated as int8: `rotate_left(64, u32[1])` gives -128, and 200 raises OverflowError.
- The shifts handle both cases.

Fix: resolve untyped operands per slot: the value becomes int64, and the count becomes a bounds-checked u32.

**F015** (low; Rust and Python).
The K45 fix rejects every negative stride, including on dims of length 0 or 1, whose stride is never used.
- NumPy keeps such a stride after `x[::-1]` of a (1, n) array and still reports C_CONTIGUOUS, so `ascontiguousarray(x)` returns `x` unchanged and is no workaround.
- jix-py's `numpy(out=)` explicitly accepts the same layout.

Secondary report on plain.rs:522: a huge length-1 stride is multiplied by the itemsize unchecked, which panics in debug builds.
Fix: set the stride of any dim with length <= 1 to 0 before the sign check and the multiply. Change the unit test to invert an axis of length > 1.

**F017** (low; Rust only).
The first paragraph of the `Array::to_ndarray` docs promises a "contiguous row-major (C-order)" result. Since #52 the result follows `read_layout_order`, and a later paragraph of the same docstring says the result is not necessarily C-contiguous.
Callers that trust the first paragraph break: `as_slice().unwrap()` panics, and FFI consumers see the elements transposed. This is the Rust counterpart of K40.
Fix: change the docs only, and delete this repro.

### Re-running a repro

The Rust repros are in `repros/`. To run one:
1. Copy it back to `jix/tests/`.
2. From the repo root, run `cargo test --manifest-path jix/Cargo.toml --all-features --test <name>`, where `<name>` is the file name without `.rs`.
3. Delete the copy afterwards. `jix/tests/repro_*.rs` is gitignored, so a leftover copy does not show up in `git status`.

Example:

    cp bug-hunt/runs/2026-10-24-batch4/repros/repro_2026_10_24_batch4__f002.rs jix/tests/
    cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_24_batch4__f002 -- --nocapture --test-threads=1
    rm jix/tests/repro_2026_10_24_batch4__f002.rs

Every repro asserts the correct behavior, so it fails while its bug is present. Control tests pass.

Per-repro notes:
- **f002**: use `--test-threads=1`. In debug all 6 tests fail. With `--release`, the var values are the same and `std` returns NaN instead of panicking.
- **f003**: all 7 tests fail.
- **f004**: use `--test-threads 1`. 4 of 5 tests fail. The inner-read counts are exact; the fifth case is a timing check.
- **f005**:
  - Use `--test-threads=1`. All 3 tests fail in debug.
  - Miri for the Plain case: `python jix/run_miri.py --all-features --test repro_2026_10_24_batch4__f005 -- plain_wide_element_type_is_rejected`. The log is in `scratch/f005_verify_miri_plain.log`.
  - If the fix seals the trait, the file stops compiling. That also counts as fixed.
- **f006**: all 5 tests fail. The repro uses its own exact rounding reference, which does not depend on `half`.
- **f007**: the crashing cases run in child processes on a 2 MiB thread. 2 of 3 tests fail in debug and in release; the multi-cell control passes. Run `-- --ignored measure --nocapture` to print the smallest stack each case needs.
- **f008, f018**: the bugs need overflow checks. In debug, all 3 f008 tests and 2 of 3 f018 tests fail. With `--release` they pass.
- **f009**: both tests fail.
- **f012**: use `--test-threads 1`. 4 of 5 tests fail.
- **f013**:
  - In debug, all 3 tests fail. With `--release` they pass, because the wrapped stride is never stepped.
  - On 32-bit, `bug-hunt/tools/cargo-i686 test ...` also fails all 3. It needs `bash bug-hunt/setup.sh` first.
- **f014**: use `--test-threads=1`. 2 of 3 tests fail; the 3-item control passes.
- **f015**: all 5 tests fail in debug. The huge-stride test depends on overflow checks.
- **f016**:
  - On x86_64, use `--test-threads 1`. 3 of 5 tests fail (ndarray panic and capacity overflow).
  - On i686, `bug-hunt/tools/cargo-i686 test ... -- --nocapture --test-threads 1` fails 4 of 5 (truncated shapes).
- **f017**: all 3 tests fail. If the fix only changes the docs, delete this repro; the in-tree test at array.rs:2374 already pins the intended layout.

To run a Python repro:
- First build the debug extension into `.venv`, with `maturin develop` in `jix-py/` or with `bash bug-hunt/setup.sh`.
- Then run, for example, `.venv/bin/python bug-hunt/runs/2026-10-24-batch4/repros/F001.py`. Each script exits non-zero while its bug is present.
- F002 and F008 show the debug extension's PanicExceptions. With a release wheel, F002's `std` returns NaN, and F008 passes.
- F004 compares timings.

## 3. Unverified findings

None. Every deduplicated finding was sent to a verifier.

## 4. Refuted and intended

No finding was refuted, judged intended, or dropped as already known.

### Verifier corrections to hunter claims

- F001: `np.asarray(jix_array)` is not a trigger. `jix.Array` has no `__array__`, so numpy builds an object array without calling `numpy()`. There is also no out-of-bounds access: the wrongly assigned strides are always <= the real C-order strides.
- F004: the read-time length test is `orig_len >= new_len`, not equality. The conclusion does not change.
- F005: the hunter's release Miri run on `write_to` (`assume` called with `false`) was not re-run. The debug assertion plus the macro definition are enough.
- F006: the comment at cast.rs:806-808 ("on x86 it also goes through f32") is wrong for this build. The no-std fallback is worse than going through f32.
- F007: `block_shape([16])` on a Plain view does not trigger it, because Plain forces `block_shape_fixed_dims` to false. A Compact array does trigger it. The hunter's 60000-byte contiguous case passes in release (it needs 2028 KiB), so the repro uses 65000 bytes.
- F011: `rotate_left(1, np.int64(1))` raising TypeError looks intended, since the docstring requires a count that casts to u32. It is not part of the finding.
- F012: severity raised from low to medium, because the cost is hang-like on ordinary input, as in F004.
- F013: the push-mode general path does not multiply strides and is not affected.
- F016: the bug is not limited to empty arrays; an empty sub-region of a non-empty array triggers it too. Python is not affected.

### Hypotheses the hunters checked and ruled out

- reduction: the unsafe `NEEDS_FINALIZE=false` promise holds for Max, Min, Sum, Product, All, Any and Reduce (Output == State, and finalize returns the state unchanged).
- reduction: no `SUPPORTS_EMPTY=false` kernel can reach `init_state(None).unwrap()` or `Mean::finalize(.., 0)`, because `new()` rejects reduced extents of 0.
- reduction: ReduceKernel now sets `NEEDS_INDICES=false`, so the prompt was stale. Only ArgMax and ArgMin read indices.
- reduction: a panicking Reduce closure unwinds cleanly and the ctx can be reused. A closure that reads arrays with the same ctx is sound.
- reduction: K62 (empty float sum gives -0.0) is fixed.
- reduction: complex product and sum seeding, and the complex mean with inf components, match numpy 2.5.4.
- reduction: bool8 inputs with bytes 2/3/128/255 match numpy in every reduction. A differential test with strided output buffers was clean.
- pipeline-simd: mask narrowing lane order, chunk arithmetic, status ORing, macro arm selection and `dispatch_baseline!` are all correct.
- pipeline-simd: every `*_bulk` body is bit-exact with its scalar fn, signed zeros included, on SSE2, SSE4.2, AVX2, forced AVX-512 and Fallback (0 mismatches).
- py-ops: two-bound `clamp` with min > max or NaN bounds agrees with the one-bound paths and with `numpy.clip`.
- py-ops: `isclose` tolerance conversions, `where` broadcasting and promotion, and typed signed shift counts all behave like numpy.
- py-ops: common types in comparisons lose no more precision than numpy's; u64 against a signed int uses exact kernels.
- py-ops: K15 (numpy bool bytes other than 0/1) is still fixed, re-checked through where, logical_xor, bitwise ops, equal, maximum and cast.
- shape-ops: the unchecked `default_logical_strides` at reshape.rs:66 cannot overflow at this commit, because of the K13 fix in `try_shape_product` and `check_shape_overflow`.
- shape-ops: the constructor scan and the read-time scan diverge only in spec hints.
- shape-ops: the 32-bit `step as usize` truncation is harmless (argued, not run).
- shape-ops: Reshape and stepped Slice give correct values in an exhaustive sub-range sweep, with pull and push into padded F-order outs and a sentinel check.
- leaf-backends: K45 holds for dims of length > 1.
- leaf-backends: the Plain `read_data` span and offset math is correct.
- leaf-backends: the Scalar push fast and strided paths write exactly the region, also into unaligned outs.
- leaf-backends: FnStorage validates its index.
- leaf-backends: padded derived structs are no longer possible since #88.
- leaf-backends: `read_layout_order` is always a permutation.
- leaf-backends: NumPy never hands jix a NULL data pointer for an empty array.
- array-read-paths: the K14 out-span check is in place.
- array-read-paths: the 32-bit compare in `check_get_buffer_size` is fixed.
- array-read-paths: an itemsize mismatch in `to_ndarray_slice` is caught and returns Err.
- array-read-paths: the `try_product().unwrap()` in `compact_with` is unreachable because of the 2^30-byte block cap.
- array-read-paths: the slow-path pieces are disjoint and cover the region: a 4500-case differential on x86_64 and i686.
- array-read-paths: an unwritten-byte check over 20+ ops found no holes.
- array-read-paths: compaction block indexing is correct.
- bitwise-cast: all 110 cast pairs and every shift, rotate and count op match Rust `as` or the scalar fn, both contiguous (AVX2) and permuted (SSE2).
- bitwise-cast: every bool8 kernel treats any nonzero byte as true.
- where-transmute: the unsafe preconditions of `Where::read_data` hold, including bool8 read with `!= 0`, empty regions and 0-d regions.
- where-transmute: SubDtype's push `get_unchecked` is sound for validated dtypes.
- where-transmute: Transmute to a larger alignment over misaligned Plain bytes is clean under Miri.
- where-transmute: types S1 (a u16 wrap leading to a SubDtype OOB read) no longer holds; only F018's debug panic remains.
- where-transmute: types S8 (duplicate field names) is fixed by K52.
- where-transmute: the len-0 inner call in where_op.rs is safe.

## 5. Coverage

- **reduction-op-kernels**: read reduction.rs in full for the kernels (2066-3443) and the two macro variants (1276-1391), and read the driver, folds and scalar traits for context. Also read the jix-py reduction wrappers. Answered all 7 focus questions, plus Python differential tests against numpy.
- **pipeline-simd**: read scalar/bool.rs, simd.rs and traits_util.rs line by line, and their callers in the op and pipeline files. Checked the fearless_simd 1.1.0 source for every primitive jix uses. A bit-exact differential test of every `*_bulk` fn and all 110 casts ran on SSE2, SSE4.2, AVX2, forced AVX-512 and Fallback. No findings.
- **py-ops-cmp-bitwise-where**: read the jix-py cmp, bitwise, where, logical1, complex and sub_dtype files in full, plus the dispatch, promotion, operand and broadcast machinery and the `numpy()` path. Also read core cmp, where_op and sub_dtype.
- **shape-ops-slice-reshape**: read slice.rs and reshape.rs in full with their tests, plus the buf, util, error, params and NdIter contracts they rely on and the jix-py slice and reshape wrappers. Ran an exhaustive sub-range correctness sweep.
- **storage-core-leaf-backends**: read plain.rs, scalar.rs and array.rs:177-517 (leaf constructors, FnStorage) line by line, tracing contracts through their callers. Ran tests of Plain pull and push against ndarray on every sub-range, Scalar push exact-region tests, and Python probes.
- **storage-core-array-read-paths**: read array.rs:519-1735 in full, with all the contracts it relies on. Proved the preconditions of every unsafe block in the unit. Ran an unwritten-byte check over 20+ ops, a 4500-case slow-path differential on x86_64 and i686, and two Miri runs.
- **elementwise-ops-bitwise-cast**: read bitwise.rs, cast.rs, simd.rs, traits_util.rs and bool.rs in full, with the Op1/Op2 framework parts they use. Checked the fearless_simd primitives per SIMD level. Ran exhaustive cast-pair and shift/rotate differentials on SSE2 and AVX2.
- **elementwise-ops-where-transmute-subdtype**: read where_op.rs, transmute.rs and sub_dtype.rs in full with their tests, plus the buf, nd_iter_unordered, pipeline, dtype, element_type and block contracts. Traced each unsafe precondition, and ran experiments under Miri.

Skipped or only skimmed:
- reduction-op-kernels:
  - The tests module (reduction.rs:3443+) was only skimmed.
  - 32-bit (i686) behavior of the kernels was not examined.
  - The NEON and AVX-512 bulk paths (Max/Min NaN handling, Sum/Product `update_bulk_simd`) were reasoned about only.
  - The driver's 32-bit suspicions (`tile_base_reduced_idx`, tmp state size) belong to the driver unit and were not examined.
- pipeline-simd:
  - NEON was not run; it was checked by reading the source only.
  - i686 was not run.
  - `count_zeros` was skipped on forced AVX-512, because the CPU lacks vpopcntdq.
  - f16 and Complex have no SIMD bodies, so they were not exercised.
  - The pipeline driver and op semantics belong to other units.
- py-ops-cmp-bitwise-where:
  - Core bitwise.rs (shift and rotate kernels, SIMD) was not read beyond the context doc.
  - Core cmp tests and core logical1.rs / complex.rs were not read.
  - The Python tests were only grepped.
  - No 32-bit or Miri runs.
- shape-ops-slice-reshape:
  - No Miri and no i686 run.
  - The Scalar inner of the plain fast path was reasoned about only.
  - The jix-py `slice_unpack` internals (K29) are out of the unit.
- storage-core-leaf-backends:
  - No Miri, i686 or release builds. The 32-bit `as usize` casts in `Plain::read_data` were reasoned about only.
  - The NdCopier broadcast and untyped inner loops were covered only by the tests above.
  - The compaction internals (array.rs:1274+) belong to another unit.
- storage-core-array-read-paths:
  - array.rs:177-518 belongs to another unit.
  - The jix-py `numpy()` path was looked at only to check reachability.
  - The archive reader side was not covered.
  - Release-mode behavior was not tested beyond the two Miri runs.
- elementwise-ops-bitwise-cast:
  - AVX-512 and NEON were not run; they were checked by reading the intrinsics only.
  - SSE4.2 was covered only by the in-tree tests.
  - No i686 run.
  - The f16 and Complex<f16> cast tables were checked by reading and the Python differential only.
  - The jix-py bitwise dispatch was only skimmed.
- elementwise-ops-where-transmute-subdtype:
  - The `REQUIRE_ALIGNED=true` paths (armv7, wasm) were checked by reading only.
  - No i686 run.
  - Python `where` with struct dtypes was tested with only one packed 13-byte struct.

## 6. Open questions from hunters

### Trust boundaries and unsafe contracts
- Should `ElementType` be sealed, as `ArrayStorage` was in #90, or should every `element_type_change` check `new_et.dtype() == old` after `from_dtype`? The call sites are plain.rs:504, block.rs:133/232, sub_dtype.rs:95/192 and transmute.rs:148. Is any public storage expected to accept a third-party `ElementType`? (array-read-paths, where-transmute)
- Does `into_type::<LyingET>()` on Transmute also over-read? This was not tested. The SubDtype path is now confirmed in F005. (leaf-backends)
- Transmute's Safety doc still names Rust `bool` as the example of a restricted representation, but `bool` is no longer `Dtyped` (#89). Should `Transmute` and `transmute_elements` become safe? `transmute_elements(_dyn)` also unwraps the itemsize check with no `# Panics` doc. (where-transmute)
- In `Array.numpy()`, the out `StridedBuf` built with aliasing strides breaks the `from_raw_parts_mut` no-alias contract. Can any consumer (NdCopier, the pipeline's `PtrMutNoalias`) turn this into real UB beyond the wrong values of F001? (py-ops)
- The `Plain::new` / `plain_ndarray_ptr` Safety docs require a non-dangling pointer, yet jix itself passes a dangling pointer for an empty owned ndarray. The docs should probably say non-null, with dangling allowed only for a zero dim. They also do not say that the memory must be initialized and must not be written concurrently. (leaf-backends)

### Numerics
- `std` takes the sqrt of the variance after casting it to the output type. So the std of f16 data overflows to inf once var > 65504, e.g. `std(f16 [0, 1000])` gives inf where 500 is representable. numpy also gives inf there, but jix advertises f64 accumulation. Should `std` take the sqrt in f64 and cast once? (reduction)
- Should a NaN `std` that comes from a negative variance go through the error-mode machinery (`check_op_status`), the way `jix.sqrt(-x)` does? (reduction; see F002)
- Max/Min over bool8 return the raw input byte (e.g. 255) rather than canonical TRUE. This is visible only through `.view(np.uint8)` / `tobytes()`. Is that acceptable? (reduction)
- How should K28 be re-fixed? Enabling half's `std` only partly helps: x86 with F16C would go back to double rounding through f32. The hunter suggests round-to-odd via f32, or rounding the 53-bit significand directly. The aarch64-only test and its comment need fixing too. (bitwise-cast; see F006)
- On NEON and on the Fallback level (armv7, ppc64le), max/min of +0.0 and -0.0 can differ by position. This is documented as "either may be returned", so it was not reported. (pipeline-simd)

### Platform coverage (aarch64, AVX-512, i686)
- NEON was never run. Re-running the bit-exact SIMD differential on aarch64 would confirm agreement. The jix/tests harnesses were deleted in this cleanup; the Fallback-level copy in `scratch/crate-pipeline_simd/tests/fallback.rs` is the closest version that survives. (pipeline-simd)
- The in-tree `cast_f64_to_f16_rounds_once` should fail on Linux aarch64, which has no compile-time fp16 and no `std` for run-time detection. Does anyone run it there? (bitwise-cast)
- Running `cast_bulk_all_levels` / `simd_bodies_all_levels` on an Ice Lake+ x86 or an aarch64 runner would close the AVX-512 / NEON gap, including `count_zeros` on real AVX-512. `SimdTestValues` still lacks mid-range values such as 200.0, 40000.0 and 3e9. (pipeline-simd, bitwise-cast)
- Does F007 trigger with default `ArrayParams` on machines with 2 MiB or more of per-core L2? This was not tested on this 1 MiB-L2 host. (reduction)
- In i686 debug builds, `Array::compact_ndarray` of any 8-byte scalar trips the `debug_assert` at util/nd_copy.rs:93. This is K8 surfacing on every Plain push. Is the aligned NdCopier path sound there when `REQUIRE_ALIGNED` is false? (array-read-paths)

### 32-bit and huge-shape sizing
- On 32-bit, `OwnedBlockTableBuilder::start` allocates `blocks_loc` with `nblocks as usize`, and `write_compressed_block` indexes with `block_index as usize`. With 2^32 or more blocks this panics instead of erroring. Does `BlockArchiveWriter::start` have the same truncation? (array-read-paths)
- jix-py `Array.numpy()` computes `read_shape` with `as usize` per dim before its `nitems == 0` early return. Check the 32-bit empty-region / huge-dim case there, the analog of F016. (array-read-paths)
- `BlockTable::new` computes `nitems = nblocks * block_size` unchecked. It is practically unreachable, but archive validation might accept such a header. (array-read-paths)
- Release builds wrap the length-1 stride overflow in `strides_from_ndarray` (F015). Does any consumer multiply that wrapped stride by something other than an index? (leaf-backends)

### Read amplification and slicing design
- Is per-element read amplification in Reshape and stepped Slice ever intended? Possible fixes: match trailing or partial-length dims in Reshape (with the `flat` decomposition at reshape.rs:355-367 revisited); read the bounding box once for Slice; or add a small block cache in `ReadContext`. (shape-ops; see F004, F012)
- For inclusive ranges: map an inclusive end of -1 to `None`? For unsigned bounds >= 2^63, should `SliceItem` carry a wider type or an out-of-range marker, given that `From` cannot fail? (shape-ops; see F003)

### Python API semantics
- The `jix.dtype_sub_field` docstring promises ValueError for a missing field or a non-struct dtype, but it raises TypeError. Change the docstring or the error kind? (py-ops)
- `isclose` rejects 0-d numpy arrays and 0-d jix arrays as `rtol` / `atol`, while numpy accepts them. Intended? (py-ops)
- `jix.real()` returns non-complex inputs unchanged, but `jix.imag()` raises TypeError for them (numpy returns zeros). This is documented but asymmetric. (py-ops)
- Fixing F001: append the inner dims to the numpy allocation and skip them when collecting strides, and let `check_output_array` accept an `out` whose trailing dims match the sub-array. Do `jix.asarray` and `write_array` round-trip Rust-written `[T; N]` arrays correctly on other paths? (where-transmute)

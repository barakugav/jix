# jix bug hunt - run 2026-10-22-trial2

- Date: 2026-10-22. Code at commit `d1fe11d`, i.e. after #85 and #86 (#86 fixed K1, K2, K5, K6
  and K9-K13 from the pilot).
- Scope: 8 hunters - 6 module units, 1 concept unit, 1 dynamic unit (listed in section 1).
- Process: Hunt, then Dedup, then Verify (one adversarial verifier per finding, which tries to
  refute it and then builds a repro), then Report. Severity follows the table in
  `bug-hunt/PRIMER.md`. No finding needed a second verifier.
- Layout of this directory:
  - `findings/<ID>.md` - one write-up per finding (all 28 have one).
  - `repros/` - 16 Rust repros (`repro_2026_10_22_trial2__fNNN.rs`), 21 Python repros (`FNNN.py`),
    and `F028_internal.rs` (an inline crate test, see "Re-running a repro").
  - `scratch/` - hunter and verifier Python experiments.
- The write-ups cite Rust repros as `jix/tests/repro_2026_10_22_trial2__fNNN.rs`, where they lived
  during the run. They have since been moved to `repros/`.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 28 |
| plausible | 0 |
| refuted | 0 |
| intended | 0 |
| unverified (not sent to a verifier) | 0 |

| severity | count | IDs |
|----------|------:|-----|
| critical | 1 | F001 |
| high | 6 | F002, F003, F004, F005, F006, F007 |
| medium | 10 | F008, F009, F010, F011, F012, F013, F014, F015, F016, F018 |
| low | 11 | F017, F019, F020, F021, F022, F023, F024, F025, F026, F027, F028 |

By category: wrong-result 7, memory-safety 6, questionable-semantics 4, error-handling 3,
overflow 3, unsound-api 2, resource 2, panic 1.

Eight hunters ran. Hunters that failed: `[]`.

| hunter | kind | raw reports | findings |
|--------|------|------------:|----------|
| reduction-fold-loops | module | 2 | F002, F028 |
| pipeline-driver | module | 3 | F003, F017, F018 |
| py-ops-shape-ops | module | 8 | F008, F009, F010, F016, F019, F020, F021, F022 |
| codec-archive-codec-pipeline | module | 3 | F004, F005, F011 |
| shape-ops-copy-ops | module | 5 | F001, F003, F009, F022, F023 |
| elementwise-ops-cmp-clamp-approx | module | 3 | F002, F012, F013 |
| concept-integer-width | concept | 3 | F003 (two reports), F006 |
| dynamic-python-differential | dynamic | 11 | F007, F008, F013, F014, F015, F019, F021, F024, F025, F026, F027 |

Dedup merged 38 raw reports into 28 findings:
- F003 combines four reports from three hunters (the 32-bit sizing sites).
- F002, F008, F009, F013, F019, F021 and F022 were each reported by two hunters.
- The source lists account for every raw report, so no report was dropped as a `KNOWN.md`
  duplicate. Three findings are new triggers that the fixes of known issues did not cover:
  F004 (K1), F003 and F006 (K2).

Reachability:
- Python (21): F002, F004, F007, F008, F009, F010, F011, F013, F014, F015, F016, F018, F019,
  F020, F021, F022, F023, F024, F025, F026, F027.
- Rust only (7): F001, F003, F005, F006, F012, F017, F028.
  - F003 and F005 are 32-bit only. They probably reach Python on the x86 / armv7 wheels too, but
    that is untested because no 32-bit Python was available.
  - F028 is crate-internal (latent).

Headlines:
- **F001 (critical)**: 100% safe Rust can overflow the heap by handing a pull result back as the
  `out` of a larger read, because nothing checks `out`'s extent. Python is not affected.
- **Two #86 fixes are incomplete**:
  - F004 (K1): a malformed archive still writes uninitialized or stale pool bytes into the
    caller's `out`, also from Python.
  - F006 (K2): `MapMultiple::new` skips the new output-itemsize check.
  - F003 is the 32-bit part of K2, which the fix never touched.
- **Silent wrong results from Python on realistic input**:
  - F007: big-endian numpy arrays are read as little-endian.
  - F002: numpy bool bytes other than 0/1 (also UB).
  - F018: `numpy(out=x)` where `x` overlaps an input (also UB).
  - F013: uint64 vs int64 comparisons above 2**53.
  - F014: complex division with very large or very small divisors.
  - F015: f64 -> f16 double rounding.
- **32-bit targets** (the x86 and armv7 wheels ship) have two heap-overflow families, F003 and
  F005, and a silently wrong slice in Python (F016).

Pilot follow-ups that this run settled:
- Big-endian numpy input (pilot section 6.A) -> F007.
- Bool validity at `where_op.rs:305` (pilot 6.A) -> F002 for the numpy path. The archive-load path
  is still open (section 6.B).
- Other itemsize-widening constructors (pilot 6.C) -> F006; `MapMultiple` is the only one missed.
- 32-bit pull regions over 4 GiB and the `array.rs:960` `nitems` truncation (pilot 6.C) -> F003.
- Reduction state init order (pilot 6.E): holds. A cell's first visit is always stream position
  0 (reduction-fold-loops). The ordering bug that does exist is F028, and it is not an uninit read.

## 2. Findings

Sorted by severity. All findings are confirmed, and ties are in ID order.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| F001 | critical | confirmed | unsound-api | Safe-API UB: a writable pull result can be reused as `out` of a larger read; `out` extent is never checked, so Plain/Compact/Flip/Repeat/Tile write past it | `jix/src/storage/buf.rs:282` | [F001](findings/F001.md) |
| F002 | high | confirmed | memory-safety | numpy bool arrays with bytes other than 0/1 are wrapped zero-copy and read as Rust `bool` (UB); reductions, comparisons, bitwise ops and `where` return wrong results | `jix-py/src/ops/common/operand.rs:183` | [F002](findings/F002.md) |
| F003 | high | confirmed | memory-safety | 32-bit: read regions over 4 GiB / 2^32 elements are sized with truncating `as usize` and wrapping usize products: heap overflow from safe code (K2's 32-bit part) | `jix/src/storage/buf.rs:312` | [F003](findings/F003.md) |
| F004 | high | confirmed | memory-safety | K1 fix incomplete: `Decoder::decode` runs the filters over a short zstd output before the length check, so a failed read writes uninit or stale pool bytes into the caller's `out` | `jix/src/codec/mod.rs:315` | [F004](findings/F004.md) |
| F005 | high | confirmed | memory-safety | 32-bit: block byte size `block_size * itemsize` is never bounded to usize; all consumers wrap alike, so size checks pass and block reads go out of bounds | `jix/src/storage/compact.rs:395` | [F005](findings/F005.md) |
| F006 | high | confirmed | memory-safety | `MapMultiple::new` skips `check_shape_overflow` with the output itemsize (K2 fix incomplete); a widening `map_multiple` pull read overflows the heap | `jix/src/ops/map.rs:237` | [F006](findings/F006.md) |
| F007 | high | confirmed | wrong-result | Big-endian numpy dtypes are silently read and written as little-endian (asarray, compact, ops, write_array, struct fields, `dtype='>f8'` / cast) | `jix-py/src/dtype.rs:101` | [F007](findings/F007.md) |
| F008 | medium | confirmed | error-handling | Python indexing rejects slice start == dim length, so `a[n:]`, `a[n:n]` and any slice on a zero-length dim (even `z[:]`) raise IndexError | `jix-py/src/ops/shape_ops.rs:342` | [F008](findings/F008.md) |
| F009 | medium | confirmed | wrong-result | Roll reduces the shift with `rem_euclid(len as i64)`: wrong values for dims > i64::MAX, panic for len == u64::MAX with shift == i64::MIN | `jix/src/ops/shape_ops/roll.rs:54` | [F009](findings/F009.md) |
| F010 | medium | confirmed | overflow | `Concatenate::new` sums concat-dim lengths unchecked: panics in debug, wraps to a tiny wrong shape (e.g. `[4]`) in release | `jix/src/ops/shape_ops/concatenate.rs:111` | [F010](findings/F010.md) |
| F011 | medium | confirmed | resource | An 8 KB Strict-valid archive declaring a 34 GB block aborts the process (Rust and Python) on its first read | `jix/src/archive/array.rs:497` | [F011](findings/F011.md) |
| F012 | medium | confirmed | unsound-api | `Op2::shape` assumes its children's shapes stay equal (`assert_unchecked`); a safe user `ArrayStorage` whose shape changes makes safe code UB | `jix/src/ops/op2.rs:182` | [F012](findings/F012.md) |
| F013 | medium | confirmed | wrong-result | Python comparisons of uint64 with int64 run in float64 and return wrong booleans above 2**53 | `jix-py/src/ops/common/dtype_promote.rs:97` | [F013](findings/F013.md) |
| F014 | medium | confirmed | wrong-result | Complex division uses num-complex's unscaled formula: NaN / inf / 0 or ~1e-5 relative error for finite operands with very large or very small divisors | `jix/src/ops/op2.rs:781` | [F014](findings/F014.md) |
| F015 | medium | confirmed | wrong-result | f64 -> f16 conversion double-rounds through f32 (cast, Complex<f64> -> Complex<f16>, asarray(dtype=f16), Python floats with f16 arrays) | `jix/src/ops/cast.rs:59` | [F015](findings/F015.md) |
| F016 | medium | confirmed | wrong-result | `slice_unpack` resolves bounds through `Py_ssize_t`: wrong slices for dims > 2^31-1 on 32-bit CPython; stops >= 2^63-1 bypass the bounds check on 64-bit | `jix-py/src/util.rs:31` | [F016](findings/F016.md) |
| F018 | medium | confirmed | memory-safety | Python `numpy(out=x)` never checks that `x` overlaps a buffer the array reads: aliasing UB and order-dependent wrong values, also without explicit `asarray` | `jix/src/storage/elementwise_pipeline.rs:280` | [F018](findings/F018.md) |
| F017 | low | confirmed | resource | Contiguous inner-loop lane count ignores element width, so wide `Dtyped` elements overflow a 2 MiB thread stack and abort | `jix/src/storage/elementwise_pipeline.rs:688` | [F017](findings/F017.md) |
| F019 | low | confirmed | error-handling | `permute_dims` with more than 8 in-range entries panics (ArrayVec capacity, `PanicException`) instead of raising | `jix-py/src/util.rs:193` | [F019](findings/F019.md) |
| F020 | low | confirmed | questionable-semantics | Python bool index (`a[True]`, `jix.slice(a, True)`) is silently taken as integer 1/0, while `np.bool_` raises TypeError | `jix-py/src/ops/shape_ops.rs:273` | [F020](findings/F020.md) |
| F021 | low | confirmed | questionable-semantics | `jix.reshape` with -1 raises on every zero-size array, even when the inferred length is unambiguously 0 | `jix-py/src/ops/shape_ops.rs:768` | [F021](findings/F021.md) |
| F022 | low | confirmed | overflow | Dim lengths > i64::MAX are cast `as i64` in `parse_basic_index`, `flatten` and core `DimSlice::resolve`: such dims cannot be indexed, `x[-2**63]` overflows, `flatten` raises | `jix-py/src/ops/shape_ops.rs:318` | [F022](findings/F022.md) |
| F023 | low | confirmed | overflow | Repeat computes `out_strides[k] * repeats` even when output dim k has length 1, so valid `out=` buffers panic in debug builds (release correct) | `jix/src/ops/shape_ops/repeat.rs:181` | [F023](findings/F023.md) |
| F024 | low | confirmed | questionable-semantics | A numpy operand on the left of an operator with a `jix.Array` never reaches jix's reflected op: object ndarray of whole jix arrays; comparisons raise | `jix-py/src/array.rs:124` | [F024](findings/F024.md) |
| F025 | low | confirmed | error-handling | 8-dim array indexed with 8 items plus Ellipsis is rejected ("9 were indexed"): the NDIM_MAX tuple guard runs before Ellipsis is discounted | `jix-py/src/ops/shape_ops.rs:250` | [F025](findings/F025.md) |
| F026 | low | confirmed | panic | `dtype_from_numpy` `debug_assert` panics on packed numpy struct dtypes whose fields all have alignment 1 (debug builds only) | `jix-py/src/dtype.rs:208` | [F026](findings/F026.md) |
| F027 | low | confirmed | questionable-semantics | `Array.numpy()` docstring promises a C-contiguous result, but the allocation follows `read_layout_order` (F-order for Fortran / permuted inputs) | `jix-py/src/array.rs:491` | [F027](findings/F027.md) |
| F028 | low | confirmed | wrong-result | Multi-cell fold feeds ArgMax/ArgMin `update` out of stream order; multi-dim argmax returns a later tie / NaN (latent, crate-internal) | `jix/src/ops/reduction.rs:1109` | [F028](findings/F028.md) |

### Bugs

**F001** (critical, Rust only).
- A pull read returns a writable pool buffer sized exactly to its region. Safe code can pass it
  back as `out` of a larger read, and `check_out_buf` checks only the stride count and writability.
- Plain/Compact/Flip/Repeat/Tile then write past the end. Miri reports a heap out-of-bounds access,
  and the flip test trips a debug precondition check.
- Fix: in `check_out_buf`, check the strided span against the buffer length, or make pull results
  non-writable.

**F002** (high, Python).
- n-d numpy bool arrays are wrapped zero-copy without checking that the bytes are 0/1.
  `.view(bool)`, `np.frombuffer` and `np.memmap` produce other values. The kernels then read them
  as Rust `bool`.
- Result: UB (Miri) and wrong `all`/`sum`/`argmax`/`equal`/`&`/`where`. `jix.compact` persists the
  bad bytes, and `Plain::new`'s safety contract omits element validity.
- Fix: normalize or reject at ingest.

**F003** (high, 32-bit).
- Region extents are cast `as usize` and multiplied unchecked (buf.rs:312, array.rs:764/960/1022,
  error.rs:246-247, elementwise_pipeline.rs:200/869), while the copy walks the full u64 region.
- On i686, pull reads, `to_ndarray_slice` and `to_ndarray` overflow the heap (SIGSEGV in release),
  and a 2^32 dim reads as empty.
- Fix: one fallible region-sizing helper used at all these sites.

**F004** (high, Python).
- Incomplete K1 fix: `Decoder::decode` runs the filters over the whole buffer before `read_block`
  checks the decoded length.
- So a malformed archive that passes Strict validation makes a failed read leave uninitialized heap
  bytes, or another array's data, in the caller's `out`. Valgrind confirms the uninitialized read.
- Fix: check `nbytes` in `decode` before any filter runs.

**F005** (high, 32-bit).
- `block_size * itemsize` is never bounded to usize. Every consumer wraps it the same way, so the
  size checks agree on a scratch buffer that is too small, while the per-element offsets do not
  wrap.
- Result: out-of-bounds reads (wrong data, then SIGSEGV) from safe `compact_with` and from a
  Strict-valid archive. `compact_with` also silently stores a truncated block.
- Fix: bound the block byte size in `tune`.

**F006** (high, Rust).
- Incomplete K2 fix: `MapMultiple::new` is the one dtype-changing constructor that lacks
  `check_shape_overflow(shape, output itemsize)`.
- A widening map over a lazy 2^61+1 shape is accepted, and its pull read writes into an 8-byte
  buffer (SIGSEGV in release). `Map` rejects the same trigger.
- Fix: add the check, and make the `materialize_out_buf` size computation checked.

**F007** (high, Python).
- `dtype_from_numpy` maps by (kind, itemsize) and never looks at byte order.
- So `'>i4'` / `'>f8'` arrays and big-endian struct fields are read as little-endian by asarray,
  compact, every op and `write_array`, and `dtype='>f8'` / `cast` label little-endian output as
  big-endian. These are silent wrong values on ordinary input (FITS, HDF5, network data).
- Fix: convert to native order, or reject.

**F008** (medium, Python).
- `parse_basic_index` applies the integer-index bound `start >= len` to a slice start.
- So `a[n:]`, `a[n:n]` and any slice of a zero-length dim (even `z[:]`) raise IndexError, contrary
  to the `numpy()` docstring and core `Slice`.
- Fix: `start_norm > len`.

**F009** (medium, Rust + Python).
- `Roll::new` uses `shift.rem_euclid(len as i64)`, which is wrong for dims > i64::MAX (valid as
  lazy 1-byte views).
- Results: wrong values; roll is the identity for len == u64::MAX; shift == i64::MIN panics even
  in release.
- Fix the reduction (in i128 or u64), then rewrite `s + l - shift` at roll.rs:94/132 as
  `s + (l - shift)`.

**F010** (medium, Rust + Python).
- `Concatenate::new` adds the concat-dim lengths with a plain `+=` before `check_shape_overflow`.
- Past u64::MAX it panics in debug. In release it returns `Ok` with a wrapped shape (`[4]`) that
  holds input 0's data, and `borders` is out of order.
- Fix: `checked_add`.

**F011** (medium, Rust + Python).
- The archive header bounds a block only by its element count (u32), so an 8 KB Strict-valid file
  can declare a 34 GB block.
- The first read allocates the full scratch buffer infallibly, and the process aborts, including
  the Python interpreter.
- Fix: fallible block buffers, plus a zstd frame-content-size check before allocating.

**F012** (medium, Rust).
- `Op2::shape` turns the construction-time shape check into `assert_unchecked`.
- A safe user `ArrayStorage` whose `shape()` later changes makes `.shape()` UB: in release, the
  optimizer's false assumption leads to a SIGSEGV. into_type.rs:88/103 do the same with `dtype()`.
- Fix: drop the assume, or store the shape at construction.

**F013** (medium, Python).
- Under Safe casting, no comparison-table slot holds both uint64 and int64, so the pair is compared
  in float64. Values above 2**53 then compare wrong, and the bool output hides the precision loss.
- Only the 64-bit pair is affected: u64 against int8/16/32 stays correct.
- Fix: an exact mixed-sign path (a kernel or i128).

**F014** (medium, Rust + Python).
- Complex division delegates to num-complex's unscaled `(c*c + d*d)` formula.
- Finite quotients come out as `nan+nanj`, `inf+nanj`, `0j`, or with ~1e-5 relative error, once
  |divisor| exceeds ~1.8e19 or drops below ~1e-19 for complex64 (about 1e154 / 1e-154 for
  complex128).
- Fix: Smith's algorithm, which numpy uses.

**F015** (medium, Rust + Python).
- Every f16 cast is `f16::from_f32(x as f32)`, so f64 -> f16 rounds twice and is 1 ulp off for
  about 4-7 in 1e5 values. Affected paths: cast, `asarray(dtype=float16)`, and Python floats mixed
  with f16 arrays.
- The test oracle uses the same formula. `half::f16::from_f64` is not correctly rounded either.
- Fix: round to f32 with round-to-odd, then `from_f32`.

**F016** (medium, Python). `PySlice_Unpack` returns `Py_ssize_t`.
- On 32-bit CPython, a dim longer than 2^31-1 gives a silently wrong slice. This comes from an exact
  model of the code and was not run.
- On 64-bit, stops >= 2^63-1 are clamped and pass the strict bounds check: `a[0:2**63]` works. This
  was run.
- Fix: read `start` / `stop` / `step` as Python ints.

**F018** (medium, Python).
- `numpy(out=x)` never checks whether `x` overlaps a numpy buffer that the lazy chain reads. That
  includes implicit zero-copy wraps of views such as `x.T`.
- So a noalias `&mut` and a `&` cover the same bytes: UB, plus wrong values that depend on visit
  order and SIMD width. The docstring warns only about the explicit `asarray` form.
- Fix: an overlap check, then a temporary copy or ValueError.

**F017** (low, Rust).
- The contiguous lane count is `8 * prev_power_of_two(reg_bytes / MAX_ITEMSIZE)`, and
  `prev_power_of_two(0) == 1`, so wide elements (e.g. `[u8; 60000]`) keep 8 lanes of by-value
  `[T; 8]` temporaries.
- In release this needs ~2.8 MiB of stack and aborts on a 2 MiB thread. The strided path needs
  ~79 KiB.
- Fix: cap the lanes by bytes, as `LanesInfo` does for the strided path.

**F019** (low, Python).
- `permute_dims` takes an unbounded list, and `normalize_dims` collects it into an 8-slot ArrayVec.
- So 9 or more in-range entries raise `PanicException`, which `except Exception` does not catch.
- Fix: check the length before collecting.

**F022** (low, Rust + Python).
- Dims > i64::MAX are cast `as i64` in `parse_basic_index`, `flatten` and core `DimSlice::resolve`.
- Such dims cannot be indexed at all (`x[0]`, `x[:]`), `x[-2**63]` panics in debug, and `flatten`
  raises. Every path ends in an error, with no wrong values.
- Fix: keep lengths as u64.

**F023** (low, Rust + Python).
- `Repeat` multiplies the out stride by `repeats` even for a length-1 out dim, whose stride is never
  used.
- So ordinary numpy `out=` views panic with multiply overflow in debug builds, e.g. one row of a
  wide buffer with `repeat(2**50)`. Release output is correct.
- Fix: `wrapping_mul`, or 0 when `g_len == 1`.

**F025** (low, Python).
- The `len > NDIM_MAX` tuple guard runs before the Ellipsis is discounted.
- So an 8-d array indexed with 8 items plus `...` raises "9 were indexed". The same form works on a
  7-d array.
- Fix: allow NDIM_MAX + 1 raw items and leave the count to the post-Ellipsis check.

**F026** (low, Python, debug builds).
- A `debug_assert` assumes jix's `is_aligned` matches numpy's `isalignedstruct`. For structs whose
  fields all have alignment 1 (also nested) they differ.
- The debug extension used by the test suite therefore panics. Release wheels only get a cosmetic
  `align=True`.
- Fix: relax the assert.

**F028** (low, latent, crate-internal).
- The multi-cell fold feeds ArgMax/ArgMin `update` in walk order. When the reduced dims are
  permuted in memory or split across tiles, that is not stream order, so a multi-dim argmax picks a
  later tie or NaN.
- Public argmax is single-dim and unaffected.
- Fix: tie-break on `idx` in `update`, or use `merge_states` for index-aware kernels.

### Questionable semantics

**F020** (low, Python).
- `extract::<i64>()` accepts a Python `bool`, so `a[True]` returns row 1 and `a[False]` row 0.
  `np.True_` raises TypeError, numpy treats a bool index as a 0-d mask, and `dim=True` behaves the
  same way.
- Decide: reject `PyBool` (simplest), or implement mask semantics.

**F021** (low, Python).
- An extra `array_size == 0` term makes `jix.reshape(z, (-1, 3))` raise for every zero-size `z`.
  When the known dims are nonzero, the inferred length 0 is unambiguous, and `flatten(z)` and
  `reshape(z, 0)` work.
- Fix: drop that term and keep the `known_size == 0` check.

**F024** (low, Python).
- `jix.Array` lacks `__array_ufunc__ = None`. So `ndarray OP jix` returns an object ndarray of
  whole jix arrays, comparisons raise, and `np.int16(3) + jix` gets a different dtype than the
  reflected op. All 17 binary operators are affected.
- Setting the attribute at runtime makes the repro pass.

**F027** (low, Python).
- The `numpy()` docstring promises a fresh C-contiguous array, but since #52 the allocation follows
  `read_layout_order` (F-order for a Fortran `asarray` or permuted inputs).
- Values are correct and the behavior is intended; only the documented contract is stale.
- Fix the docstring, or add an `order=` keyword.

### Re-running a repro

The Rust repros are in `repros/`. Copy one back to `jix/tests/`, then run
`cargo test --manifest-path jix/Cargo.toml --all-features --test <name>` from the repo root, where
`<name>` is the file name without `.rs`. Delete the copy afterwards. `jix/tests/repro_*.rs` is
gitignored, so a leftover copy does not show up in `git status`. Example:

    cp bug-hunt/runs/2026-10-22-trial2/repros/repro_2026_10_22_trial2__f004.rs jix/tests/
    cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_22_trial2__f004
    rm jix/tests/repro_2026_10_22_trial2__f004.rs

Apart from the control tests, every repro asserts the correct behavior, so it fails, panics or
crashes while the bug is present.

Per-repro notes:
- **f001**:
  - Run the two tests one at a time with `-- --nocapture --exact <name>`.
  - `flip_push_into_undersized_pull_result` aborts with SIGABRT (debug precondition check).
  - `plain_push_into_undersized_pull_result` fails, after writing past the buffer without any
    report.
  - Miri:
    `python jix/run_miri.py --all-features --test repro_2026_10_22_trial2__f001 -- --exact plain_push_into_undersized_pull_result`.
- **f002**: fails natively. To see the invalid-bool UB, run it under Miri:
  `python jix/run_miri.py --all-features --test repro_2026_10_22_trial2__f002`.
- **f003, f005**: 32-bit only (the f003 tests return early on 64-bit).
  - Run them with the i686 runner in place of `cargo`. It needs the i686 build from
    `bash bug-hunt/setup.sh`.
  - Run one test at a time:
    `bug-hunt/tools/cargo-i686 test [--release] --manifest-path jix/Cargo.toml --all-features --test <name> -- --nocapture --test-threads 1 [--exact <test>]`.
  - Several f003 tests crash the process.
  - f005's `#[ignore]`d `far_element_read_after_compact` SIGSEGVs; run it with
    `-- --ignored far_element`.
- **f004**: run with `-- --nocapture --test-threads=1`; all 3 tests fail.
- **f006**:
  - Debug panics at buf.rs:312.
  - `--release` is a real heap overflow (SIGSEGV), so use `--test-threads=1` and do not run it
    casually.
  - The control test passes.
- **f009**: run with `-- --test-threads=1`; all 3 tests fail.
- **f010**: debug panics at concatenate.rs:111. `--release` shows the wrapped shape `[4]`.
- **f011**: installs a global allocator that refuses any request over 4 GiB, so the abort
  (SIGABRT) does not depend on RAM or overcommit.
- **f012**:
  - Debug fails in the `debug_assert`.
  - With `--release --test-threads=1`, `release_miscompile_reads_out_of_bounds` SIGSEGVs.
  - The control test passes.
- **f014, f015, f022**: plain test failures. f015 includes a sweep over 1e6 values.
- **f017**:
  - Run it with `--release`. Each case runs in a child process, so the abort shows up as a test
    failure.
  - In debug, both tests fail, the strided control included, because debug builds use much more
    stack in general.
  - `-- --ignored measure --nocapture` prints the minimum stack for each path.
- **f023**: fails in debug only. `--release` passes, because the overflowed value is never used in
  release.
- **f028**:
  - The Rust test shows the driver's fold order through the public `reduce_unordered`; all 3 tests
    fail.
  - The real-kernel repro, `F028_internal.rs`, has to be pasted into the `#[cfg(test)] mod tests`
    of `jix/src/ops/reduction.rs`. That is a tracked file, so do this on a scratch copy of the
    crate.
  - Then run `cargo test --manifest-path jix/Cargo.toml --all-features --lib f028_ -- --nocapture`.

Python repros cover F002, F004, F007, F008, F009, F010, F011, F013, F014, F015, F016, F018, F019,
F020, F021, F022, F023, F024, F025, F026 and F027.
- First build the debug extension into `.venv`, with `maturin develop` in `jix-py/` or with
  `bash bug-hunt/setup.sh`.
- Then run, for example, `.venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F007.py`. Each
  script exits non-zero while its bug is present.
- Per-script notes:
  - F011 runs the crashing read in a subprocess, so the abort shows up as an exit status.
  - F016 Part B runs only on 32-bit CPython (`sys.maxsize < 2**63 - 1`); elsewhere it is skipped.
  - F010, F022 (`x[-2**63]`), F023 and F026 rely on the debug extension's overflow checks or
    `debug_assert`. A release wheel behaves as described in their write-ups.

## 3. Unverified findings

None. Every deduplicated finding was sent to a verifier.

## 4. Refuted and intended

This run no finding was refuted, judged intended, or dropped as known. The lists below are the
tuning signal for the next run.

### Close calls (judged not intended; record the maintainer's decision in KNOWN.md or a context doc)

- F013: py-ops.md lists "u64 with any signed int -> f64, lossy above 2**53" as intentional. The
  verifier ruled that this covers arithmetic, where the float64 result shows the loss, but not
  comparisons, where the bool output hides it. The maintainers already removed the same loss for
  `u64 + Python int`.
- F018: the `numpy()` docstring documents out= aliasing as caller UB. The verifier ruled the warning
  too narrow (it misses implicit wraps and views of `out`) and an overlap check cheap. The hunter
  had filed it as low for that reason.
- F027: the layout behavior is deliberate (#52, like numpy `order='K'`). Only the docstring is
  wrong, so it was confirmed as a stale contract rather than marked intended.
- F020: depends on whether Python-list "bool is an int" index semantics are intended. Open question
  in section 6.F.
- F012: depends on whether safe user `ArrayStorage` impls are a supported trust boundary (context
  doc R3; section 6.A).
- F024: the operator behavior comes from numpy's default binop-deferral rules. But jix documents
  mixing numpy data with jix arrays and already defines the reflected dunders.

### Verifier corrections to hunter claims

- F001: broader than reported. No shape op is needed: a Plain push into a 1-byte pull result
  already overflows.
- F003: the verifier re-ran the release build on i686 (SIGSEGV in memset / memcpy).
- F009: release builds panic too, because Rust checks remainder overflow regardless of the
  overflow-checks setting.
- F010: the release wrap to shape `[4]` was run, not only reasoned.
- F011: made deterministic with a 4 GiB allocator cap. Without it, the outcome depends on
  overcommit.
- F012: escalated from the hunter's debug panic to a demonstrated release SIGSEGV.
- F013: narrowed to the (u64, i64) pair. A sweep of u64 against int8/16/32 extremes found 0
  mismatches.
- F014: the hunter's special-value claims (`x/inf`, `x/0j`, no error under the default errstate)
  were re-run. They are kept as informational, because the Rust doc says errors are not reported
  yet.
- F015: `half::f16::from_f64` is not a valid oracle or fix (482 of 1e6 values wrong in the sweep).
- F016: only the 64-bit half was executed. The 32-bit numbers come from an exact model of the code.
- F017: kept at low. Part of the debug-build stack use is general debug-build bloat, not this bug.
- F018: the hunter's shifted 1-d case looked correct only because of `arange` + 1; with `* 2` the
  corruption shows. Raised from low to medium.
- F027: the hunter's `np.frombuffer(memoryview(x))` claim is wrong: it raises BufferError rather
  than returning transposed data.
- F028: the verifier found a second trigger, C-order input with the reduced dims split across
  tiles, beyond the hunter's permuted-memory one.

### Hypotheses the hunters checked and ruled out

Reduction folds (reduction-fold-loops):
- Init keyed on `base_item_idx == 0`: a cell's first visit is always stream position 0, including
  for len-1, broadcast and reversed-memory reduced dims. On 32-bit, the logical-stride wrap at
  reduction.rs:477 could only cause a re-init (lost items), not an uninit read.
- One-cell vs multi-cell coalescing: a reduced dim can never merge with a kept dim. A kept len>1 dim
  with stride 0 needs an `out` that breaks its contract, and Python rejects such strides.
- Contiguous flags and aligned reads: the `assert_unchecked` at reduction.rs:925 holds. Aligned
  reads on REQUIRE_ALIGNED targets are backed by base and stride checks.
- `fold_pairwise` halves and tails, and `merge_states` slot reads: coverage is exact, and each slot
  is read once.
- Single-dim ArgMax indices across lanes, tails, pairwise halves and tiles: correct.

Element-wise pipeline (pipeline-driver):
- Contiguous flags vs cursor stride: always consistent, so the `assert_unchecked` at
  elementwise_pipeline.rs:608 holds.
- Staging cache: each run starts at pos 0 with the largest chunk, so the cached broadcast src is
  always full.
- Static vs dyn drivers: textually identical apart from containers (checked with diff).
- Spans, pointer adds, out stride 0, the tail loop, and status ORed per chunk: all correct.
- Itemsize-0 staging corner: unreachable, because `plain_ndarray` / `compact_ndarray` reject
  `[f32; 0]`.
- Op nodes' `N_OPERANDS` vs `operands()`: no mismatch found.

Python shape ops (py-ops-shape-ops):
- The `PySlice_Unpack` FFI call is sound; only its value semantics are wrong (F016).
- Context-doc suspicion at shape_ops.rs:724 (`known_size` unchecked): no longer holds. It is now
  `checked_mul` at 752-754, so py-ops.md should be updated.
- reshape / flatten `product::<u64>()`: cannot overflow, because overflowing shapes are rejected
  even with a 0 dim.
- Edge cases of stack, concatenate, insert/remove_dim, squeeze, flip, repeat, tile, broadcast,
  reshape and swapdims (0-d, 8-d, empty, duplicate and negative inputs): all raise Python
  exceptions, except permute_dims (F019).

Codec (codec-archive):
- Byte-shuffle SIMD math: correct whole-vector zip/unzip on every fearless_simd backend. The tests
  pass at sse2, sse4.2, avx2 and avx512+vbmi.
- Bit-shuffle inverse, filter buffer routing for 1-4 filters, every dst byte written, PoolBufs
  sliced to size, decompressor `UnsafeCell` without re-entry: all correct.
- `itemsize == 0`: unreachable, because `tune` and `build_from_data` check it.
- The `cfg(miri)` zstd stand-in's length assumptions hold for every current caller. As a result,
  Miri cannot exercise a short decode.

Copy ops (shape-ops-copy-ops):
- Flip / Roll / Repeat / Tile index math on 64-bit: correct for every branch, and src offsets stay
  within the pull buffer's span.
- No u64 overflow in `s_in + total`, `first_out_k` or `(g_end-1)*n`.

Comparisons and clamp (elementwise-ops):
- Float `simd_ne` is unordered on every x86 level and on NEON, so NaN != NaN holds.
- max/min NaN results match between the SIMD body and the scalar tail.
- AVX2 mask narrowing keeps lane order.
- The Greater/GreaterEqual operand swap, Clamp (shape check, `transmute_copy`, NaN, lo > hi) and
  the ApproxEq formula all match the docs.

Integer width (concept-integer-width):
- `FnStorage` indices are u64, with no truncation.
- Every itemsize-changing constructor except `MapMultiple` has the K2 output-itemsize check.

Python differential testing (dynamic-python-differential), about 65k cases:
- No value mismatches beyond the findings and the documented intentional deviations: promotions,
  `ddof` NaN, product overflow order, f16 accumulation, saturating float->int casts.
- `numpy(index, out=view)` never wrote outside `out`'s elements.

### Observations the hunters deliberately did not report (intended or minor)

- `ReduceKernel` keeps `NEEDS_INDICES = true`, so indices are computed for nothing. Performance
  only (reduction).
- The strided one-cell path uses 16 lanes while ArgMax's `ONE_CELL_LANES` is 8 on x86. Harmless
  (reduction).
- The kernel-trait doc says a cell can be "re-seeded ... when the reduced dim spans several bulks".
  It is tiles, not bulks, and only in the one-cell path. Doc nit (reduction).
- Integer indices >= 2**63, or `np.uint64`, give TypeError instead of IndexError
  (py-ops-shape-ops).
- Maximum/Minimum of +0.0 / -0.0 can differ between the scalar tail and NEON SIMD, and between
  dtypes. The docs allow either zero (elementwise-ops; section 6.E asks whether that is
  acceptable).
- ApproxEq returns true for MAX vs -MAX with rtol > 1, because both sides overflow to inf
  (elementwise-ops).
- Python comparisons with out-of-range Python ints raise OverflowError where numpy compares exactly.
  This is a loud deviation, not a silent one (elementwise-ops).
- A small user `read_size` makes reads quadratic: 13-50 s for a 222 KB array. Documented as a
  trade-off (dynamic).
- `f16_arr == 70000` is True for inf elements, because 70000 becomes inf in f16. numpy does the
  same (dynamic).
- `jix.equal(uint64_arr, 2**64-1)` raises OverflowError, which is intentional since #85 (dynamic).

## 5. Coverage

### Per hunter

- **reduction-fold-loops**: every line of reduction.rs 18-118 and 617-1115 (kernel contract, one-cell / pairwise / multi-cell folds, merge_states), plus the driver and kernels for context; 4500 random strided/permuted/broadcast layouts on x86_64 (AVX-512) and i686, and Miri on five fold paths, all passing except for F002 and F028.
- **pipeline-driver**: every line of elementwise_pipeline.rs:1-722, plus nd_iter_unordered(_dyn), buf_pool, nd_copy.rs:1-445, buf.rs and the Dtyped derive, with each unsafe precondition traced to where it is established; experiments on wide-element stack use, 32-bit truncation, ZST arrays and Python out= aliasing.
- **py-ops-shape-ops**: all 1259 lines of jix-py/src/ops/shape_ops.rs and util.rs:1-447, plus the array.rs index / out= consumers and the core slice / roll / concatenate contracts; seven scratch scripts covered every shape op's edge cases and settled the context-doc suspicions in the unit.
- **codec-archive-codec-pipeline**: every line of codec/mod.rs, filter/mod.rs, byte_shuffle.rs and bit_shuffle.rs; shuffle SIMD math checked against the fearless_simd sources at four x86 levels; round trips over 10 item sizes, n 1..1000, blocks 1..257 and chains of 0-4 filters.
- **shape-ops-copy-ops**: every non-test line of flip.rs, roll.rs, repeat.rs and tile.rs, plus buf.rs, buf_pool.rs and array.rs:740-1066; every branch hand-checked on 64-bit, and Roll wrap, Tile case C and Repeat run on i686.
- **elementwise-ops-cmp-clamp-approx**: every line of cmp.rs:1-1085, plus op2.rs:1-321, scalar/simd.rs, traits_util.rs and the jix-py cmp / clamp / isclose bindings; fearless_simd NaN and mask semantics checked; a full-pipeline integration test passed in debug and release.
- **concept-integer-width**: the post-K2 size invariant, every shape- or itemsize-changing constructor, the read entry points, `materialize_out_buf` / `materialize_pipeline_out_buf`, `check_get_buffer_size` and the stride helpers; three heap overflows confirmed on i686; about 614 `as` casts enumerated, most not examined in depth.
- **dynamic-python-differential**: about 14k shape-op chains, 19k element-wise calls, 11k reductions, 20k `numpy(index, out=view)` calls with sentinel checks and 1.8k file round trips, all compared against numpy, plus bit-exact signed-zero / shift / rotate checks; all on the x86_64 debug build.

### Skipped or only skimmed

- reduction-fold-loops
  - Context-doc suspicions outside its lines were not settled: reduction.rs 1471-1473, 1557-1559
    and 1732-1734 (scalar traits), and 501 and 573-578 (driver).
  - NEON / armv7 (REQUIRE_ALIGNED) fold paths were reasoned about only.
  - On i686, the compact 0-d test hits the known K8 nd_copy `debug_assert`.
- pipeline-driver
  - `pick_output_layout` (elementwise_pipeline.rs:885-941) was only skimmed, and its debug overflow
    at :891 was not tested.
  - Not settled:
    - the bool-from-raw-bytes suspicion at 845-856 (since covered by F002);
    - core_trait.rs:133-134 trusting external `ArrayStorage` impls;
    - `MapMultiple` with zero arrays.
  - Op nodes (op1, cmp Clamp, where_op, arr_sequence): only `N_OPERANDS` was checked.
  - SIMD kernels (scalar/simd.rs) were not reviewed, and Miri was not used.
  - REQUIRE_ALIGNED, armv7 and ppc64le were reasoned about only.
- py-ops-shape-ops
  - test_array_ops.py was only grepped for reshape.
  - F016's 32-bit half was reasoned from the code and the pyo3-ffi typedef; no 32-bit Python.
  - Core Slice / RemoveDim read paths were not reviewed.
- codec-archive-codec-pipeline
  - No Miri, because `cfg(miri)` replaces zstd.
  - NEON and Fallback byte-shuffle paths were read, not executed.
  - 32-bit Python was not run, so F005's Python reach is inferred.
  - The other codec-archive.md suspicions (common.rs, archive block.rs / array.rs, dtype.rs) belong
    to the archive units and were not reviewed.
- shape-ops-copy-ops
  - Not reviewed in depth:
    - NdIterUnordered internals;
    - the NdIter strides-offset extension (the existing proptests were trusted for Flip's src
      offsets);
    - the nd_copy.rs inner loops after line 200;
    - the Python index parser beyond F022's lines.
  - 32-bit Tile case B was argued, not run.
- elementwise-ops-cmp-clamp-approx
  - The archive bool-load path in safe Rust: no test was written.
  - NEON / AVX-512 / fallback paths were read only. The AVX2 bool-SIMD path is covered only by the
    existing kernel test.
  - Outside the unit, not examined:
    - Python scalar / weak promotion in general;
    - cast.rs:59 (later F015);
    - the jix-py power -> sqrt routing.
- concept-integer-width
  - Enumerated but not examined:
    - archive/array.rs:468 and archive/block.rs:374 `as BlockSize` before validation;
    - archive/block.rs:450/485 `section.size as usize`;
    - common.rs uleb128 / length casts;
    - storage/block.rs:180/189/310 and compact.rs:281/396;
    - the plain.rs:289 negative-stride cast;
    - slice / roll / flip signed arithmetic;
    - reduction.rs and nd_iter_unordered coalescing products;
    - codec filter casts and most jix-py casts.
  - The jix-py numpy materialization path on 32-bit was only partly read.
- dynamic-python-differential
  - Memory-safety contracts that differential testing cannot see:
    - bool validity and out= aliasing (both later found by module hunters);
    - self-overlapping `as_strided` out;
    - free-threading / concurrent mutation;
    - `PyWriter` memoryview lifetime.
  - Not run: 32-bit, release builds, NEON / AVX-512.
  - Not covered:
    - isclose and errstate warn / raise;
    - `dtype_sub_field` / `real` / `imag` on struct dtypes, and element-wise ops on struct dtypes;
    - corrupt or truncated archives;
    - Python file-like edge cases (short writes, None returns).
  - Sizes over 2**32 elements were touched only lightly.
- Across the run:
  - No 32-bit CPython was available.
  - The armv7 / ppc64le (REQUIRE_ALIGNED = true) paths were never executed.
  - Miri runs are recorded only for the reduction hunter and the F001 / F002 verifiers.

### Housekeeping

- The 16 Rust repros were moved from `jix/tests/` to `repros/`, keeping their names.
- The 18 hunter scratch files `jix/tests/hunt_2026_10_22_trial2__*.rs` were deleted. No other file
  in `jix/tests/` was touched.
- Python scratch scripts stay in `scratch/`. No tracked file was modified.

## 6. Open questions from hunters

### A. Trust boundaries and unsafe contracts

- Should `ArrayStorage` become an `unsafe trait`, or be sealed? Or should `Op2` / `IntoType` store
  and re-check shapes and dtypes instead of using `assert_unchecked`? The same question applies to
  every consumer that trusts a child's `read_data` buffer (F012; into_type.rs:88/103).
  (elementwise-ops)
- F001 fix choice: make pull results non-writable for safe callers, or have `check_out_buf` verify
  `strided_span_bytes(region, out.strides, itemsize) <= out.len()`. The second also catches a
  mismatched itemsize, such as a u8 pull buffer reused for an f64 read. (shape-ops-copy-ops)
- Python out= validation (jix-py/src/array.rs:255-287) rejects zero and negative strides but not
  self-overlapping non-zero ones, e.g. `as_strided(buf, (3,3), (8,8), writeable=True)`.
  - That breaks the `StridedBuf::from_raw_parts_mut` no-aliasing contract, and reduction states
    stored in `out` collide.
  - Is it UB anywhere else? (reduction-fold-loops)
- Should `numpy(out=)` check for overlap with every Plain leaf of the lazy chain (a byte-range
  intersection test) and copy through a temporary like numpy, instead of documenting UB (F018)?
  (pipeline-driver)

### B. Bool validity

- Pick one fix point for F002: validate on numpy ingest (operand.rs:183, compact, asarray), or treat
  bool leaves as u8 and normalize with `!= 0`. Either way, add a value-validity clause to
  `Plain::new`'s safety contract. (reduction-fold-loops, elementwise-ops)
- Does a `.jix` archive whose bool array holds bytes other than 0/1 reach the same typed reads (cmp,
  `where`, logical ops, cast-from-bool) through the safe Rust API? This was not tested. The codec
  permutes bytes verbatim, so decoded block bytes must be treated as untrusted. (reduction,
  elementwise-ops, codec-archive)

### C. Integer width and 32-bit targets

- Should F003 be fixed centrally rather than per op? The idea: `to_ndarray_sub`,
  `to_ndarray_buf_checked`, `materialize_out_buf`, `materialize_pipeline_out_buf` and
  `check_get_buffer_size` reject extents or byte products above usize / `isize::MAX`.
  (shape-ops-copy-ops, concept-integer-width)
- Python on 32-bit: `numpy()` truncates `read_shape` with `as usize` (jix-py/src/array.rs:529-531).
  Is a selection bigger than the address space rejected anywhere before `numpy_empty_ordered` /
  `StridedBuf::from_raw_parts_mut`? Someone should trace it on a 32-bit target.
  (py-ops-shape-ops, shape-ops-copy-ops, concept-integer-width)
- Run F016 on a real i686 / armv7 CPython. With the current code,
  `jix.broadcast(np.zeros((1, 2), np.uint8), (2**31 + 10, 2))[:, 0].shape` should give `(20,)`.
  Reading `slice.start` / `stop` / `step` as Python ints would also fix the 64-bit clamp.
  (py-ops-shape-ops)
- On i686, a plain Broadcast read of `[2^32 + 8]` returns Ok with 8 elements. That is a silent
  wrong-shape result in every op, not just the copy ops (same root cause as F003).
  (shape-ops-copy-ops)
- Dims > i64::MAX (u8 / bool only) are accepted by repeat, tile and concatenate, while indexing,
  core `Slice` resolve and Roll assume they fit in i64 (F009, F010, F022). Should the constructors
  cap dim lengths at i64::MAX instead of each consumer being fixed? (py-ops-shape-ops)
- F005's bound: should it be `block_size * itemsize <= isize::MAX`, or a much smaller cap that would
  also fix F011? (codec-archive)
  - Enforce it in `tune` and in archive header validation.
  - Check whether other usize products in `compact_into_builder` need the same guard: the chunk
    bytes at array.rs:1371 and the read-size math.
- Archive load: `header.block_shape` / `block_size` are cast to u32 (`BlockSize`) before validation
  (archive/array.rs:468, archive/block.rs:374), and section sizes and offsets go through
  `as usize`. None of this was re-checked after the K5 fix. (concept-integer-width)
- Not reviewed: signed / unsigned mixing in slice (negative steps), roll, and the Python-int
  handling in jix-py shape_ops. (concept-integer-width)

### D. Untrusted archives and resource limits

- F011: check the zstd frame content size (`ZSTD_getFrameContentSize`) against the expected block
  size before allocating, or make block allocation fallible (`try_reserve`) and return Err? With
  filters, `Decoder::decode` allocates a second full block (codec/mod.rs:278). (codec-archive)
- `BlockTable::build_from_data` (storage/block.rs:378-386) and `compact_into_builder`
  (array.rs:1367-1369, 1432-1435) hand `&mut [u8]` over uninitialized memory to zstd and the
  filters. This is fine in practice, and Miri tolerates it today. A `MaybeUninit`-based API would
  be worth it if Miri ever runs in CI. (codec-archive)
- On 64-bit, `to_ndarray_buf(index, ctx, None)` of a huge broadcast region takes the slow path and
  materializes the whole region. For a 2^32 f32 region that is a 16 GiB allocation, where a direct
  pipeline pull would return a stride-0 view. Is that intended? (pipeline-driver)
- A small user `read_size` makes reads quadratic, because each piece re-decompresses the block.
  Should `read_size` have a lower bound, or the default `ReadContext` a block cache?
  (dynamic-python-differential)

### E. Platforms, SIMD and stack use

- REQUIRE_ALIGNED = true (armv7 and ppc64le, both shipped) has never run. Run the pipeline and
  reduction tests (misaligned operands, broadcast staging) under qemu-arm / qemu-ppc64le.
  (pipeline-driver, reduction-fold-loops)
- On aarch64, Maximum/Minimum of +0.0 and -0.0 depends on the element's position: NEON vmax prefers
  +0, while the scalar tail returns b. The docs allow this, but results then depend on length and
  alignment. Is that acceptable? (elementwise-ops)
- Debug-build stack use of `inner_loop_per_level` is high even for moderately wide elements:
  16 KiB elements overflow a 2 MiB test thread. It is unclear how much comes from the dispatch arms
  at O0 and how much from the lane count. Capping lanes when `MAX_ITEMSIZE > S::u8s::LEN` (the F017
  fix) may address both. (pipeline-driver)

### F. Semantics and API contracts

- Is accepting a Python bool as an integer index intended (Python list semantics), or should bool
  and `np.bool_` both be rejected? The answer decides F020. (py-ops-shape-ops)
- `isclose` on int8 dispatches to f16, and on int16 to f32 (the smallest Safe float), so rtol / atol
  are rounded to f16 / f32. numpy uses float64 for integer inputs. Is this intended?
  (elementwise-ops)
- Float `divide_trunc` / `divide_floor` beyond the exactly representable range:
  - `divide_trunc(uint8 81, float32 7.9816755e-07)` returns 101482456, while the exact quotient is
    101482452.146.
  - The docstring promises "the exact quotient rounded toward zero". Fix the doc or the kernel?
  - (`divide_floor` gets one case right where numpy is off by one.)
  (dynamic-python-differential)
- Rotates reject every signed rotate count with TypeError, while shifts accept signed counts. Is
  that intended? (dynamic-python-differential)
- `jix.equal(uint64_arr, 2**64-1)` raises OverflowError (intentional since #85), so a uint64 array
  cannot be compared with its own maximum through a Python int (`np.uint64` works). Revisit?
  (dynamic-python-differential)

# jix bug hunt - run 2026-10-29-batch6

- Date: 2026-10-29. The code is at commit `bbab817` ("Bug hunt: add units for the uncovered code"). That commit is main after #92 (`1539ac7`) plus two new hunt units for code that no unit covered: py-ops-reduction-wrappers and util-test-infra.
- Scope: 20 hunter units, 11 module units and 9 concept units, listed in section 1. No dynamic units ran. None of the 20 had run in an earlier batch. With this batch, every concept unit in `bug-hunt.js` has now run at least once.
- Process: Hunt, then Dedup, then Verify, then Report. Each finding got one adversarial verifier, which first tried to refute it and then built a repro. No finding needed a second verifier. Severity follows the table in `bug-hunt/PRIMER.md`.
- What is in this directory:
  - `findings/<ID>.md`: one write-up for each of the 44 findings.
  - `repros/` holds:
    - 28 Rust repros, `repro_2026_10_29_batch6__fNNN.rs`, for F002 (two files: `f002` and `f002_ub`), F003, F004, F007-F016, F020, F023, F029-F034, F036-F038 and F040-F042.
    - 34 Python repros, `FNNN.py`: F001, F003-F006, F008, F013-F015, F017-F026, F028-F030, F033-F044.
    - Crate-internal repros and helpers: `F027_internal.rs`, `F027_mutation.diff`, `F029_internal.rs` and `F029_mutants.py`. Also `F007_probe_fix.diff`, the one-line root-cause fix used to confirm F007.
  - `scratch/` holds:
    - 118 Python probes from hunters and verifiers.
    - Miri and build logs: `f003_miri_sb*.log`, `f032_miri.log`, `f007_build.log`.
    - The F029 mutant logs.
    - The F043 native and valgrind dumps: `util_test_infra_dump_*.txt`, `bs_native/`, `bs_vg/`.
    - Crafted archives: `untrusted_archives_files/`.
    - The derive compile-fail and layout experiments: `types_dtyped_derive/`.
    - The `f043_level/` probe of fearless_simd levels.
    - Three scratch copies of the crate, without build output:
      - `crate-f007/`: `F007_probe_fix.diff` is applied, and the f007 repro is in `tests/`.
      - `crate-f027/`: `F027_internal.rs` is mounted in `ops::reduction::tests` through an absolute `#[path]`. The `neg` mutation from `F027_mutation.diff` is applied, so its `f027_mutation_demo_*` test fails there by design.
      - `crate-f029/`: `F029_internal.rs` is appended to `src/util/test_util.rs`. The mutants are not applied; `F029_mutants.py` applies them. Its `tests/` directory holds a snapshot of this run's `jix/tests` files, taken during verification: all 57 hunter scratch tests and the repros up to f030.
- Paths in the write-ups that are now out of date:
  - The write-ups cite Rust repros as `jix/tests/repro_2026_10_29_batch6__fNNN.rs`. They have moved to `repros/` under the same names.
  - The 57 hunter scratch tests (`jix/tests/hunt_2026_10_29_batch6__*.rs`) were deleted from `jix/tests`. They include every Rust experiment cited in the coverage notes, plus the hunter tests cited by F027 (`..._util_test_infra_2.rs`) and F032 (`..._concept_threads_send_sync_1.rs`). The snapshot in `scratch/crate-f029/tests/` still has all of them.
  - This run's build output was deleted, about 18 GB in all:
    - Test binaries and incremental caches named `{repro,hunt}_2026_10_29_batch6__*`, under `jix/target` (debug, release and the Miri target) and under `target-i686/i686-unknown-linux-musl` (debug and release). Cargo keeps the per-test fingerprints in `jix-<hash>` fingerprint directories that are not named after the tests; those few KB were left in place.
    - `scratch/target-f009-rand`, `scratch/target-f029` and `scratch/target-types_dtyped_derive-rand`. The write-ups of F009, F027, F029 and F043 still cite these directories. They also cite `scratch/target-f027`, which was already gone at cleanup.

    The first re-run of a repro rebuilds its test binary. A lib-test build of a scratch crate takes about 5 GB.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 43 |
| plausible | 1 |
| refuted | 0 |
| intended | 0 |
| unverified (not sent to a verifier) | 0 |

| severity (confirmed + plausible) | count | IDs |
|----------|------:|-----|
| critical | 0 | - |
| high | 2 | F001, F003 |
| medium | 6 | F002, F004, F005, F007, F008, F036 |
| low | 36 | F006, F009-F035, F037-F044 (F044 plausible) |

By category: questionable-semantics 17, error-handling 5, test-coverage 5, memory-safety 4 (F001, F003, F016, F044), wrong-result 4, panic 3, unsound-api 2, resource 2, overflow 1, test-infra 1.

Hunters run: 20. Hunters that failed: `[]`.

| hunter | kind | raw reports | findings |
|--------|------|------------:|----------|
| types-dtyped-derive | module | 4 | F002, F009, F010, F011 |
| types-dims-errors | module | 2 | F004, F012 |
| storage-compact-params | module | 3 | F013, F014, F015 |
| util-nd-arr-sequence | module | 1 | F016 |
| py-ops-dispatch-promotion | module | 2 | F005, F017 |
| py-ops-arith-wrappers | module | 2 | F018, F019 |
| py-core-numpy-wrap-dtype | module | 3 | F006, F020, F021 |
| py-core-archive-errstate | module | 4 | F001, F022, F023, F024 |
| py-core-array-surface-util | module | 2 | F004, F025 |
| py-ops-reduction-wrappers | module | 1 | F026 |
| util-test-infra | module | 6 | F027, F028, F029, F030, F031, F043 |
| concept-aliasing-lifetimes | concept | 2 | F003, F032 |
| concept-degenerate-shapes | concept | 2 | F033, F034 |
| concept-dtype-layout | concept | 2 | F010, F035 |
| concept-arith-semantics | concept | 4 | F036, F037, F038, F039 |
| concept-tiling-coverage | concept | 0 | - |
| concept-untrusted-archives | concept | 3 | F004, F040, F041 |
| concept-python-boundary | concept | 4 | F001, F003, F004, F022 |
| concept-threads-send-sync | concept | 3 | F032, F042, F044 |
| concept-composition | concept | 2 | F007, F008 |

Dedup merged 52 raw reports into 44 findings:
- Four hunters reported F004 (32-bit `__len__`): types-dims-errors, py-core-array-surface-util, concept-untrusted-archives and concept-python-boundary. concept-degenerate-shapes raised it as an open question.
- Two hunters each reported F001, F003, F010, F022 and F032.
- No report was dropped as a duplicate of a `KNOWN.md` entry.
- Several hunters raised open questions that another hunter reported as a finding. Examples: errstate under asyncio (F024), the field-order rejection (F035), the bool8 strategy gap (F029), float Overflow reporting (F037), the free-threaded `out=` TOCTOU (F044), and the comparisons and range checks of untyped Python floats (F005, F017). See section 6.

Links to earlier fixes and entries:
- Regression from a recent fix: F036. The K51 fix (#90) replaced `powi(b as i32)` with `powf(b as f32)`, which loses the parity of odd exponents above 2^24.
- Incomplete fixes (each finding is a trigger the fix did not cover):
  - F012: the K81 fix bounded the zeros-as-1 product only inside `to_ndarray_sub`, not in the shared 32-bit helper.
  - F021: the K96 fix (#92) left out the `np.generic` exclusion that the K96 write-up asked for.
  - F026: the K33 bool guard was added to indices only, not to dims.
  - F033: the K80 fix skips the negative-stride check per dim (`len <= 1`), not for empty arrays.
  - F034: the K98 early return guards only jix-py's own copy of `strided_span_bytes`.
  - F018: K107 covered the f16 multiplier overflowing. The true result overflowing f16 is still unreported. K106 fixed the same narrowing pattern for the f16 unary kernels.
  - F019: the K92 fix covered f16/f32. Its suggested f64 change (divide, or log2/log2) was not applied.
  - F032: the K59 Safety-section fix was applied to `StridedBuf::from_raw_parts` only, not to `plain_ndarray_ptr` / `Plain::new`.
  - F005 and F017: #82's f16 skip moved the precision cliff for untyped Python floats from f16 to f32.
  - F028: its stale skip comment dates from before the K21 fix.
- The same kind of bug as an earlier entry, at a new site:
  - F002: safe user code drives jix's unsafe code into UB (K25, K50, K70).
  - F003: extends K31 (intended) to element-disjoint views and to the safe Rust API.
  - F005: inexact comparisons through a float promotion (K26).
  - F006: a native stack overflow, like K30 and K72.
  - F007: a debug-build stack abort, like K72.
  - F009: reliance on an unspecified layout (K63).
  - F020: a dtype that can be written but not read back (K53).
  - F025: the opposite direction of K37.
- Answers to earlier open questions:
  - F037: batch3 and batch5 asked whether float add/mul/square and the reductions should report Overflow / Invalid. This run confirms the asymmetry with divide.
  - F042: the pilot's question on host-dependent float results.
  - F043: the batch4/batch5 questions on SIMD level coverage and the CI CPU.
  - F044: batch5's question on a free-threaded TOCTOU in `check_output_array`.

Reachability:
- Reachable from Python (30): F001, F003, F004, F005, F006, F008, F013, F014, F015, F017-F026, F033-F042, F044.
  - F004 needs a 32-bit wheel, and F044 a free-threaded (3.14t/3.15t) wheel.
  - F034 and F041 panic only in the debug extension (`maturin develop`).
- Rust only (8): F002, F007, F009, F010, F011, F012, F016, F032.
  - F010 and F011 are compile-time failures. F009 and F016 are latent.
  - F012 needs i686 and a debug build. F007 needs a debug build.
- Test infrastructure only, with no library bug behind it (6): F027, F028, F029, F030, F031, F043.
- Memory safety:
  - F001: use-after-free of Rust heap memory from pure Python.
  - F003: aliasing UB (Miri) from safe Rust and from Python. Values are correct today.
  - F002: heap OOB and uninitialized-padding reads from code with no `unsafe`, through a name collision.
  - F016 (latent, strict Miri), F032 (a gap in the unsafe-API docs, shown with Miri) and F044 (a free-threaded race, argued only).

Headlines:
- **F001 (high): use-after-free through `write_array(a, file_like)`.** The user's `write()` gets a memoryview over Rust memory. Slices, casts and `np.frombuffer` of it outlive `release()`, so Python later reads freed heap memory (SIGSEGV in the repro).
- **F003 (high): `&[u8]` / `&mut [u8]` over the whole span of strided views.** Views that share no element but whose spans interleave alias under noalias pointers. Miri reports UB from safe Rust (`multi_slice_mut`) and from Python (`asarray(x[:, 0]).numpy(out=x[:, 1])`).
- **`#[derive(Dtyped)]` (F002, F009, F010, F011):**
  - F002 (medium): the expansion is not hygienic.
  - F009: it trusts the field order of bare `repr(packed)`.
  - F010: it panics on `r#type` fields.
  - F011: it rejects `[u8; 65535]`.
- **Wrong values from Python promotion and kernels:**
  - F005: comparisons of small ints with a Python float run in f32.
  - F036: `f32 ** int32` gets the sign wrong for odd exponents above 2^24. This is a regression.
  - F004: `len()` truncates on 32-bit.
- **Stack exhaustion from deep op chains (F007, F008).** Debug Rust builds inline 2^k copies of `Op2::shape`. Type-erased Python chains (`x = x + 1` in a loop) recurse once per level on read and on drop.
- **Error reporting and errstate (F018, F022, F024, F037, F038, F039):**
  - Float add/mul/sum never report invalid/over (F037), and log with a base near 1 never reports overflow in f16 (F018).
  - A failed write leaves an unreadable file (F038).
  - errstate leaks across asyncio tasks (F024).
  - Python exceptions from file-likes become RuntimeError (F022).
  - Warnings are lost when the same read also raises (F039).
- **Block layout params are lost (F013, F014, F015, F040).** Scalar/Plain operands, the Plain constructors, `compact_with` and file reads each drop part of an explicit block shape or codec.
- **Test infrastructure (F027-F031, F043):**
  - The comparators are weaker than documented.
  - The strategies never draw non-canonical bool8, non-finite floats or 0-d shapes.
  - SIMD levels the host lacks are skipped silently.
  - Mutants that break bool8 and signed-zero handling pass the current suite.

## 2. Findings

Sorted by severity, then confirmed before plausible, with ties in ID order.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| F001 | high | confirmed | memory-safety | PyWriter::write hands write() a memoryview over Rust memory; derived views (b[:], memoryview(b), b.cast, np.frombuffer(b), PickleBuffer(b)) survive release() and outlive the buffer - use-after-free from pure Python | `jix-py/src/archive.rs:419` | [F001](findings/F001.md) |
| F003 | high | confirmed | memory-safety | Plain::read_data (and jix-py numpy(out=)) take a &[u8] / &mut [u8] over the whole strided span, gap bytes included, so element-disjoint views whose spans interleave alias each other: UB from safe Rust and from Python (Miri) | `jix/src/storage/plain.rs:468` | [F003](findings/F003.md) |
| F002 | medium | confirmed | unsound-api | #[derive(Dtyped)] output is not hygienic: unqualified size_of/align_of/assert! and block-local consts (BASE_OFFSET, FIELDS, FIELD_*) resolve at the call site, so safe user code gets a wrong DTYPE (heap OOB read, uninitialized padding read, wrong field offsets) | `jix-macros/src/lib.rs:197` | [F002](findings/F002.md) |
| F004 | medium | confirmed | wrong-result | 32-bit: Array.__len__ truncates a dim >= 2^32 with `*len as usize`, so len() silently returns shape[0] mod 2^32 | `jix-py/src/array.rs:417` | [F004](findings/F004.md) |
| F005 | medium | confirmed | wrong-result | Comparing a bool/u8/i8/u16/i16 operand with a Python float or complex runs in float32 (Complex<f32>) and returns wrong booleans (equal/not_equal/less/less_equal/greater/greater_equal, dunder operators, isclose) | `jix-py/src/ops/common/dtype_promote.rs:96` | [F005](findings/F005.md) |
| F007 | medium | confirmed | resource | Op2::shape calls the left child's shape() twice inside #[inline(always)], so debug builds inline 2^k copies for a left-deep chain of k binary ops: a 12-term sum or a degree-6 Horner polynomial overflows a 2 MiB thread stack (SIGABRT), and rustc time and memory grow exponentially (about 3 GB at 14 terms) | `jix/src/ops/op2.rs:186` | [F007](findings/F007.md) |
| F008 | medium | confirmed | resource | Deep type-erased op chains (every Python op result) recurse once per op level on read and on drop: `x = x + 1` in a loop overflows the native stack (Python SIGSEGV from ~2k levels with the debug extension, ~7k in release; drop from ~30-50k) | `jix/src/storage/core_trait.rs:139` | [F008](findings/F008.md) |
| F036 | medium | confirmed | wrong-result | f32/f16 base ** i32 exponent rounds the exponent to f32, so odd exponents above 2^24 become even and a negative base gives the wrong sign ((-1)^odd = +1) | `jix/src/ops/op2.rs:1209` | [F036](findings/F036.md) |
| F006 | low | confirmed | panic | dtype_from_numpy recurses once per struct nesting level with no depth limit: a deeply nested numpy struct dtype segfaults the interpreter (native stack overflow) instead of raising | `jix-py/src/dtype.rs:203` | [F006](findings/F006.md) |
| F009 | low | confirmed | wrong-result | #[derive(Dtyped)] accepts a bare #[repr(packed)] (repr(Rust, packed), field order unspecified) and assumes declaration order. If rustc reorders the fields (e.g. -Zrandomize-layout), the DTYPE field offsets are wrong and dtype_sub_field silently reads the wrong bytes. | `jix-macros/src/lib.rs:127` | [F009](findings/F009.md) |
| F010 | low | confirmed | panic | #[derive(Dtyped)] panics on a raw-identifier field (r#type) because its helper-const name is invalid; the recorded field name would also be "r#type" instead of "type" | `jix-macros/src/lib.rs:147` | [F010](findings/F010.md) |
| F011 | low | confirmed | questionable-semantics | Dtyped for [T; N] rejects N == 65535 (strict `<` at dtype.rs:1210), so the valid runtime dtype u8[65535] (itemsize == Itemsize::MAX) cannot be named as a Rust type | `jix/src/dtype.rs:1210` | [F011](findings/F011.md) |
| F012 | low | confirmed | overflow | 32-bit: empty read regions skip the region-size bound, so to_ndarray_slice / to_ndarray_buf of an empty region with huge other extents overflow usize in the stride and size math (debug panic; release wraps harmlessly; a gap in the K81 fix) | `jix/src/error.rs:274` | [F012](findings/F012.md) |
| F013 | low | confirmed | questionable-semantics | Scalar and Plain leaves report an L1-sized auto block shape as their granularity, so combine_block_layout lets them override a compact operand's explicit block shape: compact() of `a * 2.0` / `a + ndarray` / where turns column blocks [1024,1] into row blocks [1,1024] | `jix/src/storage/params.rs:945` | [F013](findings/F013.md) |
| F014 | low | confirmed | questionable-semantics | An explicit block_shape given to the Plain constructors (plain_ndarray_with / _view_with / _ref_with / plain_ndarray_ptr, Python asarray(params=)) is silently discarded, though the docs say it seeds compact() / write_array() | `jix/src/storage/plain.rs:135` | [F014](findings/F014.md) |
| F015 | low | confirmed | questionable-semantics | compact_with / override_from_array silently replace a caller's block_shape_fixed_dims with the source's when block_shape is unset (compact_ndarray_with rejects the same params, archive reads honor them) | `jix/src/storage/params.rs:288` | [F015](findings/F015.md) |
| F016 | low | confirmed | memory-safety | Vec/slice MapMultiple scratch calls Vec::set_len over uninitialized items, then assigns through IndexMut, building a &mut [T] over uninitialized memory (latent; strict Miri flags it) | `jix/src/util/arr_sequence.rs:455` | [F016](findings/F016.md) |
| F017 | low | confirmed | questionable-semantics | An untyped Python float or complex next to a bool/u8/i8/u16/i16 operand is cast to f32 / complex64 at dispatch with no range check: i8 + 1e300 -> inf, u8 * 1e-46 -> 0, u8 / 1e39 -> 0 (exact 2.55e-37 fits in f32), u8 / 1e-46 raises 'divide by zero'. An out-of-range Python int raises OverflowError instead | `jix-py/src/ops/common/operand.rs:164` | [F017](findings/F017.md) |
| F018 | low | confirmed | error-handling | jix.log(f16, base) for a valid base near 1 rounds a finite result to +-inf with no Overflow report (no warning, no error under over='raise'); jix.divide on the same quotient and the f16 exp/tan/power kernels do report it | `jix-py/src/ops/op1.rs:369` | [F018](findings/F018.md) |
| F019 | low | confirmed | questionable-semantics | f64 jix.log(x, base) for a base other than e/2/10 multiplies ln(x) by a rounded 1/ln(base): exact powers come out 1 ulp off (log(64, base=4) = 2.9999999999999996), and power-of-two bases are inexact | `jix-py/src/ops/op1.rs:366` | [F019](findings/F019.md) |
| F020 | low | confirmed | error-handling | Struct dtypes nested 33+ levels are written to .jix without error but can never be read back (prost decode recursion limit); reachable from safe derive and Python | `jix/src/archive/schema/conversions.rs:129` | [F020](findings/F020.md) |
| F021 | low | confirmed | questionable-semantics | asarray/cast/compact with dtype= cast numpy scalars (np.generic) with numpy's C cast instead of jix's Cast. Float numpy scalars give different int results than the same value as a 0-d or 1-d array, and complex numpy scalars cast to float instead of raising TypeError. | `jix-py/src/ops/as_array.rs:65` | [F021](findings/F021.md) |
| F022 | low | confirmed | error-handling | File-like adapters flatten Python exceptions from read()/write()/flush() into RuntimeError (KeyboardInterrupt/SystemExit/OSError lose their type), BufWriter's Drop calls write() again after it raised, and a missing file raises FileNotFoundError or RuntimeError depending on len= | `jix-py/src/archive.rs:498` | [F022](findings/F022.md) |
| F023 | low | confirmed | questionable-semantics | read_array(file_like) and Array::read_from_reader leave the stream at an arbitrary position (core stops at the start of block_data; jix-py's internal BufReader reads up to 8 KiB ahead), so arrays written back-to-back with write_array(f) / write_to(&mut w) cannot be read back with successive reads | `jix-py/src/archive.rs:142` | [F023](findings/F023.md) |
| F024 | low | confirmed | questionable-semantics | errstate restores a construction-time snapshot of OS-thread-local modes: asyncio tasks whose with-blocks exit out of order leave the thread in a non-default mode for good (divide-by-zero then silently returns 0) | `jix-py/src/error.rs:113` | [F024](findings/F024.md) |
| F025 | low | confirmed | questionable-semantics | jix.Array binary/comparison dunders never return NotImplemented: a foreign operand's reflected op (even with __array_ufunc__ = None) is never tried, `a == None` raises ValueError, unsupported operand types raise ValueError instead of TypeError | `jix-py/src/array.rs:880` | [F025](findings/F025.md) |
| F026 | low | confirmed | questionable-semantics | A Python bool passed as a dim is silently read as dim 0/1 (jix.sum(a, True), a.mean(False), jix.argmax(a, dim=True), concatenate, squeeze, permute_dims), where numpy raises TypeError and jix's own index parser rejects bools since K33 | `jix-py/src/util.rs:289` | [F026](findings/F026.md) |
| F027 | low | confirmed | test-infra | Test comparators are weaker than documented: assert_array_matches ("bit-for-bit") and same_value / assert_same_elements ignore the sign of zero, assert_array_matches never matches NaN, bool8 compares by truthiness; f32 max/min reductions use a 0.1 tolerance and reduction_shape_strategy never draws empty or 0-d shapes | `jix/src/util/test_util.rs:893` | [F027](findings/F027.md) |
| F028 | low | confirmed | test-coverage | Python assert_array_matches checks neither dtype nor 0-d shape, and its sub-range draw still skips the K21 cases (start == len, zero-length dims) although K21 is fixed | `jix-py/python/tests/tests_util.py:284` | [F028](findings/F028.md) |
| F029 | low | confirmed | test-coverage | No bool8 test strategy (Rust or Python) draws a byte other than 0/1, so no property test feeds non-canonical bool8 into any kernel; raw-byte regressions in bitwise, where and sum pass the suite | `jix/src/util/test_util.rs:348` | [F029](findings/F029.md) |
| F030 | low | confirmed | test-coverage | Test strategy gaps: float any_strategy is finite-only despite its "full domain" doc; f16 any_strategy is 87% inf/zero, so the f16 max proptest's expected 1-D result is +inf 95% of the time; comparable/non-zero sets miss -0.0, inf and real-only/imaginary-only divisors; blocks are capped at 4 per dim; Python never draws 0-d shapes or non-C layouts | `jix/src/util/test_util.rs:243` | [F030](findings/F030.md) |
| F031 | low | confirmed | test-coverage | assert_read_rejects_invalid_index tries only 3 invalid-index classes and runs on only 4 storages (permute_dims, remove_dim, sub_dtype, where); no library bug hides behind the gap today | `jix/src/util/test_util.rs:1379` | [F031](findings/F031.md) |
| F032 | low | confirmed | unsound-api | Plain::new / plain_ndarray_ptr / compact_nd_ptr Safety sections never forbid writing the memory during a read or while a lent pull view is alive, and never cover the strided span's gap bytes: calls meeting the documented contract are UB (Miri) | `jix/src/storage/plain.rs:108` | [F032](findings/F032.md) |
| F033 | low | confirmed | questionable-semantics | Zero-element arrays are rejected on input when a dim of length > 1 has a negative stride, though no element is ever addressed (gap in the K80 fix) | `jix/src/storage/plain.rs:519` | [F033](findings/F033.md) |
| F034 | low | confirmed | panic | Empty regions overflow in strided_span_bytes (util/mod.rs:111), and on 32-bit in materialize_out_buf (buf.rs:386), because the span/size arithmetic does not stop at the zero extent: debug-only panic, release correct | `jix/src/util/mod.rs:111` | [F034](findings/F034.md) |
| F035 | low | confirmed | questionable-semantics | jix-py rejects padding-free numpy struct dtypes whose field names are not in offset order (e.g. multi-field view x[['b','a']]) with a misleading 'padding between fields' error | `jix-py/src/dtype.rs:208` | [F035](findings/F035.md) |
| F037 | low | confirmed | questionable-semantics | Float add/sub/mul/square and sum/prod/mean never report invalid/over, while divide reports the same conditions; under default modes inf/inf raises but inf-inf, 0*inf and sum([inf,-inf]) are silent NaN | `jix/src/ops/op2.rs:590` | [F037](findings/F037.md) |
| F038 | low | confirmed | error-handling | A write_to_file / write_array that fails part-way leaves a partial, unreadable .jix file; the retry then raises FileExistsError, and a failed append leaves garbage bytes in the file | `jix-py/src/archive.rs:215` | [F038](findings/F038.md) |
| F039 | low | confirmed | error-handling | A Python read that raises FloatingPointError drops the warnings recorded in the same read: an 'over' RuntimeWarning that comes with an 'invalid' (or 'divide') error is silently discarded (numpy(), compact(), write_array) | `jix-py/src/array.rs:673` | [F039](findings/F039.md) |
| F040 | low | confirmed | questionable-semantics | Arrays read from a .jix file do not carry the file's codec/filters in their spec, so compact() and writes of views over them silently re-encode with the default byte-shuffle | `jix/src/archive/array.rs:552` | [F040](findings/F040.md) |
| F041 | low | confirmed | questionable-semantics | BlockTable documents in-bounds block locations as an invariant that Minimal validation does not establish; BlockTable::new only debug_asserts it, so debug builds (incl. the dev Python extension) panic at load where release builds load | `jix/src/storage/block.rs:129` | [F041](findings/F041.md) |
| F042 | low | confirmed | questionable-semantics | Float reductions depend on the host's detected CPU cache sizes (and SIMD level): the same numpy array, or the same .jix file, sums to different f32 values on machines with different L1d/L2 | `jix/src/storage/params.rs:453` | [F042](findings/F042.md) |
| F043 | low | confirmed | test-coverage | SIMD level coverage: for_each_simd_level! skips the levels the host lacks without saying so, and level-dependent driver code (pipeline lane count, reduction folds, update_bulk fallbacks) runs only at the host's Level::new() level | `jix/src/util/test_util.rs:1270` | [F043](findings/F043.md) |
| F044 | low | plausible | memory-safety | Free-threaded builds: numpy(out=) validates out's shape/strides in one read but writes using a second read (array.rs:617/653), and plain_from_numpy reads ndim, shape and strides separately (operand.rs:424/465/471). A concurrent `.shape=` or `.strides=` becomes an out-of-bounds write or read, or a PanicException. | `jix-py/src/array.rs:617` | [F044](findings/F044.md) |

### Bugs

**F001** (high; Python).
`jix.write_array(a, file_like)` passes the user's `write()` a read-only memoryview over Rust memory. That memory is the compact array's own block data, the BufWriter buffer, or the reused compressed-block buffer. Afterwards jix calls `release()` on the view and ignores any error.
- Slices, `memoryview(b)`, `b.cast()` and `np.frombuffer(b)` survive `release()`.
- `PickleBuffer(b)` makes `release()` fail, and the failure is ignored.

So pure Python can later read freed or reused heap memory: stale bytes, then SIGSEGV in the repro. It is high rather than critical because the writer must break the io convention that `write()` touches `b` only during the call. CPython's BufferedWriter has the same hazard (bpo-15994).
Fix: pass `PyBytes::new(py, buf)`, which costs one memcpy of at most one block. At minimum, propagate a failed `release()` and document the restriction.

**F003** (high; safe Rust and Python; UB shown with Miri).
`Plain::read_data` builds a `&[u8]` over the whole strided span of a view, gap bytes included (plain.rs:458-468). jix-py `numpy(out=)` builds a `&mut [u8]` over the whole span of `out` (array.rs:651). Two views that share no element but whose spans interleave then alias under noalias pointers. Examples:
- `multi_slice_mut` in safe Rust.
- `jix.asarray(x[:, 0]).numpy(out=x[:, 1])` in Python, where `np.shares_memory` is False.

Miri (Stacked Borrows) reports UB for code with no `unsafe`. Values are correct today, and no miscompile is known. K31 (intended) covers only an `out` that overlaps memory the array reads.
Fix: give `StridedBuf` raw-pointer variants for borrowed external memory and stop building references over whole spans. Alternatively, stage non-contiguous external inputs and outs through pool buffers.

**F002** (medium; Rust, safe derive).
The `#[derive(Dtyped)]` expansion resolves `size_of`, `align_of` and `assert!` at the call site. It also declares `BASE_OFFSET` / `FIELDS` / `FIELD_*` items in the same block where it re-emits the user's field types. A user item with a colliding name changes the computed DTYPE and disables the asserts that justify the `unsafe impl`. Demonstrated:
- a heap OOB read through `Plain`, with a user `size_of`;
- a padded struct accepted with its padding claimed as field data, with a user `assert!` or `const BASE_OFFSET` (Miri reports an uninitialized read);
- wrong field offsets.

Medium, not the hunter's high: every trigger needs a name collision.
Fix: use absolute paths (`::core::mem::size_of`, ...); declare no named items next to user types, or use mixed-site hygiene; validate with `offset_of!` or a checked constructor.

**F004** (medium; 32-bit Python wheels).
`Array.__len__` returns `*len as usize`. On a 32-bit wheel, a dim of 2^32+5 (from `jix.broadcast`, `tile` or `repeat`) gives `len() == 5` with no error, while `shape` and `size` are exact. This was shown on i686 with a Rust model of `__len__` plus pyo3's conversion; no 32-bit Python was available.
Fix: `isize::try_from` and raise OverflowError.

**F005** (medium; Python).
An untyped Python float or complex next to a bool/u8/i8/u16/i16 operand Safe-casts into the f32 slot. The comparison and isclose tables list f32 before f64, so the comparison runs in float32:
- `equal(int16 [0, 1, 3], 1.00000001)` gives [F, T, F].
- `less(x, 1e-50)` misses 0.
- `uint16 [30000, 65535] > 65534.999` gives [F, F].

101 of 141 checks in the repro are wrong. int32 and wider operands, and typed np.float64 scalars, are exact.
Fix: compare untyped floats against 8/16-bit operands in f64, or exactly, as the K26 kernels do.

**F007** (medium; Rust, debug builds).
`Op2::shape` is `#[inline(always)]` and calls `self.a.shape()` twice, once inside a redundant `debug_assert_eq!`. A left-deep chain of k binary ops therefore inlines 2^k copies in debug builds:
- A 12-term i64 sum or a degree-6 Horner polynomial overflows a 2 MiB thread stack (SIGABRT under `cargo test`).
- rustc needs 3 GB to compile a 14-term sum.

Binding the shape once fixes both (`repros/F007_probe_fix.diff`). Release builds pass and Python is unaffected, so triage could lower this to low, as for K72.
Fix: `let s = self.a.shape(); debug_assert_eq!(s, self.b.shape()); s`, or drop the assert, since `Op2::new` already checks the shapes.

**F008** (medium; Python, and Rust via `into_any`).
Every Python op wraps its operands as `IntoType<ArrayStorageAny>`. So `x = x + 1` in a loop builds a type-erased chain, and reads and drops recurse through it, one group of native frames per level.
- The debug extension segfaults at about 2k levels on read and 30-50k levels on drop.
- Release Rust aborts on read at about 7k levels.

There is no depth guard. Memory use stays O(1); only the stack runs out.
Fix: track chain depth in the type-erased wrapper and materialize (or raise) past a threshold. Drop nested `Arc`s iteratively.

**F036** (medium; Rust and Python).
The f32 and f16 `Pow` kernels for integer exponents compute `a.powf(b as f32)` (op2.rs:1209, 1234). An i32 converts to f32 exactly only up to 2^24. Above that, an odd exponent becomes even and a negative base gets the wrong sign:
- `(-1f32) ** 16777217` is +1, with no flag;
- -2 gives +inf;
- -0.5 gives +0.

f64 x i32 is exact. This is a regression from the K51 fix (#90). Python reaches it with int32 arrays or np.int32 exponents.
Fix: evaluate large exponents in f64, or restore the sign when `a < 0` and `b` is odd.

**F006** (low; Python).
`dtype_from_numpy` recurses once per struct nesting level, with no limit. A deeply nested numpy struct dtype crashes the interpreter with SIGSEGV: at depth 1400 on the main thread, at depth 200 on a 1 MiB thread. This happens through `asarray`, `asarray(dtype=)` and `where`. numpy raises RecursionError for its own recursive operations on such dtypes. Low, not the hunter's medium: the dtype has to be built on purpose.
Fix: a depth cap (e.g. 32) in `dtype_from_numpy` / `dtype_to_numpy`, optionally also in `Dtype::new_struct`.

**F009** (low; Rust, latent).
`#[derive(Dtyped)]` accepts bare `#[repr(packed)]`, whose field order Rust does not specify. It computes the offsets in declaration order and never compares them with `offset_of!`. Under `-Zrandomize-layout` the DTYPE offsets are wrong and `dtype_sub_field` reads the wrong bytes. Stable rustc keeps declaration order, so the bug is latent today (like K63).
Fix: require `repr(C, packed)`, or assert every offset against `offset_of!`.

**F010** (low; Rust, compile time).
A raw-identifier field (`r#type`) makes the derive panic, because the helper-const name `FIELD_0_R#TYPE_DTYPE` is not a valid identifier. If only the panic were fixed, the recorded field name would still be `"r#type"` (from `stringify!`). That does not equal a numpy or archive field named `"type"`.
Fix: use `ident.unraw()` for the const names and for the recorded name.

**F012** (low; Rust on 32-bit, debug builds).
`check_region_size` bounds the true byte product, which is 0 for an empty region. So `to_ndarray_slice` / `to_ndarray_buf` of an empty region with huge other extents overflow usize in `default_strides`, `Compact::read_data` and `materialize_out_buf`. On i686 this panics in debug builds; release builds wrap harmlessly. `to_ndarray_sub` returns Err for the same region.
Fix: bound the zeros-as-1 product in the shared 32-bit helper.

**F016** (low; Rust, latent UB).
The Vec and slice scratch of `map_multiple` calls `Vec::set_len` before writing the items, then assigns through `IndexMut`. That creates a `&mut [T]` over uninitialized memory. It is benign today: the items are Copy and every slot is written before use. Strict Miri (`-Zmiri-recursive-validation`) flags it.
Fix: write through `spare_capacity_mut()` and call `set_len` after the loop.

**F018** (low; Python).
For a base other than e, 2 or 10, `jix.log(x, base)` narrows its f64 result to T inside an `Array::map`, which has no status channel. For f16 and a base within about 2.5e-4 of 1, a finite result rounds to +-inf with no Overflow report, even under `over='raise'`. `divide` on the same quotient reports it, and so do the f16 exp/tan/power kernels.
Fix: a status-aware map, or a core log-with-base kernel that sets Overflow when it narrows.

**F020** (low; Rust safe derive and Python).
Dtype construction, `to_proto` and the writer accept any struct nesting depth. On read, prost's decode recursion limit is 100, and each struct level costs three message levels, so depth 33 and deeper cannot be read. `write_to` / `write_array` succeed, and every later read fails with an `Io` error rather than `InvalidArchive`.
Fix: a nesting cap (e.g. 32) in the Dtype constructors or in the writer. Map header decode errors to InvalidArchive.

**F022** (low; Python).
The file-like adapters turn every Python exception from `read()` / `write()` / `flush()` / `seek()` / `tell()` into a string.
- KeyboardInterrupt, SystemExit and OSError surface as RuntimeError: they slip past `except OSError` and are caught by `except Exception`.
- After `write()` raises, BufWriter's Drop calls `write()` again with the buffered bytes.
- A missing file raises FileNotFoundError with `len=None` but RuntimeError with `len=`.

Fix: stash the PyErr in the adapter and re-raise it; discard the BufWriter buffer on error; map core Io errors to OSError.

**F032** (low; Rust, unsafe-API docs; UB shown with Miri).
The Safety sections of `Plain::new`, `plain_ndarray_ptr` and `compact_nd_ptr` require only that the elements are valid and readable. The implementation needs more: no writes to the memory during a read or while a lent pull view is alive, and rights over the gap bytes of the strided span (F003). A single-threaded call that meets the documented contract is UB under Miri: write through the owner's pointer while holding a pull view. K59 fixed the same omission on `StridedBuf::from_raw_parts` only.
Fix: extend the three Safety sections.

**F034** (low; Rust and Python, debug builds).
`strided_span_bytes` accumulates `stride * (len - 1)` over the earlier dims before it reaches a zero extent. An empty region with large strides therefore overflows, for example an empty `as_strided` numpy array as a zero-length `concatenate` member.
- Debug builds panic at util/mod.rs:111; the dev extension raises PanicException.
- On 32-bit, `materialize_out_buf` also overflows on large extents.
- Release builds return the correct empty result.

Fix: return 0 before accumulating if any dim is 0, and stop size products at a zero extent.

**F038** (low; Rust and Python).
When `write_to_file` or `write_array(path)` fails part-way, it leaves a partial file that cannot be read back. In the repro the failure is a divide-by-zero under the default errstate, and the partial file is 60 KB.
- The retry fails with FileExistsError, because the file is opened with `create_new`.
- A failed `append=True` leaves garbage after the earlier arrays: the file grows from 115 to 60373 bytes.

Fix: on Err, remove a file the write created, or for append truncate back to the original length. Alternatively, write to a temp file and rename it.

**F039** (low; Python).
Core `check_op_status` records Warn-mode warnings before it returns the first Error-mode kind. `numpy()`, `compact()` and `write_array` return that Err before calling `take_warnings()`, and the next read discards the stale warnings. So an 'over' RuntimeWarning that comes with an 'invalid' or 'divide' error is never emitted. The Rust API keeps the warnings.
Fix: take and emit the warnings whether or not the read succeeded, then propagate the error.

**F044** (low, plausible; free-threaded Python wheels only).
`numpy(out=)` validates `out`'s shape and strides, then re-reads the strides and the data pointer from the live ndarray (array.rs:617, 653). `plain_from_numpy` reads ndim, shape and strides separately. On 3.14t/3.15t, another thread doing `out.shape = ...` concurrently could turn this into an out-of-bounds write or read, or a PanicException. Not run: no free-threaded interpreter was available. Taking a snapshot only narrows the window, because numpy's own metadata setters are racy.
Fix: snapshot shape, strides and pointer once and use the snapshot. Document that such arrays must not be reshaped concurrently.

### Questionable semantics

**F011** (low; Rust, compile time).
`impl Dtyped for [T; N]` asserts `N < Itemsize::MAX` (dtype.rs:1210), so `[u8; 65535]` cannot be named as a Rust type. Its runtime dtype, with an itemsize of exactly u16::MAX, is valid, and every other itemsize limit is inclusive. The guard itself is needed: without it, `N as Itemsize` truncates.
Fix: use `<=`.

**F013** (low; Rust and Python).
Scalar and Plain seed a non-fixed block shape of all ones, which `tune` scales to an L1-sized auto shape. `combine_block_layout` takes the per-dim max, so a Scalar or Plain operand overrides a compact operand's explicit block shape. `(a * 2.0).compact()`, `(a + ndarray).compact()` and `where` turn column blocks [1024, 1] into row blocks [1, 1024].
- Values are correct.
- After compaction, column reads are about 780x slower.
- The same scalar as a broadcast 0-d array keeps the layout.

Fix: keep the all-ones block shape for non-compact leaves and scale it only in `compact` / `write_to`. Optionally, let fixed inputs win in `combine_block_layout`.

**F014** (low; Rust and Python).
`Plain::new` resets `block_shape_fixed_dims` to all-false even when `block_shape` was given, so `tune` discards the explicit block shape. This affects `plain_ndarray_with` / `_view_with` / `_ref_with` / `plain_ndarray_ptr` and Python `asarray(params=)`. The docs say the block shape seeds `compact()` / `write_array()`, and `compact_ndarray_with` honors the same params. As a side effect, invalid shapes such as `[5000, 0]` are accepted.
Fix: override the fixed dims only when `block_shape` is unset.

**F015** (low; Rust and Python).
When `block_shape` is unset, `override_from_storage` copies the source's block shape and unconditionally overwrites the caller's `block_shape_fixed_dims` (params.rs:303). That contradicts its own doc ("fields already set are not overwritten"). The same params therefore:
- raise an error in `compact_ndarray_with`;
- are silently dropped by `compact_with`;
- are honored on archive reads (K43 fix).

Fix: keep the caller's flags (`get_or_insert`), or raise an error on every path.

**F017** (low; Python).
An untyped Python float or complex next to a bool/u8/i8/u16/i16 operand is cast to f32 / complex64 with no range check; Python ints are range-checked.
- `i8 + 1e300` gives inf.
- `u8 [100] * 1e-46` gives 0.
- `u8 [255, 200] / 1e39` gives [0, 0], although 2.55e-37 fits in f32.
- `u8 / 1e-46` raises 'divide by zero', although the divisor is nonzero.

All of this is silent at every errstate setting. Same root cause as F005.
Fix: range-check untyped floats against the chosen float slot (OverflowError), or never narrow below f64 unless an operand is a typed float.

**F019** (low; Python).
For a base other than e, 2 or 10, f64 `jix.log(x, base)` multiplies `ln(x)` by a rounded `1/ln(base)`.
- `log(64, base=4)` is 2.9999999999999996, so `int()` of it is off by one.
- About 8% of random results are more than 1 ulp off (max 1.7 ulp), against 0.1-0.5% when dividing by `ln(base)`.

Fix: divide by `ln(base)`. For power-of-two bases, use `log2(x) / log2(base)`, which is exact on every power of the base.

**F021** (low; Python).
With `dtype=`, `asarray` / `cast` / `compact` treat a numpy scalar (`np.generic`) as untyped and convert it with `numpy.asarray(value, dtype=)`, which uses numpy's C cast instead of jix's saturating Cast.
- `np.float64(3e9) -> int32` is -2147483648 with a RuntimeWarning; a 0-d array gives 2147483647.
- NaN gives INT_MIN instead of 0.
- A complex scalar casts to float instead of raising TypeError.

Fix: treat `np.generic` as a typed source in `value_is_array`.

**F023** (low; Rust and Python).
`read_array(file_like)` and `Array::read_from_reader` leave the stream at an arbitrary position. Core stops at the start of block_data, inside the archive, and jix-py's 8 KiB BufReader reads ahead. Arrays written back to back with `write_array(f)` / `write_to(&mut w)` therefore cannot be read back with successive reads, unlike np.save / np.load. The documented pattern of recording offsets and seeking still works.
Fix: seek to the end of the archive after a read, in core and in jix-py, or document the position as unspecified.

**F024** (low; Python).
errstate modes live in an OS-thread-local, and `__exit__` restores a snapshot taken when the errstate object was created. Consider asyncio tasks on one thread whose `with jix.errstate(...)` blocks exit out of order. After every block has exited, the thread is still in a non-default mode: `divide='ignore'` sticks, and `x // 0` silently returns 0. numpy 2, which uses contextvars, ends at its default in the same program.
Fix: keep the modes in a `contextvars.ContextVar`.

**F025** (low; Python).
The binary and comparison dunders accept any `other` and never return NotImplemented, so a foreign type's reflected op is never tried, even when that type sets `__array_ufunc__ = None`. `a == None`, `a == 'x'`, `a + Fraction(1, 2)` and `a + Foo()` raise ValueError ('Unsupported dtype') instead of deferring or raising TypeError. K37 fixed only the direction with numpy on the left.
Fix: return NotImplemented from the dunders for foreign operands: object/str/void dtypes, or `__array_ufunc__ is None`.

**F026** (low; Python).
A Python bool passed as a dim is read as 0 or 1, because pyo3's int extraction accepts bool. `jix.sum(a, True)`, `a.mean(False)`, `jix.argmax(a, dim=True)`, `concatenate`, `squeeze` and `permute_dims` all accept it where numpy raises TypeError. The hazard is a misplaced positional flag, such as `x.var(True)`.
Fix: a dim type that rejects PyBool, used wherever dims are parsed.

**F033** (low; Rust and Python).
A zero-element array is rejected ("Negative strides are not supported") when a dim of length > 1 has a negative stride, although no element is ever addressed. numpy flags such arrays C-contiguous, so `np.ascontiguousarray(x)` returns `x` itself and the usual workaround fails. Meanwhile `numpy(out=)` accepts the same layout.
Fix: skip the sign check when any dim is 0, in `strides_from_ndarray` and in `plain_from_numpy`.

**F035** (low; Python).
`dtype_from_numpy` passes struct fields to `Dtype::new_struct` in numpy `names` order, but `new_struct` requires offset order. A padding-free dtype whose fields are listed out of order is therefore rejected with a misleading error: "padding between fields ... add an explicit padding field". numpy's own multi-field view `x[['b','a']]` produces such a dtype. `Dtype::from_fields` sorts the fields and accepts it.
Fix: sort by offset (the numpy round trip then reorders the names), or reject with an accurate message. Make `check_fields` tell out-of-order fields from gaps.

**F037** (low; Rust and Python).
Float add/sub/mul/square and sum/prod/mean never report InvalidOperation or Overflow, while divide reports both.
- Under default modes, `inf / inf` raises, but `inf - inf`, `0 * inf` and `sum([inf, -inf])` are silently NaN.
- `1e308 + 1e308`, and an f16 sum of 1000 x 100.0, are silently inf even with `all='raise'`.

The values are IEEE-correct. This has been an open question since batch3 and needs a KNOWN entry either way.
Fix: a cold classifier that runs only on non-finite results, or document that only div/pow/exp/log/trig/sqrt report.

**F040** (low; Rust and Python).
`read_from` puts only the file's block shape into the spec, so the encoder params fall back to the default: zstd level 3 with byte-shuffle. `compact()` of a file-read array, or a write of a view over one, silently re-encodes with byte-shuffle. A bit-shuffled or unfiltered file loses its filters, although the docs say codec params are preserved. The data stays correct.
Fix: seed the encoder params from the file's decoder config (codec and filters; the archive does not store the level).

**F041** (low; Rust and Python, debug builds).
`BlockTable` documents in-bounds block locations as an invariant and as a precondition of `new`. But `ArchiveValidation::Minimal` skips the location check, and `new` only `debug_assert!`s it. So a Minimal-mode archive with one bad location panics at load in debug builds, including the dev extension. Release builds load it and serve the intact blocks. There is no UB today, because every access is a checked slice.
Fix: drop the invariant from the docs and remove the debug_assert, or return InvalidArchive from the checked accesses.

**F042** (low; Rust and Python).
`tune` derives `read_size` from the host's detected L1d/L2 sizes, and archives do not store it. The reduction driver tiles by `read_size`, which also sets how the float additions are grouped. So the same array, or the same .jix file, sums to f32 values up to 4 ulps apart on hosts with different caches. This was shown with faked sysfs cache sizes under `unshare`. The SIMD lane count is a second source, argued from the code.
Fix: group the float additions independently of the I/O tiling, in fixed chunks keyed by stream position. Otherwise, document that float reductions depend on the host.

### Test-infrastructure gaps

None of these hides a library bug today; each would let a future regression through.

**F027** (low; crate-internal tests).
`assert_array_matches` is documented as "bit-for-bit" but compares with `==`. It accepts -0.0 for +0.0, never matches NaN, and compares bool8 by truthiness.
- `same_value` / `assert_same_elements` also ignore the sign of zero, although the SIMD-body tests feed both zeros.
- f32 max/min reductions use atol 0.1.
- `reduction_shape_strategy` never draws empty or 0-d shapes.

A mutant `neg` SIMD body that returns +0.0 for -(+0.0) passes all 47 op1 tests.
Fix: byte-exact comparators, with a separate +-0 allowance for Maximum/Minimum; exact max/min reductions; empty and 0-d shape arms.

**F028** (low; Python tests).
`tests_util.assert_array_matches` uses `assert_allclose` with `strict=False`, so it checks neither the dtype nor a 0-d shape: int64 vs int8, uint8 vs bool and () vs (3,) all pass. Its sub-range draw never picks `start == len` and skips zero-length dims, under a comment that has been stale since K21 was fixed. A strict helper fails only 8 reduction tests, all of which use deliberate wide-precision references.
Fix: `strict=True` (and cast those references), draw `start` from `[0, size]`, and draw several sub-ranges per call.

**F029** (low; Rust and Python tests).
No bool8 strategy draws a byte other than 0/1, so no property test feeds non-canonical bool8 into a kernel. Three raw-byte mutants pass all 258 existing tests of the affected modules:
- bool8 `& | ^` on raw bytes;
- `where` testing only bit 0 of the condition;
- sum adding raw bytes.

The new public-API test `repro_2026_10_29_batch6__f029.rs` catches each mutant and passes on the real tree.
Fix: draw any byte for bool8 (Rust `any::<u8>()`, Python uint8 views); add bool8 to the bitwise tests and to `SimdTestValues`; keep the new test.

**F030** (low; Rust and Python tests).
- Float `any_strategy` is finite-only, despite its "full domain" doc.
- f16 `any_strategy` is 87% inf or zero, so the expected 1-D result of the f16 max proptest is +inf 95% of the time.
- The comparable and non-zero sets miss -0.0, inf, and real-only or imaginary-only divisors.
- Block shapes are capped at 4 per dim.
- Python never draws 0-d shapes or non-C layouts.

Probes of the undrawn regions, about 16k cases against numpy, found nothing.
Fix: correct the doc, add a full-domain strategy, draw f16 from its bits, and widen the value sets, block shapes and layouts.

**F031** (low; Rust tests).
`assert_read_rejects_invalid_index` tries 1 + 2*ndim indices from 3 invalid classes, and only 4 storages call it. A plausible regression would pass it: an early return for empty regions placed before `check_out_buf`. The repro runs 86 invalid indices over 77 arrays, in pull and push mode, on x86_64 and i686. All are rejected correctly, so the repro can serve as the wider regression test.
Fix: extend the index list and run it for every storage, or adopt the table-driven repro.

**F043** (low; Rust tests and CI).
`for_each_simd_level!` silently skips the levels the host lacks. Driver code whose behavior depends on the level runs only at the level `Level::new()` picks: the pipeline lane count, the reduction folds, and the SSE-only fallback in Product's `update_bulk`. CI has one x86_64 leg and no `--cfg disable_dispatch_*` override. `SimdTestValues` pairs only 9 of 81 integer and 26 of 676 float edge pairs. Dumps from native AVX-512 and from AVX2 under valgrind differ only by the documented reassociation of mean.
Fix: report the skipped levels (or fail on request), add CI legs with `disable_dispatch_*`, and pair more edge values.

### Re-running a repro

The Rust repros are in `repros/`. To run one:
1. Copy it back to `jix/tests/`.
2. From the repo root, run `cargo test --manifest-path jix/Cargo.toml --all-features --test <name>`, where `<name>` is the file name without `.rs`.
3. Delete the copy afterwards. `jix/tests/repro_*.rs` is gitignored, so a leftover copy would not show up in `git status`.

Example:

    cp bug-hunt/runs/2026-10-29-batch6/repros/repro_2026_10_29_batch6__f036.rs jix/tests/
    cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_29_batch6__f036 -- --nocapture
    rm jix/tests/repro_2026_10_29_batch6__f036.rs

Unless noted otherwise, each repro asserts the correct behavior, so it fails while its bug is present, and its control tests pass.

Per-repro notes:
- **f002**: the regression test; it compiles both before and after a fix.
- **f002_ub**: NOT a regression test. Its padded structs must stop compiling once F002 is fixed. Run it under Miri: `python jix/run_miri.py --all-features --test repro_2026_10_29_batch6__f002_ub -- miri_padded_base_offset`.
- **f003, f016, f032**: pass natively; the UB shows only under Miri. Run `python jix/run_miri.py --all-features --test <name> -- <test>`. Logs: `scratch/f003_miri_sb*.log` and `scratch/f032_miri.log`.
  - f003: tests `safe_plain_push_out_inside_input_span` and `jixpy_numpy_out_column_into_column_pattern`.
  - f016: test `vec_scratch_has_no_reference_to_uninit`, with `MIRIFLAGS="-Zmiri-recursive-validation"`.
  - f032: test `lent_view_then_owner_write`.
- **f004, f012**: 32-bit. Run them with `bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test <name> -- --nocapture --test-threads 1`. This needs the i686 part of `bash bug-hunt/setup.sh`.
  - f004 passes on x86_64 and fails on i686.
  - f012: 4 of 5 tests fail in i686 debug. All pass on x86_64 and with i686 `--release`.
- **f007**: aborts the process with a stack overflow. Run each test alone: `-- --exact left_deep_sum_of_12_terms_on_default_thread` and `-- --exact horner_degree_6_on_default_thread`. Both pass with `--release`. The `#[ignore]`d `stack_probe` test bisects the stack need. `scratch/crate-f007` has the fix applied.
- **f008**: aborts in debug and in release. Run `deep_chain_read` and `deep_chain_drop` one at a time.
- **f009**: passes on stable, because the bug is latent. To see it fail, build only the test crate with layout randomization:
  `CARGO_TARGET_DIR=<dir> cargo +nightly rustc --manifest-path jix/Cargo.toml --all-features --profile test --test repro_2026_10_29_batch6__f009 -- -Zrandomize-layout`.
  Then run the test binary it built with `--nocapture`. The build output is about 1 GB.
- **f010**: fails to compile ("proc-macro derive panicked") while the bug exists.
- **f011**: at run time it compiles two small programs with rustc against the built `libjix`. The N = 65535 program fails.
- **f013, f015, f037**: use `--test-threads=1`.
  - f013: 2 tests fail and 2 controls pass.
  - f015: both tests fail. Either fix (an error, or the flags honored) makes them pass.
  - f037: all 8 tests fail.
- Simple pass/fail counts:
  - f014: 3 tests fail, 1 control passes.
  - f033: all 4 tests fail.
  - f036: 3 tests fail, the f64 control passes.
  - f020: the depth-33 test fails, the depth-32 test passes.
  - f023 and f042: both tests fail.
  - f030: all 3 tests fail.
  - f038 and f040: fail.
- **f029, f031**: pass. They are wider regression tests for gaps that have no live bug. f029 fails on each mutant from `F029_mutants.py` applied to a scratch copy of the crate.
- **f034**: in x86_64 debug, 3 of 4 tests fail. In i686 debug, all 4 fail. With `--release`, all pass on both.
- **f041**: in debug, 1 of 2 tests fails (the load panic). With `--release`, both pass.
- **F027** (crate-internal). The helpers are `pub(crate)`, so the repro runs in `scratch/crate-f027`, where it is already mounted:
  `CARGO_TARGET_DIR=bug-hunt/runs/2026-10-29-batch6/scratch/target-f027 cargo test --manifest-path bug-hunt/runs/2026-10-29-batch6/scratch/crate-f027/Cargo.toml --all-features --lib f027 -- --nocapture --test-threads 1`
  - All six `f027_helper_*` tests fail.
  - The `neg` mutation is applied in that copy, so `f027_mutation_demo_neg_of_positive_zero` fails there too. It passes on the real tree.
- **F029** (crate-internal). The repro is already appended in `scratch/crate-f029`:
  `CARGO_TARGET_DIR=bug-hunt/runs/2026-10-29-batch6/scratch/target-f029 cargo test --manifest-path bug-hunt/runs/2026-10-29-batch6/scratch/crate-f029/Cargo.toml --all-features --lib f029_`
  - Both tests fail.
  - Pass `--lib`, because that crate's `tests/` directory holds the snapshot of this run's test files.
  - To apply the mutants: `python bug-hunt/runs/2026-10-29-batch6/repros/F029_mutants.py <scratch crate> [m1] [m2] [m3]`. Never run it on the tracked tree.
- Each scratch-crate build takes about 5 GB. Delete its target dir afterwards.

To run a Python repro:
- First build the debug extension into `.venv`, with `maturin develop` in `jix-py/` or with `bash bug-hunt/setup.sh`.
- Then run the script, e.g. `.venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F005.py`. Each script exits non-zero while its bug is present, with these exceptions:
  - F003.py only shows that Python accepts the aliasing calls, with correct values. The evidence is the Rust f003 repro under Miri.
  - F004.py passes on 64-bit; it can only fail on a 32-bit wheel.
  - F044.py passes on a GIL build, where it checks the facts the argument rests on. Its race part can only fail on a free-threaded build.
- F001, F006 and F008 run the crashing cases in child processes, which die with SIGSEGV (exit -11).
- F034 and F041 show PanicExceptions from the debug extension. A release wheel should pass both (argued, not run).
- F042 needs root and `unshare -m`, because it fakes the sysfs cache sizes for each child.
- F043 also runs the prebuilt lib test binary under valgrind, if valgrind is installed. That part is a demonstration, not an assertion.

## 3. Unverified findings

None. Every deduplicated finding was sent to a verifier.

## 4. Refuted and intended

No verifier refuted a finding or judged one intended, and dedup dropped none as already known. Below are the parts of hunter claims that verifiers corrected, refuted or found intended, and the places where hunters disagreed. They are the useful signal for tuning the next run.

### Severity changes by verifiers

- F002: high -> medium. Every trigger needs a user item whose name collides with a name internal to the derive (precedents K50, K70, K25).
- F006: medium -> low. The nesting must be built on purpose; a `.npy` header caps it at about 60-100 levels. K30 and K72 were rated low.
- F036: low -> medium. The sign is silently wrong, and for base -1 the value is wrong with no flag.
- F007 stays medium. The verifier notes that triage could lower it to low, as for K72 (an abort in debug builds only).

### Sub-claims refuted or judged intended

- F017: on its own, `i8 + 1e300 -> inf` is close to intended: the f32 result dtype is documented, and "casts never set a status" is intentional. The finding stands because the early cast loses more than the f32 dtype implies: values flush to 0, and a spurious divide-by-zero is raised.
- F024: the generator variant is not specific to jix; numpy 2.5.4 leaks 'ignore' there too. The sub-claim about reusing an errstate object is intended (py-core.md:144: "applies on call, not on `__enter__`; per thread").
- F037: the hunter's sub-claim about narrowing Python scalars is intended. An f16 array plus an untyped float "can overflow to inf, by design", and casts never set a status.
- F015: setting only `block_size` on a source whose block shape is all-fixed follows the documented inheritance. Not a defect.
- F016: the "slice over uninitialized bytes" cases at `array.rs:1401` and `buf_pool.rs:116-118` are a crate-wide design choice, since pool buffers are documented as uninitialized. They are not part of this finding.
- F018: overstated. Not every f16 op reports overflow: f16 multiply, f16 add and f64 -> f16 casts are silent too, and K60 is intended. Log with a base belongs with div/pow/exp, which do report.
- F030: reduction.rs:4708-4710 relies on `any_strategy` never producing NaN. So the defect is the doc plus the missing non-finite coverage, not the strategy itself. Hand-written tests already cover some non-finite reductions and casts.
- F043: the AVX2 branch of `map_masks_to_bools` IS covered per level. The `bit_shuffle` `*_simd` functions ignore their level token and are plain scalar loops. The PRIMER and the batch4/5 reports had already noted parts of the gap.

### Verifier corrections to hunter claims

- F001: on numpy 2.5.4, `np.frombuffer` does not make `release()` raise; it just keeps the pointer. `PickleBuffer(b)` is the case that hits the ignored BufferError.
- F002: the verifier added a more natural trigger, a user `const BASE_OFFSET` used in a field type.
- F008: the read threshold with the debug extension is about 2k levels, not 5k.
- F011: the hunter called the `N < MAX` guard redundant. It is not: without it, `N as Itemsize` truncates and the unsafe impl becomes unsound. The fix is `<=`.
- F012: the K81 write-up claims that `to_ndarray_slice` / `to_ndarray_buf` "return early for empty regions". That is wrong: the stride and size products run first.
- F013: the hunter's "4 MiB tile" cost for lazy reads was not borne out. Only the layout that `compact()` / `write_to` persist suffers.
- F014: the verifier found a side effect: `asarray(params=)` accepts invalid block shapes such as `[5000, 0]`.
- F019: dividing by `ln(base)` is not exact on every power either; for base 8 it is worse. `log2(x) / log2(base)` is the method that is exact on all powers of a power-of-two base.
- F020: the hunter's Rust test used the unsafe `compact_nd_ptr`. The verifier rebuilt it on the safe public API (derive + `compact_ndarray` + `write_to`).
- F023: the cause is wider than reported. Besides the read-ahead of jix-py's BufReader, core itself stops inside the archive, at block_data.
- F026: numpy's Python-level helpers (`flip`, `roll`, `expand_dims`, `stack`, `swapaxes`) accept a bool axis. So "numpy raises" holds only for the reductions, argmax/argmin, concatenate, squeeze and transpose.
- F027: ignoring the sign of zero is load-bearing for Maximum/Minimum, which may return either zero; those ops need their own lenient comparator. The NaN leniency of `same_value` is documented.
- F028: empty slices with `start < len` are drawn. The NaN / -0.0 point is not a defect.
- F029: today no kernel with bool8 input has a hand-written SIMD bulk function, so the missing bool8 `SimdTestValues` impl matters only for future kernels.
- F031: "never checked" is slightly overstated for Op1/Op2: buf.rs:516-540 tries one reversed range on them.
- F035: the fix has a design tension. Sorting the fields changes the numpy dtype on the round trip (`x[['b','a']].dtype != x.dtype`).

### Judgment calls where hunters disagreed

These are worth a line in the PRIMER or a context doc, so the next run decides them consistently:
- Empty arrays with a negative stride on a dim of length > 1: py-core-numpy-wrap-dtype called this "too contrived to report". concept-degenerate-shapes reported it (F033, confirmed low).
- Partial files after a failed write: py-core-archive-errstate called them "consistent with np.save, not reported". concept-arith-semantics reported them (F038, confirmed low).
- The free-threaded `out=` TOCTOU: concept-python-boundary did not report it, calling it a user-level race on numpy metadata. concept-threads-send-sync did report it (F044, plausible low).
- Float statuses in reductions: py-ops-reduction-wrappers found "no status, consistent with add/mul" acceptable. concept-arith-semantics reported the asymmetry with divide (F037).

### Stale context-doc suspicions settled by hunters (update the docs)

- types:
  - S2 (field order of bare packed structs) holds -> F009.
  - S3 is K8.
  - S4 is superseded by #89 (bool8). A sweep of 35 ops over bytes 0/1/2/128/255 behaves consistently.
  - S6 (K70), S7 (K53) and S9 are fixed. S1 (K42) and S8 (K52) are fixed.
  - S5 (32-bit buffer-size truncation in error.rs) is refuted for non-empty regions. The gap for empty regions is F012.
- storage-compact:
  - params.rs:427, divide-by-zero on a zero fixed dim (K6): now returns Err.
  - params.rs:361-376, block volume > u32::MAX: unreachable, because block bytes are capped.
  - archive/array.rs:532 (K43): fixed.
- storage-core: the doc assumes that aliasing over spans needs the unsafe constructors. That is wrong: the safe `plain_ndarray_ref` / `plain_ndarray_view` reach it (F003).
- util-nd / pipeline: map.rs:238-247, empty sequences, is fixed (K87). The arr_sequence `set_len` suspicion (util-nd.md:236-238, pipeline.md:138-140) holds -> F016.
- py-ops: the big-endian cast at ops/mod.rs:96-100 is settled (K20). The f64 -> f16 double rounding is K28/K71.
- py-core:
  - Fixed: dtype.rs:138 byte order (K20), the dtype.rs:208-211 debug_assert (K39), operand.rs:185 bool bytes (K15/#89), and `normalize_dims` at util.rs:185-192 / shape_ops.rs:606 (it now rejects more than 8 dims).
  - operand.rs:176 (K80) is fixed, except for the empty-array gap F033.
  - Hold: archive.rs:437, `release()` errors ignored -> F001. The `*len as usize` open question -> F004. The deep-chain risk at py-core.md:130 -> F008.
- elementwise-ops: `power(x, 0.5)` -> Sqrt and `x ** 2` -> Square (op2.rs:681-688) are settled as intended (K60).
- reduction: the empty sum giving -0.0 (K62), the complex mean (K74), and the 32-bit tile index and state size (K86, K85) are fixed. The complex Sum/Product seeds are not identities, but they match numpy bit for bit.
- types.md:124 (from_proto depth bounded only by prost's limit) is a real defect on the write side -> F020.

### Hypotheses the hunters checked and ruled out

- types-dtyped-derive: traced the precondition of every unsafe block. The derive's `is_aligned` flag disagrees with `new_struct` only for packed structs whose fields all have alignment 1, which is cosmetic.
- types-dims-errors: `DimDyn`'s `assert_unchecked(len <= NDIM_MAX)` and every ArrayVec length write are sound. 0-d, 8-d and 9-entry dims all give Err. Arithmetic error modes are applied consistently on every Python read path.
- storage-compact-params: `tune` is idempotent (3000-iteration random round trip). `block_len_heuristic` stays in bounds. `element_cost` is always finite. Tiles never exceed their shapes.
- util-nd-arr-sequence: `N_OPERANDS` is exact for every node (checked dynamically). The `arr_ext` drop guards are correct for every needs_drop combination. Nested `map_multiple` calls never share a scratch buffer.
- py-ops-dispatch-promotion: every `into_typed::<T>().unwrap()` holds (a 38k-call brute force found no PanicException). The f16 skip is sound. `cast_impl` matches numpy for in-range values.
- py-ops-arith-wrappers: the Square/Sqrt shortcut tables and the multiply fast path give the same dtypes and bit-identical values as the general paths. The scalar path of `absolute` matches the array path.
- py-core-numpy-wrap-dtype: cached numpy descrs always match the layout of `arr.dtype()`. The `Plain::new` preconditions hold for numpy inputs. numpy's deprecated shape/strides/dtype setters keep the data pointer.
- py-core-archive-errstate: the Python seek/tell conversions, the offset/len handling, the `ContextGuard` Mutex and `gil_used=false` are sound. In append mode, `write_array` writes only at or past the old EOF, so it never modifies a live mmap of the same file.
- py-core-array-surface-util: the operand order is right in all 11 reflected dunders. `UnsafeSend` is dead code. `size()` cannot overflow.
- py-ops-reduction-wrappers: keepdim reinsertion, all/any truthiness, output dtypes and out= layouts match numpy over about 30k randomized cases.
- util-test-infra: the helpers' own unsafe blocks hold. The Safety section of `bench_util::nd_copy` is complete. Bit- and byte-shuffle archives are identical across SIMD levels.
- concept-aliasing-lifetimes: the lifetime transmute in buf.rs, the buffer pool's `UnsafeCell`s, pipeline staging, reduction state buffers and shape-op sub-buffers are sound (Miri SB+TB clean). Every buffer materialized in pull mode is fully written before it is read.
- concept-degenerate-shapes: every read of an empty region returns early or does nothing. Length-1 dims are dropped before strides are sorted. 8000 fuzzed op chains biased toward dims of length 0 and 1 pass.
- concept-dtype-layout: no path builds a Rust `bool` or another type with restricted bit patterns from array or file bytes. Every `transmute_copy` sits behind `check_dtype`. Equal dtypes have identical layouts.
- concept-arith-semantics: integer div/rem with MIN/-1, and the floor/euclid corrections, cannot overflow. Scalar and SIMD float div/rem agree on NaN residuals. The fearless_simd casts that were spot-checked round once.
- concept-tiling-coverage: every request splitter covers its request exactly (about 60k differential cases); see section 5.
- concept-untrusted-archives: every header field is validated on every entry point. Strict validation never panics under mutation and truncation fuzzing on x86_64 and i686.
- concept-python-boundary: `ContextGuard` survives an 8-thread mixed-mode stress test with no leak or hang. `check_output_array` rejects byte-swapped, misaligned, non-writeable and overlapping outs.
- concept-threads-send-sync: every `unsafe impl Send/Sync` is sound. `ReadContext` is `!Sync`, and its interior mutability is never re-entered. Nested reads on one context are Miri-clean. Pool retention is bounded.
- concept-composition: every `read_as_elementwise_pipeline` override agrees with its `read_data`, and the spec flags agree with their consumers. About 50 compositions agree with ndarray in every read mode.

### Noticed but not reported (judgment calls)

- `jix.__all__` is the extension's list, so `from jix import *` omits the Python aliases (`pow`, `abs`, `log2`, `log10`, ...). (py-core-archive-errstate)
- A bogus Python writer that reports positions >= 2^63 gets an error, or a debug overflow panic in the offset arithmetic of block.rs. (py-core-archive-errstate)
- `try_into().unwrap()` at dimension.rs:215 and :309 (jix `Dim` -> ndarray) can panic on 32-bit for dims >= 2^32, but only when users convert a jix Dim themselves. (types-dims-errors)
- `jix.log` accepts `base=NaN` and `base=+inf`. Batch5's K107 verifier ruled that neither is a bug. (py-ops-arith-wrappers)
- The asarray docstring says 0-d inputs are copied into a scalar, but 0-d struct arrays are wrapped without a copy. (py-core-numpy-wrap-dtype)
- `resize(refcheck=False)` and concurrent writers leave jix with a dangling or racing view, exactly as for numpy's own views. (py-core-numpy-wrap-dtype, concept-threads-send-sync)
- `numpy_reshape` (jix-py util.rs:159-180) and `UnsafeSend` are dead code. (concept-python-boundary)
- `iter/nd.rs:60` decides emptiness with `shape.product() == 0`. That is correct only because every NdIter range is bounded by a validated shape. (concept-degenerate-shapes)
- A `map` over 64 KiB elements needs more than a 1 MiB stack in debug builds. (concept-dtype-layout)

## 5. Coverage

- **types-dtyped-derive**:
  - Read dtype.rs:880-1219 (with 1-880 for the repr, `check_fields` and `new_struct_borrowed_unchecked`), jix-macros/src/lib.rs and element_type.rs in full.
  - Read the consumers: Transmute, SubDtype, nd_copy, Plain, the archive conversions and jix-py dtype.rs.
  - Traced every unsafe block and settled the types suspicions. Findings F002, F009, F010, F011.
- **types-dims-errors**:
  - Read dimension.rs, error.rs, lib.rs, arrayvec.rs, arr_ext.rs and jix-py error.rs in full, plus the `to_ndarray*` read paths, buf.rs and the ReadContext status handling.
  - Checked how every Python read path uses errstate. Findings F004, F012.
- **storage-compact-params**:
  - Read params.rs 1-1007 and `scale_read_shape` with its helpers.
  - Followed every spec constructor, every caller of `tune` / `into_spec` / `override_from_storage`, and every tiling caller.
  - Ran a round trip with random params. Findings F013, F014, F015.
- **util-nd-arr-sequence**:
  - Read arr_sequence.rs and arr_ext.rs in full, with map, stack, concatenate and every implementor of `N_OPERANDS` / `operands()`.
  - Ran dynamic operand-count and parity tests in debug. Finding F016.
- **py-ops-dispatch-promotion**:
  - Read dispatch.rs, dtype_promote.rs, common/mod.rs, ops/mod.rs, operand.rs, as_array.rs and the op tables in full.
  - Ran a ~38k-call brute force of dispatch and a 14x14 cast matrix against numpy. Findings F005, F017.
- **py-ops-arith-wrappers**:
  - Read jix-py op1.rs and op2.rs in full, with dispatch, operand and the core Abs/Square/Neg/Sign/Add/Sub/Mul kernels.
  - Ran bit-exact 1003-element sweeps against numpy over all unary and binary wrappers and input kinds. Findings F018, F019.
- **py-core-numpy-wrap-dtype**:
  - Read jix-py dtype.rs, operand.rs, as_array.rs, the cast path and type_hint.rs in full, with the Plain and Dtype contracts and rust-numpy's descr helpers.
  - Ran 21 probe scripts over byte order, struct alignment, subarrays, odd layouts and numpy mutation. Findings F006, F020, F021.
- **py-core-archive-errstate**:
  - Read jix-py archive.rs, error.rs, lib.rs and `__init__.py` in full, with the core archive read and write paths.
  - Traced both unsafe blocks (the memoryview and mmap). Findings F001, F022, F023, F024.
- **py-core-array-surface-util**:
  - Read jix-py array.rs 335-433 and 780-1519 (getters, every dunder, reductions, shape ops) and util.rs 150-494, with their callers.
  - Ran every dunder on bool8 arrays holding non-canonical bytes, comparing against canonical bool semantics. Findings F004, F025.
- **py-ops-reduction-wrappers**:
  - Read jix-py ops/reduction.rs in full, with dim parsing, dispatch, `cast_impl`, `numpy(out=)` and the core reduction driver and kernels.
  - Ran about 30k randomized comparisons against numpy, plus a dtype matrix, out= layouts, bool bytes 1-255 and errstate. Finding F026.
- **util-test-infra**:
  - Read test_util.rs, bench_util.rs, nd_copy.rs and tests_util.py in full, and the key parts of nd_iter_unordered.rs.
  - Experiments include a diff of AVX-512 against AVX2 (valgrind) dumps, wide invalid indices on i686, and bit-exact signed zeros.
  - Found no library bug. Findings F027-F031, F043.
- **concept-aliasing-lifetimes**:
  - Listed every transmute, `PtrNoalias` / `PtrMutNoalias`, `from_raw_parts`, `UnsafeCell` and `view_mut` site in jix and jix-py, and checked each one.
  - Ran Miri SB+TB on staged pipelines and reductions. Findings F003, F032.
- **concept-degenerate-shapes**:
  - Checked about 95 non-test sites (`- 1`, `.len() -`, step_by, chunks, divisions) in core and jix-py for 0-d, zero-length and length-1 dims.
  - Ran Python sweeps against numpy, out= buffers with odd strides on length-1 dims, an 8000-iteration op-chain fuzzer, and i686 debug and release. Findings F033, F034.
- **concept-dtype-layout**:
  - Traced dtype equality, every bool8 consumer, the `check_dtype` gates, every `element_type_change` and the alignment proofs in core and jix-py.
  - Ran struct, bool8, inner-shape, f16 and complex experiments through every op family and the archive. Findings F010, F035.
- **concept-arith-semantics**:
  - Traced the status flow through error.rs, ReadContext and the pipeline.
  - Checked the edge cases of every op1/op2/cast/cmp/bitwise kernel, the reduction traits and every jix-py read site that applies errstate. Findings F036, F037, F038, F039.
- **concept-tiling-coverage**:
  - Proved that the pieces cover the request exactly for: `to_ndarray_buf` splitting, `NdIterExtBlockOffsetSize`, `scale_read_shape`, compaction chunking, the Compact read paths, the reduction bulk/tile driver, the shape ops that split requests, and the pipeline chunk loop.
  - Ran about 60k randomized differential cases with tiny read sizes. No findings.
- **concept-untrusted-archives**:
  - Checked every value derived from an archive on every read entry point: Rust and Python, Owned and mmap, Strict and Minimal.
  - Fuzzed with byte mutations and truncations (~46k reads per run, on x86_64 debug and release and on i686), crafted struct dtypes, and tested every filter chain. Findings F004, F040, F041.
- **concept-python-boundary**:
  - Listed every detach / unsafe / `from_raw_parts` / `as usize` site in jix-py and checked each one.
  - Ran an 8-thread stress test of errstate and reads, and probes of how jix keeps numpy arrays alive. Findings F001, F003, F004, F022.
- **concept-threads-send-sync**:
  - Checked every `unsafe impl Send/Sync`, every site with interior mutability, all global state, the per-Array Mutex, and the lifetimes of Python objects during detached reads.
  - Ran Miri on nested reads and measured pool retention. Findings F032, F042, F044.
- **concept-composition**:
  - Compared every `read_as_elementwise_pipeline` override with its `read_data`, every producer of a spec flag with its consumers, and the seams of the pipeline driver.
  - Compared about 50 compositions against ndarray in pull, push, slow-path and compaction modes. Findings F007, F008.

Skipped or only skimmed:
- Across the run:
  - No free-threaded interpreter, no 32-bit CPython and no release wheel was available. F044 and the Python side of F004 are argued from code or from Rust models. The release-build thresholds of F006 and F008 were not measured in Python.
  - No armv7, aarch64 or NEON runs. The `REQUIRE_ALIGNED` staging paths, the aarch64+fp16 f16 paths and the NEON SIMD paths were reasoned about only.
  - The host is AVX-512. Valgrind exposes only AVX2, and nothing can force SSE2/SSE4.2, so those pipeline and reduction levels never ran (F043).
- types-dtyped-derive:
  - No i686 run, since K8 covers it.
  - Skimmed the rest of transmute.rs, the nd_copy inner loops and `copy_untyped`, and the pipeline's aligned reads.
  - The bool8 kernels were not audited line by line.
- types-dims-errors:
  - Skimmed elementwise_pipeline.rs (status-check sites only), the shape-op constructors (ndim checks only) and jix-py's `numpy()` stride rebuild (32-bit sizing only).
  - Did not review dtype.rs, element_type.rs or the derive.
- storage-compact-params: skimmed the params.rs inline tests (1009-2108), the reduction internals beyond tile/bulk setup, and the archive write path. No Miri or i686 run.
- util-nd-arr-sequence:
  - Skimmed the offset math of stack's `in_place` and concatenate's `inner_contiguous` paths, and params.rs `combine_*`.
  - Did not review the pipeline driver internals (`to_buf_type_erased*`, staging, inner loops).
- py-ops-dispatch-promotion:
  - The hand-written tables (power, log, isclose, clamp, rotate) were checked only for T/kind agreement.
  - broadcast.rs and `plain_from_numpy` were left to other units.
  - No release, i686 or Miri run.
- py-ops-arith-wrappers: did not review the core Pow/Ln/Log2/Log10 kernel internals, the dispatch policy beyond the tables used, or the f64 -> f16 rounding (K71).
- py-core-numpy-wrap-dtype: noted but did not pursue the theoretical UB of a `&[u8]` over never-written `np.empty` memory, or over memory that numpy threads mutate concurrently.
- py-core-archive-errstate:
  - Did not read `compact_into_builder`; see the open question on a source mutated from `write()`.
  - No Miri, i686, release or free-threaded run.
  - Did not read test_bad_input.py in full.
- py-core-array-surface-util: did not review `numpy()` / `__getitem__` / `item` / `plain` (array.rs:434-780), the callers of `strided_span_bytes`, or the op-kernel internals.
- py-ops-reduction-wrappers:
  - Covered the core fold loops and SIMD cast tables only through randomized tests.
  - Did not run the shape ops that share the bool-dim parsing; the F026 verifier did.
- util-test-infra:
  - Sampled, rather than audited, the 150 call sites of `assert_array_matches` and the 27 of approx.
  - Did not read Python test files beyond tests_util.py and the test_cmp clamp tests.
  - Checked release-only code (`for_each_unrolled`) by inspection.
- concept-aliasing-lifetimes:
  - Left the PyWriter memoryview to py-core-archive-errstate.
  - Skimmed the SIMD kernels, the filter internals beyond pool use, the archive writer and the reduction folds.
  - Argued the concurrent-mutation variants (GIL released) only.
- concept-degenerate-shapes:
  - Checked only the division sites of `scale_block_shape` / `block_len_heuristic`.
  - Skimmed the SIMD scalar layer, the NdCopier inner loops and the arr_sequence internals.
  - Did not read the trivial forwarders. No Miri run.
- concept-dtype-layout:
  - Left the derive internals to types-dtyped-derive.
  - Exercised the codec filters only through experiments.
  - Skipped f16/complex numerics and the reduction driver beyond alignment.
  - No 32-bit, armv7 or Miri run.
- concept-arith-semantics:
  - SIMD correctness on AVX-512, NEON and SSE2 was reasoned from fearless_simd, for a few ops only.
  - Did not read the dyn pipeline driver line by line.
  - Skimmed complex.rs and logical1.rs. No Miri run.
- concept-tiling-coverage:
  - Skimmed the byte_shuffle chunk/tail loop. Glanced at bit_shuffle, where_op, map.rs and sub_dtype.
  - Treated NdCopier, NdIterUnordered and the reduction folds as black boxes.
  - Did not re-run the 32-bit tiling sites on i686. No Miri run.
- concept-untrusted-archives: no Miri run. Not covered: SIMD byte-shuffle at levels the host does not run, zstd internals beyond fuzzing (window-size and dictionary frames), Windows mmap, and concurrent file modification.
- concept-python-boundary: skimmed `array_try_from_fn` / `array_try_map`, which batch5 ran under Miri, and the core copy and staging internals for unaligned Plain inputs on armv7.
- concept-threads-send-sync: did not run the multi-thread `concurrent_reads` test under Miri. Skimmed the raw-pointer guards in jix-py util.rs:423, arr_ext.rs and reduction.rs:1147/1209.
- concept-composition:
  - Did not review the reduction internals beyond spec derivation and empty handling.
  - Covered `compact_into_builder` internals only through compaction comparisons.
  - Left the composite Python ops (log with a base, mean/var, clip, where/stack promotion) to the py-ops units. No Miri run.

## 6. Open questions from hunters

### Answered during this run (became findings)
- Does `jix.errstate` leak across asyncio tasks that interleave `with` blocks? Yes: F024. (types-dims-errors, concept-python-boundary)
- Is it intended that the Python boundary and the archive reader require struct fields in offset order? It rejects valid dtypes with a misleading error: F035. (concept-dtype-layout)
- Should the bool8 strategies draw any byte? The gap hides real regressions: F029. (concept-dtype-layout)
- Does `__len__`'s `*len as usize` truncate on 32-bit? Yes: F004. (concept-degenerate-shapes)
- Can a concurrent `out.shape = ...` on 3.14t swap strides between the check and the write? Plausibly: F044. (concept-python-boundary)
- Should float add/sub/mul/square and the reductions report Overflow / Invalid? Reported as F037. A decision and a KNOWN entry are still needed. (py-ops-arith-wrappers, concept-arith-semantics)
- Is the f32 default for untyped Python floats meant to apply to comparisons, and should untyped floats be range-checked? Reported as F005 and F017. (py-ops-dispatch-promotion)
- Should Python exceptions from file-likes be preserved? Reported as F022. (concept-python-boundary)

### Memory safety, aliasing and threads
- F003 fix: does rustc emit LLVM `writable` (or anything else that allows speculative stores) for `&mut [u8]` arguments on the shipped targets? If so, the references over whole spans could become observable. Should the numpy() contract say "must not `np.may_share_memory`", or should core stop building slices over spans? (concept-python-boundary)
- Two more triggers with F003's root cause were not run (concept-aliasing-lifetimes):
  - In safe Rust, one thread reads a Plain over `col` while another writes `row` through its own `ArrayViewMut`.
  - `write_to(Cursor<&mut [u8]>)` whose buffer lies inside the span of a Plain input.
- Should jix document, or guard against, Python threads writing an asarray-wrapped numpy array or `out` while `numpy()` / `compact` / `write_array` run detached? jix holds noalias references over that memory for the whole read. numpy's GIL-releasing ufuncs have the same race in C. (concept-aliasing-lifetimes, concept-threads-send-sync, concept-composition)
- Does `compact_into_builder` / `write_to_with` keep a borrowed `&[u8]` of a Plain (numpy-backed) source alive across the Python `write()` callback? If so, a `write()` that mutates the source would write under a live shared reference. (py-core-archive-errstate)
- F001 fix choice (py-core-archive-errstate):
  - A `PyBytes` copy per write: one memcpy of at most a block.
  - Or a Python-owned bytearray: views kept by the user then see later overwrites, but there is no UB.
- F044: should `check_output_array` return the snapshot it validated, and should `plain_from_numpy` take a single snapshot? Should jix hold a buffer export on `out`, or require refcheck semantics? (concept-threads-send-sync)
- Under `-Zmiri-recursive-validation`, two more sites trip: util/aligned_vec/mod.rs:253 (`as_mut_slice` after the `set_len` at array.rs:1401), and pool buffers handed out as `&mut [u8]` over never-written memory. Does the project want soundness under strict reference validity, with APIs based on MaybeUninit? (util-nd-arr-sequence)
- `read_array(path, mmap=True)` exposes the documented contract "modifying the file while it is mapped is UB" through a call that looks safe. Add it to KNOWN as intended? (concept-aliasing-lifetimes)
- A panic while the per-Array cache Mutex is held poisons it, for example an allocation failure in `ReadContext::new` or a TLS access during thread teardown. Every later `numpy()` on that Array then raises 'Failed to acquire cache lock'. Should poisoning be ignored, since the cache is only an optimization? (concept-threads-send-sync)
- The Transmute docs (ops/transmute.rs:50-53) still say that some types, such as bool, have restricted representations. Since #89 every Dtyped type accepts every bit pattern. Should `transmute_elements` stay unsafe? (concept-dtype-layout)
- jix-macros lib.rs:220 places the user's `#[dtyped(crate = "...")]` path tokens inside the generated `unsafe {}` block. No exploit was found, but moving them out is cheap. (types-dtyped-derive)

### errstate and error reporting
- Two errstate behaviors (types-dims-errors, concept-threads-send-sync):
  - Entering the same `ErrState` object a second time runs with the restored modes, not the requested ones.
  - `__exit__` on another thread, or out of LIFO order, sets stale modes on the wrong thread. numpy rejects mismatched resets.

  Is either intended?
- Underflow is a public error kind and `errstate(under=...)` exists, but by design (commit 98b2ed3) no kernel sets it. Neither the docs nor the docstring say that `under=` has no effect. (types-dims-errors)
- Rust `ReadContext`: a failed read leaves its Warn-mode warnings in the context, and `take_warnings()` after a later successful read returns them. Document this, or stop collecting after the first Error kind? (types-dims-errors, concept-arith-semantics)
- The Rust convenience APIs (`to_ndarray()`, `compact()`, `write_to()`, `write_to_file()`, `maybe_compact()`) build a fresh ReadContext that the caller never sees. The default for overflow is Warn, so in these APIs it is effectively Ignore. Intended, or worth a line in the docs? (concept-arith-semantics)
- `numpy(out=x)` after an arithmetic Error: when the read is split into pieces, a failing piece leaves `x` partly overwritten, whereas numpy writes the whole out and then raises. Not verified. (concept-arith-semantics)
- Should `write_array` to a file-like detect append mode or an unseekable stream before it writes anything? The K49 fix detects append mode only in finalize. (py-core-archive-errstate)
- Should the archive reader report a prost decode failure (recursion limit, malformed message) as InvalidArchive rather than Io? (py-core-numpy-wrap-dtype)
- `check_ndim` on an archive header returns its generic error kind, not InvalidArchive. Cosmetic; which kind it returns was not verified. (concept-untrusted-archives)

### Python API semantics
- `np.asarray(jix_array)` silently returns a 0-d object array that wraps the Array. There is no `__array__`, and pyo3 fills `sq_item` but not `sq_length`. As a result, `np.array_equal(j, [1, 2, 3])` is silently False and `jix.compact([j0, j0])` raises 'Unsupported dtype: object'. Accepted missing feature, or a finding? (py-core-array-surface-util)
- `len()` of a 0-d array raises ValueError where numpy raises TypeError, so `list(z0d)` fails in `len` instead of in iteration. Intended? (py-core-array-surface-util)
- `argmax` / `argmin` with `dim=None` work only for 1-D input, although the docstring says "When dim is None, returns a scalar". numpy returns the flat index. Intended? (py-ops-reduction-wrappers)
- A Python bool next to a negative Python int raises OverflowError (`jix.add(True, -1)`, `jix.where(c, True, -1)`), while `jix.subtract(True, 5)` works, because its table tries i8 first. Is this covered by the intended rule that "bool_arr + 1000 raises"? (py-ops-dispatch-promotion)
- `floor` / `ceil` / `round` raise TypeError for bool (numpy returns float16), while `absolute(bool)` returns its input. Intended? (py-ops-arith-wrappers)
- `jix.log` validates with `base <= 0.0`, which lets NaN through; `!(base > 0.0)` would not. Confirm which is intended. (py-ops-arith-wrappers)
- The acceptance of bools as dims (F026) also affects insert_dim, remove_dim, roll, repeat, tile, stack and flip. numpy accepts a bool axis for some of these. Should one shared rule apply everywhere? (py-ops-reduction-wrappers)
- Should ops that select values (max, maximum, minimum, clamp, where, abs on bool, bool -> bool casts) canonicalize bool8 output to 0/1? Today they pass bytes such as 22 or 255 through to numpy bool arrays, while all/any normalize. (util-test-infra, py-ops-reduction-wrappers)
- `dtype_from_numpy` maps scalars by (kind, itemsize) only. A third-party numpy DType with kind 'f'/'i'/'u'/'c'/'b' but a different encoding would be silently reinterpreted. Checking type_num or char would close this. (py-core-numpy-wrap-dtype)
- Is the platform dependence of numpy-scalar casts (INT_MIN on x86, saturation on aarch64) acceptable for `jix.cast`, or should `np.generic` go through jix's Cast (the F021 fix)? Only x86_64 was observed. (py-core-numpy-wrap-dtype)
- `plain_from_numpy` requires the inner-shape (subarray) dims to be C-contiguous even when the outer array is empty (operand.rs:450). No numpy layout was found that reaches this; check it together with the F033 fix. (concept-degenerate-shapes)

### Numerics and determinism
- Sign of zero in float sums: `jix.sum([-0.0])` is -0.0 (the K62 seed), where numpy 2 gives +0.0. The complex seed is (-0.0, +0.0), which is inconsistent between the two components. Should the -0.0 seed cover complex too? (py-ops-reduction-wrappers)
- F042 policy: is bit-for-bit reproducibility across hosts a goal? If so, either persist `read_size` in the archive, or decouple the reduction order from `read_size` and the block shape. (concept-threads-send-sync)
- Integer mean: i64/u64 mean accumulates in f64, grouped by a lane count that depends on the SIMD level. 131 values including i64::MIN/MAX give 11134.56 at AVX-512 and 11126.75 at AVX2; the exact mean is 11157.85. Is cross-machine determinism wanted for integer input? (util-test-infra)
- F037: if the hot float kernels stay status-free, document that only div/pow/exp/log/trig/sqrt report. (concept-arith-semantics)

### Block layout and params
- The per-dim max rule of `combine_block_layout` also flips layouts with no Scalar or Plain involved. `a([1024, 1] fixed) + row.broadcast([1024, 1024])` combines to [1024, 1024], non-fixed, and `compact()` then picks [1, 1024]. Should a fixed dim of one input win over a larger non-fixed dim of another? (storage-compact-params)
- Performance: for operands with orthogonal block shapes, the per-dim max gives tiles that are misaligned with one operand's blocks, so each of its blocks is decompressed about 100x per read. Seed the layout from the costliest operand's block shape instead? (concept-tiling-coverage)
- A `block_size` given with a fully fixed block shape is ignored for the shape. It is still stored, raises the floor of the default `read_size` (up to 1 GiB), and is inherited by every derived array. Intended? (storage-compact-params)
- Should the archive store the compression level, so that F040 can be fixed completely? (concept-untrusted-archives)

### Latent hazards in internal code
- `block_len_heuristic` calls `floor_to_multiple(base_block_len)` before `base_block_len.max(1)` (params.rs:505 vs 507), so a base of 0 would divide by zero. Both callers pass 1 today. (storage-compact-params)
- `Staging::fill_from` records `filled_with_n = usize::MAX` for a broadcast source after filling only n elements. That is correct only because the first chunk of each run is the longest. Record n instead? (concept-tiling-coverage)
- `scale_read_shape` leaves dims that are missing from `scale_order` unclamped, and a layout order that is not a permutation would build overlapping strides. This is only debug_asserted in `Array::from_storage`. Worth a release-mode check? (concept-tiling-coverage)
- `iter/nd.rs:60` tests emptiness with `shape.product::<u64>() == 0`. `shape.contains(&0)` would not depend on the global overflow invariant. (concept-degenerate-shapes)
- The Dtyped Safety doc says a zero-sized dtype has no shape. But `Dtype::new_struct(vec![], &[3], 0, ..)` and `from_fields(vec![]).set_shape(&[3])` both return Ok, and an archive can carry such a field. Harmless today, but the contract and the validators disagree. (types-dtyped-derive)
- `MAX_ITEMSIZE` for `[Array<S>; N]` sequences ignores N, while `read_bulk_as_iter` keeps `[[S::Item; M]; N]` on the stack. Probably harmless; batch5 raised the same question. (util-nd-arr-sequence)
- F007: release builds with `debug-assertions` enabled, a common CI setting, get the same 2^k copies. Whether LLVM deduplicates them at -O3 was not checked. (concept-composition)
- F008 design: should jix-py track chain depth and materialize or raise, or should core evaluate type-erased chains iteratively? Either way, dropping nested `Arc<dyn AnyInner>` needs an iterative drop. (concept-composition)

### 32-bit, huge sizes and untrusted input
- Writer side: `BlockArchiveWriter::start` and `OwnedBlockTableBuilder::start` allocate 12 bytes of locations per block up front, and compute `nblocks * 12` unchecked. (concept-untrusted-archives)
  - `compact()` / `write_array` of a huge lazy array aborts on allocation failure instead of returning Err. For example, a broadcast to 2^50 u8 gives about 2^35 blocks, or 384 GiB of locations.
  - On 32-bit, `nblocks as usize` truncates, and the block index then panics.
- In Minimal mode on 32-bit, `read_block` / `write_content` truncate `offset as usize`. This stays memory-safe, but an out-of-range offset can silently select a different in-range slice. Acceptable under the trusted-source contract? (concept-untrusted-archives)
- The release-build depth thresholds of F006 are unmeasured, and so are those of `dtype_to_numpy`, core Dtype drop and eq. This matters most on musl / Alpine, where thread stacks are small. (py-core-numpy-wrap-dtype)

### Platform and test coverage
- What CPU runs the CI ubuntu x86_64 job? Without AVX-512 (e.g. on an AMD EPYC 7763), no Avx512 arm of `for_each_simd_level!` and no AVX-512 pipeline or reduction path ever runs in CI. (util-test-infra)
- The SSE2/SSE4.2 levels of the pipeline driver, the reduction folds and bit_shuffle cannot be forced. A cfg/env override of `SimdLevel`, or per-level tests of the drivers, would close this (see F043). (util-test-infra)
- The `REQUIRE_ALIGNED` paths (armv7, ppc64le) have never run. Untested cases include SubDtype pull views of packed-struct fields, Transmute to types with wider alignment, and misaligned SubDtype views that the pipeline stages. (concept-dtype-layout, concept-composition)

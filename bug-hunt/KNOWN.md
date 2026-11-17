# Known issues

Issues already reported and still open, or deliberately left as-is. Hunters skip these; the dedup
stage drops findings that match one. Add an entry when you triage a finding (open, won't fix,
intended) so later runs do not report it again. When an issue is fixed, re-run its repro, then
remove the entry (or mark it `fixed in <commit/PR>` until the next consolidation): a fixed issue
that comes back is a regression and should be reported again.

IDs are stable and never reused: K1-K183 were assigned by the 2026-10/11 scans and K184-K200 by the
two scans of #106 / #107 (the fixed ones were dropped when the scans were consolidated into
`report/`), so the next new entry is K201.

Format: one entry per issue.

```
## K<n>: <one-line title>
- Status: open | fixed in <commit/PR> | wont-fix | intended - <why>
- Location: <file:line>
- Notes: <what exactly is known; which triggers are covered>
```

## K8: i686 Linux: alignment hard-coded to 8 for f64/i64/u64/c128 while the ABI says 4
- Status: open, tracked in todo.md: i686 is a known gap (found in run 2026-10-21-pilot as F008); still open at main c705a54 (#107)
- Location: jix/src/dtype.rs:967
- Notes: bug-hunt/report/findings/K8.md; repros: K8.py, repro_k8.rs in bug-hunt/report/repros/.

## K27: Complex division uses num-complex's unscaled textbook formula: NaN/inf/0 or ~1e-5 relative error for finite operands with |divisor| beyond sqrt(MAX) or below sqrt(MIN_POSITIVE)
- Status: intended - textbook division is accepted, like the num-complex one (found in run 2026-10-22-trial2 as F014, medium)
- Location: jix/src/ops/op2.rs:781
- Notes: run 2026-10-22-trial2 F014 (its write-up was not kept: only open items were kept when the scans were consolidated). See K74 (run 2026-10-24-batch4 F009: same formula in the complex mean).

## K31: Python numpy(out=x) never checks that x overlaps a buffer the array reads: &mut [u8] over memory a live &[u8] input covers (UB) and order-dependent wrong values, also without any explicit asarray
- Status: intended - documented in c009869 (#87): out= must not overlap any numpy array in the lazy chain (found in run 2026-10-22-trial2 as F018, medium)
- Location: jix/src/storage/elementwise_pipeline.rs:280
- Notes: run 2026-10-22-trial2 F018 (its write-up was not kept: only open items were kept when the scans were consolidated). See K111 (run 2026-10-29-batch6 F003: element-disjoint views whose spans interleave also alias).

## K60: jix.power(x, 2) / x ** 2 is routed to Square and silently drops the float Overflow report that the jix.power docstring promises (f16/f32/f64)
- Status: intended - x ** 2 on floats is a plain square, without the Pow overflow report (found in run 2026-10-23-batch3 as F019, low)
- Location: jix-py/src/ops/op2.rs:688
- Notes: run 2026-10-23-batch3 F019 (its write-up was not kept: only open items were kept when the scans were consolidated).

## K64: Array::to_ndarray_buf pull mode copies the whole region for large reads of lazy views (broadcast, permute/slice over Plain, collapsed pipelines) that read_data(None) returns in O(1); a huge broadcast aborts
- Status: open, acknowledged in todo.md (#90): "pull mode to_ndarray_buf splits large reads of O(1) lazy views and copies the whole region" (found in run 2026-10-23-batch3 as F024, low); still open at main c705a54 (#107), which dropped its todo.md line: the permute-over-Plain case no longer copies, but Broadcast and element-wise pipelines still materialize the whole region
- Location: jix/src/array/read.rs:458
- Notes: bug-hunt/report/findings/K64.md; repros: repro_k64.rs in bug-hunt/report/repros/.

## K65: power with a u64 exponent promotes to float64 (integer exponent slots stop at u32)
- Status: intended (run 2026-10-23-batch3 F023; pinned by test_power_promotes_uint64_exponent_to_float64)
- Location: jix/src/ops/op2.rs:1173
- Notes: doc nit - the #81 docstring rewrite dropped the explicit u64-exponent sentence.

## K69: Reshape reads every split/merge reshape (flatten, [N]->[a,b], block-aligned splits) one element at a time, decompressing a whole block per element: 400-5000x slower reads
- Status: open, tracked in todo.md as a performance item, not a bug (found in run 2026-10-24-batch4 as F004, medium); still open at main c705a54 (#107)
- Location: jix/src/ops/shape_ops/reshape.rs:371
- Notes: bug-hunt/report/findings/K69.md; repros: K69.py, repro_k69.rs in bug-hunt/report/repros/.

## K71: f64 -> f16 via half::f16::from_f64 (no_std software fallback) still mis-rounds on every shipped target except macOS arm64, about 8x as often as the pre-K28-fix double rounding through f32
- Status: open (found in run 2026-10-24-batch4 as F006, medium); still open at main c705a54 (#107)
- Location: jix/src/ops/elementwise/cast.rs:54
- Notes: bug-hunt/report/findings/K71.md; repros: K71.py, repro_k71.rs in bug-hunt/report/repros/. Incomplete K28 fix. The in-tree test is gated to aarch64 and the property oracle calls the same function. Still open at 7ebac4c (#104): jix/src/ops/elementwise/cast.rs:55 still uses f16::from_f64.

## K72: reduce_unordered over wide Dtyped elements overflows a 2 MiB thread stack and aborts: the one-cell reduction fold keeps 16 (strided) / 8 (contiguous) lanes of T whatever the element width
- Status: fixed in release builds in 691e11b (#91); debug builds still abort (found in run 2026-10-24-batch4 as F007, low); still open at main c705a54 (#107)
- Location: jix/src/ops/reduction/common.rs:845
- Notes: bug-hunt/report/findings/K72.md; repros: repro_k72.rs in bug-hunt/report/repros/. The K30 defect in the reduction fold, which the K30 fix did not touch. After #91 the repro passes with --release (652 KiB for both one-cell cases, the multi-cell control 396 KiB), but debug builds now need 7.8 MiB (contiguous) / 8.9 MiB (strided), up from 2.1-5.7 MiB: `dispatch_lanes!` in fold_one_cell_per_simd (reduction.rs:852-860) matches on the const lane count and all ten `#[inline(always)]` arms (1..512 lanes) are probably inlined into one frame, which debug builds do not prune.

## K77: A stepped Slice over a non-plain inner makes one inner read per output position of the stepped dims, decompressing whole blocks each time: step-2 slices read 200-2100x slower than the whole array
- Status: open, tracked in todo.md as a performance item, not a bug (found in run 2026-10-24-batch4 as F012, medium); still open at main c705a54 (#107)
- Location: jix/src/ops/shape_ops/slice.rs:186
- Notes: bug-hunt/report/findings/K77.md; repros: repro_k77.rs in bug-hunt/report/repros/. Same read-amplification pattern as K69 (F004).

## K108: var/std of finite data whose deviations overflow returns NaN again for 3+ elements (var([1e308, -1e308, 0.0]) = NaN, numpy inf) - the K84 fix partly reverts K67
- Status: open (found while rebasing onto df2f918 (#92)); still open at main c705a54 (#107)
- Location: jix/src/ops/reduction/variance.rs:154
- Notes: #92 maps only `m2 < 0.0` to +inf and keeps a NaN m2 as NaN (the K84 fix). Welford can also reach a NaN m2 from finite input: for [1e308, -1e308, 0.0] the mean becomes -inf after the second item, then `0.0 - (-inf)` and `-inf + inf` make mean and m2 NaN. So `var([1e308, -1e308, 0.0])` and `var([1e308]*40 + [-1e308]*40)` return NaN (two of the six K67 repro tests fail again, the K67 repro of run 2026-10-24-batch4 (not kept; bug-hunt/report/repros/repro_k108.rs is the current repro)), while numpy gives inf; before #92 they gave inf. The sign of m2 cannot tell this from a NaN / inf input (K84), so the fix needs another signal, e.g. a flag in the state for a non-finite input. Still open at 7ebac4c (#104): jix/src/ops/reduction/variance.rs:156 is unchanged.

## K109: PyWriter::write hands write() a memoryview over Rust memory; derived views (b[:], memoryview(b), b.cast, np.frombuffer(b), PickleBuffer(b)) survive release() and outlive the buffer - use-after-free from pure Python
- Status: open: the #93 fix does not cover it (found in run 2026-10-29-batch6 as F001, high); still open at main c705a54 (#107); c705a54 (#107) only documents it ("Document that write_array's buffer release is best effort")
- Location: jix-py/src/archive.rs:423
- Notes: bug-hunt/report/findings/K109.md; repros: K109.py in bug-hunt/report/repros/. #93 calls memoryview.release() after write() and raises if it fails, but slices (b[:]) and np.frombuffer(b) register with the shared managed buffer, not as exports of b, so release() succeeds, write_array returns normally, and the views still read freed memory: re-run on 6dbeb15, 19 of 20 stale views changed and the child processes segfault. A fix needs the memory to outlive every view, e.g. pass bytes, or an object that owns the buffer. Still open at 7ebac4c (#104): jix-py/src/archive.rs:455-490 is unchanged.

## K111: Plain::read_data (and jix-py numpy(out=)) take a &[u8] / &mut [u8] over the whole strided span, gap bytes included, so element-disjoint views whose spans interleave alias each other: UB from safe Rust and from Python (Miri)
- Status: open, acknowledged in todo.md (#93) (found in run 2026-10-29-batch6 as F003, high); still open at main c705a54 (#107)
- Location: jix/src/storage/plain.rs:460
- Notes: bug-hunt/report/findings/K111.md; repros: K111.py, repro_k111.rs in bug-hunt/report/repros/. Extends K31 (intended: out must not overlap) to element-disjoint views and to the safe Rust API. Miri reports UB; values are correct today.

## K116: Deep type-erased op chains (every Python op result) recurse once per op level on read and on drop: `x = x + 1` in a loop overflows the native stack (Python SIGSEGV from ~2k levels with the debug extension, ~7k in release; drop from ~30-50k)
- Status: open, acknowledged in todo.md (#93) (found in run 2026-10-29-batch6 as F008, medium); still open at main c705a54 (#107)
- Location: jix/src/storage/core_trait.rs:113
- Notes: bug-hunt/report/findings/K116.md; repros: K116.py, repro_k116.rs in bug-hunt/report/repros/.

## K126: jix.log(f16, base) for a valid base near 1 rounds a finite result to +-inf with no Overflow report (no warning, no error under over='raise'); jix.divide on the same quotient and the f16 exp/tan/power kernels do report it
- Status: intended since e6ef7ee (#95): errstate is removed and float errors are never reported (found in run 2026-10-29-batch6 as F018, low)
- Location: jix-py/src/ops/op1.rs:369
- Notes: run 2026-10-29-batch6 F018 (its write-up was not kept: only open items were kept when the scans were consolidated). K107 covered the f16 multiplier overflowing, not the result; K106 fixed the same narrowing for the f16 unary kernels. Re-checked on 6dbeb15: jix.log(f16, base=1.0001) still overflows to inf with no Overflow report.

## K133: jix.Array binary/comparison dunders never return NotImplemented: a foreign operand's reflected op (even with __array_ufunc__ = None) is never tried, `a == None` raises ValueError, unsupported operand types raise ValueError instead of TypeError
- Status: partly fixed in 6dbeb15 (#93) and by c705a54 (#107): binary ops return NotImplemented for operands with __array_ufunc__ = None, and unsupported operand types raise TypeError; still open: a == None, a != None and a == 'x' raise TypeError where numpy compares element-wise (found in run 2026-10-29-batch6 as F025, low)
- Location: jix-py/src/array.rs:894
- Notes: bug-hunt/report/findings/K133.md; repros: K133.py in bug-hunt/report/repros/. The opposite direction of K37. Still open: a + None, a + 'x', a + object(), a * Fraction(1, 2) raise ValueError ('Unsupported dtype') instead of TypeError, and a == None / a != None raise.

## K145: Float add/sub/mul/square and sum/prod/mean never report invalid/over, while divide reports the same conditions; under default modes inf/inf raises but inf-inf, 0*inf and sum([inf,-inf]) are silent NaN
- Status: intended since e6ef7ee (#95): errstate is removed and float errors are never reported (found in run 2026-10-29-batch6 as F037, low)
- Location: jix/src/ops/op2.rs:590
- Notes: run 2026-10-29-batch6 F037 (its write-up was not kept: only open items were kept when the scans were consolidated). Answers the batch3 / batch5 question on float Overflow / Invalid reporting.

## K150: Float reductions depend on the host's detected CPU cache sizes (and SIMD level): the same numpy array, or the same .jix file, sums to different f32 values on machines with different L1d/L2
- Status: open (found in run 2026-10-29-batch6 as F042, low); still open at main c705a54 (#107)
- Location: jix/src/storage/params.rs:456
- Notes: bug-hunt/report/findings/K150.md; repros: K150.py, repro_k150.rs in bug-hunt/report/repros/. Answers the pilot's question on host-dependent float results. Re-checked on 6dbeb15: still host-dependent.

## K151: SIMD level coverage: for_each_simd_level! skips the levels the host lacks without saying so, and level-dependent driver code (pipeline lane count, reduction folds, update_bulk fallbacks) runs only at the host's Level::new() level
- Status: open (found in run 2026-10-29-batch6 as F043, low); still open at main c705a54 (#107)
- Location: jix/src/util/test_util.rs:1291
- Notes: bug-hunt/report/findings/K151.md; repros: K151.py in bug-hunt/report/repros/. Test infrastructure only.

## K152: Free-threaded builds: numpy(out=) validates out's shape/strides in one read but writes using a second read (array.rs:617/653), and plain_from_numpy reads ndim, shape and strides separately (operand.rs:424/465/471). A concurrent `.shape=` or `.strides=` becomes an out-of-bounds write or read, or a PanicException.
- Status: open, plausible (found in run 2026-10-29-batch6 as F044, low); still open at main c705a54 (#107)
- Location: jix-py/src/array.rs:590
- Notes: bug-hunt/report/findings/K152.md; repros: K152.py in bug-hunt/report/repros/. Plausible: argued from the code; needs a free-threaded build.

## K158: Every element-wise op deep-copies (Arc::make_mut) the cache-id set it shares with its input, so a left-deep lazy chain of n ops holds about n^2/2 ids: memory grows quadratically with chain depth (~120 MB at 4000 ops, ~740 MB at 10000; `x = x + 1` in a Python loop)
- Status: open (found in run 2026-11-15-batch7-1 as F006, low); still open at main c705a54 (#107)
- Location: jix/src/storage/params.rs:817
- Notes: bug-hunt/report/findings/K158.md; repros: K158.py, repro_k158.rs in bug-hunt/report/repros/ (Rust and Python). Regression from #100 (per-node cache-id sets). Shape ops that call with_cache_scope reset the set. K116 bounds the readable depth (~7k ops in release). Re-checked: Rust ratio 15.2 for 4x depth; Python +117 MB vs +9 MB.

## K161: Scalar::new and Plain::new run tune's padded-shape check on an auto block shape they then overwrite with all-ones, so `y + 1` (any scalar operand) on a valid lazy array of ~2^63+ elements raises InvalidShapeOperation while `y + y` works
- Status: open (found in run 2026-11-15-batch7-2 as F002, low); still open at main c705a54 (#107)
- Location: jix/src/storage/params.rs:387
- Notes: bug-hunt/report/findings/K161.md; repros: K161.py, repro_k161.rs in bug-hunt/report/repros/ (Rust and Python). Side effect of the K101 fix. Python: any scalar operand; Rust: only through the doc-hidden __private::Scalar::new or unsafe plain_ndarray_ptr with zero strides. Also hits empty shapes like (0, 2**32 - 1, 2**32 + 1). No wrong values. Re-checked on x86_64: 2 Rust tests fail, Python 4 of 4 scalar ops raise.

## K169: After an integer ArithmeticError, the caller's push destination (numpy(out=), to_buf_slice, to_buf_strided_into) holds a host-dependent mix of results, old bytes and raw reduction states (int mean leaves i64 partial sums read as f64 bits; where() leaves x where cond picks y), and no Errors / Raises section mentions ErrorKind::Arithmetic or the destination's state
- Status: intended - documented in c705a54 (#107): a read that fails leaves the destination buffer unspecified (found in run 2026-11-15-batch7-4 as F004, low)
- Location: jix/src/array/read.rs:556
- Notes: bug-hunt/report/findings/K169.md; repros: K169.py, repro_k169.rs in bug-hunt/report/repros/ (Rust and Python). Every push consumer that fills out in several steps propagates the first Err with `?`: the slow-path piece loop (read.rs:555, cut by host cache sizes, cf. K150), reductions with states in out (reduction/common.rs:479, :563), Select (select.rs:142), Concatenate (concatenate.rs:320). Only the fused pipeline writes the whole region before returning Err (pipeline.rs:310). to_ndarray / numpy() without out are unaffected. No UB. Suggested fix: docs - list Arithmetic in the Errors / Raises sections of the read entry points and state that out is unspecified after any error. The repros assert numpy's full-write behavior, so they stay red after a docs-only fix. Re-checked on x86_64: Rust 3/3 fail (2113536 of 4194304 written; out[..4] as i64 = [96010, 99850, 95990, 99830]; out[..3] = [1, 2, 3]), F004.py checks A/B/C FAIL.

## K170: Python ops reject an untyped Python int in [2**63, 2**64) with OverflowError even next to a uint64 operand (jix.add(u64_arr, 2**63)), where numpy accepts it; jix.asarray(2**63, dtype=np.uint64) accepts the same int since the K96 fix
- Status: intended since dbb99e2 (#85, "Raise OverflowError for a Python int outside int64") (found in run 2026-11-15-batch7-4 as F001, low)
- Location: jix-py/src/ops/common/operand.rs:55
- Notes: no write-up; run 2026-11-15-batch7-4, section 4 of its report (not kept) and the verifier's scratch script (not kept). Pinned by test_python_int_outside_int64_raises (jix-py/python/tests/test_type_promotion.py:520-528), whose docstring states the numpy difference; the error names the value and np.uint64(...) works. Left for the maintainers: the asarray-vs-op inconsistency (both pinned by tests). Do not re-report.

## K171: f16 ** x is computed in f32 (f64 for i32 exponents) and rounded to f16, so 7 of 319,445 results near an f16 midpoint come out 1 ulp off (max error 0.500055 ulp); f16 exp / log / sin / cos misround at a similar rate
- Status: intended - "computed in f32, rounded back to f16" is the f16 design (found in run 2026-11-15-batch7-4 as F002, low)
- Location: jix/src/ops/elementwise/arith.rs:380
- Notes: no write-up; run 2026-11-15-batch7-4, section 4 of its report (not kept) and the verifier's scratch script (not kept). Pinned by the property-test oracle at arith.rs:980-987 and test_type_promotion.py:707; numpy 2.x returns the same values. A correctly rounded fix needs a correct f64 -> f16 rounding, which is the open K71. Do not re-report.

## K176: Layout-identical numpy struct dtypes (default vs align=True, or a Rust #[derive(Dtyped)] file vs a default numpy dtype) become unequal jix dtypes, so cast / concatenate / stack / where between them raise TypeError, while .dtype == and numpy(out=) treat them as the same
- Status: open (found in run 2026-11-15-batch7-5 as F005, low); still open at main c705a54 (#107)
- Location: jix-py/src/dtype.rs:217
- Notes: bug-hunt/report/findings/K176.md; repros: K176.py in bug-hunt/report/repros/. dtype_from_numpy copies numpy's alignment (1 for every non-align=True struct), and core Dtype::eq compares alignment; jix-py cast_impl (casting.rs:56) and promotion_target (dtype_promote.rs:177) then refuse. Keep core eq (into_typed soundness depends on it); add a jix-py "same layout" test with a transmute_elements_dyn retag. Re-checked: F005.py 5 TypeErrors (asarray(dtype=), cast, concatenate, stack, where).

## K179: where() reads both branches, so an integer error in an element it does not select raises ArithmeticError, except for some spellings of a 0-d condition (Python bool, numpy bool scalar, 0-d numpy array skip the other branch; a 0-d jix array reads it); numpy's guard idiom where(b != 0, a // b, 0) always raises, and neither is documented
- Status: open (found in run 2026-11-15-batch7-5 as F008, low); still open at main c705a54 (#107)
- Location: jix-py/src/ops/elementwise/select.rs:72
- Notes: bug-hunt/report/findings/K179.md; repros: K179.py in bug-hunt/report/repros/. Settles batch 7-4's question (c). Related to K169 (the same eager reads). The scalar shortcut is from #95; core Array::select has no shortcut. rounding_mode trunc / euclid, % and negative integer powers behave the same. Suggested fix: docs first (workaround a // where(b != 0, b, 1)); optionally take the shortcut for a 0-d jix bool array too. Re-checked: F008.py checks A and B FAIL.

## K181: power() with a float base rejects a Python int exponent outside i32 (f64_arr ** 2**31 -> OverflowError "out of bounds for i32") although the result is float either way and the later (float, float) slot holds it; every other op accepts any i64 Python int next to a float array
- Status: open (found in run 2026-11-15-batch7-5 as F012, low); still open at main c705a54 (#107)
- Location: jix-py/src/ops/common/dispatch.rs:249
- Notes: bug-hunt/report/findings/K181.md; repros: K181.py in bug-hunt/report/repros/. The Pow table lists (float, i32) before (float, float), an untyped Python int passes the Safe check into i32, and dispatch_args does not fall through on a range error. The integer-base case (int64_arr ** 2**32 -> "out of bounds for u32") is intended, like the pinned power(int_arr, -1). Suggested fix: send an untyped int exponent outside i32 to (float, float) for a float base, or document and pin the limit. Re-checked: F012.py 5 checks FAIL (2**31, -2**31 - 1, f32 base, np.float64 base, 10**10).

## K182: subtract(bool_arr, python_int) picks int8 (first match in the i8-first bool-subtract table), so b - 200 raises OverflowError instead of using a wider dtype
- Status: intended - follows the deliberate i8-first ordering of dbb99e2 ("Subtract two bools as int8"), with first-match-then-raise for an int that does not fit documented in test_type_promotion.py:430-452 (refuted in run 2026-11-15-batch7-5 as F010, low)
- Location: jix-py/src/ops/elementwise/arith.rs:319
- Notes: no write-up; run 2026-11-15-batch7-5, section 4 of its report (not kept) and the verifier's scratch script (not kept). Pinned by the docstring (arith.rs:316-317) and test_arith.py:665-672. The u8 slot the hunter proposed would silently give b - 1 -> [0, 255]. Do not re-report.

## K183: numpy fromnumeric reductions on a jix.Array (np.sum / prod / max / amax / min / amin / mean / std / var / all / any, and np.squeeze(axis=)) raise "Array.sum() got an unexpected keyword argument 'axis'", while np.argmax / nanmax / median / average / round / cumsum / sort work through __array__
- Status: open (found in run 2026-11-16-batch7-6 as F001, low); still open at main c705a54 (#107)
- Location: jix-py/src/array.rs:1336
- Notes: bug-hunt/report/findings/K183.md; repros: K183.py in bug-hunt/report/repros/. numpy calls the same-named method of any non-ndarray with numpy keywords (`_wrapreduction`, `mean`, `std`, `var`, `squeeze` have no TypeError fallback; `_wrapfunc` does), and jix's methods take `(dim=None, *, keepdim=False[, ddof])`; same mismatch at all 1266, any 1277, max 1287, min 1297, mean 1331, prod 1341, std 1351, var 1362, squeeze 1442. Raises even with no axis given, naming a method the user never called. Nothing silently wrong; np.squeeze(z) returns a lazy jix.Array. Not a regression: #94's `__array__` made the sibling functions work, which is what makes the surface inconsistent; jix-py/docs/module.md:14 says the library is NumPy-compatible and no doc or test says np.* reductions are unsupported. np.ptp raises the deliberate `__array_ufunc__ = None` error instead (ufunc reduce on the object). Suggested fix: an `__array_function__` that converts via `__array__`, or numpy keyword aliases on these methods (axis / keepdims, out / dtype / where only at their defaults), or at least documentation; add a test to test_array_protocol.py. Re-checked: F001.py 13 failures (numpy 2.5.4), 5 controls ok; np.reshape / transpose / swapaxes / clip / take / trace also return ndarrays via the `_wrapfunc` fallback.

## K184: #106 reused FileMetadata field 2 of the archive schema (uint64 varint -> Semver message), so archives written by jix 0.0.7 fail to decode on current main (InvalidArchive, worded like corruption) and jix 0.0.7 cannot read new archives; both call themselves 0.0.7
- Status: open (found in run 2026-11-16-scan-106-107 as F001, medium)
- Location: jix/schema/proto/jix/v1/structure.proto:7
- Notes: bug-hunt/report/findings/K184.md; repros: repro_k184.rs, K184.py (fixture K184_written_by_jix_0.0.7.jix; the reverse check needs the K184_oldjix_cli build) in bug-hunt/report/repros/. A #106 regression, unreleased so far, so the fix is free: `reserved 2; reserved "lib_version_semver"; Semver lib_version = 3;` (verified with real 0.0.7 and current binaries). The "(archive written by X, read by Y)" suffix wraps only read_body, so the metadata decode error does not say which version wrote the file.

## K185: 32-bit: the #107 isize::MAX read-size cap does not bound read tiles or compaction chunks: scale_read_shape grows broadcast / tiled dims to 2 * max and compaction floors its chunk at one target block, so compact_with panics with 'capacity overflow' and reductions over long broadcasts return a spurious 'does not fit in memory' Err
- Status: open (found in run 2026-11-16-scan-106-107 as F002, low)
- Location: jix/src/storage/params.rs:26
- Notes: bug-hunt/report/findings/K185.md; repros: repro_k185.rs (i686 only) in bug-hunt/report/repros/. The remainder of K154 (whose out-of-bounds read #107 fixed; K154 is merged here) and of K153's path. Growth at util/mod.rs:505-509; the panic at array/compact.rs:264 -> util/aligned_vec/raw.rs:33. With 1 GiB blocks it happens at the default read size. Fix: clamp the grown tile / chunk bytes to isize::MAX, or check before reserving and return an Err.

## K186: The #107 K168 fix makes approx_equal / isclose reject every pair whose a - b overflows, even when the tolerance covers it exactly (atol = inf, rtol = inf, rtol >= 2, or 1 + min/max <= rtol < 2); f16 hits it at ordinary magnitudes (40000 vs -40000)
- Status: open (found in run 2026-11-16-scan-106-107 as F003, low)
- Location: jix/src/ops/elementwise/cmp.rs:230
- Notes: bug-hunt/report/findings/K186.md; repros: repro_k186.rs, K186.py in bug-hunt/report/repros/. A regression from the K168 fix: `diff.is_finite()` is ANDed into the result. Through Python any f16 atol >= 65520 casts to inf. The Rust test reference was changed the same way, so the tests pin the wrong answer. Fix: compare at half scale when the difference overflows.

## K187: 32-bit: a stepped Slice keeps its inner's ReadTiling::ZeroCopy (#106) even when its K156 fallback (#107) materializes the region, so an outer stepped Slice pulls the whole bounding box from it: a pull of 2 elements allocates and reads O(step) elements
- Status: open (found in run 2026-11-16-scan-106-107 as F004, low)
- Location: jix/src/ops/shape_ops/slice.rs:103
- Notes: bug-hunt/report/findings/K187.md; repros: repro_k187.rs (tracking allocator; fails on i686, passes on x86_64) in bug-hunt/report/repros/. Values are correct and push reads are unaffected; before #106 stepped Slices cleared their flags. Fix: do not classify a stepped Slice as ZeroCopy when its box can fail check_region_size (32-bit).

## K188: ArchiveReader::new takes the archive start from stream_position() (tell() for a Python file-like) and restores it with an absolute seek, so with a file-like whose tell() is wrong (K177's tell()==0 reader, which #107 supports) a second back-to-back read_array silently returns the first archive again
- Status: open (found in run 2026-11-16-scan-106-107 as F005, low)
- Location: jix/src/archive/common.rs:120
- Notes: bug-hunt/report/findings/K188.md; repros: K188.py in bug-hunt/report/repros/. The #107 K177 fix turned a loud panic into silent wrong data; the #107 test packs the same archive twice, so it cannot notice. Fix: use the positions seek() returns (Current(0), End(0), then a relative restore) and pack two different archives in the test. By design (not part of this entry): a file-like whose seek(0, 2) raises now fails even with len=, and seek() returning None with tell() == 0 fails loudly with 'jix file too short'.

## K189: 32-bit: Owned::read_section checks a section's element count against usize, not its byte size against isize::MAX, so a block_data or blocks_loc section over 2 GiB panics with 'capacity overflow' in Vec::resize after reading 1-1.5 GiB, instead of returning Err
- Status: open (found in run 2026-11-16-scan-106-107 as F006, low)
- Location: jix/src/archive/block.rs:394
- Notes: bug-hunt/report/findings/K189.md; repros: repro_k189.rs (i686; ignored on 64-bit, run with --include-ignored) in bug-hunt/report/repros/. Reachable with a real archive from a 64-bit host or a sparse file; Python gets a PanicException. Fix: reject sections over isize::MAX bytes with the existing 'section size ... is too large' error and grow with try_reserve, as BlockLocation::alloc_table does.

## K190: Setting one codec key in params (compression_level, codec or filters) resets the source's other codec settings to their defaults (e.g. compact(src, compression_level=3) brings byte-shuffle back on a filter-less source), while the Python compact / write_array and Rust compact_with docs promise unset keys are inherited
- Status: open (found in run 2026-11-16-scan-106-107 as F007, low)
- Location: jix/src/storage/params.rs:233
- Notes: bug-hunt/report/findings/K190.md; repros: repro_k190.rs, K190.py in bug-hunt/report/repros/. Dates from #55, not a #106 / #107 regression. ArrayParams::codec / level / filters each seed the whole EncoderParams group from defaults, and inherit_unset_from_storage inherits the group only when wholly unset; only ArrayParams::codec's doc mentions the reset. Data still round-trips. Decision needed: inherit per key, or document the group reset everywhere and pin it with a test.

## K191: 64-bit: for an empty region whose non-zero extents exceed isize::MAX bytes, the push read entries (to_buf_slice, to_buf_strided_into) return Ok while the pull and ndarray entries (to_buf_strided, to_ndarray_sub) return InvalidShapeOperation
- Status: intended - each entry's error docs state this split (the push entries list the size error only on a 32-bit target), #107 made it deliberately ('Bound the region size of pull reads on every target'), and an empty push touches no memory (judged intended in run 2026-11-16-scan-106-107 as F008, low)
- Location: jix/src/array/read.rs:293
- Notes: run 2026-11-16-scan-106-107 F008 (its write-up and repro were not kept: only open items are kept in bug-hunt/report/). The trigger needs more than 8 EiB of extents; the in-tree test read_entries_agree_on_an_empty_region_with_huge_extents is the 32-bit K157 test, not a cross-target contract. Do not re-report.

## K192: Select is classified ReadTiling::Fusable (#106), but its read_data pulls the condition and y into region-sized temporaries, so a top-level read of cond_op.select(x, y_op) runs untiled and allocates them for the whole region
- Status: open (found in run 2026-11-17-scan-106-107-b as F001, low)
- Location: jix/src/ops/elementwise/select.rs:109
- Notes: bug-hunt/report/findings/K192.md; repros in bug-hunt/report/repros/: repro_k192.rs. A #106 regression (before #106 Select cleared its flags); the same class as K64 / K187 with its own root cause; Rust API only (jix-py's Any demotes the class). A 16 MiB push read allocates 20 MiB, against 160 KiB tiled. Fix: classify Select as Tiled unless the condition and y are ZeroCopy, or stream read_data through its own pipeline. The concept-read-contract audit found Select the only storage that pulls its inputs while classified fused().

## K193: 32-bit: since #106 a ZeroCopy / Fusable / SinglePass read is not tiled, so a narrowing op over broadcast leaves (a struct field, a map or cast to a smaller type, a comparison) fails with a spurious 'does not fit in memory' Err: each leaf checks the whole region at its own wider itemsize
- Status: open (found in run 2026-11-17-scan-106-107-b as F002, low)
- Location: jix/src/array/read.rs:458
- Notes: bug-hunt/report/findings/K193.md; repros in bug-hunt/report/repros/: repro_k193.rs (i686). A #106 regression, confirmed against a build of a7b0656 (4/4 pass there). Same family as K185 / K187: check_out_buf (buf.rs:359, broadcast.rs:130) checks region * leaf itemsize even for 0-stride pull views that allocate nothing. Fix: check only memory that is really allocated, or tile when nitems * max leaf itemsize > isize::MAX (one fix may also cover K194).

## K194: 32-bit: a stepped Slice over a SubDtype checks its zero-copy bounding box at the field itemsize, but the SubDtype (ZeroCopy since #106) pulls that box from its parent at the parent itemsize, so a 2-element pull fails with InvalidShapeOperation instead of taking #107's K156 fallback
- Status: open (found in run 2026-11-17-scan-106-107-b as F003, low)
- Location: jix/src/ops/shape_ops/slice.rs:159
- Notes: bug-hunt/report/findings/K194.md; repros in bug-hunt/report/repros/: repro_k194.rs (i686). #106's ReadTiling meets #107's K156 fallback; K187's suggested fix does not cover it. The push read of the same view works; no 64-bit analogue. Fix: do not classify a narrowing SubDtype as ZeroCopy on 32-bit, or check the box at the largest itemsize in the view chain.

## K195: A dim of length exactly PY_SSIZE_T_MAX accepts out-of-range slice bounds: PySlice_Unpack clamps them to the dim length, so a[2**100:] / a[n+1:] return empty and a[5:2**100] / a[0:n+1] select to the end instead of raising IndexError
- Status: open (found in run 2026-11-17-scan-106-107-b as F004, low)
- Location: jix-py/src/util.rs:36
- Notes: bug-hunt/report/findings/K195.md; repros in bug-hunt/report/repros/: K195.py. Older code, not a #106 / #107 regression; breaks the contract test_slice_huge_bounds pins for shorter dims. A 32-bit variant (a dim of length 2**31 - 1) is argued from code. Fix: read slice.start / slice.stop as Python ints instead of PySlice_Unpack (one fix could also cover K196 and batch 7-5's 32-bit slice question).

## K196: An integer index outside i64 (a[2**63], a[-(2**63)-1], a[np.uint64(2**63)], jix.slice(a, 2**64)) raises TypeError 'only integers, slices ... are valid indices' instead of the IndexError that the docstrings promise and that a[2**63-1] / a[0:2**63] raise
- Status: open (found in run 2026-11-17-scan-106-107-b as F005, low)
- Location: jix-py/src/ops/shape_ops.rs:274
- Notes: bug-hunt/report/findings/K196.md; repros in bug-hunt/report/repros/: K196.py. parse_basic_index extracts int items as i64 and sends any failure to the catch-all TypeError. Fix: classify the item by type first and map overflow to IndexError.

## K197: numpy()'s GIL decision can never see ReadTiling::Fusable: every jix-py array is an ArrayAny, which demotes Fusable to SinglePass, so tiny fused reads over numpy memory always release the GIL (#106's attach-for-cheap-reads arm is dead for ops)
- Status: open (found in run 2026-11-17-scan-106-107-b as F006, low)
- Location: jix-py/src/array.rs:633
- Notes: bug-hunt/report/findings/K197.md; repros in bug-hunt/report/repros/: repro_k197.rs (deterministic), K197.py (GIL hand-off latency; inconclusive on some hosts). Performance only, not a regression. AnyMeta::new demotes the class (jix/src/storage/any.rs:105-108). Fix: expose the pre-erasure tiling to jix-py (or drop the Fusable arm) and pin the attach decision with a test.

## K198: Identity Slice / Flip / Roll and `&dyn ArrayStorage` pass their inner's ReadTiling::Fusable through (#106) without forwarding read_as_elementwise_pipeline, so a parent element-wise op reads untiled and pulls a temporary the size of the whole region
- Status: open (found in run 2026-11-17-scan-106-107-b as F007, low)
- Location: jix/src/ops/shape_ops/slice.rs:229
- Notes: bug-hunt/report/findings/K198.md; repros in bug-hunt/report/repros/: repro_k198.rs. A #106 regression (argued from the a7b0656 source). A 16 MiB region needs a 16 MiB temporary, against 128 KiB through a non-identity view. Fix: forward the pipeline in the identity case (preferred), or demote the tiling with forwarded() as Any and Transmute do. Not measured: ident.view() + ident.view() (the identity Slice's own cache scope may compute the inner twice).

## K199: #107's K180 fix computes the numpy dtype of every jix operand that gets broadcast, so on i686 Linux any broadcasting op with a jix f8 / i8 / u8 / c16 operand (x + y with broadcasting, a > a.mean()) raises ValueError when the op is built
- Status: open (found in run 2026-11-17-scan-106-107-b as F008, low)
- Location: jix-py/src/ops/common/broadcast.rs:54
- Notes: bug-hunt/report/findings/K199.md; repros in bug-hunt/report/repros/: K199.py (simulates i686 by making numpy.dtype fail; no i686 Python was available). A #107 regression and a new trigger of K8 (fixing K8 fixes it). Fix: carry over a numpy dtype only if one is already cached, through a peek that does not compute it.

## K200: jix.reshape / jix.broadcast (and Array.reshape / Array.broadcast) accept a Python bool as a shape entry (True -> 1, False -> 0), unlike numpy and unlike jix's own Dim and index arguments
- Status: open (found in run 2026-11-17-scan-106-107-b as F009, low)
- Location: jix-py/src/ops/shape_ops.rs:721
- Notes: bug-hunt/report/findings/K200.md; repros in bug-hunt/report/repros/: K200.py. The shape arguments are ItemOrTuple<i64>, which accepts bool: a.broadcast((False, 6)) silently returns a (0, 6) array. jix rejects bool for Dim since #93 (K134, fixed). Fix: a bool-rejecting shape-entry newtype; maybe also for the read_size / block_shape params.

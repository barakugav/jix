# Known issues

Issues already reported, accepted, or deliberately left as-is. Hunters skip these; the dedup
stage drops findings that match one. For a FIXED entry, do not re-report the original trigger;
only report a new trigger the fix does not cover (a regression or an incomplete fix), and say so. Add an entry when you triage a finding (fixed, won't fix,
intended) so later runs do not report it again.

Format: one entry per issue.

```
## K<n>: <one-line title>
- Status: open | fixed in <commit/PR> | wont-fix | intended
- Location: <file:line>
- Notes: <what exactly is known; which triggers are covered>
```

## K1: Short zstd block decode only debug_asserted; release reads of a malformed .jix return uninit/stale destination bytes
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F001)
- Location: jix/src/storage/block.rs:193
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F001.md; repro in that run's repros/.

## K2: Pull-mode buffer size product(shape)*itemsize wraps after an itemsize-widening op (cast/Op2/Map skip check_shape_overflow)
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F002)
- Location: jix/src/storage/buf.rs:312
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F002.md; repro in that run's repros/.

## K3: Padded struct dtypes: NdCopier copies elements as [u32;2]/[u64;2], reading uninit padding
- Status: open (found in run 2026-10-21-pilot as F003)
- Location: jix/src/util/nd_copy.rs:31
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F003.md; repro in that run's repros/.

## K4: read_from_reader hands uninitialized buffers to the caller's safe Read impl (value_from_io, read_slice, Owned::read_section)
- Status: open (found in run 2026-10-21-pilot as F004)
- Location: jix/src/util/mod.rs:577
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F004.md; repro in that run's repros/.

## K5: mmap reader bounds sections by the caller's len, not the mapping: truncated file + explicit len segfaults
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F005)
- Location: jix/src/archive/block.rs:487
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F005.md; repro in that run's repros/.

## K6: ArrayParams::tune divides by zero / overflows on an invalid fixed block dim when another dim is auto-scaled
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F006)
- Location: jix/src/storage/params.rs:427
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F006.md; repro in that run's repros/.

## K7: mmap reader rejects archives whose block-locations section is not 8-aligned in the file
- Status: open (found in run 2026-10-21-pilot as F007)
- Location: jix/src/archive/block.rs:489
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F007.md; repro in that run's repros/.

## K8: i686 Linux: alignment hard-coded to 8 for f64/i64/u64/c128 while the ABI says 4
- Status: open (found in run 2026-10-21-pilot as F008)
- Location: jix-py/src/dtype.rs:80, jix/src/dtype.rs:967
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F008.md; repro in that run's repros/.

## K9: RuntimeAlign::new accepts align 0 (latent)
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F009)
- Location: jix/src/util/aligned_vec/mod.rs:28
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F009.md; repro in that run's repros/.

## K10: RawAlignedBytes::shrink_to reallocs to size 0, no null check, leaks at capacity 0 (latent)
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F010)
- Location: jix/src/util/aligned_vec/raw.rs:155
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F010.md; repro in that run's repros/.

## K11: ArrayVec::extend_from_iter double-drops ZST elements; reachable via Dimension::vec
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F011)
- Location: jix/src/util/arrayvec.rs:416
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F011.md; repro in that run's repros/.

## K12: ArrayParams::block_shape / block_shape_fixed_dims panic on more than 8 entries
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F012)
- Location: jix/src/storage/params.rs:121
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F012.md; repro in that run's repros/.

## K13: A leading 0 dim hides an overflowing shape product in try_product
- Status: fixed in 5b3a9f5 (#86) (found in run 2026-10-21-pilot as F013)
- Location: jix/src/util/mod.rs:145
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F013.md; repro in that run's repros/.

## K14: Safe-API UB: a writable pull result can be reused as `out` of a larger read; `out` extent is never checked, so Plain/Compact/Flip/Repeat/Tile write past it
- Status: open (found in run 2026-10-22-trial2 as F001, critical)
- Location: jix/src/storage/buf.rs:282
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F001.md; repro in that run's repros/.

## K15: numpy bool arrays with bytes other than 0/1 are wrapped zero-copy and read as Rust bool (UB); reductions, comparisons, bitwise ops and where silently return wrong results
- Status: open (found in run 2026-10-22-trial2 as F002, high)
- Location: jix-py/src/ops/common/operand.rs:183
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F002.md; repro in that run's repros/.

## K16: 32-bit: read regions over 4 GiB / 2^32 elements are sized with truncating `as usize` and wrapping usize products (pull allocation, to_ndarray_slice length check, to_ndarray shape, pipeline region), causing a heap overflow from safe code. This is K2's 32-bit part, which its fix did not cover.
- Status: open (found in run 2026-10-22-trial2 as F003, high)
- Location: jix/src/storage/buf.rs:312
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F003.md; repro in that run's repros/. Covers the 32-bit part of K2 that the #86 fix did not cover.

## K17: K1 fix is incomplete: Decoder::decode runs the filter chain over a short zstd decompression before read_block checks the length, so a failed read writes uninitialized or stale pool bytes into the caller's out buffer
- Status: open (found in run 2026-10-22-trial2 as F004, high)
- Location: jix/src/codec/mod.rs:315
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F004.md; repro in that run's repros/. K1 fix is incomplete: the filters run before the length check, so a failed read still writes stale bytes into out.

## K18: 32-bit: block byte size block_size * itemsize is never bounded to usize; every consumer wraps it the same way, so the size checks pass and block reads go out of bounds (wrong data, then SIGSEGV), from safe compact_with and from a Strict-valid archive
- Status: open (found in run 2026-10-22-trial2 as F005, high)
- Location: jix/src/storage/compact.rs:395
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F005.md; repro in that run's repros/.

## K19: MapMultiple::new skips check_shape_overflow with the output itemsize (incomplete fix of K2); a widening map_multiple pull read overflows the heap (SIGSEGV in release)
- Status: open (found in run 2026-10-22-trial2 as F006, high)
- Location: jix/src/ops/map.rs:237
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F006.md; repro in that run's repros/. K2 fix is incomplete for MapMultiple.

## K20: Big-endian numpy dtypes are silently read and written as little-endian (asarray/compact/ops/write_array, struct fields, and dtype='>f8' / cast)
- Status: open (found in run 2026-10-22-trial2 as F007, high)
- Location: jix-py/src/dtype.rs:101
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F007.md; repro in that run's repros/.

## K21: Python indexing rejects slice start == dim length, so a[n:], a[n:n] and any slice on a zero-length dim (even z[:]) raise IndexError
- Status: open (found in run 2026-10-22-trial2 as F008, medium)
- Location: jix-py/src/ops/shape_ops.rs:342
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F008.md; repro in that run's repros/.

## K22: Roll reduces shift with `rem_euclid(len as i64)`: wrong values for dims > i64::MAX, panic for len == u64::MAX with shift == i64::MIN
- Status: open (found in run 2026-10-22-trial2 as F009, medium)
- Location: jix/src/ops/shape_ops/roll.rs:54
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F009.md; repro in that run's repros/.

## K23: Concatenate::new sums concat-dim lengths unchecked: a sum past u64::MAX panics in debug and wraps to a tiny wrong shape (e.g. [4]) in release
- Status: open (found in run 2026-10-22-trial2 as F010, medium)
- Location: jix/src/ops/shape_ops/concatenate.rs:111
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F010.md; repro in that run's repros/.

## K24: An 8 KB Strict-valid archive that declares a 34 GB block aborts the process (Rust and Python) on its first read, with an allocation failure instead of an error
- Status: open (found in run 2026-10-22-trial2 as F011, medium)
- Location: jix/src/archive/array.rs:497
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F011.md; repro in that run's repros/.

## K25: Op2::shape assumes its children's shapes are still equal (assert_unchecked). A safe user ArrayStorage whose shape changes after construction makes safe code UB (SIGSEGV in release).
- Status: open (found in run 2026-10-22-trial2 as F012, medium)
- Location: jix/src/ops/op2.rs:182
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F012.md; repro in that run's repros/.

## K26: Python comparisons of uint64 with int64 run in float64 and return wrong booleans above 2**53 (narrower signed ints are unaffected)
- Status: open (found in run 2026-10-22-trial2 as F013, medium)
- Location: jix-py/src/ops/common/dtype_promote.rs:97
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F013.md; repro in that run's repros/.

## K27: Complex division uses num-complex's unscaled textbook formula: NaN/inf/0 or ~1e-5 relative error for finite operands with |divisor| beyond sqrt(MAX) or below sqrt(MIN_POSITIVE)
- Status: open (found in run 2026-10-22-trial2 as F014, medium)
- Location: jix/src/ops/op2.rs:781
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F014.md; repro in that run's repros/.

## K28: f64 -> f16 conversion double-rounds through f32 (Array::cast, Complex<f64>->Complex<f16>, jix.cast/asarray(dtype=f16), Python floats mixed with f16 arrays)
- Status: open (found in run 2026-10-22-trial2 as F015, medium)
- Location: jix/src/ops/cast.rs:59
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F015.md; repro in that run's repros/.

## K29: slice_unpack resolves slice bounds through Py_ssize_t: on 32-bit CPython a dim longer than 2^31-1 gives a silently wrong slice; on 64-bit, stops >= 2^63-1 bypass the strict bounds check
- Status: open (found in run 2026-10-22-trial2 as F016, medium)
- Location: jix-py/src/util.rs:31
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F016.md; repro in that run's repros/.

## K30: Contiguous inner-loop lane count ignores element width (prev_power_of_two(0)==1 keeps 8 lanes), so wide Dtyped elements overflow a 2 MiB thread stack and abort
- Status: open (found in run 2026-10-22-trial2 as F017, low)
- Location: jix/src/storage/elementwise_pipeline.rs:688
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F017.md; repro in that run's repros/.

## K31: Python numpy(out=x) never checks that x overlaps a buffer the array reads: &mut [u8] over memory a live &[u8] input covers (UB) and order-dependent wrong values, also without any explicit asarray
- Status: open (found in run 2026-10-22-trial2 as F018, medium)
- Location: jix/src/storage/elementwise_pipeline.rs:280
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F018.md; repro in that run's repros/.

## K32: permute_dims with more than 8 in-range entries panics (ArrayVec capacity, PanicException) instead of raising ValueError/RuntimeError
- Status: open (found in run 2026-10-22-trial2 as F019, low)
- Location: jix-py/src/util.rs:193
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F019.md; repro in that run's repros/.

## K33: Python bool index (a[True], a[False], jix.slice(a, True)) is silently taken as integer 1/0, while np.bool_ raises TypeError
- Status: open (found in run 2026-10-22-trial2 as F020, low)
- Location: jix-py/src/ops/shape_ops.rs:273
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F020.md; repro in that run's repros/.

## K34: jix.reshape with -1 raises on every zero-size array even when the inferred length is unambiguously 0 (flatten / explicit 0 work)
- Status: open (found in run 2026-10-22-trial2 as F021, low)
- Location: jix-py/src/ops/shape_ops.rs:768
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F021.md; repro in that run's repros/.

## K35: Dim lengths > i64::MAX are cast `as i64` in parse_basic_index, flatten and core DimSlice::resolve, so such dims cannot be indexed, x[-2**63] overflows, and flatten raises
- Status: open (found in run 2026-10-22-trial2 as F022, low)
- Location: jix-py/src/ops/shape_ops.rs:318
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F022.md; repro in that run's repros/.

## K36: Repeat computes out_strides[k] * repeats even when the output dim k has length 1 (stride never stepped), so it overflows: debug-build panic on valid out= buffers, release result correct
- Status: open (found in run 2026-10-22-trial2 as F023, low)
- Location: jix/src/ops/shape_ops/repeat.rs:181
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F023.md; repro in that run's repros/.

## K37: A numpy operand on the left of an operator with a jix.Array never reaches jix's reflected op: ndarray gives an object ndarray of whole jix arrays and comparisons raise; numpy scalar gives a different dtype
- Status: open (found in run 2026-10-22-trial2 as F024, low)
- Location: jix-py/src/array.rs:124
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F024.md; repro in that run's repros/.

## K38: 8-dim array indexed with 8 items plus Ellipsis is rejected ('9 were indexed') - NDIM_MAX tuple guard runs before Ellipsis is discounted
- Status: open (found in run 2026-10-22-trial2 as F025, low)
- Location: jix-py/src/ops/shape_ops.rs:250
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F025.md; repro in that run's repros/.

## K39: dtype_from_numpy debug_assert panics on packed numpy struct dtypes whose fields all have alignment 1 (incl. nested sub-structs); release only gets a cosmetic align=True descr
- Status: open (found in run 2026-10-22-trial2 as F026, low)
- Location: jix-py/src/dtype.rs:208
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F026.md; repro in that run's repros/.

## K40: Array.numpy() docstring promises a C-contiguous result, but the allocation follows read_layout_order (F-order for Fortran asarray / permuted inputs)
- Status: open (found in run 2026-10-22-trial2 as F027, low)
- Location: jix-py/src/array.rs:491
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F027.md; repro in that run's repros/.

## K41: Multi-cell fold feeds ArgMax/ArgMin update out of stream order (permuted reduced dims, or C-order with reduced dims split across tiles): multi-dim argmax returns a later tie/NaN (latent, crate-internal)
- Status: open (found in run 2026-10-22-trial2 as F028, low)
- Location: jix/src/ops/reduction.rs:1109
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F028.md; repro in that run's repros/.

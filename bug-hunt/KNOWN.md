# Known issues

Issues already reported, accepted, or deliberately left as-is. Hunters skip these; the dedup
stage drops findings that match one. For a FIXED entry, do not re-report the original trigger;
only report a new trigger the fix does not cover (a regression or an incomplete fix), and say so.
Add an entry when you triage a finding (fixed, won't fix, intended) so later runs do not report
it again.

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
- Status: fixed in 6f8e220 (#88) (found in run 2026-10-21-pilot as F003)
- Location: jix/src/util/nd_copy.rs:31
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F003.md; repro in that run's repros/.

## K4: read_from_reader hands uninitialized buffers to the caller's safe Read impl (value_from_io, read_slice, Owned::read_section)
- Status: fixed in c009869 (#87) (found in run 2026-10-21-pilot as F004)
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
- Status: fixed in c009869 (#87) (found in run 2026-10-21-pilot as F007)
- Location: jix/src/archive/block.rs:489
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F007.md; repro in that run's repros/.

## K8: i686 Linux: alignment hard-coded to 8 for f64/i64/u64/c128 while the ABI says 4
- Status: open, tracked in todo.md: i686 is a known gap (found in run 2026-10-21-pilot as F008)
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
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F001, critical)
- Location: jix/src/storage/buf.rs:282
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F001.md; repro in that run's repros/.

## K15: numpy bool arrays with bytes other than 0/1 are wrapped zero-copy and read as Rust bool (UB); reductions, comparisons, bitwise ops and where silently return wrong results
- Status: fixed in 7e93487 (#89) (found in run 2026-10-22-trial2 as F002, high)
- Location: jix-py/src/ops/common/operand.rs:183
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F002.md; repro in that run's repros/.

## K16: 32-bit: read regions over 4 GiB / 2^32 elements are sized with truncating `as usize` and wrapping usize products (pull allocation, to_ndarray_slice length check, to_ndarray shape, pipeline region), causing a heap overflow from safe code. This is K2's 32-bit part, which its fix did not cover.
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F003, high)
- Location: jix/src/storage/buf.rs:312
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F003.md; repro in that run's repros/. Covers the 32-bit part of K2 that the #86 fix did not cover. See K81 (run 2026-10-24-batch4 F016: empty regions with huge dims are not covered).

## K17: K1 fix is incomplete: Decoder::decode runs the filter chain over a short zstd decompression before read_block checks the length, so a failed read writes uninitialized or stale pool bytes into the caller's out buffer
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F004, high)
- Location: jix/src/codec/mod.rs:315
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F004.md; repro in that run's repros/. K1 fix is incomplete: the filters run before the length check, so a failed read still writes stale bytes into out.

## K18: 32-bit: block byte size block_size * itemsize is never bounded to usize; every consumer wraps it the same way, so the size checks pass and block reads go out of bounds (wrong data, then SIGSEGV), from safe compact_with and from a Strict-valid archive
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F005, high)
- Location: jix/src/storage/compact.rs:395
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F005.md; repro in that run's repros/.

## K19: MapMultiple::new skips check_shape_overflow with the output itemsize (incomplete fix of K2); a widening map_multiple pull read overflows the heap (SIGSEGV in release)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F006, high)
- Location: jix/src/ops/map.rs:237
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F006.md; repro in that run's repros/. K2 fix is incomplete for MapMultiple.

## K20: Big-endian numpy dtypes are silently read and written as little-endian (asarray/compact/ops/write_array, struct fields, and dtype='>f8' / cast)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F007, high)
- Location: jix-py/src/dtype.rs:101
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F007.md; repro in that run's repros/.

## K21: Python indexing rejects slice start == dim length, so a[n:], a[n:n] and any slice on a zero-length dim (even z[:]) raise IndexError
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F008, medium)
- Location: jix-py/src/ops/shape_ops.rs:342
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F008.md; repro in that run's repros/.

## K22: Roll reduces shift with `rem_euclid(len as i64)`: wrong values for dims > i64::MAX, panic for len == u64::MAX with shift == i64::MIN
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F009, medium)
- Location: jix/src/ops/shape_ops/roll.rs:54
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F009.md; repro in that run's repros/.

## K23: Concatenate::new sums concat-dim lengths unchecked: a sum past u64::MAX panics in debug and wraps to a tiny wrong shape (e.g. [4]) in release
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F010, medium)
- Location: jix/src/ops/shape_ops/concatenate.rs:111
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F010.md; repro in that run's repros/.

## K24: An 8 KB Strict-valid archive that declares a 34 GB block aborts the process (Rust and Python) on its first read, with an allocation failure instead of an error
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F011, medium)
- Location: jix/src/archive/array.rs:497
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F011.md; repro in that run's repros/.

## K25: Op2::shape assumes its children's shapes are still equal (assert_unchecked). A safe user ArrayStorage whose shape changes after construction makes safe code UB (SIGSEGV in release).
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F012, medium)
- Location: jix/src/ops/op2.rs:182
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F012.md; repro in that run's repros/.

## K26: Python comparisons of uint64 with int64 run in float64 and return wrong booleans above 2**53 (narrower signed ints are unaffected)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F013, medium)
- Location: jix-py/src/ops/common/dtype_promote.rs:97
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F013.md; repro in that run's repros/.

## K27: Complex division uses num-complex's unscaled textbook formula: NaN/inf/0 or ~1e-5 relative error for finite operands with |divisor| beyond sqrt(MAX) or below sqrt(MIN_POSITIVE)
- Status: intended - textbook division is accepted, like the num-complex one (found in run 2026-10-22-trial2 as F014, medium)
- Location: jix/src/ops/op2.rs:781
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F014.md; repro in that run's repros/. See K74 (run 2026-10-24-batch4 F009: same formula in the complex mean).

## K28: f64 -> f16 conversion double-rounds through f32 (Array::cast, Complex<f64>->Complex<f16>, jix.cast/asarray(dtype=f16), Python floats mixed with f16 arrays)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F015, medium)
- Location: jix/src/ops/cast.rs:59
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F015.md; repro in that run's repros/. See K71 (run 2026-10-24-batch4 F006: the fix is incomplete, f16::from_f64 still mis-rounds without std).

## K29: slice_unpack resolves slice bounds through Py_ssize_t: on 32-bit CPython a dim longer than 2^31-1 gives a silently wrong slice; on 64-bit, stops >= 2^63-1 bypass the strict bounds check
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F016, medium)
- Location: jix-py/src/util.rs:31
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F016.md; repro in that run's repros/.

## K30: Contiguous inner-loop lane count ignores element width (prev_power_of_two(0)==1 keeps 8 lanes), so wide Dtyped elements overflow a 2 MiB thread stack and abort
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F017, low)
- Location: jix/src/storage/elementwise_pipeline.rs:688
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F017.md; repro in that run's repros/. See K72 (run 2026-10-24-batch4 F007: same defect in the reduction fold).

## K31: Python numpy(out=x) never checks that x overlaps a buffer the array reads: &mut [u8] over memory a live &[u8] input covers (UB) and order-dependent wrong values, also without any explicit asarray
- Status: intended - documented in c009869 (#87): out= must not overlap any numpy array in the lazy chain (found in run 2026-10-22-trial2 as F018, medium)
- Location: jix/src/storage/elementwise_pipeline.rs:280
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F018.md; repro in that run's repros/.

## K32: permute_dims with more than 8 in-range entries panics (ArrayVec capacity, PanicException) instead of raising ValueError/RuntimeError
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F019, low)
- Location: jix-py/src/util.rs:193
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F019.md; repro in that run's repros/.

## K33: Python bool index (a[True], a[False], jix.slice(a, True)) is silently taken as integer 1/0, while np.bool_ raises TypeError
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F020, low)
- Location: jix-py/src/ops/shape_ops.rs:273
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F020.md; repro in that run's repros/.

## K34: jix.reshape with -1 raises on every zero-size array even when the inferred length is unambiguously 0 (flatten / explicit 0 work)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F021, low)
- Location: jix-py/src/ops/shape_ops.rs:768
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F021.md; repro in that run's repros/.

## K35: Dim lengths > i64::MAX are cast `as i64` in parse_basic_index, flatten and core DimSlice::resolve, so such dims cannot be indexed, x[-2**63] overflows, and flatten raises
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F022, low)
- Location: jix-py/src/ops/shape_ops.rs:318
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F022.md; repro in that run's repros/.

## K36: Repeat computes out_strides[k] * repeats even when the output dim k has length 1 (stride never stepped), so it overflows: debug-build panic on valid out= buffers, release result correct
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F023, low)
- Location: jix/src/ops/shape_ops/repeat.rs:181
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F023.md; repro in that run's repros/. See K78 (run 2026-10-24-batch4 F013: same defect in Slice's plain fast path).

## K37: A numpy operand on the left of an operator with a jix.Array never reaches jix's reflected op: ndarray gives an object ndarray of whole jix arrays and comparisons raise; numpy scalar gives a different dtype
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F024, low)
- Location: jix-py/src/array.rs:124
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F024.md; repro in that run's repros/.

## K38: 8-dim array indexed with 8 items plus Ellipsis is rejected ('9 were indexed') - NDIM_MAX tuple guard runs before Ellipsis is discounted
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F025, low)
- Location: jix-py/src/ops/shape_ops.rs:250
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F025.md; repro in that run's repros/.

## K39: dtype_from_numpy debug_assert panics on packed numpy struct dtypes whose fields all have alignment 1 (incl. nested sub-structs); release only gets a cosmetic align=True descr
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F026, low)
- Location: jix-py/src/dtype.rs:208
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F026.md; repro in that run's repros/.

## K40: Array.numpy() docstring promises a C-contiguous result, but the allocation follows read_layout_order (F-order for Fortran asarray / permuted inputs)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F027, low)
- Location: jix-py/src/array.rs:491
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F027.md; repro in that run's repros/. See K82 (run 2026-10-24-batch4 F017: same doc promise on the Rust to_ndarray).

## K41: Multi-cell fold feeds ArgMax/ArgMin update out of stream order (permuted reduced dims, or C-order with reduced dims split across tiles): multi-dim argmax returns a later tie/NaN (latent, crate-internal)
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F028, low)
- Location: jix/src/ops/reduction.rs:1109
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F028.md; repro in that run's repros/.

## K42: Struct layout validation sums field ends in unchecked u16; a field list past 65535 wraps and validates against a tiny itemsize, so SubDtype reads up to 64 KiB OOB (crafted .jix -> SIGSEGV in release; also reachable from 100% safe Rust via from_fields + a safe user ArrayStorage)
- Status: fixed in 6f8e220 (#88) (found in run 2026-10-23-batch3 as F001, high)
- Location: jix/src/dtype.rs:673
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F001.md; repro in that run's repros/. See K83 (run 2026-10-24-batch4 F018: one unchecked u16 add left in Dtype::from_fields).

## K43: Archive read keeps the caller's block_shape_fixed_dims, so tune rescales the file's block shape and blocks are decoded with the wrong geometry (silently wrong values, panic, or every read failing; write_to then persists a corrupt Strict-valid file)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F002, high)
- Location: jix/src/archive/array.rs:537
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F002.md; repro in that run's repros/.

## K44: A Strict-valid .jix array of bool (or a struct with a bool field) whose bytes are not 0/1 decodes into typed Rust bool: UB from safe code (Miri-confirmed), and where/cast/not/sum/any give inconsistent results
- Status: fixed in 7e93487 (#89) (found in run 2026-10-23-batch3 as F003, high)
- Location: jix/src/storage/block.rs:201
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F003.md; repro in that run's repros/. Archive-decode trigger of the bool-validity problem; K15 covers the numpy trigger.

## K45: Safe ndarray Plain/compact constructors accept negative strides (reversed views): `s as usize * size_of::<T>()` wraps (plain.rs:289); debug panics, release returns a wrong window when the wrapped span is 0 and violates from_raw_parts/get_unchecked preconditions; the owned variant panics on unwrap (plain.rs:248)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F004, high)
- Location: jix/src/storage/plain.rs:289
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F004.md; repro in that run's repros/. See K80 (run 2026-10-24-batch4 F015: the fix also rejects negative strides on length-0/1 dims).

## K46: check_section_bounds adds section offset + size unchecked: a crafted TOC panics (debug: add overflow; release: capacity overflow in Owned::read_section) instead of returning InvalidArchive, also from default Python read_array
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F005, medium)
- Location: jix/src/archive/common.rs:264
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F005.md; repro in that run's repros/.

## K47: Archive message length (uleb128) is allocated and zero-filled before it is checked against the archive length, so an 11-byte file aborts the process (Rust and Python)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F006, medium)
- Location: jix/src/archive/common.rs:210
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F006.md; repro in that run's repros/.

## K48: With len=None (the Python default for file-like readers), Owned::read_section allocates and zero-fills the untrusted TOC section size before reading it: a 106-byte BytesIO aborts the interpreter, and a 1 GiB size commits 1 GiB first
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F007, medium)
- Location: jix/src/archive/block.rs:433
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F007.md; repro in that run's repros/.

## K49: Writing an archive into an append-mode (O_APPEND) file or file object returns Ok but leaves the TOC as zeros, so the archive cannot be read back
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F008, medium)
- Location: jix/src/archive/block.rs:231
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F008.md; repro in that run's repros/.

## K50: ArrayStorage is a safe, unsealed trait, but jix's unsafe code trusts its read_data buffers: a user storage with no unsafe causes OOB reads (default elementwise pipeline, pull) and uninitialized reads (to_ndarray_sub assume_init, push)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F009, medium)
- Location: jix/src/storage/core_trait.rs:134
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F009.md; repro in that run's repros/. ArrayStorage is now sealed (ArrayStorageImpl marker), so user storages can no longer reach the unsafe code. See K70 (run 2026-10-24-batch4 F005: ElementType is still unsealed).

## K51: Float base ** integer exponent uses powi: error grows with |n| (tens to thousands of ulps) and representable subnormal results are flushed to 0, silently
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F010, medium)
- Location: jix/src/ops/op2.rs:1201
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F010.md; repro in that run's repros/.

## K52: Dtype::new_struct, and so the archive reader, accepts duplicate struct field names that from_fields rejects. The second field cannot be selected, and in Python the dtype is unusable.
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F011, low)
- Location: jix/src/dtype.rs:606
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F011.md; repro in that run's repros/.

## K53: Derived structs with a [T; 0] field, or a zero-sized array field whose shape product exceeds u16, write to .jix but cannot be read back; Dtype::set_shape panics (debug) on such dtypes
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F012, low)
- Location: jix/src/archive/schema/conversions.rs:125
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F012.md; repro in that run's repros/.

## K54: NdCopier::copy trusts that its `dtype` matches the one given to `new`; scalar routes only debug_assert it, so a wider copier with spans sized from copy()'s dtype reads and writes past both spans in release (latent, crate-internal)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F013, low)
- Location: jix/src/util/nd_copy.rs:89
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F013.md; repro in that run's repros/. Fixed by documenting the contract on NdCopier::copy (crate-internal).

## K55: read_uleb128 has no length cap: an 11-byte length prefix panics in debug (shift overflow) and is silently accepted in release
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F014, low)
- Location: jix/src/archive/common.rs:184
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F014.md; repro in that run's repros/.

## K56: 32-bit: section sizes and uleb128 message lengths truncated with `as usize`; Owned reads panic (assert_eq at common.rs:240 / capacity overflow) and a malformed archive loads silently, where 64-bit returns Err
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F015, low)
- Location: jix/src/archive/block.rs:430
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F015.md; repro in that run's repros/.

## K57: The write_array docstring names gzip.GzipFile as a supported writer, but the archive writer seeks backwards in finalize, so every write_array to a GzipFile raises "Negative seek in write mode"
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F016, low)
- Location: jix-py/src/archive.rs:162
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F016.md; repro in that run's repros/. Fixed in the docstring (GzipFile no longer listed).

## K58: Writable pull results with aliasing 0 strides (Broadcast pull, elementwise-pipeline pull) break the from_pool/with_strides no-aliasing contract and are accepted back as `out` by check_out_buf, so push reads return Ok with the region not in out
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F017, low)
- Location: jix/src/ops/shape_ops/broadcast.rs:156
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F017.md; repro in that run's repros/. New trigger next to K14 (fixed): 0-stride writable pull results are still accepted as out.

## K59: StridedBuf::from_raw_parts(_mut) Safety sections omit non-null (even for an empty region) and exclusive access / no overlap with any input: calls meeting the documented contract are UB (debug abort, Miri SB error)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F018, low)
- Location: jix/src/storage/buf.rs:84
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F018.md; repro in that run's repros/. Fixed by documenting the full safety contract.

## K60: jix.power(x, 2) / x ** 2 is routed to Square and silently drops the float Overflow report that the jix.power docstring promises (f16/f32/f64)
- Status: intended - x ** 2 on floats is a plain square, without the Pow overflow report (found in run 2026-10-23-batch3 as F019, low)
- Location: jix-py/src/ops/op2.rs:688
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F019.md; repro in that run's repros/.

## K61: 32-bit: check_out_buf runs check_region_size (`r.end - r.start`) before the index is validated; a reversed range passed to read_data on Op1/Op2/MapMultiple/Clamp/InsertDim/RemoveDim/PermuteDims/Where/SubDtype panics (debug) or returns InvalidShapeOperation instead of InvalidIndex (release)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F020, low)
- Location: jix/src/error.rs:278
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F020.md; repro in that run's repros/. Regression from #87: check_region_size runs before index validation (32-bit).

## K62: Empty float/complex sum returns -0.0 (sign bit set) instead of +0.0 (f16/f32/f64; complex gives (-0.0, +0.0))
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F021, low)
- Location: jix/src/ops/reduction.rs:1483
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F021.md; repro in that run's repros/.

## K63: Owned::read_section transmutes Vec<MaybeUninit<T>> to Vec<T>, relying on Vec layout that std does not guarantee (latent; no current rustc config, -Zrandomize-layout included, breaks it)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F022, low)
- Location: jix/src/archive/block.rs:439
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F022.md; repro in that run's repros/.

## K64: Array::to_ndarray_buf pull mode copies the whole region for large reads of lazy views (broadcast, permute/slice over Plain, collapsed pipelines) that read_data(None) returns in O(1); a huge broadcast aborts
- Status: open, acknowledged in todo.md (#90): "pull mode to_ndarray_buf splits large reads of O(1) lazy views and copies the whole region" (found in run 2026-10-23-batch3 as F024, low)
- Location: jix/src/array.rs:1034
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F024.md; repro in that run's repros/.

## K65: power with a u64 exponent promotes to float64 (integer exponent slots stop at u32)
- Status: intended (run 2026-10-23-batch3 F023; pinned by test_power_promotes_uint64_exponent_to_float64)
- Location: jix/src/ops/op2.rs:1173
- Notes: doc nit - the #81 docstring rewrite dropped the explicit u64-exponent sentence.

## K66: Array.numpy() gives wrong values, including uninitialized bytes, for dtypes with an inner shape: numpy expands the sub-array descr into trailing dims and their strides are taken as the array's strides (permuted layouts raise, numpy(out=) never works)
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F001, high)
- Location: jix-py/src/array.rs:570
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F001.md; repro in that run's repros/.

## K67: Welford variance goes negative/NaN when |x - mean| overflows f64 (var([1e308, -1e308]) = -inf, numpy inf); std fails a debug_assert (PanicException in the dev extension) and returns NaN silently in release
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F002, medium)
- Location: jix/src/ops/reduction.rs:1874
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F002.md; repro in that run's repros/. The fix maps a NaN m2 to +inf too, so NaN / inf inputs now give inf: see K84.

## K68: Rust range -> SliceItem: inclusive end -1 (`..=-1`, `0..=-1`) silently selects nothing and `-k..=-1` errors (Array::slice panics); u64/usize bounds >= 2^63 wrap to end-relative indices; `..=i64::MAX` overflows
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F003, medium)
- Location: jix/src/ops/shape_ops/slice.rs:324
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F003.md; repro in that run's repros/.

## K69: Reshape reads every split/merge reshape (flatten, [N]->[a,b], block-aligned splits) one element at a time, decompressing a whole block per element: 400-5000x slower reads
- Status: open, tracked in todo.md as a performance item, not a bug (found in run 2026-10-24-batch4 as F004, medium)
- Location: jix/src/ops/shape_ops/reshape.rs:296
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F004.md; repro in that run's repros/.

## K70: ElementType is a safe, unsealed trait, but Plain/BlockTable/SubDtype/Transmute take their dtype from a user ET::dtype() without checking it: heap OOB read (Plain::read_data, SubDtype) and assert_unchecked(false) (BlockTable::decoder_config via write_to) from safe code
- Status: fixed in 691e11b (#91): ElementType is sealed (found in run 2026-10-24-batch4 as F005, medium)
- Location: jix/src/element_type.rs:16
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F005.md; repro in that run's repros/. Same kind of bug as K50 (ArrayStorage, sealed in #90); ElementType was left unsealed. The repro no longer compiles (sealed trait), which is the fix.

## K71: f64 -> f16 via half::f16::from_f64 (no_std software fallback) still mis-rounds on every shipped target except macOS arm64, about 8x as often as the pre-K28-fix double rounding through f32
- Status: open (found in run 2026-10-24-batch4 as F006, medium)
- Location: jix/src/ops/cast.rs:55
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F006.md; repro in that run's repros/. Incomplete K28 fix. The in-tree test is gated to aarch64 and the property oracle calls the same function.

## K72: reduce_unordered over wide Dtyped elements overflows a 2 MiB thread stack and aborts: the one-cell reduction fold keeps 16 (strided) / 8 (contiguous) lanes of T whatever the element width
- Status: fixed in release builds in 691e11b (#91); debug builds still abort (found in run 2026-10-24-batch4 as F007, low)
- Location: jix/src/ops/reduction.rs:768
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F007.md; repro in that run's repros/. The K30 defect in the reduction fold, which the K30 fix did not touch. After #91 the repro passes with --release (652 KiB for both one-cell cases, the multi-cell control 396 KiB), but debug builds now need 7.8 MiB (contiguous) / 8.9 MiB (strided), up from 2.1-5.7 MiB: `dispatch_lanes!` in fold_one_cell_per_simd (reduction.rs:852-860) matches on the const lane count and all ten `#[inline(always)]` arms (1..512 lanes) are probably inlined into one frame, which debug builds do not prune.

## K73: ReductionOp::new adds 4 to the reduced element count unchecked: reducing 2+ dims of a lazy 1-byte-dtype shape with more than 2^64 - 5 reduced elements panics in debug builds (release only wraps a cost heuristic)
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F008, low)
- Location: jix/src/ops/reduction.rs:177
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F008.md; repro in that run's repros/.

## K74: Complex mean divides the sum by Complex(n, 0) with num-complex's textbook division, so sum * n overflows: mean([1.7e308+0j, 0j]) = inf+0j, mean of 1e6 copies of 1e302+1e302j = inf+infj (numpy and the f64 mean are finite)
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F009, low)
- Location: jix/src/ops/reduction.rs:1766
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F009.md; repro in that run's repros/. The K27 formula at a separate call site; a K27 fix in op2.rs does not cover it.

## K75: clamp(bool, min, max) returns uint8 while one-bound clamp, maximum and minimum return bool: bool8 is last in the two-bound clamp dispatch table
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F010, low)
- Location: jix-py/src/ops/cmp.rs:447
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F010.md; repro in that run's repros/.

## K76: bitwise_rotate_left/right with a Python int value: all-Python-int calls raise TypeError (the shifts work), and a Python int value next to a typed u32 count is rotated as int8 (-128 for 64, OverflowError for 200)
- Status: open, tracked in todo.md (#91): "bitwise_rotate_left/right type promotion is wrong, (int,int) raise" (found in run 2026-10-24-batch4 as F011, low)
- Location: jix-py/src/ops/bitwise.rs:432
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F011.md; repro in that run's repros/.

## K77: A stepped Slice over a non-plain inner makes one inner read per output position of the stepped dims, decompressing whole blocks each time: step-2 slices read 200-2100x slower than the whole array
- Status: open, tracked in todo.md as a performance item, not a bug (found in run 2026-10-24-batch4 as F012, medium)
- Location: jix/src/ops/shape_ops/slice.rs:181
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F012.md; repro in that run's repros/. Same read-amplification pattern as K69 (F004).

## K78: Slice plain pull fast path computes inner_stride * step unchecked: a step larger than the dim (output length 1) overflows - debug panic, release wraps a stride that is never stepped
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F013, low)
- Location: jix/src/ops/shape_ops/slice.rs:161
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F013.md; repro in that run's repros/. The K36 defect at a site the K36 fix did not cover; on 32-bit a step of ~1e6 suffices.

## K79: SliceSpec::new / From<&[I]> panic (ArrayVec capacity) on more than 8 items, before the fallible Slice::new can return InvalidIndex
- Status: fixed in 691e11b (#91): SliceSpec::new returns Result, From<&[I]> became TryFrom (found in run 2026-10-24-batch4 as F014, low)
- Location: jix/src/ops/shape_ops/slice.rs:252
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F014.md; repro in that run's repros/. Same kind of bug as K12 / K32. The repro no longer compiles (API change), which is the fix; covered by the in-tree test slice_spec_too_many_items_errors.

## K80: A negative stride on a never-stepped (length 0/1) dim is rejected on input (Rust ndarray constructors, Python asarray/compact/operands), although numpy reports the array C-contiguous and numpy(out=) accepts the same layout; the same line multiplies a huge length-1 stride unchecked (debug panic)
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F015, low)
- Location: jix/src/storage/plain.rs:520
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F015.md; repro in that run's repros/. Gap in the K45 fix.

## K81: to_ndarray_sub sizes its ndarray with a per-dim `as usize` and bounds the region only on 32-bit, via the element product (0 for an empty region): empty regions with huge dims get a truncated shape on i686 (Ok [0, 5] for [0, 2^32+5]) and panic inside ndarray on 64-bit
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F016, low)
- Location: jix/src/array.rs:774
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F016.md; repro in that run's repros/. Gap in the K16 fix (empty regions).

## K82: Array::to_ndarray docs promise a contiguous row-major (C-order) result, but the result follows read_layout_order (F-order for permuted views and F-order Plain sources)
- Status: fixed in 691e11b (#91) (docs no longer promise C order) (found in run 2026-10-24-batch4 as F017, low)
- Location: jix/src/array.rs:577
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F017.md; repro in that run's repros/. Rust counterpart of K40; docs-only fix. Delete the repro: it asserts C order, which is no longer promised.

## K83: Dtype::from_fields computes the struct itemsize with an unchecked u16 add before check_fields, so debug builds panic instead of returning Err on oversized or malformed field lists
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F018, low)
- Location: jix/src/dtype.rs:491
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F018.md; repro in that run's repros/. An unchecked add the K42 fix left behind; release returns the correct Err.

## K84: var/std of data containing NaN or inf returns +inf instead of NaN (regression from the K67 fix)
- Status: open (found while rebasing onto 691e11b (#91))
- Location: jix/src/ops/reduction.rs:1933
- Notes: The #91 variance finalize maps every `m2` that is not `>= 0.0` to +inf, to turn an overflowed (negative) m2 into inf. A NaN m2 also fails `>= 0.0`, so `var([nan, 1.0])`, `std([1.0, nan, 2.0])` and `var([inf, 1.0])` return inf; numpy, and jix before #91, give nan. Same through Python (`jix.var`, `jix.std`) and for complex inputs. The sign of m2 alone cannot tell overflow from a NaN / inf input (F002 also had finite inputs giving a NaN m2), so the fix needs another signal, e.g. a non-finite-input flag in the state or a finite mean.

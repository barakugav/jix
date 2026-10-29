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
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F002.md; repro in that run's repros/. See K88 (run 2026-10-27-batch5 F004: jix-py still calls the unwrapping Array::cast).

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
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F013.md; repro in that run's repros/. See K101 (run 2026-10-27-batch5 F017: the reader's zeros-as-1 padded-shape check rejects empty arrays the writer accepts).

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
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F005.md; repro in that run's repros/. See K105 (run 2026-10-27-batch5 F021: the 1 GiB block cap rejects block shapes a widening view inherits).

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
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F018.md; repro in that run's repros/. See K111 (run 2026-10-29-batch6 F003: element-disjoint views whose spans interleave also alias).

## K32: permute_dims with more than 8 in-range entries panics (ArrayVec capacity, PanicException) instead of raising ValueError/RuntimeError
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F019, low)
- Location: jix-py/src/util.rs:193
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F019.md; repro in that run's repros/.

## K33: Python bool index (a[True], a[False], jix.slice(a, True)) is silently taken as integer 1/0, while np.bool_ raises TypeError
- Status: fixed in c009869 (#87) (found in run 2026-10-22-trial2 as F020, low)
- Location: jix-py/src/ops/shape_ops.rs:273
- Notes: bug-hunt/runs/2026-10-22-trial2/findings/F020.md; repro in that run's repros/. See K134 (run 2026-10-29-batch6 F026: a Python bool is still accepted as a dim).

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
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F010.md; repro in that run's repros/. See K144 (run 2026-10-29-batch6 F036: regression, odd exponents above 2^24 lose their sign).

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
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F017.md; repro in that run's repros/. New trigger next to K14 (fixed): 0-stride writable pull results are still accepted as out. See K103 (run 2026-10-27-batch5 F019: check_out_buf still has no aliasing check).

## K59: StridedBuf::from_raw_parts(_mut) Safety sections omit non-null (even for an empty region) and exclusive access / no overlap with any input: calls meeting the documented contract are UB (debug abort, Miri SB error)
- Status: fixed in 4d11918 (#90) (found in run 2026-10-23-batch3 as F018, low)
- Location: jix/src/storage/buf.rs:84
- Notes: bug-hunt/runs/2026-10-23-batch3/findings/F018.md; repro in that run's repros/. Fixed by documenting the full safety contract. See K140 (run 2026-10-29-batch6 F032: the Plain::new / plain_ndarray_ptr Safety sections).

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
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F001.md; repro in that run's repros/. See K89 (run 2026-10-27-batch5 F005 and F015 (K99): the new numpy() shapes overflow 8-entry buffers, and plain() changed shape and dtype).

## K67: Welford variance goes negative/NaN when |x - mean| overflows f64 (var([1e308, -1e308]) = -inf, numpy inf); std fails a debug_assert (PanicException in the dev extension) and returns NaN silently in release
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F002, medium)
- Location: jix/src/ops/reduction.rs:1874
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F002.md; repro in that run's repros/. The fix maps a NaN m2 to +inf too, so NaN / inf inputs now give inf: see K84. Partly regressed by the K84 fix in #92: see K108.

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
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F015.md; repro in that run's repros/. Gap in the K45 fix. See K141 (run 2026-10-29-batch6 F033: empty arrays with a negative stride on a longer dim are still rejected).

## K81: to_ndarray_sub sizes its ndarray with a per-dim `as usize` and bounds the region only on 32-bit, via the element product (0 for an empty region): empty regions with huge dims get a truncated shape on i686 (Ok [0, 5] for [0, 2^32+5]) and panic inside ndarray on 64-bit
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F016, low)
- Location: jix/src/array.rs:774
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F016.md; repro in that run's repros/. Gap in the K16 fix (empty regions). See K120 (run 2026-10-29-batch6 F012: the shared 32-bit region helper still skips empty regions).

## K82: Array::to_ndarray docs promise a contiguous row-major (C-order) result, but the result follows read_layout_order (F-order for permuted views and F-order Plain sources)
- Status: fixed in 691e11b (#91) (docs no longer promise C order) (found in run 2026-10-24-batch4 as F017, low)
- Location: jix/src/array.rs:577
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F017.md; repro in that run's repros/. Rust counterpart of K40; docs-only fix. Delete the repro: it asserts C order, which is no longer promised.

## K83: Dtype::from_fields computes the struct itemsize with an unchecked u16 add before check_fields, so debug builds panic instead of returning Err on oversized or malformed field lists
- Status: fixed in 691e11b (#91) (found in run 2026-10-24-batch4 as F018, low)
- Location: jix/src/dtype.rs:491
- Notes: bug-hunt/runs/2026-10-24-batch4/findings/F018.md; repro in that run's repros/. An unchecked add the K42 fix left behind; release returns the correct Err.

## K84: var/std of data containing NaN or inf returns +inf instead of NaN (regression from the K67 fix)
- Status: fixed in df2f918 (#92) (found while rebasing onto 691e11b (#91))
- Location: jix/src/ops/reduction.rs:1933
- Notes: The #91 variance finalize maps every `m2` that is not `>= 0.0` to +inf, to turn an overflowed (negative) m2 into inf. A NaN m2 also fails `>= 0.0`, so `var([nan, 1.0])`, `std([1.0, nan, 2.0])` and `var([inf, 1.0])` return inf; numpy, and jix before #91, give nan. Same through Python (`jix.var`, `jix.std`) and for complex inputs. The sign of m2 alone cannot tell overflow from a NaN / inf input (F002 also had finite inputs giving a NaN m2), so the fix needs another signal, e.g. a non-finite-input flag in the state or a finite mean.

## K85: 32-bit: the reduction's temp state buffer size `out_nitems * size_of::<State>()` wraps, so var/std states are written far past a few-byte pool buffer (heap overflow, SIGSEGV from safe read_data / compact_with)
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F001, high)
- Location: jix/src/ops/reduction.rs:516
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F001.md; repro in that run's repros/. check_region_size bounds the region by the output itemsize only. Reproduced on i686: debug panics at the multiply, release SIGSEGV.

## K86: 32-bit: the reduction driver computes reduced-stream positions in usize; a reduction over 2^32+ items silently drops everything before the last tile at position k*2^32 (all kernels), and argmax indices are truncated mod 2^32
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F002, medium)
- Location: jix/src/ops/reduction.rs:588
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F002.md; repro in that run's repros/.

## K87: MapMultiple::new panics (index out of bounds in combine_hints) on an empty array sequence instead of returning Err like Stack::new / Concatenate::new
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F003, low)
- Location: jix/src/ops/map.rs:256
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F003.md; repro in that run's repros/. Same kind of bug as K79.

## K88: Widening cast of a huge lazy array raises PanicException instead of RuntimeError: jix-py cast_impl calls the panicking convenience Array::cast
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F004, low)
- Location: jix-py/src/ops/mod.rs:116
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F004.md; repro in that run's repros/. Since the K2 fix made Cast::new fallible.

## K89: Array.numpy() / __getitem__ / plain() panic (ArrayVec capacity) when the array's ndim plus its dtype's inner ndim exceeds 8 (item() is not affected)
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F005, medium)
- Location: jix-py/src/util.rs:103
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F005.md; repro in that run's repros/. Regression from the K66 fix (#91): numpy shapes now have up to NDIM_MAX + DTYPE_MAX_NDIM entries.

## K90: Array.numpy_dtype holds the per-Array std::sync::Mutex while dtype_to_numpy runs Python code (py.import -> builtins.__import__); if that code yields the GIL to a thread using the same Array, or uses the same Array itself, the process hangs forever
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F006, medium)
- Location: jix-py/src/array.rs:301
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F006.md; repro in that run's repros/. GIL builds need an __import__ hook and an aligned struct dtype; the free-threaded deadlock is argued from the code only.

## K91: Broadcast reads and evaluates its inner row for an EMPTY request (broadcast dim 0..0 becomes inner 0..1), so zero-element reads raise spurious arithmetic errors, do O(row) work, or abort
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F007, medium)
- Location: jix/src/ops/shape_ops/broadcast.rs:137
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F007.md; repro in that run's repros/. From Python through concatenate with a zero-length broadcast member.

## K92: jix.log(x, base), jix.log2 and jix.log10 round three times: exact powers come out 1 ulp low (log10(1000) = 2.9999999999999996, f16 log2(32) = 4.996), up to 2.7 ulp error in f64
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F008, medium)
- Location: jix-py/src/ops/op1.rs:362
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F008.md; repro in that run's repros/. See K127 (run 2026-10-29-batch6 F019: the f64 path still multiplies by a rounded 1/ln(base)).

## K93: NdIterExtBlockOffsetSize doc prescribes block_end = end.div_ceil(b), which visits a phantom size-0 block for an empty range with an unaligned begin; every caller uses calc_block_end instead (latent, crate-internal doc)
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F009, low)
- Location: jix/src/util/iter/block.rs:12
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F009.md; repro in that run's repros/.

## K94: std of f16/f32/complex64 casts the f64 variance to the narrow output dtype before sqrt, giving inf, 0, or results tens of ulps off where the correctly rounded std is representable
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F010, low)
- Location: jix/src/ops/reduction.rs:2853
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F010.md; repro in that run's repros/. NumPy gives the same values. Answers batch4's open question on std.

## K95: RemoveDim::new / new_array panics (dim_arr NDIM_MAX assert) on a DimsArg with more than 8 entries instead of returning Err
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F011, low)
- Location: jix/src/ops/shape_ops/remove_dim.rs:58
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F011.md; repro in that run's repros/. Same kind of bug as K12 / K32 / K79; Python is protected by normalize_dims.

## K96: asarray / compact / cast with dtype= silently wrap out-of-range Python ints (asarray(300, dtype=uint8) -> 44, compact([-1, 5], dtype=uint32) -> [4294967295, 5]), while every jix op bounds-checks Python ints and raises OverflowError
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F012, low)
- Location: jix-py/src/ops/as_array.rs:60
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F012.md; repro in that run's repros/. Also the reverse: asarray(2**63, dtype=uint64) raises although the value fits. See K129 (run 2026-10-29-batch6 F021: numpy scalars still use numpy's C cast).

## K97: numpy(out=) accepts self-overlapping destinations (positive strides that alias), breaking StridedBuf::from_raw_parts_mut's no-alias precondition; reductions folded into such an out leave values that belong to no output cell
- Status: fixed in df2f918 (#92): numpy(out=) rejects overlapping outs (found in run 2026-10-27-batch5 as F013, low)
- Location: jix-py/src/array.rs:266
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F013.md; repro in that run's repros/. No UB found. The self-overlapping out question left open by trial2 and batch3. The repro's last, unguarded call now raises ValueError, which its docstring counts as correct.

## K98: numpy(out=) with an empty out that has a negative stride on an extent>1 dim (or a huge stride) panics in debug builds: the stride rebuild runs before the nitems==0 early return
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F014, low)
- Location: jix-py/src/array.rs:607
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F014.md; repro in that run's repros/. See K142 (run 2026-10-29-batch6 F034: core strided_span_bytes still overflows on empty regions).

## K99: Array.plain() of a dtype with an inner shape returns a different array: the inner dims become array dims and the dtype becomes the base dtype, contradicting its docstring and compact()
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F015, low)
- Location: jix-py/src/array.rs:733
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F015.md; repro in that run's repros/. The K66 fix (#91) changed numpy(), which plain() goes through.

## K100: read_from rounds untrusted header dims up with unchecked next_multiple_of: a dim within b - 1 of u64::MAX panics in debug builds (Rust and dev Python) instead of returning InvalidArchive; release wraps and returns InvalidShapeOperation
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F016, low)
- Location: jix/src/archive/array.rs:490
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F016.md; repro in that run's repros/. Same kind of bug as K55 / K83.

## K101: An empty array that jix writes cannot be read back when its padded shape product (zero dims counted as 1) overflows u64 - the archive reader is stricter than the writer
- Status: fixed in df2f918 (#92) by rejection: ArrayParams::tune rejects a shape whose block-padded product overflows u64, so such arrays can no longer be created or written (found in run 2026-10-27-batch5 as F017, low)
- Location: jix/src/archive/array.rs:493
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F017.md; repro in that run's repros/. Regression from the K13 fix (#86). The repro, which expects a round trip, now fails at creation (InvalidShapeOperation); e.g. compact_fn of shape [0, 2^32-1, 2^32+1] with the automatic block shape is rejected.

## K102: Archive reader truncates u64 header fields to u32 before validating them, so a block dim or block_size of 2^32 + 2 is silently read as 2 (also under Strict validation)
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F018, low)
- Location: jix/src/archive/array.rs:471
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F018.md; repro in that run's repros/.

## K103: check_out_buf still accepts a writable out whose strides alias over the requested region (InsertDim pull results with 0 strides, any pool-backed pull reused for a wider region), so push reads return Ok with the region not in out
- Status: fixed in df2f918 (#92): check_out_buf rejects outs whose strides overlap over the region (found in run 2026-10-27-batch5 as F019, low)
- Location: jix/src/storage/buf.rs:307
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F019.md; repro in that run's repros/. Incomplete K58 fix. Reported at jix/src/ops/shape_ops/insert_dim.rs:188; the defect is in check_out_buf. The check is conservative: some interleaved layouts that do not overlap are rejected too.

## K104: The f32 element_cost hint has no upper bound, so it becomes +inf; a zero-length concatenate member then turns it into NaN (0 * inf), and the next multi-input constructor panics in max_cost_layout_order (partial_cmp().unwrap())
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F020, low)
- Location: jix/src/storage/params.rs:912
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F020.md; repro in that run's repros/. Debug builds panic earlier, at the is_finite debug_assert at array.rs:1662.

## K105: A widening element-wise view keeps the source's explicit block shape (in elements, all fixed), so compact()/write_to() with default params fails with InvalidArgument once that block is over 1 GiB at the wider itemsize
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F021, low)
- Location: jix/src/ops/op1.rs:36
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F021.md; repro in that run's repros/. Since the 1 GiB block cap added with the K18 fix (#87).

## K106: f16 tan(+-177.5) rounds to +-inf with no Overflow report: Tan's error set has no overflow condition while the f16 kernel rounds the f32 result to f16
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F022, low)
- Location: jix/src/ops/op1.rs:674
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F022.md; repro in that run's repros/.

## K107: jix.log silently accepts an invalid base (1, 0, -0.0, negative): every element becomes inf/NaN/signed zero with no error, despite the documented 'positive and not 1' precondition
- Status: fixed in df2f918 (#92) (found in run 2026-10-27-batch5 as F023, low)
- Location: jix-py/src/ops/op1.rs:378
- Notes: bug-hunt/runs/2026-10-27-batch5/findings/F023.md; repro in that run's repros/. For f16, a valid base within ~1.5e-5 of 1 overflows the multiplier to inf. See K126 (run 2026-10-29-batch6 F018: an f16 result that rounds to inf is not reported).

## K108: var/std of finite data whose deviations overflow returns NaN again for 3+ elements (var([1e308, -1e308, 0.0]) = NaN, numpy inf) - the K84 fix partly reverts K67
- Status: open (found while rebasing onto df2f918 (#92))
- Location: jix/src/ops/reduction.rs:1942
- Notes: #92 maps only `m2 < 0.0` to +inf and keeps a NaN m2 as NaN (the K84 fix). Welford can also reach a NaN m2 from finite input: for [1e308, -1e308, 0.0] the mean becomes -inf after the second item, then `0.0 - (-inf)` and `-inf + inf` make mean and m2 NaN. So `var([1e308, -1e308, 0.0])` and `var([1e308]*40 + [-1e308]*40)` return NaN (two of the six K67 repro tests fail again, bug-hunt/runs/2026-10-24-batch4/repros/repro_2026_10_24_batch4__f002.rs), while numpy gives inf; before #92 they gave inf. The sign of m2 cannot tell this from a NaN / inf input (K84), so the fix needs another signal, e.g. a flag in the state for a non-finite input.

## K109: PyWriter::write hands write() a memoryview over Rust memory; derived views (b[:], memoryview(b), b.cast, np.frombuffer(b), PickleBuffer(b)) survive release() and outlive the buffer - use-after-free from pure Python
- Status: open (found in run 2026-10-29-batch6 as F001, high)
- Location: jix-py/src/archive.rs:419
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F001.md; repro in that run's repros/.

## K110: #[derive(Dtyped)] output is not hygienic: unqualified size_of/align_of/assert! and block-local consts (BASE_OFFSET, FIELDS, FIELD_*) resolve at the call site, so safe user code gets a wrong DTYPE (heap OOB read, uninitialized padding read, wrong field offsets)
- Status: open (found in run 2026-10-29-batch6 as F002, medium)
- Location: jix-macros/src/lib.rs:197
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F002.md; repro in that run's repros/. Same kind of bug as K25 / K50 / K70: safe user code drives jix's unsafe code into UB.

## K111: Plain::read_data (and jix-py numpy(out=)) take a &[u8] / &mut [u8] over the whole strided span, gap bytes included, so element-disjoint views whose spans interleave alias each other: UB from safe Rust and from Python (Miri)
- Status: open (found in run 2026-10-29-batch6 as F003, high)
- Location: jix/src/storage/plain.rs:468
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F003.md; repro in that run's repros/. Extends K31 (intended: out must not overlap) to element-disjoint views and to the safe Rust API. Miri reports UB; values are correct today.

## K112: 32-bit: Array.__len__ truncates a dim >= 2^32 with `*len as usize`, so len() silently returns shape[0] mod 2^32
- Status: open (found in run 2026-10-29-batch6 as F004, medium)
- Location: jix-py/src/array.rs:417
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F004.md; repro in that run's repros/.

## K113: Comparing a bool/u8/i8/u16/i16 operand with a Python float or complex runs in float32 (Complex<f32>) and returns wrong booleans (equal/not_equal/less/less_equal/greater/greater_equal, dunder operators, isclose)
- Status: open (found in run 2026-10-29-batch6 as F005, medium)
- Location: jix-py/src/ops/common/dtype_promote.rs:96
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F005.md; repro in that run's repros/. #82's f16 skip moved the precision cliff for untyped Python floats from f16 to f32. Same kind of bug as K26.

## K114: dtype_from_numpy recurses once per struct nesting level with no depth limit: a deeply nested numpy struct dtype segfaults the interpreter (native stack overflow) instead of raising
- Status: open (found in run 2026-10-29-batch6 as F006, low)
- Location: jix-py/src/dtype.rs:203
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F006.md; repro in that run's repros/. A native stack overflow, like K30 / K72.

## K115: Op2::shape calls the left child's shape() twice inside #[inline(always)], so debug builds inline 2^k copies for a left-deep chain of k binary ops: a 12-term sum or a degree-6 Horner polynomial overflows a 2 MiB thread stack (SIGABRT), and rustc time and memory grow exponentially (about 3 GB at 14 terms)
- Status: open (found in run 2026-10-29-batch6 as F007, medium)
- Location: jix/src/ops/op2.rs:186
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F007.md; repro in that run's repros/. A debug-build stack abort, like K72.

## K116: Deep type-erased op chains (every Python op result) recurse once per op level on read and on drop: `x = x + 1` in a loop overflows the native stack (Python SIGSEGV from ~2k levels with the debug extension, ~7k in release; drop from ~30-50k)
- Status: open (found in run 2026-10-29-batch6 as F008, medium)
- Location: jix/src/storage/core_trait.rs:139
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F008.md; repro in that run's repros/.

## K117: #[derive(Dtyped)] accepts a bare #[repr(packed)] (repr(Rust, packed), field order unspecified) and assumes declaration order. If rustc reorders the fields (e.g. -Zrandomize-layout), the DTYPE field offsets are wrong and dtype_sub_field silently reads the wrong bytes.
- Status: open (found in run 2026-10-29-batch6 as F009, low)
- Location: jix-macros/src/lib.rs:127
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F009.md; repro in that run's repros/. Reliance on an unspecified layout, like K63.

## K118: #[derive(Dtyped)] panics on a raw-identifier field (r#type) because its helper-const name is invalid; the recorded field name would also be "r#type" instead of "type"
- Status: open (found in run 2026-10-29-batch6 as F010, low)
- Location: jix-macros/src/lib.rs:147
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F010.md; repro in that run's repros/.

## K119: Dtyped for [T; N] rejects N == 65535 (strict `<` at dtype.rs:1210), so the valid runtime dtype u8[65535] (itemsize == Itemsize::MAX) cannot be named as a Rust type
- Status: open (found in run 2026-10-29-batch6 as F011, low)
- Location: jix/src/dtype.rs:1210
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F011.md; repro in that run's repros/.

## K120: 32-bit: empty read regions skip the region-size bound, so to_ndarray_slice / to_ndarray_buf of an empty region with huge other extents overflow usize in the stride and size math (debug panic; release wraps harmlessly; a gap in the K81 fix)
- Status: open (found in run 2026-10-29-batch6 as F012, low)
- Location: jix/src/error.rs:274
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F012.md; repro in that run's repros/. Gap in the K81 fix: the zeros-as-1 bound was added only inside to_ndarray_sub.

## K121: Scalar and Plain leaves report an L1-sized auto block shape as their granularity, so combine_block_layout lets them override a compact operand's explicit block shape: compact() of `a * 2.0` / `a + ndarray` / where turns column blocks [1024,1] into row blocks [1,1024]
- Status: open (found in run 2026-10-29-batch6 as F013, low)
- Location: jix/src/storage/params.rs:945
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F013.md; repro in that run's repros/.

## K122: An explicit block_shape given to the Plain constructors (plain_ndarray_with / _view_with / _ref_with / plain_ndarray_ptr, Python asarray(params=)) is silently discarded, though the docs say it seeds compact() / write_array()
- Status: open (found in run 2026-10-29-batch6 as F014, low)
- Location: jix/src/storage/plain.rs:135
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F014.md; repro in that run's repros/.

## K123: compact_with / override_from_array silently replace a caller's block_shape_fixed_dims with the source's when block_shape is unset (compact_ndarray_with rejects the same params, archive reads honor them)
- Status: open (found in run 2026-10-29-batch6 as F015, low)
- Location: jix/src/storage/params.rs:288
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F015.md; repro in that run's repros/.

## K124: Vec/slice MapMultiple scratch calls Vec::set_len over uninitialized items, then assigns through IndexMut, building a &mut [T] over uninitialized memory (latent; strict Miri flags it)
- Status: open (found in run 2026-10-29-batch6 as F016, low)
- Location: jix/src/util/arr_sequence.rs:455
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F016.md; repro in that run's repros/.

## K125: An untyped Python float or complex next to a bool/u8/i8/u16/i16 operand is cast to f32 / complex64 at dispatch with no range check: i8 + 1e300 -> inf, u8 * 1e-46 -> 0, u8 / 1e39 -> 0 (exact 2.55e-37 fits in f32), u8 / 1e-46 raises 'divide by zero'. An out-of-range Python int raises OverflowError instead
- Status: open (found in run 2026-10-29-batch6 as F017, low)
- Location: jix-py/src/ops/common/operand.rs:164
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F017.md; repro in that run's repros/. See K113 (F005), same root cause.

## K126: jix.log(f16, base) for a valid base near 1 rounds a finite result to +-inf with no Overflow report (no warning, no error under over='raise'); jix.divide on the same quotient and the f16 exp/tan/power kernels do report it
- Status: open (found in run 2026-10-29-batch6 as F018, low)
- Location: jix-py/src/ops/op1.rs:369
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F018.md; repro in that run's repros/. K107 covered the f16 multiplier overflowing, not the result; K106 fixed the same narrowing for the f16 unary kernels.

## K127: f64 jix.log(x, base) for a base other than e/2/10 multiplies ln(x) by a rounded 1/ln(base): exact powers come out 1 ulp off (log(64, base=4) = 2.9999999999999996), and power-of-two bases are inexact
- Status: open (found in run 2026-10-29-batch6 as F019, low)
- Location: jix-py/src/ops/op1.rs:366
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F019.md; repro in that run's repros/. The K92 fix covered f16/f32 only.

## K128: Struct dtypes nested 33+ levels are written to .jix without error but can never be read back (prost decode recursion limit); reachable from safe derive and Python
- Status: open (found in run 2026-10-29-batch6 as F020, low)
- Location: jix/src/archive/schema/conversions.rs:129
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F020.md; repro in that run's repros/. A dtype that can be written but not read back, like K53.

## K129: asarray/cast/compact with dtype= cast numpy scalars (np.generic) with numpy's C cast instead of jix's Cast. Float numpy scalars give different int results than the same value as a 0-d or 1-d array, and complex numpy scalars cast to float instead of raising TypeError.
- Status: open (found in run 2026-10-29-batch6 as F021, low)
- Location: jix-py/src/ops/as_array.rs:65
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F021.md; repro in that run's repros/. Gap in the K96 fix: np.generic is not excluded.

## K130: File-like adapters flatten Python exceptions from read()/write()/flush() into RuntimeError (KeyboardInterrupt/SystemExit/OSError lose their type), BufWriter's Drop calls write() again after it raised, and a missing file raises FileNotFoundError or RuntimeError depending on len=
- Status: open (found in run 2026-10-29-batch6 as F022, low)
- Location: jix-py/src/archive.rs:498
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F022.md; repro in that run's repros/.

## K131: read_array(file_like) and Array::read_from_reader leave the stream at an arbitrary position (core stops at the start of block_data; jix-py's internal BufReader reads up to 8 KiB ahead), so arrays written back-to-back with write_array(f) / write_to(&mut w) cannot be read back with successive reads
- Status: open (found in run 2026-10-29-batch6 as F023, low)
- Location: jix-py/src/archive.rs:142
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F023.md; repro in that run's repros/.

## K132: errstate restores a construction-time snapshot of OS-thread-local modes: asyncio tasks whose with-blocks exit out of order leave the thread in a non-default mode for good (divide-by-zero then silently returns 0)
- Status: open (found in run 2026-10-29-batch6 as F024, low)
- Location: jix-py/src/error.rs:113
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F024.md; repro in that run's repros/.

## K133: jix.Array binary/comparison dunders never return NotImplemented: a foreign operand's reflected op (even with __array_ufunc__ = None) is never tried, `a == None` raises ValueError, unsupported operand types raise ValueError instead of TypeError
- Status: open (found in run 2026-10-29-batch6 as F025, low)
- Location: jix-py/src/array.rs:880
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F025.md; repro in that run's repros/. The opposite direction of K37.

## K134: A Python bool passed as a dim is silently read as dim 0/1 (jix.sum(a, True), a.mean(False), jix.argmax(a, dim=True), concatenate, squeeze, permute_dims), where numpy raises TypeError and jix's own index parser rejects bools since K33
- Status: open (found in run 2026-10-29-batch6 as F026, low)
- Location: jix-py/src/util.rs:289
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F026.md; repro in that run's repros/. Gap in the K33 fix: the bool guard covers indices, not dims.

## K135: Test comparators are weaker than documented: assert_array_matches ("bit-for-bit") and same_value / assert_same_elements ignore the sign of zero, assert_array_matches never matches NaN, bool8 compares by truthiness; f32 max/min reductions use a 0.1 tolerance and reduction_shape_strategy never draws empty or 0-d shapes
- Status: open (found in run 2026-10-29-batch6 as F027, low)
- Location: jix/src/util/test_util.rs:893
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F027.md; repro in that run's repros/. Test infrastructure only.

## K136: Python assert_array_matches checks neither dtype nor 0-d shape, and its sub-range draw still skips the K21 cases (start == len, zero-length dims) although K21 is fixed
- Status: open (found in run 2026-10-29-batch6 as F028, low)
- Location: jix-py/python/tests/tests_util.py:284
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F028.md; repro in that run's repros/. Test infrastructure only.

## K137: No bool8 test strategy (Rust or Python) draws a byte other than 0/1, so no property test feeds non-canonical bool8 into any kernel; raw-byte regressions in bitwise, where and sum pass the suite
- Status: open (found in run 2026-10-29-batch6 as F029, low)
- Location: jix/src/util/test_util.rs:348
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F029.md; repro in that run's repros/. Test infrastructure only.

## K138: Test strategy gaps: float any_strategy is finite-only despite its "full domain" doc; f16 any_strategy is 87% inf/zero, so the f16 max proptest's expected 1-D result is +inf 95% of the time; comparable/non-zero sets miss -0.0, inf and real-only/imaginary-only divisors; blocks are capped at 4 per dim; Python never draws 0-d shapes or non-C layouts
- Status: open (found in run 2026-10-29-batch6 as F030, low)
- Location: jix/src/util/test_util.rs:243
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F030.md; repro in that run's repros/. Test infrastructure only.

## K139: assert_read_rejects_invalid_index tries only 3 invalid-index classes and runs on only 4 storages (permute_dims, remove_dim, sub_dtype, where); no library bug hides behind the gap today
- Status: open (found in run 2026-10-29-batch6 as F031, low)
- Location: jix/src/util/test_util.rs:1379
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F031.md; repro in that run's repros/. Test infrastructure only.

## K140: Plain::new / plain_ndarray_ptr / compact_nd_ptr Safety sections never forbid writing the memory during a read or while a lent pull view is alive, and never cover the strided span's gap bytes: calls meeting the documented contract are UB (Miri)
- Status: open (found in run 2026-10-29-batch6 as F032, low)
- Location: jix/src/storage/plain.rs:108
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F032.md; repro in that run's repros/. Gap in the K59 fix: the Safety sections of plain_ndarray_ptr / Plain::new were not updated.

## K141: Zero-element arrays are rejected on input when a dim of length > 1 has a negative stride, though no element is ever addressed (gap in the K80 fix)
- Status: open (found in run 2026-10-29-batch6 as F033, low)
- Location: jix/src/storage/plain.rs:519
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F033.md; repro in that run's repros/. Gap in the K80 fix: the negative-stride check is skipped per dim (len <= 1), not for empty arrays.

## K142: Empty regions overflow in strided_span_bytes (util/mod.rs:111), and on 32-bit in materialize_out_buf (buf.rs:386), because the span/size arithmetic does not stop at the zero extent: debug-only panic, release correct
- Status: open (found in run 2026-10-29-batch6 as F034, low)
- Location: jix/src/util/mod.rs:111
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F034.md; repro in that run's repros/. Gap in the K98 fix: only jix-py's own copy of strided_span_bytes returns early.

## K143: jix-py rejects padding-free numpy struct dtypes whose field names are not in offset order (e.g. multi-field view x[['b','a']]) with a misleading 'padding between fields' error
- Status: open (found in run 2026-10-29-batch6 as F035, low)
- Location: jix-py/src/dtype.rs:208
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F035.md; repro in that run's repros/.

## K144: f32/f16 base ** i32 exponent rounds the exponent to f32, so odd exponents above 2^24 become even and a negative base gives the wrong sign ((-1)^odd = +1)
- Status: open (found in run 2026-10-29-batch6 as F036, medium)
- Location: jix/src/ops/op2.rs:1209
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F036.md; repro in that run's repros/. Regression from the K51 fix (#90): powi(b as i32) became powf(b as f32), losing the parity of odd exponents above 2^24.

## K145: Float add/sub/mul/square and sum/prod/mean never report invalid/over, while divide reports the same conditions; under default modes inf/inf raises but inf-inf, 0*inf and sum([inf,-inf]) are silent NaN
- Status: open (found in run 2026-10-29-batch6 as F037, low)
- Location: jix/src/ops/op2.rs:590
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F037.md; repro in that run's repros/. Answers the batch3 / batch5 question on float Overflow / Invalid reporting.

## K146: A write_to_file / write_array that fails part-way leaves a partial, unreadable .jix file; the retry then raises FileExistsError, and a failed append leaves garbage bytes in the file
- Status: open (found in run 2026-10-29-batch6 as F038, low)
- Location: jix-py/src/archive.rs:215
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F038.md; repro in that run's repros/.

## K147: A Python read that raises FloatingPointError drops the warnings recorded in the same read: an 'over' RuntimeWarning that comes with an 'invalid' (or 'divide') error is silently discarded (numpy(), compact(), write_array)
- Status: open (found in run 2026-10-29-batch6 as F039, low)
- Location: jix-py/src/array.rs:673
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F039.md; repro in that run's repros/.

## K148: Arrays read from a .jix file do not carry the file's codec/filters in their spec, so compact() and writes of views over them silently re-encode with the default byte-shuffle
- Status: open (found in run 2026-10-29-batch6 as F040, low)
- Location: jix/src/archive/array.rs:552
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F040.md; repro in that run's repros/.

## K149: BlockTable documents in-bounds block locations as an invariant that Minimal validation does not establish; BlockTable::new only debug_asserts it, so debug builds (incl. the dev Python extension) panic at load where release builds load
- Status: open (found in run 2026-10-29-batch6 as F041, low)
- Location: jix/src/storage/block.rs:129
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F041.md; repro in that run's repros/.

## K150: Float reductions depend on the host's detected CPU cache sizes (and SIMD level): the same numpy array, or the same .jix file, sums to different f32 values on machines with different L1d/L2
- Status: open (found in run 2026-10-29-batch6 as F042, low)
- Location: jix/src/storage/params.rs:453
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F042.md; repro in that run's repros/. Answers the pilot's question on host-dependent float results.

## K151: SIMD level coverage: for_each_simd_level! skips the levels the host lacks without saying so, and level-dependent driver code (pipeline lane count, reduction folds, update_bulk fallbacks) runs only at the host's Level::new() level
- Status: open (found in run 2026-10-29-batch6 as F043, low)
- Location: jix/src/util/test_util.rs:1270
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F043.md; repro in that run's repros/. Test infrastructure only.

## K152: Free-threaded builds: numpy(out=) validates out's shape/strides in one read but writes using a second read (array.rs:617/653), and plain_from_numpy reads ndim, shape and strides separately (operand.rs:424/465/471). A concurrent `.shape=` or `.strides=` becomes an out-of-bounds write or read, or a PanicException.
- Status: open, plausible (found in run 2026-10-29-batch6 as F044, low)
- Location: jix-py/src/array.rs:617
- Notes: bug-hunt/runs/2026-10-29-batch6/findings/F044.md; repro in that run's repros/. Plausible: argued from the code; needs a free-threaded build.

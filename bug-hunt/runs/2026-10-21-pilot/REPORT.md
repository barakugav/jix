# jix bug hunt - run 2026-10-21-pilot

- Date: 2026-10-21. Code at commit `502e9a4`.
- Scope: pilot. Units were the three `util-memory-*` module units and the `concept-alignment`
  concept unit. No dynamic hunters ran.
- Process: Hunt, then Dedup, then Verify (adversarial verifier plus repro for every finding),
  then Report. Severity follows the table in `bug-hunt/PRIMER.md`.
- Layout of this directory:
  - `findings/<ID>.md` - one write-up per finding.
  - `repros/` - Rust repros (`repro_2026_10_21_pilot__fNNN.rs`) and Python repros (`FNNN.py`).
  - `scratch/` - hunter experiments in Python.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 13 |
| plausible | 0 |
| refuted | 0 |
| verifier failed | 0 |
| unverified (not sent to a verifier) | 0 |

| severity | count | IDs |
|----------|------:|-----|
| critical | 0 | - |
| high | 6 | F001, F002, F003, F004, F005, F011 |
| medium | 2 | F008, F013 |
| low | 5 | F006, F007, F009, F010, F012 |

Four hunters ran, and none failed (failed list: `[]`):

| hunter | kind | raw reports | IDs |
|--------|------|------------:|-----|
| util-memory-alloc-pool | module | 4 | F001, F002, F009, F010 |
| util-memory-arrayvec-bitmap-cache | module | 3 | F006, F011, F012 |
| util-memory-helpers | module | 5 | F002, F003, F004, F006, F013 |
| concept-alignment | concept | 4 | F005, F007, F008 (two reports) |

Dedup merged the 16 raw reports into 13 findings:
- F002 and F006 were each reported by two hunters.
- concept-alignment filed two reports on the i686 alignment root cause, and dedup merged them into
  F008.
- `KNOWN.md` is empty, so no finding was dropped as known.

Headlines:
- Six high-severity memory-safety or unsound-API defects. Two are reachable from Python:
  - **F001**: a malformed `.jix` makes release wheels return uninitialized or stale bytes,
    including another array's data.
  - **F005**: an mmap read of a truncated file with an explicit `len` segfaults.
- The other four highs need the safe public Rust API plus unusual input:
  - F002: a huge shape followed by a widening cast.
  - F003: a padded struct dtype.
  - F004: a `Read` impl that inspects `buf`.
  - F011: a ZST with `Drop` in a `Dimension::vec` container.
- Nothing is critical: every UB path needs malformed input, an extreme shape, a niche feature or a
  non-standard reader.
- One more bug sits outside the pilot's scope: big-endian numpy input is silently reinterpreted as
  little-endian. A hunter demonstrated it, but this run did not verify it (see section 6, theme A).
  Make sure the Python hunters cover it in the full run.

## 2. Findings

Sorted by severity. All 13 are confirmed, so the confirmed-before-plausible rule changes nothing;
ties are listed in ID order.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| F001 | high | confirmed | memory-safety | Short zstd block decode is only `debug_assert`ed; release reads return uninitialized or stale destination bytes as data | `jix/src/storage/block.rs:193` | [F001](findings/F001.md) |
| F002 | high | confirmed | memory-safety | Pull-mode buffer size `product(shape) * itemsize` wraps after an itemsize-widening op, so a too-small pooled buffer is written at full-shape strides | `jix/src/storage/buf.rs:312` | [F002](findings/F002.md) |
| F003 | high | confirmed | memory-safety | Padded struct dtypes: `NdCopier` moves elements as `[u32; 2]` / `[u64; 2]` and reads uninitialized padding as integers | `jix/src/util/nd_copy.rs:31` | [F003](findings/F003.md) |
| F004 | high | confirmed | unsound-api | `Array::read_from_reader` hands uninitialized buffers to the caller's safe `Read` impl | `jix/src/util/mod.rs:577` | [F004](findings/F004.md) |
| F005 | high | confirmed | memory-safety | mmap reader bounds sections by the caller's `len`, not by the mapping; a truncated file gives an OOB read and a segfault | `jix/src/archive/block.rs:487` | [F005](findings/F005.md) |
| F011 | high | confirmed | unsound-api | `ArrayVec::extend_from_iter` double-drops ZST elements; safe public code reaches it through `Dimension::vec` | `jix/src/util/arrayvec.rs:416` | [F011](findings/F011.md) |
| F008 | medium | confirmed | portability | i686 Linux: alignment hard-coded to 8 for f64/i64/u64/c128, but the ABI says 4; Python dtype conversion fails and derived-struct archives cannot be read back | `jix-py/src/dtype.rs:80` | [F008](findings/F008.md) |
| F013 | medium | confirmed | overflow | A 0 dim hides an overflowing shape product: stride overflow panics in debug, and reducing the 0 dim away panics in release | `jix/src/util/mod.rs:145` | [F013](findings/F013.md) |
| F006 | low | confirmed | panic | `ArrayParams::tune` divides by zero (or overflows) on an invalid fixed block dim when another dim is auto-scaled | `jix/src/storage/params.rs:427` | [F006](findings/F006.md) |
| F007 | low | confirmed | portability | mmap reader rejects valid archives whose block-locations section is not 8-aligned in the file | `jix/src/archive/block.rs:489` | [F007](findings/F007.md) |
| F009 | low | confirmed | unsound-api | `RuntimeAlign::new` accepts align 0, so safe `AlignedBytes` constructors build a null `NonNull` or an align-0 `Layout` (latent) | `jix/src/util/aligned_vec/mod.rs:28` | [F009](findings/F009.md) |
| F010 | low | confirmed | memory-safety | `RawAlignedBytes::shrink_to` reallocs to size 0, skips the null check and leaks at capacity 0 (latent) | `jix/src/util/aligned_vec/raw.rs:155` | [F010](findings/F010.md) |
| F012 | low | confirmed | error-handling | `ArrayParams::block_shape` / `block_shape_fixed_dims` unwrap `check_ndim`, so more than 8 entries panic | `jix/src/storage/params.rs:121` | [F012](findings/F012.md) |

### Bugs

**F001** (high, reachable from Python).
- **Defect.** `BlockTable::read_block` checks the decoded length only with `debug_assert_eq!`.
  zstd accepts a short frame padded by a skippable frame, so the archive's offsets stay valid,
  Strict validation passes, and there is no checksum to catch it.
- **Effect in release builds.** The unwritten tail of the destination is returned as data, from
  one of two sources:
  - freed heap memory, through `ndarray` uninit plus `assume_init`, which is UB;
  - another array's bytes, from the pooled tmp or filter scratch buffers.
- **Evidence.** Reproduced in debug (panic at block.rs:193) and release (stale 0x77 / 0x5A bytes),
  and through `jix.read_array(...).numpy()`.
- **Fix direction.** Return `InvalidArchive` instead of asserting.

**F002** (high).
- **Defect.** `check_shape_overflow` runs only in broadcast/tile/repeat/stack/concatenate, and it
  uses the input itemsize. `Op1`/`Op2`/`Map` (and therefore `cast`) widen the itemsize without a
  re-check, so `u8.broadcast([2^61+1]).cast::<f64>()` is accepted.
- **Effect.** A pull read sizes its buffer with a wrapping `product * itemsize` (buf.rs:312) and
  then writes tiles at full-shape strides. That is a heap overflow in release.
- **Evidence.**
  - Debug overflow panic: reproduced.
  - Release SIGSEGV: reported by the hunter, not re-run by the verifier.
  - 32-bit wheels (any pull read over 4 GiB): argued from the code, not run.
- **Python.** Not claimed to be reachable, because `numpy()` pushes into a numpy destination.

**F003** (high).
- **Defect.** `#[derive(Dtyped)]` accepts padded `#[repr(C)]` structs. `NdCopier` picks its kernel
  from `(itemsize, alignment)` alone:
  - `{a: u8, b: u32}` is copied as `[u32; 2]`, and `(16, 8)` as `[u64; 2]`;
  - `fill_contiguous` dispatches on itemsize alone.
- **Effect.** Every read path goes through `NdCopier`, so a plain `to_ndarray()` of a Rust-built
  array reads uninitialized padding as integers. That is UB from safe code (Miri).
- **Scope.** Values come out correct natively, and Python cannot trigger it.
- **Fix note.** Untyped copies alone are not enough: byte-shuffle and zstd also read the padding
  bytes.

**F004** (high).
- **Defect.** `value_from_io`, `ArchiveReader::read_slice` and `Owned::read_section` pass
  `&mut [u8]` over uninitialized memory to `read_exact` on the user's reader. `Read` is a safe
  trait, so a safe impl that looks at `buf` before filling it causes UB through the safe
  `read_from_reader`.
- **Evidence.** Natively, a snooping reader saw 594 of 610 bytes from a freed `Vec`, plus glibc
  tcache metadata (a mangled heap pointer). Miri reports the UB.
- **Scope.** `File`, `Cursor` and `BufReader` are unaffected, and so is Python.

**F005** (high, reachable from Python).
- **Defect.** `Mmap::read_section` checks `offset + size` against the caller's `len`
  (`check_section_bounds`), not against `mmap.len()`. `MmapData::as_ref` then builds a slice of that
  length, so zstd gets a source pointer past the mapping.
- **Trigger.** `jix.read_array(p, len=<original size>, mmap=True)` on a truncated file segfaults.
  This is a safe call. With the default `len=None` the read fails cleanly.
- **Related, not run.** The unchecked add at common.rs:267 can wrap in release, which a crafted TOC
  could exploit even with the default `len`.
- **Fix direction.** Also bound sections by the mapping length.

**F011** (high).
- **Defect.** `extend_from_iter` skips `ptr.write` for ZSTs but still counts the element in `len`.
  A ZST with `Drop` is therefore dropped at once and again on clear/truncate/drop.
- **Reachability.** The hunter thought it crate-internal. The verifier reached it from safe public
  code through `Dimension::vec` -> `DimArray<T>`, using `Extend`, `collect`, `clone_from` and
  `IntoIter::clone`.
- **Evidence.** The verifier turned it into a glibc double free with a sound user-side token type.
- **Fix.** Add `else { mem::forget(elt) }`, as `Clone` already does. Do not make the write
  unconditional (over-aligned ZST pointer).

**F008** (medium).
- **Defect.** `ScalarKind::alignment()` returns 8 for i64/u64/f64/c128 on every target. On i686
  Linux (the shipped manylinux/musllinux x86 wheels) the ABI, Rust `align_of` and numpy all say 4.
- **Effect.**
  - `dtype_to_numpy` requires exact `(itemsize, alignment)` equality, so `.dtype`, `.numpy()`,
    shape ops and `where` raise ValueError for numpy's default dtypes.
  - Derived structs with 8-byte fields write archives that read back as `InvalidArchive`.
- **Evidence.**
  - Rust half: confirmed on a cross-built i686-musl run.
  - Python half: not executed, because no i686 Python was available.
- **Severity.** No UB.

**F013** (medium).
- **Defect.** `try_product` folds `checked_mul` from the left, so a leading 0 masks overflow:
  `[0, 2^40, 2^40]` passes every shape check, while `[2^40, 2^40, 0]` is rejected.
- **Effect.**
  - The unchecked C-order stride helpers then overflow. `reshape`, compact of a broadcast and
    `compact_fn` panic in debug; in release the wrap is harmless.
  - `broadcast(..).sum(0)` yields a non-empty `[2^40, 2^40]` array, and its `.compact()` panics in
    release (array.rs:1300).
- **Python.** Reachable too, with an extra unchecked product at
  `jix-py/src/ops/shape_ops.rs:724`.
- **Side issue.** `to_ndarray()` of that array panics inside ndarray.

**F006** (low).
- **Defect.** `tune` runs `scale_block_shape` before it validates the user's fixed block dims.
  - A fixed 0 combined with any auto-scaled dim divides by zero (params.rs:427) in every profile.
  - Huge fixed dims overflow the unchecked product (:400).
- **Contrast.** The all-fixed path returns a clean `InvalidArgument`.
- **Trigger.** Needs an explicit invalid `block_shape` plus `block_shape_fixed_dims` with a `False`.
  Python sees `PanicException`.

**F007** (low).
- **Defect.** The writer pads the block-locations section relative to the absolute stream position,
  while the TOC stores base-relative offsets. `Mmap::read_section` then needs the absolute file
  offset to be 8-aligned.
- **Effect.** `mmap=True` rejects, while `mmap=False` accepts:
  - relocated archives (embedded, concatenated, saved from BytesIO);
  - archives written by the i686 Linux wheel, which pads to 4.
- **Severity.** A clean error with no memory-safety impact.

**F009** (low, latent).
- **Defect.** `RuntimeAlign::new` lets 0 through, so the safe `AlignedBytes` constructors reach
  `NonNull::new_unchecked(null)` and `Layout::from_size_align_unchecked(n, 0)`.
- **Why latent.** The type is crate-internal, and every caller passes an alignment derived from
  `Alignment` or `CACHE_LINE_SIZE`.
- **Also.** The doc at mod.rs:46-48 ("0 = minimum alignment") is stale.

**F010** (low, latent).
- **Defect.** `shrink_to(0)`:
  - calls `realloc` with `new_size == 0` (UB per Miri);
  - wraps the result in `NonNull::new_unchecked` without a null check;
  - leaks the block once capacity is 0.
- **Why latent.** `shrink_to` / `shrink_to_fit` have no callers outside the inline tests.

**F012** (low).
- **Defect.** The public builder setters call `check_ndim(len).unwrap()`, so more than 8 entries
  panic instead of returning `TooManyDimensions`.
- **Python.** The params extraction forwards lists unchecked, so `params={"block_shape": [1]*9}`
  raises `PanicException`, a `BaseException` that `except Exception` does not catch.
- **Severity.** Wrong error class only.

### Questionable semantics

None reported this run. One candidate came up as an open question: cross-machine reproducibility of
float reductions (section 6, theme F).

### Re-running a repro

Each Rust repro is in `repros/`. The test name is the file name without `.rs`. To run one, copy it
back to `jix/tests/` and run it from the repo root:

    cp bug-hunt/runs/2026-10-21-pilot/repros/repro_2026_10_21_pilot__f001.rs jix/tests/
    cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_21_pilot__f001

Delete the copy afterwards. `jix/tests/repro_*.rs` is gitignored, so a leftover copy does not show
up in `git status`.

The repros assert the correct behavior, so they fail, panic or abort while the bug is present.
Per-repro notes:

- **F001**
  - Debug: panics at block.rs:193.
  - With `--release`, the reads return stale bytes.
- **F002**: the debug build panics on the overflow. A release run is an out-of-bounds heap write,
  so do not run it casually.
- **F003**
  - Passes natively.
  - To see the UB, run it under Miri:
    `python jix/run_miri.py --all-features --test repro_2026_10_21_pilot__f003 -- padded_plain_to_ndarray`
- **F004**
  - Native run: add `-- --nocapture --test-threads=1`.
  - Miri: `python jix/run_miri.py --all-features --test repro_2026_10_21_pilot__f004 -- miri_`
- **F005**: `read_truncated_mmap_last_block` kills the test process with SIGSEGV. Run the tests one
  at a time with `-- --nocapture --test-threads 1 <test name>`.
- **F008**
  - On x86_64 only `numpy_i686_struct_layouts` fails.
  - The `*_on_target` tests fail only when built for i686. The cross-build steps are in
    `findings/F008.md`.
- **F009 and F010**
  - Both include the unmodified `jix/src/util/aligned_vec/` sources via
    `#[path = "../src/..."]`, so they compile only from `jix/tests/`.
  - F009 re-runs its `#[ignore]`d children as subprocesses.
  - F010's null-realloc variant: `-- --ignored f010_shrink_null --nocapture`.
- **F011**: the double-free demo is `#[ignore]`d because glibc aborts the process. Run it with
  `-- --ignored double_free`.

Python repros cover F001, F005, F006, F007, F008, F012 and F013.
- First build the bindings into `.venv`, with `maturin develop` in `jix-py/` or with
  `bash bug-hunt/setup.sh`.
- Then run, for example, `.venv/bin/python bug-hunt/runs/2026-10-21-pilot/repros/F005.py`.
- Per-script notes:
  - F005 runs each case in a child process, so the segfault shows up as an exit status.
  - F001 panics with the debug extension and returns stale bytes with a release wheel.
  - F008 shows the bug only on i686 Linux.

## 3. Unverified findings

None. Every deduplicated finding was sent to a verifier: `verify` was on and `minConfidence` was
`low`.

## 4. Refuted and intended

- No finding was refuted or judged intended.
- No finding matched `KNOWN.md`, which has no entries yet.

The lists below are the tuning signal for the next run.

### Verifier corrections to hunter claims

- F006: lowered from medium to low. No valid input reaches the panic: internally propagated block
  shapes are clamped with `.max(1)`, and archive load validates `b > 0`.
- F007: rated low, not medium. The failure is a clean `InvalidArchive`, there is no memory impact,
  and `mmap=False` works around it.
- F011: the hunter said "crate-internal". The verifier found a public path through the associated
  type `Dimension::Vec<T>` and rated it high.
  - Lesson: a type from a `pub(crate)` module can still leak through a public associated type.
  - The context docs and the primer should say this.
- F013: raised from low to medium. The verifier found a release-build panic (`sum` over the 0 dim,
  then `compact`), beyond the debug-only stride overflow.
- F002, F005: hunter claims not re-run by the verifier:
  - F002's release SIGSEGV;
  - F005's wrapping-add variant.
  Both findings stand on other evidence.

### Hypotheses the hunters checked and ruled out

From the coverage notes, one line each with the reason:

- `strides_for_layout_order` fed a non-permutation: refuted. Every producer emits a permutation:
  plain, permute_dims, insert/remove_dim, stack, reshape, reduction, and the forwarding ops.
- Tile/Repeat on 8-D arrays overflowing `dim_arr(ndim+1)`: refuted. They are rejected at
  construction with a clear error.
- Reduction `state_in_out_buf` aliasing: clean under Miri for mean i32->f64, with both C-order and
  permuted outputs.
- `AlternatingBuffers` filter parity and the raw-pointer move in `edit()`: clean under Miri for
  1-4 filters.
- `cast_slice` empty-slice alignment assert at scalar.rs:120: unreachable, because
  `is_contiguous_aligned` guards it.
- `value_as_bytes` exposing padding: all callers are padding-free (Header, `[Section; 2]`,
  BlockLocation2 on 64- and 32-bit).
- `SendSyncPtr`: the current callers are sound. Plain keeps its allocation alive, and Mmap is held
  in an Arc.
- Pipeline and Staging noalias spans: they match what the kernels touch, and the n == 0 spans
  cannot occur.
- REQUIRE_ALIGNED kernels: every aligned access is backed by a runtime base+stride check or by
  staging. This covers where_op, reduction `items_aligned`, pipeline `compute_need_staging` and
  nd_copy.
- `scale_read_shape` and the `calc_block_end` callers: clean. Products are bounded, block shapes
  are validated > 0, and the tile sizes are >= 1.
- Alloc/dealloc Layout pairing in aligned_vec: consistent on every path (fresh, grow, clone, pool
  reuse).
  - The only violations are F009 and F010.
- BufferPool `UnsafeCell` aliasing and thread safety: no user code runs while a `&mut` is live, and
  ReadContext is `!Sync`.
- BufferPool `get(0)` leaving a dangling cap-0 buffer in the free list: harmless. A later larger
  `get` frees it and replaces it.
- Full-capacity PoolBuf slices: no consumer uses the slice length as the logical size. The extra
  length only widens noalias spans over buffers that never overlap.
- Staging broadcast cache with `filled_with_n = MAX`: safe, because the first fill of a run is
  always the largest chunk.
  - Worth stating as a contract in the pipeline context doc.
- ArrayVec `len <= CAP`: every `len` write holds the invariant.
  - `extend_from_slice` truncates in release, but all three callers pre-check.
  - The only defect is the ZST case (F011).
- DimBitmap `filled(0, false).all() == true`, which makes a 0-d spec count as fully fixed: harmless
  in `tune`.
- cpu_cache accepting "0K" on Linux: effectively unreachable, because kernels hide a sysfs `size`
  of 0.
- `l1_data == 0` or `l1_data < itemsize`: no panic. It yields 1-element blocks, and a 56000-byte
  itemsize round-trips correctly.
- `CacheSizes` doc ("a missing level equals the next level down") disagrees with the code: no
  impact, because nothing reads `l3`. This is a doc nit.
- Misaligned inputs and outputs (offsets 0-5, odd strides) through sum, argmax, var, max,
  elementwise ops, where and compact: no misaligned-deref panic in a debug-build sweep.
- Python: unaligned numpy arrays and packed-struct field views work correctly.

## 5. Coverage

### Per hunter

- **util-memory-alloc-pool**
  - Read every line of `aligned_vec/raw.rs`, `aligned_vec/mod.rs` and `buf_pool.rs`, tests
    included.
  - Checked:
    - Layout pairing on every alloc, realloc and dealloc;
    - BufferPool aliasing and thread safety;
    - every consumer of full-capacity PoolBuf slices;
    - write-before-read for the encoder, compaction, reduction and staging buffers.
- **util-memory-arrayvec-bitmap-cache**
  - Read every line of `arrayvec.rs`, `bitmap.rs`, `hint.rs` and `cpu_cache.rs`.
  - Checked every `len` write against `len <= CAP`, plus the DimBitmap invariants and their callers.
  - Traced the cpu_cache parsing and the `l1_data == 0` path through `tune`, with a runtime harness
    on the real files.
- **util-memory-helpers**
  - Read every line of `util/mod.rs` and its tests, and traced each unsafe block to its callers.
  - Miri-checked AlternatingBuffers and the reduction state aliasing.
  - Checked the noalias spans, the REQUIRE_ALIGNED backing checks, `scale_read_shape`,
    `value_as_bytes`, `cast_slice` and the `SendSyncPtr` callers.
- **concept-alignment**
  - Traced the alignment of every typed access in `jix/src` and `jix-py/src`: nd_copy,
    NdIterUnordered, pipeline staging, reduction, where_op, FnStorage, Scalar, the mmap typed slices,
    the codec filters and the jix-py dtype seam.
  - A debug-build Rust sweep and a Python sweep over misaligned inputs and outputs both passed.

### Skipped or only skimmed

- util-memory-alloc-pool
  - Skimmed only:
    - reduction.rs fold/finalize internals;
    - elementwise_pipeline beyond Staging and materialization;
    - archive parsing.
  - Not done:
    - NdIterUnordered's visit order, on which reduction state init depends, was not verified.
    - The 32-bit variant of F002 was argued from the code, not built.
    - Miri was not run.
- util-memory-arrayvec-bitmap-cache
  - Consulted, not exhaustively reviewed:
    - `dimension.rs:505-545`;
    - parts of `dtype.rs`, `params.rs` and `elementwise_pipeline.rs:860-935`;
    - `info.rs`, `plain.rs:100-190`, `scalar.rs:40-75`, `codec/mod.rs:140-320`;
    - `archive/block.rs:330-370` and `archive/array.rs:440-520`;
    - the jix-py params extraction.
  - Not reviewed: macOS sysctl at runtime (the host is not macOS).
  - Miri was not run.
- util-memory-helpers
  - Not reviewed beyond the span and alignment questions: the pipeline static/dyn drivers.
  - SIMD kernels were not reviewed.
  - The 32-bit variant of F002 was argued from the code only, because no i686 C headers were
    available for zstd-sys.
  - The Scalar/Broadcast padded paths and the `to_ndarray_slice` de-initialization of user bytes
    were not Miri-tested.
  - `StridedBuf::from_raw_parts[_mut]` silently zips shape and stride lists of different lengths in
    `strided_span_bytes`. This public unsafe API has an ambiguous contract; the hunter noted it but
    did not report it.
- concept-alignment
  - Checked by grep only: shape_ops (tile, repeat, flip, roll, concatenate, stack, reshape).
  - Skimmed:
    - `scalar/simd.rs` and `traits_util.rs`, load/store paths only;
    - the `arr_sequence` tuple impls.
  - Reasoned from the code only, because there was no cross target at hunt time (the F008 verifier
    later cross-built for i686-musl):
    - the NEON and AVX-512 specifics;
    - the armv7/ppc64le `REQUIRE_ALIGNED = true` paths.
  - Miri was not run; debug-build ub-checks were used instead.
- Context-doc suspicion that no hunter settled: `where_op.rs:305` (bool validity). There is no bool
  validation in archive loading (`archive/schema/conversions.rs`).
  - All the other suspicions in these units were settled:
    - storage/block.rs:193 -> F001;
    - scalar.rs:111 -> F003;
    - value_from_io -> F004;
    - aligned_vec -> F009 and F010;
    - arrayvec ZST -> F011.

### Housekeeping

- The hunter scratch files `jix/tests/hunt_2026_10_21_pilot__*.rs` (12 files) were deleted.
- The Rust repros were moved to `repros/`.
- The hunters' Python scratch scripts remain in `scratch/`.
- The alloc-pool hunter left a release build in `jix/target/release`.

## 6. Open questions from hunters

### A. Possible bugs outside this run's units (route to the full run)

- **Big-endian numpy input is silently reinterpreted as little-endian** (concept-alignment).
  - Demonstrated, but not verified as a finding:
    `jix.asarray(np.arange(3, dtype='>i4')).numpy()` returns `[0, 16777216, 33554432]`
    (`scratch/concept_alignment_3.py`).
  - Already listed as a suspicion in `py-core.md:183` and `py-ops.md:197`.
  - Likely high: silent wrong results on input numpy accepts.
- Context-doc suspicion `where_op.rs:305` (bool validity) is still open; see section 5.

### B. Untrusted archives (codec-archive unit)

- `check_section_bounds` (archive/common.rs:267) computes `section.offset as u64 + section.size`
  unchecked, and block.rs:487 computes `base_offset as i64 + section.offset` and then casts with
  `as usize`. Both can wrap with a crafted TOC. Re-check them once F005's fix clamps `len` to the
  file. (helpers, concept-alignment)
- uleb128 decoding (common.rs:184): `<< (i*7)` overflows at the 10th byte. `read_slice` (:209)
  reserves an attacker-controlled length before reading, so a tiny file can abort the process on
  allocation failure. (helpers)
- archive/array.rs:468 casts `header.block_shape[dim]` with `as BlockSize` before validating it,
  so an out-of-range u64 is silently truncated. Check the proto field width. (arrayvec)
- archive/block.rs `read_content`: `header.block_size as BlockSize` truncation, and
  `section.size as usize` on 32-bit. (alloc-pool, noted only)
- `<[T; 0]>::DTYPE` builds a dtype with shape [0] and itemsize 0, which the runtime constructors
  reject. A derived struct with a zero-length array field might therefore write an archive whose
  dtype fails to load back. (arrayvec)
- Does zstd's `decompress_to_buffer` ever write past `nbytes` into dst? The F001 repro saw 18
  0xEE bytes, not 16. This matters for any "decode straight into a larger out buffer"
  optimization. (alloc-pool)

### C. Size arithmetic, overflow and 32-bit targets

- Fix scope for F002: should the Op1/Op2/Map/Cast constructors run `check_shape_overflow` with
  the output itemsize? And should `materialize_out_buf`, `materialize_pipeline_out_buf` and
  `pick_output_layout` switch to `checked_mul`? (helpers, alloc-pool)
- Other constructors that widen the itemsize without a re-check: Op2 promotion (e.g. broadcast u8
  + f64), where_op promotion, transmute, and reductions whose state is larger than the input. They
  all feed the unchecked sizes at buf.rs:312, elementwise_pipeline.rs:878 and reduction.rs:501.
  (alloc-pool)
- `pick_output_layout` (elementwise_pipeline.rs:890): its `product * itemsize` is unchecked, so a
  pull read of a mostly-broadcast huge region panics in debug and wraps in release. (arrayvec)
- 32-bit (x86 and armv7 wheels):
  - `check_shape_overflow` bounds only the u64 product, so a pull region over 4 GiB wraps
    `product::<usize>() * itemsize`.
  - array.rs:960 truncates nitems, which can misroute a huge request into the small_read fast
    path.
  - The u64->usize narrowings (`(end-start) as usize`, `Idx::from_u64`) should be checked.
  - Needs someone with a 32-bit target. (alloc-pool, helpers)
- `block_shape.try_product().unwrap()` over u32 `BlockSize` (array.rs:1293, archive/array.rs:421)
  panics for a user-chosen block of 2^32 or more elements, e.g. `[70000, 70000]`. Should `tune`
  reject it cleanly? (helpers)

### D. Platform and alignment

- F008 follow-up:
  - Confirm on a real i686 Linux wheel, e.g. in an i386 docker image:
    `np.dtype('f8').alignment` should be 4, and `jix.asarray(np.zeros(3)).numpy()` should raise.
  - Decide whether a platform-independent alignment of 8 is the intended model. If it is, the fix
    is to compare only itemsize, or to accept numpy alignment <= jix alignment
    (jix-py/src/dtype.rs:78-86).
  - The derive would then need offsets computed from `F::DTYPE.alignment()`, or it should reject
    such structs on i686.
  (concept-alignment)
- `REQUIRE_ALIGNED = true` targets (armv7, ppc64le): every aligned access is guarded on paper, but
  none of it has ever run. Cheap insurance would be a cross-compiled run of the existing
  unaligned-buffer tests under qemu (`--target armv7-unknown-linux-gnueabihf`).
  (concept-alignment)

### E. Uninitialized memory and unsafe contracts

- Reduction state init (ops/reduction.rs:803-806, `fold_multi_cell` init flag) assumes the row with
  `base_item_idx == 0` is visited first for each cell. That holds only if
  `NdIterUnordered::foreach_inner_1d` visits every outer dim in increasing order, which nobody has
  verified. If it does not, `assume_init_read` reads an uninitialized pooled state. (alloc-pool)
- F003 follow-up: after Scalar's typed `*item = self.data` writes into a caller's `&mut [u8]`
  (`to_ndarray_slice`), the padding bytes of the user's own buffer become uninitialized, so a later
  safe read by the user would be UB. Not Miri-tested. (helpers)
- `Plain::plain_ndarray_ptr` / `Plain::new` are unsafe, but their contracts say nothing about
  concurrent mutation, while `SendSyncPtr` makes Plain unconditionally Send + Sync. Should the
  contracts require that the data is not mutated concurrently? (helpers)

### F. Semantics and resources

- Cache-size-dependent results: the default block size comes from the building machine's L1d and
  the read window from its L2. So float `sum()` / `mean()` of the same data can differ in the last
  bits between hosts (e.g. 32K vs 48K L1d). Is cross-machine bitwise reproducibility a goal? If so,
  this is a questionable-semantics item for the reduction/params owners. (arrayvec)
- The Python cached `ReadContext` (jix-py/src/array.rs:163-176) keeps its BufferPool, and every
  pooled scratch buffer, alive for the Python Array's lifetime. If any pull path allocates
  output-sized buffers, that memory is retained indefinitely. Not verified. (alloc-pool)

# jix bug hunt - run 2026-10-23-batch3

- Date: 2026-10-23. The code is at commit `2bdbcd4`, after #87 (c009869). #87 fixed K4, K7, K14, K16-K26, K28-K30 and K32-K41, and documented K31 as intended. K3, K8, K15 and K27 are still open.
- Scope: 8 hunters. There were 6 module units, 1 concept unit and 1 dynamic unit (listed in section 1).
- Process: Hunt, then Dedup, then Verify, then Report. Each finding got one adversarial verifier, which tried to refute it and then built a repro. Severity follows the table in `bug-hunt/PRIMER.md`.
- Layout of this directory:
  - `findings/<ID>.md`: one write-up for every confirmed finding (23). F023 has none because it was judged intended.
  - `repros/`:
    - 20 Rust repros (`repro_2026_10_23_batch3__fNNN.rs`);
    - 13 Python repros (`FNNN.py`);
    - `F013_internal.rs` and `F013_internal_bin.rs`, which are crate-internal (see "Re-running a repro").
  - `scratch/`: hunter and verifier experiments, Miri logs, the prepared F013 scratch crate (`scratch/crate-F013`) and the F022 layout probe.
- Stale paths in the write-ups:
  - They cite Rust repros as `jix/tests/repro_2026_10_23_batch3__fNNN.rs`, where they lived during the run. They have since moved to `repros/`.
  - Hunter scratch tests (`jix/tests/hunt_2026_10_23_batch3__*.rs`) mentioned in write-ups and coverage notes were deleted.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 23 |
| plausible | 0 |
| refuted | 0 |
| intended | 1 |
| unverified (not sent to a verifier) | 0 |

| severity (confirmed + plausible) | count | IDs |
|----------|------:|-----|
| critical | 0 | - |
| high | 4 | F001, F002, F003, F004 |
| medium | 6 | F005, F006, F007, F008, F009, F010 |
| low | 13 | F011, F012, F013, F014, F015, F016, F017, F018, F019, F020, F021, F022, F024 |

F023 (low, intended) is not counted above.

By category (confirmed): questionable-semantics 6, wrong-result 4, unsound-api 3, memory-safety 2, error-handling 2, resource 2, panic 2, overflow 1, unsafe-hygiene 1.

Eight hunters ran. Hunters that failed: `[]`.

| hunter | kind | raw reports | findings |
|--------|------|------------:|----------|
| types-dtype-layout | module | 4 | F001, F011, F012 (two reports) |
| storage-compact-read-path | module | 3 | F002, F003, F005 |
| util-nd-copier | module | 1 | F013 |
| codec-archive-archive-io | module | 8 | F005, F006, F007, F008, F014, F015, F016, F022 |
| storage-core-buf-contract | module | 3 | F009, F017, F018 |
| elementwise-ops-op2-arith-div-pow | module | 3 | F010, F019, F023 |
| concept-push-pull | concept | 2 | F020, F024 |
| dynamic-rust-differential | dynamic | 3 | F004 (two reports), F021 |

Dedup merged 27 raw reports into 24 findings:
- F005 was reported by two hunters.
- F004 and F012 each merge two reports from one hunter.
- No report was dropped as a `KNOWN.md` duplicate.
- Four findings are new triggers that earlier fixes or entries did not cover:
  - F005: the part of K5's suggested fix that was never applied.
  - F003: the archive-load half of K15. K15 covers numpy ingest only.
  - F009: the `read_data` part of the K25 trust boundary.
  - F017: the aliasing check that the K14 fix left out.

Reachability:
- Python (15): F001, F002, F003, F005, F006, F007, F008, F010, F011, F014, F015, F016, F019, F021, F022.
  - F015 only affects the 32-bit wheels.
  - F022 is latent.
- Rust only (8): F004, F009, F012, F013, F017, F018, F020, F024.
  - F013 is crate-internal and latent.
  - F020 is 32-bit only.
- Eight findings are triggered by an untrusted `.jix` file: F001, F003, F005, F006, F007, F011, F014 and F015.

Headlines:
- **Untrusted archives are still the main exposure.**
  - F001: a struct dtype whose layout check wraps in u16 is accepted, and a field read then goes up to 64 KiB out of bounds (SIGSEGV in release).
  - F003: bool bytes other than 0/1 pass Strict validation and reach typed Rust `bool` (UB, confirmed by Miri).
  - F005, F006, F007: a tiny malformed file panics or aborts the process, because sizes are allocated before they are validated.
- **F002: silent wrong values on valid archives.** Reading with documented `params` that leave a block dim unfixed re-tunes the file's block shape. `write_to` then saves the corruption into a file that passes Strict validation.
- **F004: reversed ndarray views in the Rust API.** Negative strides wrap. Release builds return wrong windows and break `from_raw_parts` / `get_unchecked` preconditions.
- **Rust-side trust boundaries are still open:**
  - F009: a safe user `ArrayStorage` can cause out-of-bounds and uninitialized reads.
  - F017: pull buffers with aliasing strides are accepted as `out`.
  - F018: the `StridedBuf::from_raw_parts(_mut)` safety docs are incomplete.
- **F010: float ** int uses `powi`.** Errors reach thousands of ulps, and subnormal results are flushed to 0 silently. The common `arr ** 3` in Python is affected.

Trial2 follow-ups settled by this run:
- 6.B (bool bytes in archives) -> F003.
- 6.A (trust in `read_data` buffers) -> F009, shown with a native SIGSEGV and with Miri.
- 6.C (archive casts after K5):
  - F015 covers the 32-bit truncation of section sizes and message lengths.
  - The u64 -> u32 truncation of header `block_shape` / `block_size` was judged harmless, because the truncated geometry stays self-consistent.
- 6.D (a huge broadcast pull materializes the whole region) -> F024 (low).
- Fixes re-checked by hunters and holding:
  - K4: buffers handed to user `Read` impls are zero-filled.
  - K5: the mmap reader is bounded by the mapping. F005 is the leftover in the shared check.
  - K14: the extent of `out` is checked, and the original trigger now errors.
  - K17: the decode length is checked before the filters run.
  - K25: no `assert_unchecked` remains in `Op2::shape`.
  - K16: on 32-bit, `check_region_size` runs inside `check_out_buf`.
  - The 2^30-byte block cap in `tune` (the K18 / K24 fixes) holds, except through F002's drift, where reads fail before any large allocation.

## 2. Findings

All findings in the table are confirmed. They are sorted by severity, with ties in ID order.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| F001 | high | confirmed | memory-safety | Struct layout validation sums field ends in unchecked u16; a field list past 65535 wraps and validates against a tiny itemsize, so SubDtype reads up to 64 KiB out of bounds (crafted .jix -> SIGSEGV in release; also from safe Rust) | `jix/src/dtype.rs:673` | [F001](findings/F001.md) |
| F002 | high | confirmed | wrong-result | Archive read keeps the caller's `block_shape_fixed_dims`, so `tune` rescales the file's block shape and blocks are decoded with the wrong geometry (silently wrong values, panic, or every read failing; `write_to` then persists a corrupt Strict-valid file) | `jix/src/archive/array.rs:537` | [F002](findings/F002.md) |
| F003 | high | confirmed | memory-safety | A Strict-valid .jix bool array (or struct with a bool field) whose bytes are not 0/1 decodes into typed Rust `bool`: UB from safe code (Miri), and where/cast/not/sum/any disagree | `jix/src/storage/block.rs:201` | [F003](findings/F003.md) |
| F004 | high | confirmed | wrong-result | Safe ndarray Plain/compact constructors accept negative strides: `s as usize * size_of::<T>()` wraps; debug panics, release returns a wrong window or violates `from_raw_parts` / `get_unchecked` preconditions; the owned variant panics on `unwrap` | `jix/src/storage/plain.rs:289` | [F004](findings/F004.md) |
| F005 | medium | confirmed | error-handling | `check_section_bounds` adds section offset + size unchecked: a crafted TOC panics (debug add overflow; release capacity overflow in `Owned::read_section`) instead of returning InvalidArchive, also from default Python `read_array` | `jix/src/archive/common.rs:264` | [F005](findings/F005.md) |
| F006 | medium | confirmed | resource | Archive message length (uleb128) is allocated and zero-filled before it is checked against the archive length; an 11-byte file aborts the process (Rust and Python) | `jix/src/archive/common.rs:210` | [F006](findings/F006.md) |
| F007 | medium | confirmed | resource | With `len=None` (the Python default for file-like readers), `Owned::read_section` allocates and zero-fills the untrusted TOC section size before reading: a 106-byte BytesIO aborts the interpreter, 1 GiB is committed first | `jix/src/archive/block.rs:433` | [F007](findings/F007.md) |
| F008 | medium | confirmed | wrong-result | Writing an archive into an append-mode (O_APPEND) file or file object returns Ok but leaves the TOC as zeros, so the archive cannot be read back | `jix/src/archive/block.rs:231` | [F008](findings/F008.md) |
| F009 | medium | confirmed | unsound-api | `ArrayStorage` is a safe, unsealed trait, but jix's unsafe code trusts its `read_data` buffers: a user storage with no `unsafe` causes out-of-bounds reads (pull) and uninitialized reads (`to_ndarray_sub` `assume_init`, push) | `jix/src/storage/core_trait.rs:134` | [F009](findings/F009.md) |
| F010 | medium | confirmed | wrong-result | Float base ** integer exponent uses `powi`: error grows with \|n\| (tens to thousands of ulps) and representable subnormal results are flushed to 0, silently | `jix/src/ops/op2.rs:1201` | [F010](findings/F010.md) |
| F011 | low | confirmed | questionable-semantics | `Dtype::new_struct` (and so the archive reader) accepts duplicate struct field names that `from_fields` rejects; the second field cannot be selected, and in Python the dtype is unusable | `jix/src/dtype.rs:606` | [F011](findings/F011.md) |
| F012 | low | confirmed | error-handling | Derived structs with a `[T; 0]` field, or a zero-sized array field whose shape product exceeds u16, write to .jix but cannot be read back; `Dtype::set_shape` panics (debug) on such dtypes | `jix/src/archive/schema/conversions.rs:125` | [F012](findings/F012.md) |
| F013 | low | confirmed | unsound-api | `NdCopier::copy` trusts that its `dtype` matches the one given to `new`; scalar routes only `debug_assert` it, so a mismatch reads and writes past both spans in release (latent, crate-internal) | `jix/src/util/nd_copy.rs:89` | [F013](findings/F013.md) |
| F014 | low | confirmed | overflow | `read_uleb128` has no length cap: an 11-byte length prefix panics in debug (shift overflow) and is silently accepted in release | `jix/src/archive/common.rs:184` | [F014](findings/F014.md) |
| F015 | low | confirmed | panic | 32-bit: section sizes and uleb128 message lengths are truncated with `as usize`; Owned reads panic and a malformed archive loads silently, where 64-bit returns Err | `jix/src/archive/block.rs:430` | [F015](findings/F015.md) |
| F016 | low | confirmed | questionable-semantics | The `write_array` docstring names `gzip.GzipFile` as a supported writer, but the writer seeks backwards in finalize, so every `write_array` to a GzipFile raises "Negative seek in write mode" | `jix-py/src/archive.rs:162` | [F016](findings/F016.md) |
| F017 | low | confirmed | questionable-semantics | Writable pull results with aliasing 0 strides (Broadcast pull, pipeline pull) break the `from_pool` / `with_strides` no-aliasing contract and are accepted back as `out` by `check_out_buf`, so push reads return Ok with the region not in `out` | `jix/src/ops/shape_ops/broadcast.rs:156` | [F017](findings/F017.md) |
| F018 | low | confirmed | unsound-api | `StridedBuf::from_raw_parts(_mut)` Safety sections omit non-null (even for an empty region) and exclusive access / no overlap with any input: calls meeting the documented contract are UB | `jix/src/storage/buf.rs:84` | [F018](findings/F018.md) |
| F019 | low | confirmed | questionable-semantics | `jix.power(x, 2)` / `x ** 2` is routed to Square and silently drops the float Overflow report that the `jix.power` docstring promises (f16/f32/f64) | `jix-py/src/ops/op2.rs:688` | [F019](findings/F019.md) |
| F020 | low | confirmed | panic | 32-bit: `check_out_buf` runs `check_region_size` (`r.end - r.start`) before the index is validated; a reversed range passed to `read_data` on nine op storages panics (debug) or returns InvalidShapeOperation instead of InvalidIndex (release) | `jix/src/error.rs:278` | [F020](findings/F020.md) |
| F021 | low | confirmed | questionable-semantics | An empty float/complex sum returns -0.0 (sign bit set) instead of +0.0 (f16/f32/f64; complex gives (-0.0, +0.0)) | `jix/src/ops/reduction.rs:1483` | [F021](findings/F021.md) |
| F022 | low | confirmed | unsafe-hygiene | `Owned::read_section` transmutes `Vec<MaybeUninit<T>>` to `Vec<T>`, relying on a Vec layout that std does not guarantee (latent; no current rustc config breaks it) | `jix/src/archive/block.rs:439` | [F022](findings/F022.md) |
| F024 | low | confirmed | questionable-semantics | `Array::to_ndarray_buf` pull mode copies the whole region for large reads of lazy views that `read_data(None)` returns in O(1); a huge broadcast aborts | `jix/src/array.rs:1034` | [F024](findings/F024.md) |

### Bugs

**F001** (high; Rust, and Python through a crafted .jix).
- `is_aligned_struct`, `is_packed_struct` and `from_fields` add field ends into a u16 unchecked. In release, a field list that runs past 65535 wraps and is accepted with a tiny itemsize. Nothing downstream re-checks the field bounds: not `from_proto`, and not `SubDtype::new`.
- Result:
  - `dtype_sub_field` reads up to 64 KiB past the buffer (SIGSEGV in release). Debug builds panic instead of returning an error.
  - Safe Rust reaches the same read through `Dtype::from_fields` plus a safe user storage.
- Fix: do the layout arithmetic checked or in a wider type in all three validators. Optionally, also bound offset + size in `SubDtype::new`.

**F002** (high; Rust and Python).
- `read_from` sets only `block_shape` and passes the caller's `block_shape_fixed_dims` / `block_size` to `tune`. `tune` rescales the file's block shape, and `read_data` then decodes the stored blocks with that wrong geometry.
- Result:
  - silently permuted values (3584 of 4096 wrong in the repro);
  - or `InvalidBufferSize` on every read;
  - or an index-out-of-bounds panic.
  `write_to` then persists the drifted header into a file that passes Strict validation.
- Fix: in `read_from` the file geometry must win (force all dims fixed, or reject the params). Also have `new` check the spec against the `BlockTable`.

**F003** (high; Rust and Python).
- Archive load never validates decoded values. A bool array, or a struct with a bool field, whose stored bytes are not 0/1 passes Strict validation and is read as typed Rust `bool`: UB from safe code (Miri). `where`, `cast`, `not`, `sum` and `any` disagree about the same bytes.
- This is the archive half of K15, which covers numpy ingest only.
- Fix: normalize or reject bool bytes in `BlockTable::read_block` for dtypes that contain a bool, in both Strict and Minimal mode.

**F004** (high; Rust only).
- The ndarray constructors (`plain_ndarray_ref(_with)` / `_view`, `compact_ndarray`) convert negative strides with a wrapping `as usize`. The owned variant unwraps a `TryFromIntError`.
- Result:
  - Debug builds panic.
  - Release builds return a wrong window when the wrapped span is exactly 0 (reversed 1-d pairs, two rows of a flipud matrix). Other spans violate `from_raw_parts` / `get_unchecked` preconditions.
  - jix-py already rejects negative strides; the Rust side does not.
- Fix: reject or normalize negative strides in the three constructors, and treat length-1 axes as stride 0. Make Plain's empty-region shortcut test the shape, not `src_span == 0`.

**F005** (medium; Rust, and Python `read_array` with default arguments).
- `check_section_bounds` computes offset + size with a plain `+`.
  - Debug builds panic on the add.
  - Release builds wrap, pass the check, and then `Owned::read_section` panics with "capacity overflow".
  - Python gets a `PanicException` (a `BaseException`, so `except Exception` misses it). The mmap reader errors cleanly in release.
- This is the uncovered remainder of K5's suggested fix.
- Fix: `checked_add` in `check_section_bounds`.

**F006** (medium; Rust and Python).
- `try_read_message` resizes a buffer to the untrusted uleb128 message length before comparing it with the bytes left in the archive. Every entry point and both validation modes go through this path.
- Result: a file of about 10 bytes aborts the process (uncatchable), panics with capacity overflow, or zero-fills 256 MiB before the EOF error comes back.
- Fix:
  - Reject `length > reader.limit()` before reading.
  - For `len=None`, add a header-size cap or an incremental read.

**F007** (medium; Rust `read_from_reader(.., None, ..)` and Python `read_array(file_like)` with default arguments).
- With `len=None`, `Owned::read_section` allocates and zero-fills a Vec of the TOC section size before reading any of it.
- Result: a 106-byte BytesIO aborts the interpreter, and a declared 1 GiB section commits 1 GiB before the error. With `len` given, the same file gives a clean error.
- Fix:
  - Do a bounded read (`take` + `read_to_end`, or `try_reserve`).
  - Check the blocks_loc size against `nblocks` first.
  - Optionally, have Python default `len` to the stream length.

**F008** (medium; Rust and Python).
- The archive writer patches the block locations and the TOC with seek-then-write. O_APPEND sends those writes to EOF, so `write_to` / `write_array` return success but leave a zero TOC.
- Result:
  - The archive cannot be read back ("block locations length 0 ...").
  - In Python, `f.tell()` after the write is short of EOF, so offsets recorded for packed arrays are wrong.
  - The failure is loud on read; no wrong values are returned.
- Fix: check the stream position after the patches and return an error. Document that the writer must not be in append mode.

**F009** (medium; Rust, through a safe user storage that breaks the documented contract).
- `ArrayStorage` is safe and unsealed, yet unsafe code relies on what its `read_data` returns:
  - The default `read_as_elementwise_pipeline` wraps a pull buffer in an unsafe operand without a span check.
  - `to_ndarray_sub` calls `assume_init` without checking that the push wrote anything.
- Result: out-of-bounds reads (SIGSEGV at 2^22 elements) and uninitialized reads (Miri), from code with no `unsafe`. This is the same boundary as K25 and K4.
- Fix: make the trait `unsafe` (or sealed) with a Safety section, or validate pull spans and zero-fill the push destination.

**F010** (medium; Rust and Python).
- Float base with an integer exponent uses `powi`.
- Result:
  - The error grows with |n|: 36 ulps at n=50, about 790 at n=1000, thousands at n=10000. numpy stays at 1 ulp.
  - Subnormal results of negative exponents come back as 0, with no status bit.
  - `x ** -1074` and `x ** -1074.0` disagree.
- Python's `arr ** 3` takes this path.
- Fix: use `powf` on the exactly converted exponent (f64 for i32), or keep a fast path only for small |n|.

**F012** (low; Rust only).
- The `[T; N]` `Dtyped` impl accepts N == 0, and zero-sized-T shapes whose product exceeds u16.
  - Derived structs with such fields can be written to .jix, but `from_proto` refuses to read them back.
  - `Dtype::set_shape` uses an unchecked product, so it panics in debug. In release, a 256x256 shape returns a misleading error.
- Fix: const-assert N > 0 and that the product fits, or validate before writing. Use a checked product in `set_shape`.

**F013** (low; crate-internal, latent).
- `NdCopier::copy` takes its own `dtype` argument, separate from the one given to `new`. The scalar routes only `debug_assert` that the two match, and size every byte count from `new`'s type. On a mismatch, release builds read and write 4 bytes past both documented spans.
- No caller mismatches today, but `Staging` passes the dtype again on every call, which invites the slip.
- Fix: store the itemsize in the copier and drop `copy()`'s `dtype` parameter. Short of that, promote the itemsize assert (not the alignment one, because of K8).

**F014** (low; Rust and Python).
- `read_uleb128` has no 10-byte cap. An 11th byte overflows the shift: debug panics (`PanicException` in the debug extension), and release silently accepts overlong prefixes that prost itself rejects.
- Fix: cap at 10 bytes and validate the 10th byte, as prost's `decode_varint` does.

**F015** (low; 32-bit only, including the x86 and armv7 wheels).
- `Owned::read_section` and `try_read_message` truncate u64 sizes with `as usize`.
- Result: a malformed archive panics (the release `assert_eq` at common.rs:240, or capacity overflow), or loads silently where 64-bit returns Err. No UB.
- Fix: `usize::try_from` with an `isize::MAX` bound, mapped to InvalidArchive.

**F018** (low; Rust, public unsafe API).
- The Safety sections of `StridedBuf::from_raw_parts(_mut)` omit two requirements:
  - a non-null pointer, even for an empty region;
  - exclusive access for `'a`, including no overlap between `out` and any input.
- Calls that meet the documented contract are therefore UB: debug precondition aborts, and a Miri Stacked Borrows error for an in-place negate. This is the Rust-side counterpart of K31.
- Fix: use a dangling pointer for a zero span. Document the exclusivity and overlap rules on `from_raw_parts_mut`, `to_ndarray_buf`, `read_data` and `plain_ndarray_ptr`.

**F020** (low; 32-bit, direct `read_data` calls only).
- `check_out_buf` runs `check_region_size` (`end - start`) before the ranges are validated. Nine op storages call it before checking the index.
- Result: a reversed range panics on i686 debug, and gives InvalidShapeOperation instead of InvalidIndex on i686 release. The `Array`-level entry points and Python validate first, so they are not affected.
- Fix: call `check_get_range` first inside `check_out_buf`, or use `checked_sub` in `check_region_size`.

**F022** (low; latent, runs on every non-mmap read).
- `Owned::read_section` transmutes `Vec<MaybeUninit<T>>` to `Vec<T>`. That relies on a `Vec` layout std explicitly does not guarantee; the transmute docs use this exact case as their example.
- No current toolchain breaks it, `-Zrandomize-layout` included, and Miri is clean.
- Fix: `ManuallyDrop` + `Vec::from_raw_parts`, which works on MSRV 1.89.

### Questionable semantics

**F011** (low; Rust, and Python through a crafted .jix).
- `new_struct`, which the archive reader uses, does not reject duplicate field names, while `from_fields` does.
- Result: the second field can never be selected. In Python, `arr.dtype` and `arr.numpy()` raise ValueError at first use instead of InvalidArchive at load.
- Fix: share the duplicate-name check with `new_struct`.

**F016** (low; Python).
- The `write_array` docstring lists `gzip.GzipFile` as a supported writer. The writer seeks backwards in finalize, so every write to a GzipFile raises "Negative seek in write mode". The error is clean, with no corruption.
- Fix: remove it from the docstring and require random-access seek, or change the format to use a trailer. Add a test either way.

**F017** (low; Rust only).
- Broadcast pull and pipeline pull hand out writable buffers with 0 strides on dims longer than 1. That breaks the `from_pool` / `with_strides` no-aliasing contract.
- `check_out_buf` was hardened for extent (K14) but not for aliasing, so it accepts these buffers as `out`. Pushes then return Ok with meaningless contents.
- Miri finds no UB today. Python already rejects such `out` buffers.
- Fix: reject aliasing `out` in `check_out_buf`, and/or make such pull results read-only.

**F019** (low; Python).
- `power(x, 2)`, `x ** 2`, `power(x, 2.0)` and `power(x, np.int32(2))` switch to Square, which has no float status.
- Result: the Overflow report that the docstring promises is lost. `power(x, 3)` and numpy both report it. Values are identical.
- Fix: limit the shortcut to integer pairs, or give Square a float status kernel.

**F021** (low; Rust and Python).
- An empty float or complex sum returns the fold identity -0.0 instead of +0.0. The sign shows in `1/x`, `signbit` and printing. The docs say "returns 0", and numpy gives +0.0.
- Fix: return +0.0 from `init_state(None)`.

**F024** (low; Rust only).
- `to_ndarray_buf` pull mode bypasses splitting only for `plain_read` storages. Large pulls of broadcast, permute, slice and pipeline views therefore copy the whole region, and a huge broadcast aborts. `read_data(None)` returns an O(1) view for the same region. Values are correct.
- Fix: propagate a "pull is a view" flag, or collapse broadcast dims in the slow path. At least document the behavior.

### Re-running a repro

The Rust repros are in `repros/`. To run one:
1. Copy it back to `jix/tests/`.
2. Run `cargo test --manifest-path jix/Cargo.toml --all-features --test <name>` from the repo root, where `<name>` is the file name without `.rs`.
3. Delete the copy afterwards. `jix/tests/repro_*.rs` is gitignored, so a leftover copy does not show in `git status`.

Example:

    cp bug-hunt/runs/2026-10-23-batch3/repros/repro_2026_10_23_batch3__f002.rs jix/tests/
    cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_23_batch3__f002 -- --nocapture
    rm jix/tests/repro_2026_10_23_batch3__f002.rs

Every repro asserts the correct behavior, so it fails, panics or crashes while its bug is present. There are three exceptions:
- f022 passes, because the issue is latent.
- f020 passes on 64-bit.
- Control tests pass.

Per-repro notes:
- **f001**:
  - Debug: the 6 non-control tests fail with overflow panics.
  - `--release`: run the tests one at a time by name.
    - `crafted_archive_field_read` crashes with SIGSEGV.
    - `safe_api_relabel_field_read` reads out of bounds silently.
- **f002, f008, f010, f011, f021**: plain test failures.
- **f003**:
  - Fails natively.
  - For the UB report, run `python jix/run_miri.py --all-features --test repro_2026_10_23_batch3__f003 -- miri_cast_reads_invalid_bool`.
- **f004**:
  - Debug: all 5 tests fail.
  - `--release`: 3 tests fail; `reversed_full_read` and `compact_ndarray_reversed` pass by accident.
  - To see the precondition violations, use a separate target dir with `CARGO_PROFILE_DEV_OVERFLOW_CHECKS=false CARGO_PROFILE_TEST_OVERFLOW_CHECKS=false` and run each test alone.
- **f005**:
  - Use `--test-threads 1`.
  - Debug: 3 tests fail on the add overflow.
  - `--release`: the 2 heap-reader tests fail with capacity overflow; the mmap test passes.
- **f006**:
  - Use `RUST_BACKTRACE=0`.
  - The aborting case runs itself in a child process (an ignored helper test), so the harness survives.
  - All 3 tests fail in any profile.
- **f007**: the 2^44 cases abort the binary. Run each test alone: `-- --nocapture --test-threads 1 --exact <test>`.
- **f009**:
  - Run the tests one at a time by name.
  - `zz_pull_short_buffer_large_region_is_rejected` crashes with SIGSEGV.
  - Miri on the push test: `python jix/run_miri.py --all-features --test repro_2026_10_23_batch3__f009 -- push_ignored_out_is_rejected`.
- **f012**: the two `set_shape` tests rely on debug overflow checks.
- **f014**:
  - Debug: both tests panic.
  - `--release`: only `valid_archive_with_overlong_length_prefix_is_rejected` fails.
- **f015**:
  - 32-bit only. The bug tests are `#[ignore]`d on 64-bit.
  - Run with `bug-hunt/tools/cargo-i686` in place of `cargo` (it needs `bash bug-hunt/setup.sh`): `bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_23_batch3__f015 -- --nocapture --test-threads 1`.
- **f017**:
  - Use `--test-threads 1`; all 3 tests fail.
  - Miri reaches the same assertions and reports no UB.
- **f018**:
  - Run each test with `--exact <name>`. Three tests abort with SIGABRT natively.
  - `in_place_neg_4` passes natively; it needs `python jix/run_miri.py --all-features --test repro_2026_10_23_batch3__f018 -- --exact in_place_neg_4`.
- **f020**:
  - On x86_64, all 8 tests pass.
  - With `bug-hunt/tools/cargo-i686 ... -- --test-threads=1`, debug panics and `--release` returns the wrong error kind (7 tests fail).
- **f022**: passes natively and under Miri. The layout probe is in `scratch/f022-layout/`.
- **f024**:
  - Use `--test-threads=1`; the 4 non-ignored tests fail.
  - The ignored `pull_of_huge_broadcast` aborts; run it alone with `-- --ignored pull_of_huge_broadcast`.
- **F013** (crate-internal):
  - `scratch/crate-F013` is a copy of `jix/` that already has the `nd_copy_split_dtype` helper in `src/util/bench_util.rs` and the driver as `src/bin/f013.rs`. The driver is the same as `repros/F013_internal_bin.rs`.
  - Run with debug asserts off: `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false CARGO_TARGET_DIR=<scratch dir> cargo run --manifest-path bug-hunt/runs/2026-10-23-batch3/scratch/crate-F013/Cargo.toml --all-features --bin f013`. It exits 1.
  - Run it again without the assertions override to see the debug_assert panic.
  - `F013_internal.rs` is the inline-test form. It was not run, because a full `--lib` test build did not fit the shared disk.

Python repros cover F001, F002, F003, F005, F006, F007, F008, F010, F011, F014, F016, F019 and F021.
- First build the debug extension into `.venv`, with `maturin develop` in `jix-py/` or with `bash bug-hunt/setup.sh`.
- Then run, for example, `.venv/bin/python bug-hunt/runs/2026-10-23-batch3/repros/F002.py`. Each script exits non-zero while its bug is present.
- Per-script notes:
  - F001, F006 and F007 run their crashing cases in subprocesses and report exit statuses. Run F005 with `RUST_BACKTRACE=0`.
  - F001, F005 and F014 show debug-extension behavior (overflow panics). A release wheel behaves as in the write-ups:
    - F001 accepts the dtype and reads out of bounds;
    - F005 panics with capacity overflow;
    - F014 silently accepts the prefix.
  - F010 runs under `jix.errstate(all="raise")`, which raises nothing.

## 3. Unverified findings

None. Every deduplicated finding was sent to a verifier.

## 4. Refuted and intended

- **F023** (low / intended / questionable-semantics, `jix/src/ops/op2.rs:1173`): a u64 exponent in `power` promotes to float64, because the integer exponent slots stop at u32.
  - Why intended: it is deliberate and pinned by tests (`test_type_promotion.py:592-597`, `test_power_promotes_uint64_exponent_to_float64`). The docstring before #81 stated it, and it follows the general Safe-casting rule.
  - Python int and `np.uint32` exponents stay exact.
  - Leftover doc nit: the #81 docstring rewrite dropped the explicit u64-exponent sentence. Restate it, or add a (u64, u64) pair.

No finding was refuted or dropped as known.

### Close calls (judged not intended; record the maintainer's decision in KNOWN.md or a context doc)

- F004: the context doc lists "no negative strides by design" as intentional. The verifier ruled that this design requires the Rust boundary to reject them, as jix-py does.
- F007: "`len=None` skips section bounds checks" is listed as intended. The verifier ruled that this covers skipping the caller-length check, not sizing an up-front allocation from untrusted bytes.
- F009: depends on whether safe user `ArrayStorage` impls are a supported trust boundary (as with K25). It is rated medium, like K25.
- F010: the trait doc names `powi`, but the same doc says floats follow `powf`, and the Python docstring says "as numpy".
- F013, F022: latent today. They are kept because the unsafe contract or layout assumption is unstated or explicitly disclaimed.
- F016, F019: contradictions between docstring and behavior. Either can be resolved by changing the doc instead of the code.
- F024: the contract lets pull choose any strides, but the docs promise "the cheapest buffer it can". Kept at low.

### Verifier corrections to hunter claims

- F001:
  - The verifier added a path from safe Rust with no crafted file (`from_fields` plus a safe user storage).
  - The hunter's line numbers moved: sub_dtype.rs:135/149 are now :140/:154.
  - Through numpy, Python reaches only a debug panic; release wheels need a crafted .jix.
- F002: the hunter's side claim that drift bypasses the 2^30-byte block bound is true at load time, but has no further impact. Reads fail before any large allocation.
- F003: the hunter did not run Miri; the verifier did, and Miri reports UB in the bool -> i32 cast kernel.
- F004: the verifier ran a build with overflow checks off and UB checks on, which shows the `from_raw_parts` / `get_unchecked` violations. Physical accesses appear to stay inside the allocation, so the severity is not critical.
- F005: the release behavior (capacity overflow) was re-run. The 32-bit behavior was argued from the code only.
- F008: the verifier also found that Python `f.tell()` after the write is short of EOF (235 vs 267).
- F010: the hunter's f64 example `1.0123456789 ** 100` is 0 ulps off, not off. The finding stands on the other cases.
- F012: the duplicate `set_shape` report reproduces too. In release, 300x300 wraps to a nonzero value that happens to work, and 256x256 wraps to 0 and returns a misleading error.
- F013: the real path is the strided `inner_loop`, with a 4-byte tail overrun, not a 32-byte `copy_nonoverlapping`.
- F014: the hunter's 15-byte file is rejected in release only by accident. The verifier built an overlong but otherwise valid prefix, which loads Ok in release.
- F017: the hunter claimed the reduction's shared state sums both rows (36). It actually gives [26, 26], because row 1's fold re-initializes the state.
- F018: the side claim about stride < itemsize is only a wording gap (order-dependent values, no noalias UB). It is not part of the verdict.
- F020: storage-core-buf-contract noted this but did not run it. concept-push-pull reported it, and the verifier reproduced it on i686 in debug and release.
- F022: the hunter's claim that `-Zrandomize-layout` breaks it is wrong. The seed does not depend on T, as shown over seeds 1-7 with build-std.

### Hypotheses the hunters checked and ruled out

Dtype layout (types-dtype-layout):
- `Alignment::new` / `as_usize` transmute: sound.
- The `set_shape` assert at dtype.rs:867 cannot fire for any constructible dtype, and the zero-element current shape is guarded.
- `from_proto` calling `new_struct` and then `set_shape`: idempotent.
- `PartialEq` ignoring `is_aligned`: only cosmetic.
- Recursion depth: bounded by prost.
- numpy dtypes cannot make the validators wrap to a passing result.

Compact read path (storage-compact-read-path):
- Every unsafe block in compact.rs and storage/block.rs holds, including the direct-decode path, which was proven for all out layouts.
- The `assert_unchecked_eq` in block.rs holds for every constructor.
- `MmapData::as_ref`: sound.
- Double tune in `compact_with`: idempotent.
- About 2.5M randomized read cases pass on x86_64:
  - ndim 0-8;
  - pull, and push into C / F / padded / permuted / sub-tile / unaligned outs;
  - 7 dtypes and 4 filter chains.
  On i686 everything passes except u64, which trips the K8 assert.

NdCopier (util-nd-copier):
- The `assert_unchecked` routes and the ALIGNED choice: sound.
- `const_len LEN == len` always holds.
- Untyped broadcast doubling and `copy_untyped` routing: consistent.
- `bytes_offset` stays in bounds; the stride coalescing math cannot overflow.
- No caller passes overlapping src/dst.

Archive I/O (codec-archive-archive-io):
- Byte casts of Header, the Section array and BlockLocation: sound.
- `compressed.len() as u32`: unreachable, because of the 2^30 block cap.
- Strict rejects bad locations without overflow. Minimal gives clean panics, never UB.
- `base_offset` arithmetic is consistent across writer, reader and mmap, including an odd file offset.

Buffer contract (storage-core-buf-contract):
- The transmute in `read_data_and_map_strides` is sound in pull mode.
- `materialize_out_buf` overflow is covered on 32-bit and 64-bit.
- `is_contiguous(_aligned)` is exact.
- Forwarders, the `Any` `assert_unchecked` and the `MaybeCompact` unwrap are consistent.

Op2 arithmetic (elementwise-ops-op2-arith-div-pow):
- The integer div/rem family has no overflow; an exhaustive i8/u8 test of all 7 ops passes with overflow checks on.
- Float `div_rem_trunc` is exact below EXACT_LIMIT and within one ulp beyond it.
- Scalar and SIMD error classification agree.
- f16 + - * / do not double-round.
- Pow casts do not change classification.
- The 0.5 -> Sqrt shortcut matches numpy.

Push/pull (concept-push-pull):
- The push/pull contract holds at every `read_data` site.
- The differential harness (about 80 op chains x 60 regions, pull and push, untouched-byte checks) passes.
- 22 Python ops with `numpy(out=)` pass.

Dynamic (dynamic-rust-differential):
- More than 150k random op-chain seeds and the other 12 harnesses show no mismatches beyond F004 and F021.

### Observations the hunters deliberately did not report (intended or minor)

- Zero-sized fields at a shared offset make `from_fields` accept or reject depending on the order of the fields. It is contrived and only ever errors (types).
- Valid numpy struct dtypes whose names are listed out of offset order are rejected with a generic layout error. This is a limitation (types).
- `version.rs`: `VERSION_U64` is written but never read, and `parse_pre_release` maps several pre-release tags to the same value. Harmless while nobody decodes it (codec-archive).
- `read_byte` does not retry `ErrorKind::Interrupted` (codec-archive).
- 32-bit: `nblocks as usize` in the writer and builder truncates for more than 2^32 blocks. That ends in a panic, not UB (codec-archive, storage-compact).
- Header u64 fields are truncated to u32 before validation. The result is self-consistent (storage-compact, codec-archive).
- The DivideTrunc doc says an infinite dividend sets InvalidOperation, but inf / +-0 returns +-inf with no flag. That matches numpy and IEEE, so it is a doc fix only (elementwise-ops).
- An aliasing `out` in reductions gives garbage values only, with no UB (storage-core).
- The stale docs at nd_copy.rs:9-11 are intentional per the context doc (util-nd-copier).

## 5. Coverage

### Per hunter

- **types-dtype-layout**: all of dtype.rs:1-1198 (constructors, the three layout validators, `set_shape`, the `Dtyped` impls), plus contracts in util/arrayvec, archive conversions, jix-py dtype and sub_dtype.rs. Context suspicions S1, S7, S8 and S9 were settled.
- **storage-compact-read-path**: every line of storage/compact.rs and storage/block.rs, with each unsafe precondition re-derived. It also ran the 2.5M-case randomized read differential on x86_64 and i686.
- **util-nd-copier**: all of nd_copy.rs and nd_iter_unordered.rs and the Ptr helpers in util/mod.rs. The dtype and span were checked at every NdCopier call site, and 8 focus questions were settled.
- **codec-archive-archive-io**: every line of archive/common.rs, archive/block.rs, version.rs and mod.rs, plus storage/block.rs, the read entry points and the jix-py `read_array` / `write_array` / `PyReader` / `PyWriter`. Seven context suspicions were settled.
- **storage-core-buf-contract**: every line of core_trait.rs, buf.rs, storage/mod.rs, any.rs and info.rs, with contracts followed into about 20 `read_data` impls and the array front-end. The K14 fix was re-tested, and Miri ran on the findings.
- **elementwise-ops-op2-arith-div-pow**: every line of op2.rs:323-1594 (arithmetic, the div/rem families, Pow), plus scalar/traits_util.rs, simd.rs, the relevant tests and the fearless_simd bodies. It ran an exhaustive i8/u8 integer-op test.
- **concept-push-pull**: the `read_data` push/pull contract across every `ArrayStorage` impl and jix-py `numpy(out=)`. It ran the differential harness on x86_64 and i686, and 22 Python ops with `numpy(out=)`.
- **dynamic-rust-differential**: 13 randomized differential files. They covered:
  - random op chains (more than 150k seeds) through every read path;
  - other dtypes, reductions and the pipeline driver;
  - casts and bitwise ops against Rust;
  - huge lazy arrays up to 2^50 elements, also on i686;
  - struct fields.

### Skipped or only skimmed

- types-dtype-layout:
  - The inline tests in dtype.rs after line 1200, and the jix-macros derive.
  - S2-S6 were not re-examined: derive, i686 alignment (K8), bool validation (K15), error.rs and element_type.rs.
- storage-compact-read-path:
  - The nd_copy.rs internals; the hunter relied on the contract plus the differential test.
  - The NdIter core advance logic, and the params.rs heuristics beyond geometry.
  - The hunter did not run Miri; the F003 verifier did.
- util-nd-copier:
  - reduction.rs:660-800 (the `bytes_offset` caller).
  - The pipeline inner loops outside Staging, and the `strides_for_layout_order` producers.
  - The other util-nd.md suspicions (iter/nd.rs:61, block.rs, map.rs, arr_sequence) and util-memory.md were not settled.
- codec-archive-archive-io:
  - The header validation arithmetic in archive/array.rs and the schema/conversions.rs dtype checks (the array-format-dtype unit).
  - The codec filter internals.
  - The params-leak and bool-byte suspicions, which belong to other units and became F002 and F003.
- storage-core-buf-contract:
  - Plain and Scalar internals beyond the `read_data` entry.
  - The Flip / Repeat / Tile / Reshape copy math, the NdCopier internals and the pipeline `to_buf_impl` iteration.
  - Not re-examined: plain.rs negative strides (later F004), padding (K3), ElementType S6, and 32-bit `to_ndarray_slice` (K16).
  - No i686 or release runs.
- elementwise-ops-op2-arith-div-pow:
  - The NEON and wasm SIMD paths (trunc, mul_add_precise, copysign, select) were reviewed by reading only.
  - The bool-validity and cast.rs:59 suspicions are outside the unit.
- concept-push-pull:
  - The pipeline internals beyond `out` handling (read_bulk kernels, staging cache, SIMD).
  - REQUIRE_ALIGNED (armv7) paths were reasoned about, not run.
  - No Miri.
  - The NdCopier internals, and archive/mmap decode beyond Compact's read path.
- dynamic-rust-differential:
  - Python only for the empty-sum check.
  - Release builds only for the negative-stride repros; i686 only for files 2 and 9; no Miri (per its task).
  - Not tested:
    - `read_from_file_mmap` and `read_from_file_section`;
    - ReadContext error modes and status propagation;
    - f16 / complex arithmetic beyond copies and reductions;
    - Transmute, `out=` aliasing and multi-threaded reads;
    - SIMD levels other than the host's;
    - element-wise op semantics (div / pow / rem / sqrt).
- Across the run:
  - No 32-bit CPython was available.
  - armv7 / ppc64le (REQUIRE_ALIGNED) and aarch64 / NEON were never executed.
  - Python results come from the debug extension. Release-wheel behavior is inferred from Rust `--release` runs.

### Housekeeping

- The 20 Rust repros were moved from `jix/tests/` to `repros/`, keeping their names.
- The 36 hunter scratch files `jix/tests/hunt_2026_10_23_batch3__*.rs` were deleted. No other file in `jix/tests/` was touched.
- Python repros and scratch experiments were already under this run directory. No tracked file was modified.

## 6. Open questions from hunters

### A. Trust boundaries and unsafe contracts

- Should `ArrayStorage` become an `unsafe trait` (or be sealed)? Its Safety section would state:
  - shape, dtype and spec are stable;
  - pull buffers cover the region, initialized and valid for the dtype;
  - push writes every element of the region into `out`.

  The alternative is to have every consumer re-validate pull spans and zero-fill `to_ndarray_sub`'s destination. This covers F009, and every NdCopier caller that copies from a child pull buffer: Tile case C, Repeat, Flip, SubDtype, Broadcast. (storage-core-buf-contract, util-nd-copier)
- `StridedBuf` has no length accessor and `check_out_buf` is `pub(crate)`, so a user storage cannot validate a direct `out`. Should one of them be exposed? (storage-core-buf-contract)
- F017 fix choice: make 0-stride pull results read-only, or have `check_out_buf` reject `out` strides that alias on dims longer than 1? The second also covers misuse of `from_raw_parts_mut`. Or should the documented contract be relaxed to match the code? (storage-core-buf-contract, concept-push-pull)
- `PoolBuf::as_slice` and `Staging::write_to` build `&[u8]` over the whole pool capacity, which may include bytes never written. Miri accepts this today. Should the pool hand out only the requested prefix? (util-nd-copier)
- Push-mode padding (K3 family): `compact_into_builder_dyn` hands uninitialized memory to push writers that store whole typed values, so struct padding stays uninitialized and is then encoded. Does K3 cover this, or is it a separate entry? Miri would decide. (concept-push-pull)
- Python `numpy(out=)` rejects zero and negative strides but not self-overlapping non-zero strides. Reject them or document them? (concept-push-pull)
- `read_data_and_map_strides` discards the inner push result and assumes the inner wrote into `inner_out`. This is covered by the F009 decision. (concept-push-pull)
- Should `SubDtype::new` also check offset + field size <= parent itemsize, as defense in depth? `new_struct_borrowed_unchecked`, which the derive uses, bypasses validation entirely. (types-dtype-layout)
- Do any consumers besides SubDtype trust struct field offsets or sizes without re-checking them? `dtype_to_numpy` would hand numpy out-of-bounds offsets, which numpy should reject. (types-dtype-layout)
- F004 policy: reject negative strides with an Err, or normalize them (move the base pointer and add a lazy flip)? The same applies to `plain_ndarray_ptr`'s contract. Also open:
  - Does compaction from a reversed view ever hit the zero-span window and corrupt compressed blocks?
  - Does any release path compute an address outside the allocation?
  (dynamic-rust-differential)
- `StridedBuf::from_raw_parts(_mut)` panics on more than 8 strides, a limit the safety docs do not mention. (dynamic-rust-differential)

### B. Untrusted archives

- F001 fix: checked `ceil_to_multiple` / adds in all three layout walkers. (types-dtype-layout)
- F002 fix choice: should `read_from` ignore the geometry params (force all fixed dims and keep the caller's bitmap only as a propagation hint)? Or should `ArrayBlockTableStorageBase::new` assert that the tuned spec equals the BlockTable geometry? The latter needs a new argument. (storage-compact-read-path)
- F003 fix point:
  - validate bool bytes after decode in `read_block` for any dtype that contains a bool, at O(block) per read; or
  - treat bool leaves as u8 and normalize in the kernels, matching whatever K15's fix chooses.
  (storage-compact-read-path)
- Should `ArchiveReader` cap message sizes, check `length <= reader.limit()`, and read sections incrementally (`take` + `read_to_end` / `try_reserve`)? That would close F006, F007 and F015 together. (codec-archive-archive-io)
- Should Python `read_array` default `len` for seekable file-like objects to the stream length, so the default call gets section bounds checks? (codec-archive-archive-io)
- Forward compatibility: `lib_version_semver` is never checked on read, and prost ignores unknown fields. Should readers reject files whose major/minor version is newer than their own? (codec-archive-archive-io)
- A failed write leaves a partial archive with a zero TOC in the stream; with `write_array(append=True)`, that junk stays in the user's file. Truncate back to `base_offset` on error, or document it? (codec-archive-archive-io)
- Header `block_shape` of 2^32+8 is accepted as 8, which is self-consistent. Should such headers be rejected? (storage-compact-read-path)
- Minimal validation on 32-bit: `offset as usize` in `read_block` can decode a different in-bounds block instead of panicking. That is within Minimal's "trusted file" contract, but it is a silent wrong read. (codec-archive-archive-io)

### C. Integer width and 32-bit targets

- `nblocks as usize` in `OwnedBlockTableBuilder::start` and `BlockArchiveWriter::start` truncates for more than 2^32 blocks; an explicit error would be cleaner. (storage-compact-read-path, codec-archive-archive-io)
- K8 side effect: on i686 debug builds, the `debug_assert` at nd_copy.rs:90 fires for every 64-bit scalar copy (u64, [u64;2], i64 reduction outputs). Should K8's notes say that 32-bit debug test runs panic there, and will K8's fix cover it? (storage-compact-read-path, concept-push-pull)
- `NdIterUnordered` coalescing multiplies `group_len` unchecked. For zero-itemsize dtypes broadcast to a huge shape, this can exceed usize on 64-bit (debug panic, release wrap). Confirm that no consumer uses the wrapped `inner_len` with a nonzero itemsize. (util-nd-copier)
- The `[T; N]` impl asserts `N < Itemsize::MAX`, which looks off by one. A `[u8; 65535]` dtype loads from archives or numpy but cannot be `into_typed::<[u8; 65535]>()`. Not compile-tested. (types-dtype-layout)
- storage-core-buf-contract's open item about the unverified 32-bit reversed-range panic is now settled as F020.

### D. Numerics, platforms and API semantics

- F010: is `powi` an intended speed trade-off? A fix could use `powf` on the f64-converted exponent, and for f32 evaluate in f64 and round once, as numpy does. Accuracy vs numpy parity needs a decision. (elementwise-ops-op2-arith-div-pow)
- Float Add / Sub / Mul / Square never report Overflow, while Div, Pow and Exp do. Is that deliberate? It is not listed under intentional behaviors, and F019 is one symptom. (elementwise-ops-op2-arith-div-pow)
- The NEON (aarch64) and AVX-512 implementations of trunc / mul_add_precise / copysign used by `div_rem_trunc_v` are tested only on whatever SIMD levels the CI CPU has; aarch64 wheels are never tested. (elementwise-ops-op2-arith-div-pow)
- An explicit `ArrayParams::block_shape` larger than an array dim is rejected with InvalidArgument, even on length-0/1 dims. That happened for 1977 of 3000 random shapes. The doc says the shape is clamped at the array edges. Should it clamp to max(shape, 1) instead? (dynamic-rust-differential)

# jix bug hunt - consolidated report of the 2026-10/11 scans

- Date: 2026-11-17. This report unions the 14 runs of the scans (2026-10-21 to 2026-11-17) and keeps only the issues that are still open. Every item was re-checked against main at `c705a54` (#107) on 2026-11-16/17, by re-running its repro (Rust repros ported to the current API where #94-#106 broke them, i686 for the 32-bit items) or, for K111, K116 and K152, from code that #107 did not change (K111 was last run under Miri at `1169965`). All 36 still reproduce (K199 in a simulation of i686, K197 deterministically only through its Rust repro), or (K152, plausible) are still argued from unchanged code.
- Scope: 14 runs over the code of main after #84 to #107. The first six (pilot, trial2, batch3-batch6, after #84-#92) ran each of the 60 hunter units once; the six 6-unit batches of batch 7 (after #105) re-ran the 36 units over code new or changed in #94-#105; two last 6-unit scans (`2026-11-16-scan-106-107` and `2026-11-17-scan-106-107-b`) covered the code #106 and #107 added or changed, and checked #107's fixes for completeness. Section 1 lists the runs and section 5 their units.
- Process: each run went Hunt, then Dedup, then Verify (an adversarial verifier per finding, which first tried to refute it, then built a repro), then Report; severity follows the table in `bug-hunt/PRIMER.md`. Every finding was then triaged into `KNOWN.md`. The consolidation dropped every entry fixed by #86-#107 (152; #107 alone fixed 22 of the 44 items this report listed at #106, and K154, whose out-of-bounds read #107 fixed, was merged into K185, the panic that remains) and the 11 judged intended (they stay in `KNOWN.md` so later runs do not re-file them), and kept the 36 below.
- IDs are the stable `KNOWN.md` ids (Kn), shared with `KNOWN.md` and the `[CONFIRMED in <run> as Kn]` tags in `bug-hunt/context/`. Each write-up keeps the text of its run; its `Consolidated:` line names the run and the run's own finding id, the re-check result and the current location. Line numbers inside a write-up are at its run's commit; the location column below is at `c705a54`.
- What is in this directory:
  - `findings/Kn.md`: one write-up per item (K108's was written for this report: it was found by hand while rebasing, not by a hunter).
  - `repros/`: Rust repros `repro_kn*.rs` (integration tests: copy into `jix/tests/` to run), Python repros `Kn.py`, the jix 0.0.7 archive and the jix 0.0.7 CLI source that K184's repros use. The repros of K8, K64, K69, K72, K77, K111 and K150 were ported to the API of `1169965` (a `Ported to main` line in their header says so); all `repro_k*.rs` compile at `c705a54`. Paths and F-ids inside the repros were rewritten to the new names; the per-ID list is at the end of section 2.
  - Not kept: the run directories (their reports, the write-ups of fixed or intended findings, and the hunters' scratch). Write-ups still cite some scratch files and hunter tests (`scratch/...`, `jix/tests/hunt_*.rs`, `<scratch>/...`); those were never committed.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 35 |
| plausible | 1 |
| refuted | 0 (not listed, see section 4) |
| intended | 0 (not listed, see section 4) |
| unverified | 0 |

| severity (confirmed + plausible) | count | IDs |
|----------|------:|-----|
| critical | 0 | - |
| high | 2 | K109, K111 |
| medium | 6 | K8, K69, K71, K77, K116, K184 |
| low | 28 | K64, K72, K108, K133, K150, K151, K158, K161, K176, K179, K181, K183, K185, K186, K187, K188, K189, K190, K192, K193, K194, K195, K196, K197, K198, K199, K200, K152 |

By category:
- questionable-semantics: 11 (K64, K133, K150, K176, K179, K181, K183, K190, K195, K196, K200).
- resource: 6 (K116, K72, K158, K187, K192, K198).
- error-handling: 4 (K161, K193, K194, K199).
- wrong-result: 4 (K71, K108, K186, K188).
- memory-safety: 3 (K109, K111, K152).
- panic: 2 (K185, K189).
- pathological-performance: 2 (K69, K77).
- compatibility: 1 (K184).
- performance: 1 (K197).
- portability: 1 (K8).
- test-coverage: 1 (K151).

Reachability:
- Python API only: 12 (K109, K133, K152, K176, K179, K181, K183, K195, K196, K197, K199, K200). K188 needs a file-like whose tell() is wrong, so in practice it is Python-only too.
- 32-bit targets only: 7 (K8, K185, K187, K189, K193, K194, K199; K8 and K199 are i686 Linux, the rest any 32-bit target). None of them is memory-unsafe any more: #107 turned K153 / K154's out-of-bounds reads into the K185 panic.
- Debug builds only: K72. Free-threaded Python only: K152. Test infrastructure only (no library defect): K151.
- The rest reach through the public Rust API and, mostly, through Python as well.

Runs (all found items, and which of them are still open):

| run | code | units | findings | still open |
|-----|------|------:|---------:|------------|
| 2026-10-21-pilot | main before #85 | 4 | 13 | K8 |
| 2026-10-22-trial2 | after #86 | 8 | 26 (+2 intended) | - |
| 2026-10-23-batch3 | after #87 | 8 | 22 (+2 intended) | K64 |
| 2026-10-24-batch4 | after #90 | 8 | 18 | K69, K71, K72, K77 |
| 2026-10-27-batch5 | after #91 | 12 | 23 | - |
| 2026-10-29-batch6 | after #92 | 20 | 42 (+2 intended) | K109, K111, K116, K133, K150, K151, K152 |
| 2026-11-15-batch7-1 | after #105 | 6 | 7 | K158, K154 -> K185 |
| 2026-11-15-batch7-2 | after #105 | 6 | 3 | K161 |
| 2026-11-15-batch7-3 | after #105 | 6 | 5 | - |
| 2026-11-15-batch7-4 | after #105 | 6 | 1 (+3 intended) | - |
| 2026-11-15-batch7-5 | after #105 | 6 | 10 (+1 intended) | K176, K179, K181 |
| 2026-11-16-batch7-6 | after #105 | 6 | 1 | K183 |
| (by hand, while rebasing) | after #92 | - | 2 | K108 |
| 2026-11-16-scan-106-107 | after #107 | 6 | 7 (+1 intended) | K184, K185, K186, K187, K188, K189, K190 |
| 2026-11-17-scan-106-107-b | after #107 | 6 | 9 | K192, K193, K194, K195, K196, K197, K198, K199, K200 |

Themes:
- #107 fixed 22 of the 44 items this report listed at #106, but the last scan found three of its fixes incomplete or overreaching: the isize::MAX read-size cap (K185, the rest of K153 / K154), the K168 isclose fix (K186: now rejects pairs the tolerance covers) and the K177 seek fix (K188: a loud panic became silently re-reading the first archive). #107's K180 fix also made every broadcasting op fail on i686 Linux (K199, a new K8 trigger).
- #106's ReadTiling levels decide whether a read is tiled, and five storages claim a cheaper level than they deliver: a stepped Slice (K187) and a SubDtype under one (K194) on 32-bit, Select (K192), identity views (K198), and on 32-bit any narrowing op over broadcast leaves (K193). The effect is region-sized temporaries or spurious errors, never wrong values; jix-py's GIL decision cannot see the levels at all (K197).
- Archive compatibility: K184 (medium, from #106, unreleased) makes every archive written by the published jix 0.0.7 unreadable and new archives unreadable by 0.0.7; a fresh protobuf field number fixes both directions for free while #106 is unreleased.
- Python memory safety: K109 (high: `write_array(a, file_like)` lets pure Python keep views of freed Rust memory; #107 only documents it) and K111 (high: `Plain::read_data` and `numpy(out=)` build slices over whole strided spans, so element-disjoint views alias; UB under Miri at `1169965`, code unchanged since).
- 32-bit size arithmetic: K185 and K189 panic with 'capacity overflow' on valid input, K187 over-reads, K8 breaks i686 dtypes. CI tests only x86_64.
- Resource blow-ups and read amplification: deep op chains recurse (K116) and copy cache-id sets (K158); wide elements overflow a debug stack (K72); split / merge reshapes and stepped slices read one element or row at a time (K69, K77); pulls of broadcasts copy (K64).
- Semantics: K190 (codec params reset each other), K176 (struct dtype identity), K179 (where() reads unselected branches), K181 (power()'s i32 limit), K183 (numpy reductions on a jix.Array), K195 / K196 / K200 (Python index and shape arguments: huge slice bounds, i64 overflow, bools), K133 (== None), K150 (host-dependent float sums), K71 / K108 (f16 rounding, var NaN).

## 2. Findings

Sorted by severity, then confirmed before plausible, then by ID. Locations are at main `c705a54`.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| K109 | high | confirmed | memory-safety | PyWriter::write hands write() a memoryview over Rust memory; derived views (b[:], memoryview(b), b.cast, np.frombuffer(b), PickleBuffer(b)) survive release() and outlive the buffer - use-after-free from pure Python | `jix-py/src/archive.rs:423` | [K109](findings/K109.md) |
| K111 | high | confirmed | memory-safety | Plain::read_data (and jix-py numpy(out=)) take a &[u8] / &mut [u8] over the whole strided span, gap bytes included, so element-disjoint views whose spans interleave alias each other: UB from safe Rust and from Python (Miri) | `jix/src/storage/plain.rs:460` | [K111](findings/K111.md) |
| K8 | medium | confirmed | portability | i686 Linux: alignment hard-coded to 8 for f64/i64/u64/c128, but the ABI says 4; Python dtype conversion fails and derived-struct archives cannot be read back | `jix/src/dtype.rs:967` | [K8](findings/K8.md) |
| K69 | medium | confirmed | pathological-performance | Reshape reads every split/merge reshape (flatten, [N]->[a,b], block-aligned splits) one element at a time, decompressing a whole block per element: 400-5000x slower reads | `jix/src/ops/shape_ops/reshape.rs:371` | [K69](findings/K69.md) |
| K71 | medium | confirmed | wrong-result | K28 fix is incomplete: f64 -> f16 via half::f16::from_f64 (no_std software fallback) still mis-rounds on every shipped target except macOS arm64, about 8x as often as the pre-fix f32 double rounding | `jix/src/ops/elementwise/cast.rs:54` | [K71](findings/K71.md) |
| K77 | medium | confirmed | pathological-performance | A stepped Slice over a non-plain inner makes one inner read per output position of the stepped dims, decompressing whole blocks each time: step-2 slices read 200-1500x slower than the whole array | `jix/src/ops/shape_ops/slice.rs:186` | [K77](findings/K77.md) |
| K116 | medium | confirmed | resource | Deep type-erased op chains (every Python op result) recurse once per op level on read and on drop: `x = x + 1` in a loop overflows the native stack (Python SIGSEGV from ~2k levels with the debug extension, ~7k in release; drop from ~30-50k) | `jix/src/storage/core_trait.rs:113` | [K116](findings/K116.md) |
| K184 | medium | confirmed | compatibility | #106 reused FileMetadata field 2 (uint64 varint -> Semver message), so archives written by jix 0.0.7 fail to decode on current main, and 0.0.7 cannot read new archives | `jix/schema/proto/jix/v1/structure.proto:7` | [K184](findings/K184.md) |
| K64 | low | confirmed | questionable-semantics | Pull reads (to_buf_strided) of a large Broadcast or element-wise pipeline region go to the splitting slow path and materialize the whole region, where the storage itself returns an O(1) 0-stride view (the permute-over-Plain case is fixed in #106) | `jix/src/array/read.rs:458` | [K64](findings/K64.md) |
| K72 | low | confirmed | resource | Debug builds: reduce_unordered over wide Dtyped elements overflows a 2 MiB thread stack and aborts: the one-cell fold inlines every lane-count arm into one frame (7.8-8.9 MiB); release builds are fixed since #91 | `jix/src/ops/reduction/common.rs:845` | [K72](findings/K72.md) |
| K108 | low | confirmed | wrong-result | var/std of finite data whose deviations overflow returns NaN for 3+ elements (var([1e308, -1e308, 0.0]) = NaN, numpy inf) - the K84 fix partly reverts K67 | `jix/src/ops/reduction/variance.rs:154` | [K108](findings/K108.md) |
| K133 | low | confirmed | questionable-semantics | a == None, a != None and a == 'x' on a jix.Array raise TypeError where numpy compares element-wise (returning NotImplemented for __array_ufunc__ = None operands was fixed in #93; TypeError for unsupported operands in #107) | `jix-py/src/array.rs:894` | [K133](findings/K133.md) |
| K150 | low | confirmed | questionable-semantics | Float reductions depend on the host's detected CPU cache sizes (and SIMD level): the same numpy array, or the same .jix file, sums to different f32 values on machines with different L1d/L2 | `jix/src/storage/params.rs:456` | [K150](findings/K150.md) |
| K151 | low | confirmed | test-coverage | SIMD level coverage: for_each_simd_level! skips the levels the host lacks without saying so, and level-dependent driver code (pipeline lane count, reduction folds, update_bulk fallbacks) runs only at the host's Level::new() level | `jix/src/util/test_util.rs:1291` | [K151](findings/K151.md) |
| K158 | low | confirmed | resource | Every element-wise op deep-copies the cache-id set of everything below it (#100), so a left-deep lazy chain of n ops holds about n^2/2 ids: memory grows quadratically with chain depth (about 120 MB at 4000 ops, 740 MB at 10000) | `jix/src/storage/params.rs:817` | [K158](findings/K158.md) |
| K161 | low | confirmed | error-handling | Scalar::new and Plain::new run tune's padded-shape check even though they never pad. So `y + 1` on a valid lazy array of ~2^63+ elements raises InvalidShapeOperation, while `y + y` works | `jix/src/storage/params.rs:387` | [K161](findings/K161.md) |
| K176 | low | confirmed | questionable-semantics | Layout-identical numpy struct dtypes (default vs `align=True`) become unequal jix dtypes, so cast / concatenate / stack / where between them raise TypeError, while jix's own `.dtype ==` and numpy(`out=`) treat them as the same | `jix-py/src/dtype.rs:217` | [K176](findings/K176.md) |
| K179 | low | confirmed | questionable-semantics | `where()` raises ArithmeticError for an integer error in an element it does not select, but only for some spellings of a 0-d condition (Python bool, numpy scalar or numpy 0-d skip the other branch; a jix 0-d array reads it), and numpy's guard idiom `where(b != 0, a // b, 0)` always raises; neither is documented | `jix-py/src/ops/elementwise/select.rs:72` | [K179](findings/K179.md) |
| K181 | low | confirmed | questionable-semantics | `power()` with a float base rejects a Python int exponent outside i32 ("out of bounds for i32"), though the result dtype is float either way and the later (float, float) slot holds it; every other op accepts any i64 int next to a float array | `jix-py/src/ops/common/dispatch.rs:249` | [K181](findings/K181.md) |
| K183 | low | confirmed | questionable-semantics | NumPy reductions (`np.sum` / `mean` / `max` / `min` / `prod` / `std` / `var` / `all` / `any`, `np.squeeze(axis=)`) raise "Array.sum() got an unexpected keyword argument 'axis'" on a `jix.Array`, while `np.argmax` / `nanmax` / `median` / `average` / `round` work via `__array__` | `jix-py/src/array.rs:1336` | [K183](findings/K183.md) |
| K185 | low | confirmed | panic | 32-bit: the #107 isize::MAX read-size cap does not bound tiles or chunks. scale_read_shape grows broadcast/tile dims to 2 * max, and compaction to at least 2 target blocks, so compact_with panics with "capacity overflow" and reductions fail with a spurious InvalidShapeOperation | `jix/src/storage/params.rs:26` | [K185](findings/K185.md) |
| K186 | low | confirmed | wrong-result | The K168 fix makes approx_equal / isclose reject every pair whose difference a - b overflows, even when atol = inf, rtol = inf, rtol >= 2, or 1 + min/max <= rtol < 2 covers it exactly | `jix/src/ops/elementwise/cmp.rs:230` | [K186](findings/K186.md) |
| K187 | low | confirmed | resource | 32-bit: a stepped Slice is classified ReadTiling::ZeroCopy even when its K156 fallback materializes the region, so an outer stepped Slice pulls its whole bounding box from it. A pull of 2 elements allocates and reads O(step) elements | `jix/src/ops/shape_ops/slice.rs:103` | [K187](findings/K187.md) |
| K188 | low | confirmed | wrong-result | ArchiveReader::new seeks back to tell() with an absolute seek, so a file-like whose tell() is wrong (K177's tell()==0 reader) silently gets the first archive again on a back-to-back read | `jix/src/archive/common.rs:120` | [K188](findings/K188.md) |
| K189 | low | confirmed | panic | 32-bit: Owned::read_section checks a section's element count against usize, not its byte size against isize::MAX. A block_data or blocks_loc section over 2 GiB therefore panics with "capacity overflow" in Vec::resize, after 1 GiB or 1.5 GiB has been read, instead of returning Err | `jix/src/archive/block.rs:394` | [K189](findings/K189.md) |
| K190 | low | confirmed | questionable-semantics | Setting one codec key in params (compression_level, codec or filters) resets the source's other codec settings to their defaults, while the docs promise unset keys are inherited | `jix/src/storage/params.rs:233` | [K190](findings/K190.md) |
| K192 | low | confirmed | resource | Select is classified `ReadTiling::Fusable` (#106), but its `read_data` pulls condition and y into region-sized temporaries, so a top-level read of `cond_op.select(x, y_op)` runs untiled and allocates them for the whole region | `jix/src/ops/elementwise/select.rs:109` | [K192](findings/K192.md) |
| K193 | low | confirmed | error-handling | 32-bit: since #106 a ZeroCopy / Fusable / SinglePass read is not tiled, so a narrowing op over broadcast leaves (a struct field, map or cast to a smaller type, a comparison) fails with a spurious 'does not fit in memory' Err, because each leaf checks the whole region at its own wider itemsize | `jix/src/array/read.rs:458` | [K193](findings/K193.md) |
| K194 | low | confirmed | error-handling | 32-bit: a stepped Slice over a SubDtype checks its zero-copy bounding box at the field itemsize, but the SubDtype pulls that box from its parent at the parent itemsize, so a 2-element pull fails with InvalidShapeOperation instead of taking #107's K156 fallback | `jix/src/ops/shape_ops/slice.rs:159` | [K194](findings/K194.md) |
| K195 | low | confirmed | questionable-semantics | A dim of length exactly `PY_SSIZE_T_MAX` accepts out-of-range slice bounds: `PySlice_Unpack` clamps them to the dim length, so `a[2**100:]` / `a[n+1:]` return empty and `a[5:2**100]` / `a[0:n+1]` select to the end instead of raising IndexError | `jix-py/src/util.rs:36` | [K195](findings/K195.md) |
| K196 | low | confirmed | questionable-semantics | An integer index outside i64 (`a[2**63]`, `a[-(2**63)-1]`, `a[np.uint64(2**63)]`, `jix.slice(a, 2**64)`) raises TypeError 'only integers, slices ... are valid indices' instead of the IndexError that docstrings promise and that `a[2**63-1]` / `a[0:2**63]` raise | `jix-py/src/ops/shape_ops.rs:274` | [K196](findings/K196.md) |
| K197 | low | confirmed | performance | numpy()'s GIL decision can never see `ReadTiling::Fusable`: every jix-py array is an ArrayAny, which demotes Fusable to SinglePass, so tiny fused reads over numpy memory always release the GIL | `jix-py/src/array.rs:633` | [K197](findings/K197.md) |
| K198 | low | confirmed | resource | Identity Slice / Flip / Roll and `&dyn ArrayStorage` pass their inner's `ReadTiling::Fusable` through (#106) without forwarding its pipeline, so a parent element-wise op reads untiled and pulls a temporary the size of the whole region | `jix/src/ops/shape_ops/slice.rs:229` | [K198](findings/K198.md) |
| K199 | low | confirmed | error-handling | #107's K180 fix computes the numpy dtype of every jix operand that gets broadcast, so on i686 Linux any broadcasting op with a jix f8/i8/u8/c16 operand now raises ValueError when the op is built (new K8 trigger, regression from a7b0656) | `jix-py/src/ops/common/broadcast.rs:54` | [K199](findings/K199.md) |
| K200 | low | confirmed | questionable-semantics | reshape / broadcast (and Array.reshape / Array.broadcast) accept a Python bool as a shape entry (True -> 1, False -> 0), unlike numpy and unlike jix's own Dim and index arguments | `jix-py/src/ops/shape_ops.rs:721` | [K200](findings/K200.md) |
| K152 | low | plausible | memory-safety | Free-threaded builds: numpy(out=) validates out's shape/strides in one read but writes using a second read (array.rs:617/653), and plain_from_numpy reads ndim, shape and strides separately (operand.rs:424/465/471). A concurrent `.shape=` or `.strides=` becomes an out-of-bounds write or read, or a PanicException. | `jix-py/src/array.rs:590` | [K152](findings/K152.md) |

### Bugs

**K109** (high; Python).
#107 documents that the buffer release is best effort; the use-after-free is still reachable.
`jix.write_array(a, file_like)` passes the user's `write()` a read-only memoryview over Rust memory. That memory is the compact array's own block data, the BufWriter buffer, or the reused compressed-block buffer. Afterwards jix calls `release()` on the view and ignores any error.
- Slices, `memoryview(b)`, `b.cast()` and `np.frombuffer(b)` survive `release()`.
- `PickleBuffer(b)` makes `release()` fail, and the failure is ignored.

So pure Python can later read freed or reused heap memory: stale bytes, then SIGSEGV in the repro. It is high rather than critical because the writer must break the io convention that `write()` touches `b` only during the call. CPython's BufferedWriter has the same hazard (bpo-15994).
Fix: pass `PyBytes::new(py, buf)`, which costs one memcpy of at most one block. At minimum, propagate a failed `release()` and document the restriction.

**K111** (high; safe Rust and Python; UB shown with Miri).
`Plain::read_data` builds a `&[u8]` over the whole strided span of a view, gap bytes included (plain.rs:458-468). jix-py `numpy(out=)` builds a `&mut [u8]` over the whole span of `out` (array.rs:651). Two views that share no element but whose spans interleave then alias under noalias pointers. Examples:
- `multi_slice_mut` in safe Rust.
- `jix.asarray(x[:, 0]).numpy(out=x[:, 1])` in Python, where `np.shares_memory` is False.

Miri (Stacked Borrows) reports UB for code with no `unsafe`. Values are correct today, and no miscompile is known. K31 (intended) covers only an `out` that overlaps memory the array reads.
Fix: give `StridedBuf` raw-pointer variants for borrowed external memory and stop building references over whole spans. Alternatively, stage non-contiguous external inputs and outs through pool buffers.

**K8** (medium).
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

**K71** (medium; Rust and Python).
`half` is built without `std`, so `f16::from_f64` uses its software fallback on every shipped target except aarch64-apple-darwin. The fallback drops the low 32 mantissa bits before rounding: about 5e-4 of uniform random inputs come out 1 ulp off, 8x the pre-fix rate. The original K28 trigger still fails on x86_64 Linux.
The only correctness test is gated to aarch64, and the property oracle calls the same function.
Fix: a correctly rounded f64 -> f16 in jix (round-to-odd to f32, then `from_f32`). Make the test unconditional and give the oracle an independent reference.

**K116** (medium; Python, and Rust via `into_any`).
Every Python op wraps its operands as `IntoType<ArrayStorageAny>`. So `x = x + 1` in a loop builds a type-erased chain, and reads and drops recurse through it, one group of native frames per level.
- The debug extension segfaults at about 2k levels on read and 30-50k levels on drop.
- Release Rust aborts on read at about 7k levels.

There is no depth guard. Memory use stays O(1); only the stack runs out.
Fix: track chain depth in the type-erased wrapper and materialize (or raise) past a threshold. Drop nested `Arc`s iteratively.

**K184 (medium, compatibility; a #106 regression, not yet released).**
- **Cause:** `structure.proto:7` gives field 2 to `Semver lib_version`, a length-delimited message. Field 2 used to be `uint64 lib_version_semver`, a varint that every writer since 0.0.5 stored as non-zero. prost rejects the wire-type mismatch.
- **Effect on reads:** every public read path (`read_from*`, mmap, `jix.read_array`) fails on a 0.0.7 archive. The `InvalidArchive` error looks like corruption, because #106's "(archive written by X, read by Y)" suffix wraps only `read_body`. A real 0.0.7 binary also fails on every new file. Both formats call themselves "jix 0.0.7".
- **Fix (verified with real binaries):** `reserved 2; reserved "lib_version_semver"; Semver lib_version = 3;`.
- **Repros:**
  - `repro_k184.rs` embeds the 113-byte file a real 0.0.7 build wrote. Its control, which re-encodes only field 2, loads.
  - `K184.py` also checks the reverse direction.
  - The fixture is `K184_written_by_jix_0.0.7.jix`.

**K72** (low; Rust only).
Release builds were fixed in #91; debug builds (including the dev Python extension) still abort.
The one-cell reduction fold keeps 16 (strided) or at least 8 (contiguous) lanes of items and states on the stack, whatever the element width. `reduce_unordered` over 65000-byte elements needs 2.1-5.7 MiB of stack and aborts (SIGABRT) on a 2 MiB thread; the multi-cell control needs 396 KiB.
This is the K30 defect, whose fix did not touch reduction.rs. Debug builds abort with elements of only 4-8 KiB.
Fix: cap the lane count by bytes on both paths.

**K108** (low; Rust and Python).
`var([1e308, -1e308, 0.0])` and `var([1e308]*40 + [-1e308]*40)` return NaN; numpy returns inf, and so did jix before #92 (K67). Welford's running mean overflows to -inf on the second item, so the mean and m2 become NaN, and since the K84 fix `finalize` keeps a NaN m2 as NaN to serve NaN / inf inputs. The #107 fix of K166 scales Chan's merge but does not touch Welford's update.
Fix: track a "saw a non-finite input" flag in the state and map a NaN m2 to +inf when it is clear.

**K158** (low; all targets; Python and Rust `into_any` chains).
Since #100 every element-wise op `Arc::make_mut`s the `arrays_to_cache_potentially` set it shares with its input (params.rs:834, :880), deep-copying it. A left-deep chain of n ops, such as `x = x + 1` in a Python loop, therefore holds about n^2/2 ids: about 120 MB at 4000 ops and 740 MB at 10000. Shape ops that call `with_cache_scope` reset the set.

Fix: share the structure, compute `arrays_to_cache` once at read time, or cap the set (it only steers the cache).

**K161** (low; all targets; Python with any scalar operand; Rust only through the doc-hidden `__private::Scalar::new` or the unsafe `plain_ndarray_ptr` with zero strides).
- Cause: `Scalar::new` and `Plain::new` build their spec through `into_spec` -> `tune`. The padded-shape check (params.rs:379-388) tests an auto block shape that both constructors overwrite with all-ones on the next line.
- Trigger: `y = jix.broadcast(x, (8589934588, 2**31 + 1))`. The element product fits u64, and Broadcast accepts the shape.
- What happens:
  - `y + 1`, `y * 2`, `y == 7` and `jix.maximum(y, 3)` raise "padded to block_shape [1, 49152] has too many items".
  - `y + y` reads back correctly.
  - There are no wrong values and no panic.
  - Compaction re-tunes with the real block shape anyway, so the check in the leaf constructors protects nothing.
- Fix, either of:
  - Skip the block-geometry checks for `plain_read` leaves.
  - Tune these leaves with an all-ones block that is fixed in every dim, then reset the fixed flags (K121's `combine_block_layout` relies on them).

**K185 (low, panic, 32-bit; #107's K153 / K154 fix is incomplete).**
- **Cause:** #107 clamps `ReadSize.max` to isize::MAX. But `scale_read_shape` (`util/mod.rs:505-509`) lets dims that Broadcast / Tile weight FULL grow to `2 * max`. Compaction floors `max_chunk` at one target block, so with 1 GiB blocks the chunk doubles to 2 GiB even at the default read size.
- **Effect:**
  - `compact_with` panics in `chunk_buf.reserve` (compact.rs:264, raw.rs:33).
  - `sum` over a long broadcast returns a spurious "read region ... does not fit in memory" Err. The same sum with a 512 MiB read size succeeds.
- **Relation to KNOWN:** this refines K154's open remainder (the cause is the 2x growth) and reopens K153's path.
- **Repro:** `repro_k185.rs` (i686 only).

**K186 (low, wrong-result; a regression from #107's K168 fix).**
- **Cause:** cmp.rs:230 now ANDs `diff.is_finite()` into the result. So finite operands of opposite sign whose difference overflows are never "close".
- **Why it is wrong:** the documented formula `abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))` holds exactly when atol = inf, rtol = inf, rtol >= 2, or rtol >= 1 + min/max. All of these returned true before #107.
- **Effect:** f16 hits this at ordinary magnitudes (40000 vs -40000), and through Python any f16 atol >= 65520 casts to inf. The Rust test reference was changed the same way, so the tests pin the wrong answer.
- **Fix:** compare at half scale when the difference overflows.
- **Repros:** `repro_k186.rs` and `K186.py`. The Python repro finds 25 wrong (dtype, tolerance) combinations against an exact `Fraction` reference.

**K187 (low, resource, 32-bit, Rust API only).**
- **Cause:** since #106 a stepped Slice forwards its inner's ReadTiling, so a stepped Slice over Plain / Scalar / Broadcast is classified ZeroCopy. #107's K156 fallback makes it materialize when its bounding box fails `check_region_size`.
- **Effect:**
  - An outer stepped Slice trusts the flag (slice.rs:161) and pulls the whole box, `(n-1)*step+1` elements.
  - On i686 a 2-element pull allocated 256 KiB in one test and 64 MiB in another; the slowest took 3.6 s.
  - Values are correct, and push reads are unaffected.
  - Before #106, stepped Slices cleared their flags.
- **Fix:** do not classify a stepped Slice as ZeroCopy on 32-bit.
- **Repro:** `repro_k187.rs` uses a tracking allocator. It fails on i686 and passes on x86_64.

**K188 (low, wrong-result; #107's K177 fix is incomplete).**
- **Cause:** `ArchiveReader::new` takes the archive start from `stream_position()`, which is `tell()` for a Python file-like. `stream_len` then restores the position with `SeekFrom::Start(tell())`.
- **Effect:** with the tell()==0 reader that #107 explicitly supports, a second back-to-back `read_array` silently returns the first archive again. K177's loud panic became silent wrong data. The #107 test packs the same archive twice, so it cannot notice.
- **Fix:**
  - Use the values `seek()` returns: `Current(0)`, `End(0)`, then a relative restore.
  - Make the test pack two different archives.
- **Repro:** `K188.py`.

**K189 (low, panic, 32-bit).**
- **Cause:** `Owned::read_section` (block.rs:394-395) checks the element count `section.size / size_of::<T>()` against usize, not the byte size against isize::MAX.
- **Effect:** for a file over 2 GiB (a real archive from a 64-bit host, or a sparse file), the doubling `Vec::resize` (block.rs:404) panics with "capacity overflow". By then it has read 1 GiB (block_data) or 1.5 GiB (blocks_loc). Python gets a `PanicException`.
- **Fix:**
  - Reject sections over isize::MAX bytes with the existing "section size ... is too large" error.
  - Grow with `try_reserve`, as `BlockLocation::alloc_table` already does.
- **Repro:** `repro_k189.rs` (i686; ignored on 64-bit).

**K192** (low; Rust API; #106 regression; reported by 3 hunters).
- **Defect.** Select's `spec()` uses `ReadTiling::fused([c, x, y])`, but its `read_data` pulls `condition` and `y` with `read_data_pull` (select.rs:143-144). When those are element-wise ops, each pull materializes a buffer the size of the region. Because Select is Fusable, read.rs:458 does not tile the top-level read.
- **Evidence.** A 16 MiB push read allocates 20 MiB; the same expression read tiled allocates 160 KiB. Values are correct. Python is not affected, because Any demotes the class.
- **Fix.** Classify Select as Tiled unless condition and y are ZeroCopy, or stream `read_data` through its own pipeline.
- **Repro:** `repros/repro_2026_11_17_scan_106_107_b__f001.rs`.

**K193** (low; Rust on 32-bit, Python on the x86 / armv7 wheels argued; #106 regression).
- **Defect.** A narrowing op over broadcast leaves is now one untiled `read_data`. Examples: a struct field, a map or cast to a smaller type, a comparison. Each leaf's `check_out_buf` checks region * leaf itemsize against isize::MAX (buf.rs:359, broadcast.rs:130). That check runs even for 0-stride pull views, which allocate nothing. The read fails with "read region ... does not fit in memory", although the result is only 32-256 MiB.
- **Evidence.** The same test file passes 4/4 at `a7b0656` on i686.
- **Fix.** Check only memory that is really needed, for example the Broadcast's inner region. Alternatively, tile when nitems * max leaf itemsize > isize::MAX.
- **Repro:** `repros/repro_2026_11_17_scan_106_107_b__f002.rs` (i686).

**K194** (low; Rust API on 32-bit only; #106 meets #107's K156 fallback).
- **Defect.** The stepped Slice's bounding-box check (slice.rs:153-160) runs at the field itemsize. Since #106 the SubDtype is ZeroCopy and pulls that box from its parent at the parent's 16-byte itemsize, so the parent's check fails. A 2-element pull returns InvalidShapeOperation instead of taking the general fallback.
- **Evidence.** The push read of the same view works. There is no 64-bit analogue.
- **Fix.** On 32-bit, do not classify a narrowing SubDtype as ZeroCopy. Or check the box at the largest itemsize in the view chain. K187's suggested fix does not cover this case.
- **Repro:** `repros/repro_2026_11_17_scan_106_107_b__f003.rs` (i686).

**K198** (low; Rust API; #106 regression).
- **Defect.** The identity Slice / Flip / Roll and `impl ArrayStorage for &dyn ArrayStorage` return their inner's spec unchanged, so an element-wise inner stays Fusable. They do not forward `read_as_elementwise_pipeline`, so a parent op takes the default whole-region pull into a temporary (core_trait.rs:113-125), and nothing tiles the read.
- **Evidence.** A 16 MiB region needs a 16 MiB temporary, against 128 KiB through a non-identity view.
- **Fix.** Forward the pipeline in the identity case (preferred), or demote the tiling with `forwarded()` as Any and Transmute do.
- **Repro:** `repros/repro_2026_11_17_scan_106_107_b__f007.rs`.

**K199** (low; Python on i686 Linux; #107 regression; new K8 trigger).
- **Defect.** `broadcast_operand` now calls `PyArray::numpy_dtype`, which computes and caches the dtype when nothing is cached. Every broadcast jix operand therefore runs `dtype_to_numpy`. On i686 Linux that fails for f8/i8/u8/c16 (K8), so ops like `x + y` with broadcasting, or `a > a.mean()`, raise ValueError when the op is built. Only `where` uses the carried dtype.
- **Evidence.** Simulated on x86_64 by making `numpy.dtype` raise. No i686 Python was available.
- **Fix.** Carry over only a dtype that is already cached.
- **Repro:** `repros/K199.py`.

**K152** (low, plausible; free-threaded Python wheels only).
`numpy(out=)` validates `out`'s shape and strides, then re-reads the strides and the data pointer from the live ndarray (array.rs:617, 653). `plain_from_numpy` reads ndim, shape and strides separately. On 3.14t/3.15t, another thread doing `out.shape = ...` concurrently could turn this into an out-of-bounds write or read, or a PanicException. Not run: no free-threaded interpreter was available. Taking a snapshot only narrows the window, because numpy's own metadata setters are racy.
Fix: snapshot shape, strides and pointer once and use the snapshot. Document that such arrays must not be reshaped concurrently.

### Questionable semantics

**K64** (low; Rust only).
Partly fixed: #106 removed the todo.md item and a permuted Plain view now stays a view, but a Broadcast or an element-wise pipeline is still materialized (repro_k64).
- `to_ndarray_buf` pull mode bypasses splitting only for `plain_read` storages. Large pulls of broadcast, permute, slice and pipeline views therefore copy the whole region, and a huge broadcast aborts. `read_data(None)` returns an O(1) view for the same region. Values are correct.
- Fix: propagate a "pull is a view" flag, or collapse broadcast dims in the slow path. At least document the behavior.

**K133** (low; Python).
Partly fixed in #93 and #107; what remains is `==` / `!=` with None or a string, which raise TypeError.
The binary and comparison dunders accept any `other` and never return NotImplemented, so a foreign type's reflected op is never tried, even when that type sets `__array_ufunc__ = None`. `a == None`, `a == 'x'`, `a + Fraction(1, 2)` and `a + Foo()` raise ValueError ('Unsupported dtype') instead of deferring or raising TypeError. K37 fixed only the direction with numpy on the left.
Fix: return NotImplemented from the dunders for foreign operands: object/str/void dtypes, or `__array_ufunc__ is None`.

**K150** (low; Rust and Python).
`tune` derives `read_size` from the host's detected L1d/L2 sizes, and archives do not store it. The reduction driver tiles by `read_size`, which also sets how the float additions are grouped. So the same array, or the same .jix file, sums to f32 values up to 4 ulps apart on hosts with different caches. This was shown with faked sysfs cache sizes under `unshare`. The SIMD lane count is a second source, argued from the code.
Fix: group the float additions independently of the I/O tiling, in fixed chunks keyed by stream position. Otherwise, document that float reductions depend on the host.

**K176** (low, confirmed; Python only, also hits Rust-written files). `dtype_from_numpy` copies numpy's alignment, so a default struct (alignment 1) and the same fields with `align=True` (alignment 4) differ under core `Dtype::eq`. jix-py's `cast_impl` (casting.rs:56) and `promotion_target` (dtype_promote.rs:177) then raise TypeError for cast / concatenate / stack / where, while `.dtype ==` and numpy(`out=`) call them the same. `#[derive(Dtyped)]` picks the aligned layout, so a Rust-written `.jix` cannot be combined with a default numpy dtype of the same fields. Fix: add a "same layout" test in jix-py with a `transmute_elements_dyn` retag, and keep core `eq` (`into_typed` soundness depends on it).

**K179** (low, confirmed; Python only). The #95 shortcut in `where` skips the unselected operand only when the condition is a Python bool, a numpy bool scalar or a 0-d numpy array. A 0-d jix Array, or any array condition, builds core Select, which reads both branches, so an integer error in an element the condition does not select raises ArithmeticError. numpy's guard idiom `where(b != 0, a // b, 0)` therefore always raises, and no docstring says so. Fix: docs first, with the workaround `a // where(b != 0, b, 1)`; optionally treat a 0-d jix bool array as a scalar.

**K181** (low, confirmed; Python only). For a float base the Pow table lists (float, i32) before (float, float), an untyped Python int always passes the Safe check into i32, and dispatch does not fall through on a range error. So `f64_arr ** 2**31` raises "out of bounds for i32" although the result is float either way, while every other op accepts any i64 Python int next to a float array. Fix: send an untyped int exponent outside i32 to the (float, float) slot when the base is a float, or at least document the limit and pin it with a test.

**K183** (low, confirmed; Python only).
- Location: `jix-py/src/array.rs:1320` is the `#[pyo3(signature)]` of `sum`; the write-up cites the `fn` line, 1321. The other methods with the same mismatch:

  | method | line |
  |--------|-----:|
  | `all` | 1266 |
  | `any` | 1277 |
  | `max` | 1287 |
  | `min` | 1297 |
  | `mean` | 1331 |
  | `prod` | 1341 |
  | `std` | 1351 |
  | `var` | 1362 |
  | `squeeze` | 1442 |

- What goes wrong:
  - These methods take jix keywords (`dim`, `keepdim`, `ddof`).
  - For any object that is not an ndarray, NumPy's fromnumeric calls the method of the same name with NumPy keywords, through `_wrapreduction` or, for `mean` / `std` / `var` / `squeeze`, directly. Neither path falls back on a TypeError.
  - So `np.sum(z)` raises a TypeError, even with no axis. The error names a jix method the user never called.
- What still works: functions that go through `_wrapfunc` (`np.argmax`, `np.round`, `np.clip`) or through `asarray` (`np.nanmax`, `np.median`, `np.average`) fall back to `__array__`. So `np.max` fails while `np.nanmax` works.
- Fix options (pick one, then add a `test_array_protocol.py` case for it):
  - Define `__array_function__`.
  - Accept NumPy aliases: `axis`, `keepdims`, and `out` / `dtype` / `where` only at their defaults.
  - At least, document that `np.*` reductions are unsupported.

**K190 (low, questionable-semantics; dates from #55, not a regression).**
- **Cause:** `ArrayParams::codec`, `level` and `filters` each seed the whole `EncoderParams` group from its defaults. `inherit_unset_from_storage` (params.rs:232-233) inherits the group only when it is wholly unset.
- **Effect:**
  - `jix.compact(src, params={"compression_level": 3})` brings byte-shuffle back on a source that had no filters. In the trigger the output got 57x smaller.
  - `filters=` on a level-19 source drops the level to 3, and the output got 39% larger.
  - Data still round-trips.
- **Docs:** the Python `compact` / `write_array` docs and the Rust `compact_with` doc promise per-key inheritance. Only `ArrayParams::codec`'s doc mentions the reset.
- **Decision needed:** either inherit per key, or document the group reset everywhere and pin it with a test.
- **Repros:** `repro_k190.rs` and `K190.py`.

**K195** (low; Python; reported by 2 hunters).
- **Defect.** `slice_unpack` relies on `PySlice_Unpack`, which clamps explicit bounds to `PY_SSIZE_T_MAX`. On a dim of length exactly `sys.maxsize`, the clamped bound equals the length and passes the strict check, so `b[2**100:]` is empty and `b[5:2**100]` selects to the end. This breaks the contract that `test_slice_huge_bounds_raise` pins. On the 32-bit wheels the length is 2**31-1, a plausible lazy broadcast.
- **Fix.** Read `slice.start` / `slice.stop` directly as Python ints.
- **Repro:** `repros/K195.py`.

**K196** (low; Python).
- **Defect.** `parse_basic_index` extracts int items as i64 and sends any failure to the catch-all TypeError. So `a[2**63]`, `a[np.uint64(2**63)]` and `jix.slice(a, 2**64)` raise "only integers, slices ... are valid indices" instead of the IndexError that the docstrings promise and that `a[2**63-1]` raises. An exception raised by a user `__index__` is swallowed the same way.
- **Fix.** Classify the item by type first, and map overflow to IndexError.
- **Repro:** `repros/K196.py`.

**K200** (low; Python).
- **Defect.** The shape arguments of `jix.reshape` / `jix.broadcast` / `Array.reshape` / `Array.broadcast` are `ItemOrTuple<i64>`, which accepts bool: True means 1 and False means 0. `a.broadcast((False, 6))` silently returns a (0, 6) array. numpy raises TypeError here, and jix itself rejects bool for `Dim` (#93) and for index items.
- **Fix.** Add a bool-rejecting shape-entry newtype.
- **Repro:** `repros/K200.py`.

### Pathological performance

**K69** (medium; Rust and Python).
Reshape's read-time scan runs past the last orig dim on its first miss. Any reshape whose leading new dim splits or merges dims (flatten, `[N]->[a,b]`, `[6,4]->[2,3,4]`) therefore makes one 1-element inner read per output element, and each read decompresses a whole block.
That is 400-5000x slower than reading the array, about 7 us per element in release: `flatten` of a 256 MB array takes about 8 minutes. It happens even when the split is block-aligned, contrary to the code's own comments. The values are correct.
Fix: a scan that only skips orig dims with larger strides, matching on divisor lengths, `flat` computed over all dims, and range splitting for merges. Alternatively, a block cache in `ReadContext`.

**K77** (medium; Rust only).
A Slice with any step > 1 uses the general path unless it is a pull over a PLAIN_READ inner. That path makes one inner `read_data` per output position of the stepped dims, and on a Compact inner each read decodes whole blocks.
`a[::2]` reads 750-2100x slower than the whole array (about 15 minutes for 1 GiB). The verifier raised the severity from low to match K69.
Fix: read the bounding box once (in chunks) and copy it with `stride * step`, or at least document the cost.

**K197** (low; Python; performance; not a regression).
- **Defect.** #106's numpy() stays attached for ZeroCopy or Fusable reads under a cost threshold. But every `PyArray` is an `ArrayAny`, and `AnyMeta::new` demotes Fusable to SinglePass (any.rs:105-108). The Fusable arm is therefore dead, and a 64-byte `(jix.asarray(x) + 1).numpy()` still releases the GIL.
- **Evidence.** With a second thread spinning, the read waits about one switch interval (50 ms with `setswitchinterval(0.05)`).
- **Fix.** Expose the pre-erasure tiling to jix-py and add a test that pins the attach decision.
- **Repros:** `repros/repro_2026_11_17_scan_106_107_b__f006.rs` and `repros/K197.py`.

### Test-infrastructure gaps

**K151** (low; Rust tests and CI).
`for_each_simd_level!` silently skips the levels the host lacks. Driver code whose behavior depends on the level runs only at the level `Level::new()` picks: the pipeline lane count, the reduction folds, and the SSE-only fallback in Product's `update_bulk`. CI has one x86_64 leg and no `--cfg disable_dispatch_*` override. `SimdTestValues` pairs only 9 of 81 integer and 26 of 676 float edge pairs. Dumps from native AVX-512 and from AVX2 under valgrind differ only by the documented reassociation of mean.
Fix: report the skipped levels (or fail on request), add CI legs with `disable_dispatch_*`, and pair more edge values.

### Re-running a repro

Each repro asserts the correct behavior, so it fails while its bug is present (unless its header says otherwise), and its control tests pass.

- Rust (`repro_kn.rs`): copy it to `jix/tests/`, run `cargo test --manifest-path jix/Cargo.toml --all-features --test repro_kn -- --nocapture --test-threads=1` from the repo root, then delete the copy (`jix/tests/repro_*.rs` is gitignored). Several tests are `#[ignore]`d because they abort or crash the process; the header of each file says how to run them.
- 32-bit items (K8, K185, K187, K189, K193, K194): the same with `bug-hunt/tools/cargo-i686 test ...` in place of `cargo test` (K189 needs `-- --include-ignored`).
- Miri (K111): `python jix/run_miri.py --all-features --test repro_k111 -- --exact safe_plain_push_out_inside_input_span` (from the repo root, with the repro copied into `jix/tests/`).
- Python (`Kn.py`): after `maturin develop` in `jix-py`, run `.venv/bin/python -I bug-hunt/report/repros/Kn.py` (exit status 1 while the bug exists). K184.py's reverse check (jix 0.0.7 reading a new archive) runs only with `JIX_0_0_7_CLI` set; `repros/K184_oldjix_cli/README.md` says how to build it. K199.py simulates i686 on any host; K197.py measures GIL hand-off latency and can be inconclusive (use repro_k197.rs).

Repros per ID (original name in its run -> name here):

- K8 (run 2026-10-21-pilot F008): F008.py -> K8.py; repro_2026_10_21_pilot__f008.rs -> repro_k8.rs (ported).
- K64 (run 2026-10-23-batch3 F024): repro_2026_10_23_batch3__f024.rs -> repro_k64.rs (ported).
- K69 (run 2026-10-24-batch4 F004): F004.py -> K69.py; repro_2026_10_24_batch4__f004.rs -> repro_k69.rs (ported).
- K71 (run 2026-10-24-batch4 F006): F006.py -> K71.py; repro_2026_10_24_batch4__f006.rs -> repro_k71.rs.
- K72 (run 2026-10-24-batch4 F007): repro_2026_10_24_batch4__f007.rs -> repro_k72.rs (ported).
- K77 (run 2026-10-24-batch4 F012): repro_2026_10_24_batch4__f012.rs -> repro_k77.rs (ported).
- K108: repro_k108.rs (new).
- K109 (run 2026-10-29-batch6 F001): F001.py -> K109.py.
- K111 (run 2026-10-29-batch6 F003): F003.py -> K111.py; repro_2026_10_29_batch6__f003.rs -> repro_k111.rs (ported).
- K116 (run 2026-10-29-batch6 F008): F008.py -> K116.py; repro_2026_10_29_batch6__f008.rs -> repro_k116.rs.
- K133 (run 2026-10-29-batch6 F025): F025.py -> K133.py.
- K150 (run 2026-10-29-batch6 F042): F042.py -> K150.py; repro_2026_10_29_batch6__f042.rs -> repro_k150.rs (ported).
- K151 (run 2026-10-29-batch6 F043): F043.py -> K151.py.
- K152 (run 2026-10-29-batch6 F044): F044.py -> K152.py.
- K158 (run 2026-11-15-batch7-1 F006): F006.py -> K158.py; repro_2026_11_15_batch7_1__f006.rs -> repro_k158.rs.
- K161 (run 2026-11-15-batch7-2 F002): F002.py -> K161.py; repro_2026_11_15_batch7_2__f002.rs -> repro_k161.rs.
- K176 (run 2026-11-15-batch7-5 F005): F005.py -> K176.py.
- K179 (run 2026-11-15-batch7-5 F008): F008.py -> K179.py.
- K181 (run 2026-11-15-batch7-5 F012): F012.py -> K181.py.
- K183 (run 2026-11-16-batch7-6 F001): F001.py -> K183.py.
- K184 (run 2026-11-16-scan-106-107 F001): F001.py -> K184.py; F001_written_by_jix_0.0.7.jix -> K184_written_by_jix_0.0.7.jix; repro_2026_11_16_scan_106_107__f001.rs -> repro_k184.rs; scratch/f001-old/oldjix-cli -> K184_oldjix_cli/ (source only).
- K185 (run 2026-11-16-scan-106-107 F002): repro_2026_11_16_scan_106_107__f002.rs -> repro_k185.rs.
- K186 (run 2026-11-16-scan-106-107 F003): F003.py -> K186.py; repro_2026_11_16_scan_106_107__f003.rs -> repro_k186.rs.
- K187 (run 2026-11-16-scan-106-107 F004): repro_2026_11_16_scan_106_107__f004.rs -> repro_k187.rs.
- K188 (run 2026-11-16-scan-106-107 F005): F005.py -> K188.py.
- K189 (run 2026-11-16-scan-106-107 F006): repro_2026_11_16_scan_106_107__f006.rs -> repro_k189.rs.
- K190 (run 2026-11-16-scan-106-107 F007): F007.py -> K190.py; repro_2026_11_16_scan_106_107__f007.rs -> repro_k190.rs.
- K192 (run 2026-11-17-scan-106-107-b F001): repro_2026_11_17_scan_106_107_b__f001.rs -> repro_k192.rs.
- K193 (run 2026-11-17-scan-106-107-b F002): repro_2026_11_17_scan_106_107_b__f002.rs -> repro_k193.rs.
- K194 (run 2026-11-17-scan-106-107-b F003): repro_2026_11_17_scan_106_107_b__f003.rs -> repro_k194.rs.
- K195 (run 2026-11-17-scan-106-107-b F004): F004.py -> K195.py.
- K196 (run 2026-11-17-scan-106-107-b F005): F005.py -> K196.py.
- K197 (run 2026-11-17-scan-106-107-b F006): F006.py -> K197.py; repro_2026_11_17_scan_106_107_b__f006.rs -> repro_k197.rs.
- K198 (run 2026-11-17-scan-106-107-b F007): repro_2026_11_17_scan_106_107_b__f007.rs -> repro_k198.rs.
- K199 (run 2026-11-17-scan-106-107-b F008): F008.py -> K199.py.
- K200 (run 2026-11-17-scan-106-107-b F009): F009.py -> K200.py.

## 3. Unverified findings

None. K108 was found by hand, not by a hunter, so no verifier saw it; its repro, written for this report, reproduces it at `c705a54`.

## 4. Refuted and intended

Not listed: this report keeps only open items. Each run's refuted findings and the hunters' discarded hypotheses were in its own report, which was not kept; the lessons from them are folded into `bug-hunt/context/*.md` ("Intentional behaviors" and the `[RULED OUT in <run>: ...]` tags) and `bug-hunt/README.md`. The 11 findings judged intended stay in `KNOWN.md` as `intended` entries so later runs skip them: K27, K31, K60, K65, K126, K145, K169 (documented in #107), K170, K171, K182, K191.

## 5. Coverage

Units per run:

- 2026-10-21-pilot: util-memory-alloc-pool, util-memory-arrayvec-bitmap-cache, util-memory-helpers, concept-alignment.
- 2026-10-22-trial2: reduction-fold-loops, pipeline-driver, py-ops-shape-ops, codec-archive-codec-pipeline, shape-ops-copy-ops, elementwise-ops-cmp-clamp-approx, concept-integer-width, dynamic-python-differential.
- 2026-10-23-batch3: types-dtype-layout, storage-compact-read-path, util-nd-copier, codec-archive-archive-io, storage-core-buf-contract, elementwise-ops-op2-arith-div-pow, concept-push-pull, dynamic-rust-differential.
- 2026-10-24-batch4: pipeline-simd, elementwise-ops-bitwise-cast, elementwise-ops-where-transmute-subdtype, reduction-op-kernels, storage-core-array-read-paths, storage-core-leaf-backends, shape-ops-slice-reshape, py-ops-cmp-bitwise-where.
- 2026-10-27-batch5: elementwise-ops-kernel-framework, elementwise-ops-op1-unary-classify-complex, pipeline-nodes, reduction-driver, reduction-scalar-traits, util-nd-iter-unordered, util-nd-iter-ordered, shape-ops-forwarders, shape-ops-join, py-core-read-path, py-ops-operand-broadcast, codec-archive-array-format-dtype.
- 2026-10-29-batch6: types-dtyped-derive, types-dims-errors, storage-compact-params, util-nd-arr-sequence, py-ops-dispatch-promotion, py-ops-arith-wrappers, py-core-numpy-wrap-dtype, py-core-archive-errstate, py-core-array-surface-util, py-ops-reduction-wrappers, util-test-infra, concept-aliasing-lifetimes, concept-degenerate-shapes, concept-dtype-layout, concept-arith-semantics, concept-tiling-coverage, concept-untrusted-archives, concept-python-boundary, concept-threads-send-sync, concept-composition.
- 2026-11-15-batch7-1: storage-core-buf-contract, array-frontend-read-paths, array-frontend-read-cache, shape-ops-forwarders, shape-ops-copy-ops, concept-read-contract.
- 2026-11-15-batch7-2: storage-core-leaf-backends, array-frontend-compaction, storage-compact-read-path, storage-compact-params, shape-ops-slice-reshape, shape-ops-join.
- 2026-11-15-batch7-3: pipeline-driver, pipeline-nodes, pipeline-plumbing, reduction-driver, reduction-variance-bools-custom, concept-read-cache.
- 2026-11-15-batch7-4: elementwise-ops-div-rem, elementwise-ops-arith-pow-unary, elementwise-ops-cmp-minmax-classify, elementwise-ops-select-transmute-subdtype, types-dims-errors-root, concept-arith-semantics.
- 2026-11-15-batch7-5: py-core-read-path, py-core-dtype-surface, py-core-archive-params, py-ops-dispatch-promotion, py-ops-arith-wrappers, py-ops-cmp-bitwise-select.
- 2026-11-16-batch7-6: util-nd-copier, util-nd-arr-sequence, concept-degenerate-shapes, concept-composition, dynamic-rust-differential, dynamic-python-differential.
- 2026-11-16-scan-106-107: codec-archive-archive-io, storage-compact-params, array-frontend-read-paths, array-frontend-compaction, shape-ops-join, py-core-archive-params (over the #106 / #107 diff, including util/page_size.rs and the jix-py array.rs GIL-detach changes).
- 2026-11-17-scan-106-107-b: concept-python-boundary, concept-read-contract, py-ops-dispatch-promotion, py-ops-cmp-bitwise-select, py-ops-shape-ops, elementwise-ops-select-transmute-subdtype (the rest of the #106 / #107 diff: every GIL-detach site, the ReadTiling class of every storage, the jix-py op wrappers, transmute / sub_dtype / select).

Totals: 108 hunter runs (60 units once, 36 of them again after #105, 12 over #106 / #107), 341 agents, about 321M input-token equivalents (the cost table is in `bug-hunt/README.md`).

Differential testing in batch 7 (main after #105), all with 0 mismatches: about 190k random Rust op chains (debug, release, i686 release), 13k shape-op chains per element type, about 1.18G bit-for-bit SIMD-level kernel comparisons in release (SSE2 / SSE4.2 / AVX2 / AVX-512, fearless_simd 1.2.0, x86_64 and i686), 160k composition chains, a 36k-case nd_copy fuzz, about 400k Python degenerate-shape checks and 25k random Python ops against NumPy 2.5.4. The scan of #106 / #107 added, also clean: 3000 random compactions and 12000 random lazy expressions through every read entry, an 800-case Stack / Concatenate differential (x86_64 and i686), a 3000-mix `tune` fuzz and an archive truncation sweep at every length; the second scan added a 9000-chain Python shape-op differential, a 14x14 dtype-pair promotion check against `np.result_type`, push-completeness checks of 17 ops into C / F / strided `out=` and a 6-thread GIL-detach stress test.

Never covered, on any run: NEON / aarch64 and armv7 (REQUIRE_ALIGNED) builds, fearless_simd's Fallback level, free-threaded or multi-threaded Python, the release Python extension, Python on a 32-bit target, and shapes beyond 2^32 on 64-bit. The GIL-detach sites were reviewed in full (no Python object is touched while detached) but only on a GIL build of CPython 3.13. Areas only reasoned about: armv7 staging with fully collapsed regions, and whether the pipeline's staged `fill_contiguous` branch runs for odd-itemsize elements.

## 6. Open questions from hunters

The questions the hunters of batch 7 and of the two #106 / #107 scans left open, grouped by theme; the questions of the earlier runs concerned code that #94-#105 rewrote and are not repeated. F-ids were replaced by K-ids where the finding is in `KNOWN.md`. The batch 7 questions were asked at `a7b0656` and were not re-checked against #107, which fixed many of the items they mention.

### 32-bit, huge sizes and unbounded geometry

From batch7-1, "32-bit sizes and the read budget":

- How to fix the root enabler of K153 and K154 (concept-read-contract)? Either compute in u64 and run `usize::try_from` / `check_region_size` before each Index is built (reduction tiles, compaction chunks), or cap `read_size` at about isize::MAX / 4 on 32-bit.
- K153's release consequences depend on where the reduction state lives: the caller's uninitialized ndarray with `state_in_out_buf`, otherwise a stale pool buffer. Other zero-size consumers were not explored: Plain's `(size - 1) * stride`, and Repeat's `g_end - 1` when `g_end == 0` (concept-read-contract).
- K156's release i686 outcome (box_size wraps to 0, so the Index gets a zero `NonZeroUsize` size) is argued from the code, not run (storage-core-buf-contract).
- K155 vs K64: bound the region on every target in `check_region_size`, as `to_ndarray_sub` already does, or fold this into the K64 fix? Allocations between available memory and isize::MAX still abort through `handle_alloc_error`, and no bound prevents that (array-frontend-read-paths).

From batch7-2, "Unbounded geometry: block count and read_size":

- Where should K160's block-count bound go (storage-compact-read-path)?
  - In `tune`: this covers both builders, but it would also reject archives that the writer could otherwise stream.
  - In each builder's `start`: fallible allocation, `usize::try_from`, and a checked `nblocks * 12`.
  - Note that `BlockArchiveWriter::start` (archive/block.rs:171-203) has already written the header and the TOC placeholder by the time it allocates.
- Should `tune` (or ReadSize) cap a user-set `read_size` (storage-compact-read-path)? It is the common cause of K153 / K154, although the Compact read path itself never uses it.
- K154 extension on 32-bit, with the same root cause and no UB (array-frontend-compaction). With a user `read_size` of 2-4 GiB and chunk dims below 2^32, compaction panics before any storage check can return an Err:
  - A 1-D chunk of 2^31..2^32-1 u8 elements hits "capacity exceeds isize::MAX" in `AlignedBytes::reserve` (compact.rs:258-260).
  - For a multi-dim chunk whose product passes 2^32, the usize product in `default_strides` overflows in debug (compact.rs:262, util/mod.rs:150). In release it wraps, and `check_region_size` then returns an Err.
  - A K154 fix that converts u64 to usize with checks would cover these.
- 64-bit (storage-compact-params): a user `read_size` such as (2^40, 2^40) makes compaction size its chunk buffer at up to `read_size` bytes (compact.rs:258-260). Compacting a large lazy view then aborts or panics with capacity overflow instead of returning an Err. Same root cause; not listed in K153 / K154; not run.

From batch7-3, "32-bit, alignment and test infrastructure":

- On 32-bit, `read_data_impl` materializes the pull output (up to isize::MAX bytes) before `check_region_size(index.size(), state_size)` can fail. Moving the state-size check before `materialize` would make the failure cheap. Long-standing, and only reached by unsplit reads (reduction-driver).
- The REQUIRE_ALIGNED = true paths (armv7 / wasm), including the staged-output scatter, were checked only by reasoning. Nothing in CI or in this environment runs them (pipeline-driver).
- bools.rs has no active Rust tests: its `test_reduction!` blocks for all / any have been commented out since before 6dbeb15 and reference the removed multi-dtype All / Any. The kernels are covered only by doctests and the Python tests, so a regression would pass CI (reduction-variance-bools-custom).
- reduction-variance-bools-custom also asked whether the 32-bit test at common.rs:2398-2407 still compiles. It does not: this is now K165.

From batch7-5, "32-bit platforms":

- On 32-bit CPython (x86 / armv7 / win32 wheels), `slice_unpack` (util.rs:26-30) raises IndexError "cannot slice a dim of length N on this platform" for any slice on a dim longer than `2**31-1`, even `a[0:1]` or `jix.slice(a, slice(0, 2))` on a lazily broadcast dim, while `a[5]` and `a[..., :]` work. The K29 write-up suggested extracting the bounds as `Option<i64>`; the fix chose to reject. Is the over-rejection intended? Argued from code only (py-core-read-path).

From batch7-6, "32-bit, alignment-required targets and unproven coverage":

- armv7 (REQUIRE_ALIGNED) staging with a fully collapsed region (concept-degenerate-shapes):
  - The case: staging rule (a), with the all-size-1 sentinel and its faked strides.
  - The staged-output scatter then uses `inner_strides[0] = itemsize`.
  - Argued correct for length 1, but nobody has run it on hardware.
- 32-bit empty regions (concept-degenerate-shapes):
  - `check_ranges` / `check_region_size` reject empty regions whose zeros-as-1 product exceeds isize::MAX, e.g. `[0, 2^31, 2^31]` u8.
  - So `to_ndarray` of a valid empty array errors where numpy can create it.
  - This is the intended K81 / K120 behavior. Should empty regions be exempt?
- Not done: running the composition fuzz on i686 in release with 4-byte dtypes, which covers the 32-bit seams with REQUIRE_ALIGNED=false. The sizes would need to be much larger to reach the truncation sites. (concept-composition)
- Not proven: whether the pipeline's broadcast staging (`fill_contiguous`) is actually reached for odd-itemsize Map / MapMultiple elements. Experiment `_3` passes, but `src/` was not instrumented to show that the staged branch ran rather than the strided fallback. (util-nd-copier)

From the #106 / #107 scan, "Read-size cap, tiles and compaction chunks (K185)":

- **Fix scope.** Options:
  - cap the weighted-dim target in `scale_read_shape` at isize::MAX bytes / itemsize;
  - clamp `ReadSize.max` to isize::MAX / 2;
  - check `chunk_bytes` in `compact_into_builder_dyn` before the reserve, then shrink the chunk or return Err.

  Shrinking or capping also turns the K153 reduction Err into a successful zero-copy read. (array-frontend-read-paths, array-frontend-compaction)
- **64-bit analog:** a read_size >= 2^62 gives a tile over isize::MAX bytes and a capacity-overflow panic in `alloc_out_buf`. This was argued, not run. The storage hunter did not treat it as a bug, because it needs arrays far beyond any real memory. (array-frontend-read-paths, storage-compact-params)
- **The compaction chunk follows the source spec's read_size** (compact.rs:223), not the compaction params. So `compact_with(params.read_size(..))` and `jix.compact(params={'read_size': ...})` cannot avoid the panic. Is that intended? (array-frontend-compaction)
- **`KNOWN.md`'s interim K154 status** says "a chunk near isize::MAX bytes" panics. The measured chunk / tile is about 2 * isize::MAX, so the status should name the 2x weighted growth as the cause. (array-frontend-compaction)

### Read path, read cache and performance

From batch7-1, "Read cache: efficiency and depth":

- Should identity Slice / Roll / Flip skip entering their own cache scope, given that they advertise the inner's ids? Raised by storage-core-buf-contract, array-frontend-read-cache and concept-read-contract.
- In the static Rust API, element-wise ops fuse through `read_as_elementwise_pipeline` and never reach `read_cached`. So a repeated op (xy in `xy * xy`, asserted to be in `arrays_to_cache` at storage/compact.rs:960) never hits. Only leaves and ops behind a `read_data` boundary benefit. Is that intended (array-frontend-read-cache)?
- `read_cached_impl` is `#[inline(never)]` and adds a stack frame per op level on type-erased chains. Does it lower the K116 depth limits measured before #100 (array-frontend-read-cache)?
- Broadcast and PermuteDims / RemoveDim cache-scope and copy choices lose caching opportunities but never change values (shape-ops-forwarders).

From batch7-1, "Copy-op performance and consistency":

- A 1-D or innermost-dim flip copies element by element. Worth a perf note if a benchmark shows a large gap against NumPy (shape-ops-copy-ops).
- jix-py `repeat` / `tile` with repeats == 1 skip the NDIM_MAX check. This is harmless, and the restriction itself is documented (shape-ops-copy-ops).

From batch7-2, "Read-path performance (no value impact)":

- A stepped Slice over a Plain inner, read in push mode (`to_ndarray`, `to_ndarray_sub`, `to_buf_strided_into`), takes the general path instead of the box view. In debug it took 1.22 s, against 51 ms for a pull and 4.8 ms for step 1. The K77 todo item could be widened: in push mode, the fast path could pull the box and copy it into `out` (shape-ops-slice-reshape; not measured in release).
- A non-identity step-1 Slice clears PLAIN_READ (slice.rs:225-228), so `Slice(Slice(Plain))` loses the plain box fast path and `small_read`. Is that intended (shape-ops-slice-reshape)?
- Should the identity Slice skip its own cache scope, as identity Flip / Roll already do (slice.rs:136 vs 222-224)? shape-ops-slice-reshape raised this again after batch7-1.
- Stack's in-place fast path checks the array shape (stack.rs:203-204), while Concatenate's checks the request shape. Align them (shape-ops-join).
- Concatenate's `repeats_across_inners = false` assumes members are read at different regions. Members are actually read at local coordinates, so the same storage inside two members could hit the cache (shape-ops-join).

From batch7-3, "Lane counts, staging and stack use (K163 / K164 / K30 family)":

- Should broadcast staging rule (c) (pipeline.rs:486-497) be limited to elements that actually get a wider contiguous loop (`MAX_ITEMSIZE <= S::u8s::LEN`, or a cap on `chunk_len_max * itemsize`)? And should a huge staging buffer go back to the context's BufferPool at all? Today it stays cached there for the context's lifetime (pipeline-driver).
- Fix direction for K164 (pipeline-driver): should fixed-size sequences report `MAX_ITEMSIZE = N * D::MAX_ITEMSIZE`, the size of the `[T; N]` the closure receives? Or should the lane count also be capped by the total bytes per `read_bulk` call? Either choice affects the K30 lane formula.
- Large fixed-size sequences stay stack-heavy even at 4 lanes: about 330 B per array in release and 2.3 KiB in debug, from the inline `[D; N]` pipelines. Should an K164 fix also route large N through the heap (Vec) path, or document a practical N limit for `[Array<S>; N]` (pipeline-plumbing)?
- LLVM does not share stack slots across the per-SIMD-level arms that `fearless_simd::dispatch!` inlines into `inner_loop`: the measured extra stack is about the sum over levels, not the max. This multiplies every lane-sized buffer in the K30 / K72 family. Would making `inner_loop_per_level` `#[inline(never)]` per level shrink every pipeline's frame (pipeline-plumbing)?
- pipeline.rs:695: the non-contiguous lane formula uses `size_of::<T>()` of the output, not MAX_ITEMSIZE. A deep pipeline with wide intermediate values and a narrow output was not measured (pipeline-driver).

From batch7-3, "Read cache: misses, memory and depth":

- Broadcast and InsertDim keep their inner's cache ids without opening a cache scope, although they read the inner at a different region. A storage reached through two paths at different regions within one scope therefore misses or thrashes on every read. Values stay correct; is this only extra reads and memory, or can it break something (pipeline-nodes, reduction-driver; someone covering the params.rs cache ids should check)?
- Should K116's notes record that the read cache halves the depth limit in debug builds (Python dev extension 837 vs 1654 levels; Rust debug 828 vs 1685; release about 2.5% lower)? This answers batch7-1's open question (concept-read-cache).
- Memory retention: an expression with k shared subexpressions pins k region buffers per piece, each up to `read_size.max` = L2/2. jix-py caches one ReadContext per Python Array, so repeated squaring (`y = y * y` k times) leaves about k * L2/2 bytes held by that Array object. This is a design trade-off against exponential uncached time; not measured (concept-read-cache).
- Repeated shape-op subexpressions are never cached: `t = (t + t).permute_dims(..)` nested n times still reads the base 2^n times. The context doc calls this intended. Should a shape op that is reached twice get a cache id of its own (concept-read-cache)?

From batch7-6, "Performance":

- A tiny inherited `read_size` makes numpy() very slow (dynamic-python-differential):
  - A lazy view inherits its leaves' `read_size`. With a tiny one (e.g. `compact(..., params={'read_size': 1})`), numpy() / `to_buf_strided_into` splits the read into pieces through `to_buf_strided_slow`.
  - Each piece re-evaluates every nested broadcast reduction. The read cache never holds them, because each is reached only once in the expression.
  - Example: `dynamic_py_8.py` seed 2361, shape (5,3,2,2,5) f64. compact() took 2.2 s; numpy() was still running after more than 5 minutes.
  - Is the 100x+ gap between the two read paths acceptable?
- A large explicit block may be decompressed many times (concept-composition):
  - No block cache exists, although CLAUDE.md says `ReadContext` carries an optional one.
  - `read_shape_heuristic` / `scale_read_shape` (`util/mod.rs:487-497`) shrink the read tile below the block when the block is bigger than `read_size.max` (default L2/2). So a 32 MiB block read in ~1 MiB tiles is decompressed about 32 times.
  - The read_size docs describe this trade-off. Is it pathological? Not measured.
- Concatenate disables caching for ids repeated across members (`concatenate.rs:155-160`), yet `concatenate((a.view(), a.view()))` reads `a` at the same region twice. Stack has no single-member forwarding path, so a pull read of one stacked slice always materializes. Is either worth changing? (util-nd-arr-sequence)
- An identity Slice returns its inner's spec, cache ids included, but still enters its own cache scope (`slice.rs:137`, `:222`), unlike identity Flip and Roll. So in `slice_identity(x) + x`, `x` is read twice. Values are unaffected. (concept-composition)

From the #106 / #107 scan, "ReadTiling classification (K187)":

- **ZeroCopy for ops whose pull is zero-copy only conditionally.** The storage hunter found only the stepped Slice conditional; Plain, Scalar, Broadcast, SubDtype, Transmute, PermuteDims, Insert/RemoveDim and Any are unconditional views. The read-paths hunter, who owns `slice.rs` / K156, should confirm the fix choice. (storage-compact-params)
- **Ops above such a Slice:** Flip / Tile / Repeat / fused pipelines also skip tiling and fill a region-sized temporary. As far as the hunter saw, this is performance only, with no wrong values. (array-frontend-read-paths)

From the #106 / #107 scan, "Read entries on empty, huge regions (K191)":

- **Should `to_buf_slice` / `to_buf_strided_into` reject such regions on 64-bit** to match the pull entries? The verifier judged the split intended (section 4). Consider renaming `read_entries_agree_on_an_empty_region_with_huge_extents`, or noting in its doc that it is a 32-bit test, since its name suggests a cross-target contract. (array-frontend-read-paths)

From the #106 / #107 scan, "Leftovers of earlier fixes (performance, armv7)":

- **K163:** rule (a) in `compute_need_staging` (pipeline.rs:479-484) still stages a stride-0 operand of any width when REQUIRE_ALIGNED is set and the base is misaligned. That affects armv7 and the Rust API only. No safe-API way to get such a base was found, so it is unconfirmed. (shape-ops-join)
- **K163, performance:** when a wide (> 32-byte) broadcast is left unstaged, narrow broadcast operands are still staged although the pipeline stays strided. (shape-ops-join)
- **K164:** the tuple sequence impl (arr_sequence.rs:679) still takes the max item size, not the sum (a few KiB at most). Counting N also lowers the contiguous lane count for fixed-size sequences: 8 f32 arrays on AVX2 go from 64 to 8 lanes, so `map_multiple` over `[Array; N]` may be slower. Not measured. (shape-ops-join)

### Contracts, invariants and documentation

From batch7-1, "Read contract and API hardening":

- `ArrayStorage::read_as_elementwise_pipeline` is public (doc-hidden) and takes a user-built `Index` (`Index::new` does not check the shape), so it skips the entry-point checks. Should it be pub(crate) / sealed, or should the contract say leaves must re-check (array-frontend-read-paths)?
- `check_out_buf_impl` ignores the stride of len-1 dims, so `to_buf_strided_into` forwards arbitrary strides there. Repeat (K36), the Slice fast path (K78), tile, roll, flip and reshape were checked. Not yet audited: the PermuteDims/InsertDim stride remaps, reduction outs and the pipeline's output-layout picking (array-frontend-read-paths).
- Latent: the pipeline driver now collapses a length-0 dim to 1 when every operand has stride 0 there (pipeline.rs:201-209, 325-333). A zero-size push read with a 0 out stride would run the kernel once into the caller's out. No producer is known today (concept-read-contract).
- Compaction (array/compact.rs:253-273) runs `set_len` on an uninitialized `AlignedBytes` and builds a `&mut [u8]` over it before the push read fills it. Is that accepted? array-frontend-compaction (batch 2) should confirm (storage-core-buf-contract).

From batch7-2, "Docs and safety contracts":

- Plain gap bytes (storage-core-leaf-backends): `Plain::read_data` builds a `&[u8]` over the whole strided span, including the gap bytes between elements.
  - The Safety sections of `Plain::new` and `plain_ndarray_ptr` (plain.rs:97-108, 402-412) require the gap bytes to be in one allocation and not written. They do not require them to be initialized.
  - No code path reads a gap byte. Whether a reference over uninitialized gap bytes is itself UB is an open question in Rust.
  - Should the docs also require initialized gap bytes, next to K111 / K140?
- Errors sections (storage-core-leaf-backends):
  - `Plain::new` omits dims > i64::MAX.
  - `plain_ndarray_ref` / `_view` list only ndim and negative strides, yet a valid broadcast f64 view of more than 2^61 elements fails `check_shape_overflow`.
  - The limit is intended library-wide, but these constructors do not document it.
- `tune`'s docs say InvalidArgument "if a block dim is 0". But a 0 in a non-fixed dim of a partially fixed explicit `block_shape` (e.g. [0, 64] with fixed [false, true]) is silently rescaled. This is harmless; the docs overstate the check (storage-compact-params).

From batch7-3, "Safety contracts":

- `materialize_pipeline_out_buf` builds a writable PoolBufMut with aliasing 0 strides through `from_pool`, whose Safety section forbids aliasing strides. It becomes read-only only after a successful run, so on an Err it stays writable in the slot. Nothing can observe it today. Should `from_pool`'s contract be relaxed to allow this, or should the buffer be made read-only before the run (pipeline-nodes)?

From batch7-4, "Safety contracts and the public surface":

- SubDtype push (sub_dtype.rs:105) and every pull consumer rely on no storage building a zero-size Index through `Index::new_unchecked`. That is only debug_asserted (index.rs:45). K153 / K154 cover the 32-bit truncations to zero. The roughly 20 `new_unchecked` sites in shape ops, reductions and compaction were not audited for 64-bit zero-size paths. A zero-size Index under SubDtype push would make `get_unchecked` read out of bounds (select-transmute-subdtype).
- `Dtype::new_struct_borrowed_unchecked` (dtype.rs:624) is a public, doc-hidden `unsafe fn` used by `derive(Dtyped)`, and it has no `# Safety` section. SubDtype relies on the invariants it skips: field offset + size <= itemsize, and fields tiling the element. Should the contract be documented, even if the fn is meant for the macro only (select-transmute-subdtype)?
- `PoolBuf::as_slice` / `as_mut_slice` (buf_pool.rs:108-117) build `&[u8]` / `&mut [u8]` over the whole pool capacity, including bytes never written. Select, SubDtype push and NdCopier wrap these slices. Miri accepts this today, but Rust's rules have not settled whether such references are valid. This affects the whole crate (select-transmute-subdtype).
- `pub use util::*` also exports the doc-hidden `bench_util` at the crate root, as `jix::bench_util` besides `jix::__private::bench_util`. The Safety section of `nd_copy` does not require `shape.len() == src_strides.len() == dst_strides.len()` or `shape.len() <= 8`. Whether a mismatch panics or is UB was not checked; it is reachable only through the unsafe fn (types-dims-errors-root).
- The doc-hidden public `ArrayStorage::read_as_elementwise_pipeline` accepts a user-built `Index` (`Index::new` checks no shape). It is safe only because every leaf validates the index first and the returned pipeline cannot be used outside the crate. Should it check the shape itself, or become pub(crate) (types-dims-errors-root)?

From batch7-4, "Errors and their documentation":

- When a push read fails with an ArithmeticStatus error, should `to_buf_strided_slow_dyn` (read.rs:540-556) first read the remaining pieces, so that `out` ends up fully written, as it does in the pipeline (pipeline.rs:310)? Or should `to_buf_strided_into` / `to_buf_slice` / `numpy(out=)` document that `out` is unspecified after an Err (concept-arith-semantics; K169)?
- The Errors sections of `to_ndarray_sub`, `to_buf_slice`, `to_buf_strided` and `to_buf_strided_into` do not mention `ErrorKind::Arithmetic`. Yet all four return it for a lazy view with integer division by zero or a negative integer power (types-dims-errors-root; covered by K169's suggested fix).
- With more than 8 ranges, `check_ranges` checks ndim before comparing lengths. So all four read entry points return TooManyDimensions, where their docs promise InvalidIndex for a wrong number of ranges. Should the docs or the order of the checks change (types-dims-errors-root)?
- The `ErrorKind::Arithmetic` doc (error.rs:75-76) says "hit a result it can't represent", yet MIN / -1 also has no representation and wraps silently. The public `ElementwisePipeline::read_bulk` docs (pipeline.rs:46-48) link to the pub(crate) `ArithmeticErrorKind` (concept-arith-semantics).

From batch7-5, "Internal invariants and dtype conversion":

- `read_layout_order` must be a permutation of 0..ndim (array.rs:570-578, util.rs:145-177). A duplicate makes `PyArray_Transpose` raise, and an out-of-range entry panics at `drop_dims[dim]` / `np_dim[dim]`. Only its length is debug_asserted (params.rs:791). Should the permutation itself be asserted (py-core-read-path)?
- A gap in the K143 fix: `dtype_from_numpy` sorts fields by offset with a stable sort, so a zero-size field at the same offset as a non-empty field is accepted or rejected depending on numpy's names order ("field b is at offset 0, expected 4"). Sorting by (offset, itemsize) would fix it; very rare input (py-core-dtype-surface).
- Latent: `dtype_from_numpy` picks a scalar kind from (descr.kind, itemsize), never from the type number. A third-party numpy user dtype that reports kind 'f' / 'i' / 'u' / 'c' with a standard itemsize but a different encoding would be read without error. No concrete instance found; ml_dtypes uses kind 'V' (py-core-dtype-surface).

From batch7-6, "Documentation and invariants":

- Stale NdCopier docs (util-nd-copier):
  - The comment at `nd_copy.rs:94-95` and the NdCopier type doc (`:6-13`) still say NdIterUnordered drops unit dims.
  - Since c28c62a the pre-pass does it for 0 or 1 long dims, and it skips the stride / shape length check there.
  - A `debug_assert_eq` on the lengths at the top of `copy()` would restore the check.
- `fill_contiguous` (`nd_copy.rs:441`) is a `pub(crate) unsafe fn` with no Safety section. Its typed arms read `*src` even when `n == 0`, while the untyped arm returns early. Should the contract (n, src, dst, overlap) be documented before a second caller appears? (util-nd-copier)
- An undocumented invariant (concept-composition):
  - "`spec().flags().is_compact()` implies `as_compact().is_some()`" is documented nowhere, yet `MaybeCompact::as_compact` unwraps on it (`maybe_compact.rs:106`).
  - A new forwarder that keeps its inner's spec but the default `as_compact()` would panic in `maybe_compact().write_to()`.
  - Should it be documented on `ArrayStorage::as_compact`?
- Should the docs say a map_multiple closure must be pure? (util-nd-arr-sequence)
  - MapMultiple, like every fused pipeline, computes a position only once along dims where every operand has stride 0 (`pick_output_layout`).
  - So a side-effecting `map_multiple` closure over broadcast inputs runs fewer times than the logical element count.
  - The closure is only bounded by `Fn`, and the docs do not say it must be pure.
- The PRIMER / context docs assume fearless_simd 1.1.0, but `jix/Cargo.lock` resolves 1.2.0. (dynamic-rust-differential)
  - In release, 1.2.0's software `mul_add_precise` (SSE2 / SSE4.2) agreed with the scalar fn and hardware FMA on every input class, on x86_64 and i686.
  - The NEON and Fallback paths remain unexercised.

### Params and archives

From batch7-2, "Compaction params semantics":

- Is the chunk-size source intended (array-frontend-compaction)? Compaction chunks follow the SOURCE spec's `read_size`, never the caller's `params.read_size` (compact.rs:219). `compact_ndarray_with` always uses the default, while `compact_fn_with` / `compact_ndarray_ptr` use the caller's. This affects only performance and memory, but it is inconsistent.
- Should `tune`'s padded-shape and block-size checks run only when blocks are actually built, that is in `compact_with`, `write_to` and archive reads (storage-compact-params)? Scalar::new and Plain::new compute an auto block shape only to discard it (K161). A fix could build their spec without `tune`, or skip the padded check when the flags are not those of a compact storage.
- Two usability traps follow the documented inheritance; they are worth a maintainer opinion (storage-compact-params):
  - All-fixed flags with no `block_shape`, on a source derived from Plain or Scalar, give 1-element blocks (the K121 granularity).
  - A tiny all-fixed source block makes `compact()` of a broadcast produce millions of tiny blocks.

From batch7-5, "Archive I/O":

- Should file-like positions be computed without `tell()` at all (`seek(SeekFrom::Current(-buffered))`)? Then a seekable reader with an inconsistent `tell()` also works, and the K177 panic goes away without validating `tell()` (py-core-archive-params).
- For K178: decide "created by this call" from the open itself (create_new first, then fall back to opening the existing file), resolve the path to an absolute one before detaching, truncate through the fd (`set_len(0)`) before any unlink, and unlink only if the path still names the same dev / inode. Core `write_to_file` needs the same change (py-core-archive-params).

From batch7-6, "Archive parameters":

- `Array::read_from` and the other archive readers ignore a reader-side `block_shape` of any length, as documented. But a `block_shape_fixed_dims` of the wrong length returns Err ("ndim does not match block_shape_fixed_dims length"). Should the mask be ignored too, since its block shape is? The probe was the now-deleted `jix/tests/hunt_2026_11_16_batch7_6__dynamic_rust_differential_10.rs`. (dynamic-rust-differential)

From the #106 / #107 scan, "Archive format compatibility and versioning (K184)":

- **Is breaking pre-#106 compatibility intended for 0.0.x?** `todo.md` still lists "document archive compatibility".
  - If yes, the format still needs a version marker or a fresh field number, so the next change fails with a clear "unsupported version" instead of a prost wire-type error.
  - If no, `reserved 2;` plus a new field number for Semver fixes both directions.
  - The verifier did not check whether the PR skipped the buf breaking-change job with the `schema-skip-breaking` label. (codec-archive-archive-io)
- **`lib_version()` drops Cargo build metadata** (there is no CARGO_PKG_VERSION_BUILD), so `lib_version_matches_cargo` would fail for a `+meta` version. Cosmetic today. (codec-archive-archive-io)

From the #106 / #107 scan, "File-like readers and writers (K188)":

- **`read_array(file_like)` now requires SEEK_END** (for `stream_len`) **and nonzero SEEK_CUR** (for the K177 restore). Is that the intended minimum contract? If so, the docstring should say so. Also, when the restore seek fails, the decoded array is discarded and the stream is left after the read-ahead. (py-core-archive-params)
- **`write_array` to a file opened in 'ab'** detects append mode only in finalize. By then the header, locations and TOC are already appended, and nothing truncates them on the file-like path. Is that trailing garbage acceptable? (codec-archive-archive-io)

From the #106 / #107 scan, "Params semantics (K190 and nearby)":

- **Codec params:** inherit them per key (merge the source's EncoderParams with the explicitly set fields)? Or document, in the Python `compact` / `write_array` docs and the Rust `level()` / `filters()` docs, that setting any codec key resets the others? A test should pin it either way. (py-core-archive-params)
- **`compact_with(ArrayParams::new().block_size(X))` on a source with an all-fixed block shape** (any Compact built with an explicit block_shape) ignores X, with no error. `tune` rescales only non-fixed dims. The docs say a compact source keeps its block shape by default, so this is arguably intended; it needs a decision or a doc line on `block_size`. (storage-compact-params)
- **`tune`'s doc** says it fails with InvalidArgument "if a block dim is 0", but a 0 in a non-fixed dim is silently rescaled. Doc nit. (storage-compact-params)

### Numerics and SIMD

From batch7-3, "Semantics and consistency":

- bool8 Max / Min / Reduce pass a non-canonical true byte (e.g. 2) through, while All / Any normalize to 0 / 1. Is that intended? Still open since batch6 (reduction-variance-bools-custom).
- K108 complex variant: for complex data whose mean difference overflows in one component, `delta * Complex(r, 0)` makes both mean components NaN via inf * 0, one merge level earlier than for real data. A K108 fix should cover complex too, e.g. by not scaling a complex delta through a complex multiply (reduction-variance-bools-custom).
- MapMultiple reports a shape mismatch as InvalidArgument but zero arrays as InvalidShapeOperation, while Op2 and Stack use their own kinds. The error kinds of multi-input constructors are inconsistent; cosmetic (pipeline-plumbing).

From batch7-4, "SIMD dependency and level coverage":

- fearless_simd now resolves to 1.2.0 (div-rem, arith-pow-unary, cmp-minmax-classify).
  - Cargo.toml says `fearless_simd = "1.1"` and the per-crate locks are gitignored, so wheels may build against any 1.x.
  - On the SSE2, SSE4.2 and Fallback levels, 1.2.0's `mul_add_precise_f32x4` / `_f64x2` are software FMAs: round-to-odd through f64 for f32, and Graillat-Muller for f64, with a scalar fallback outside 2^-400..2^400.
  - They run in every strided float divide_trunc / floor / euclid / remainder read on x86, and in every such read on armv7 and ppc64le.
  - A 2^18-pair comparison agreed with hardware FMA. But in-tree tests cover these levels with only 64 values, and only on the host's levels (K151).
  - Should the R8 assumptions (wrap, saturate, unordered ne) be re-checked for 1.2.0, or the version pinned?
- SIMD-vs-scalar agreement was verified only on x86 levels. Still unexercised: the NEON cast chains, NEON saturating narrows / truncates, and the native f16 arithmetic on aarch64+fp16 (concept-arith-semantics).
- Maximum / Minimum on aarch64: should the sign of zero for mixed signed zeros at least not depend on the element's position (SIMD body vs N=1 tail), even though the docs allow either (cmp-minmax-classify)?

From batch7-4, "Float accuracy and rounding policy":

- Should f16 Pow be correctly rounded? A fix needs a correct single f64 -> f16 rounding, and `half::f16::from_f64` does not provide one (K71; with F16C, half's `f64_to_f16` itself goes through f32 first). numpy has the same misround, so this is a policy call (arith-pow-unary; K171).
- f32 ** i32 goes through f64, while f32 ** i8 / u8 / i16 / u16 use f32 powf (arith.rs:358-368). So the same exponent value can round differently by an ulp depending on its dtype. Should the narrow exponent arms use the f64 path too (arith-pow-unary, concept-arith-semantics)?
- Past `EXACT_LIMIT`, the residual of `div_rem_trunc` (div_rem.rs:404-407) is not `a - q*b` when `qt - 1.0` rounds back to `qt`: the fma result is negative and `copysign(a)` flips it. Is it worth a comment, so no future change relies on that residual's sign (div-rem)?
- [SUPERSEDED by batch7-5 K169 / K175: such inputs run in f32, not f16] `jix.isclose` on bool / u8 input computes in f16 (the first Safe float in the table), so tolerances are rounded to f16. Should integer inputs go to f32 / f64 instead (cmp-minmax-classify)?

From batch7-6, "SIMD bit-exactness contract":

- Signaling NaNs in `Floor` / `Ceil` (dynamic-rust-differential):
  - The AVX2 / AVX-512 bulk body returns a quieted NaN, but the scalar fn returns the signaling NaN unchanged. Example: f32 0xffab4161 becomes 0xffeb4161.
  - The scalar fn also runs for the N=1 tail and for strided reads at the SSE2 baseline. So an element's quiet bit depends on its position in the array, which breaks the pipeline's "SIMD == scalar, bit for bit" contract.
  - Seen 14 + 114 times in ~59M comparisons on x86_64 release; not seen on i686 musl.
  - Should the contract exempt NaN payloads explicitly, or should the scalar floor / ceil quiet signaling NaNs?

### Python API semantics

From batch7-4, "Python scalar operands":

- Is it intended that every Python op rejects out-of-range Python ints with OverflowError, while numpy 2 compares them exactly (`u8_arr < 300` -> all True, `u64_arr > -1` -> all True) (cmp-minmax-classify)? The [2**63, 2**64) case is pinned as intended (K170).
- `where()` with a scalar bool condition returns the selected operand and skips the other branch (jix-py select.rs:68-75). With an array condition (a 0-d jix array included), core Select evaluates both branches, and an error in the unselected branch raises. Is that difference intended (concept-arith-semantics)?

From batch7-5, "Python ints and scalar vs list inputs (K170)":

- `jix.asarray(2**63)` and `jix.compact(2**63)` without `dtype=` raise OverflowError ("out of bounds for int64", operand.rs:55), while `jix.asarray([2**63])` gives uint64, as NumPy does for both. Likewise `jix.sum(2**63)` raises but `jix.sum([2**63])` works. Same root cause as K170, but a new trigger inside `asarray` itself. Should the bare-int path follow NumPy like the list path, or does the K170 decision cover it? (py-core-read-path, py-core-dtype-surface, py-core-archive-params, py-ops-dispatch-promotion, py-ops-arith-wrappers, py-ops-cmp-bitwise-select)
- Lists skip the Python-int policy, because numpy converts them into typed arrays first: `jix.equal(i8_arr, [1000])` works but `jix.equal(i8_arr, 1000)` raises, and `jix.add(u64_arr, [2**63])` and `concatenate([u, [2**63]])` succeed (py-ops-cmp-bitwise-select, py-ops-arith-wrappers).
- Other edges near K170: `roll` raises OverflowError from its i64 parameter with a different message (py-core-dtype-surface), and `log(base=2**63)` is accepted because base is a float parameter (py-ops-arith-wrappers).

From batch7-5, "Dispatch and promotion policy":

- Power: should the (float, i32) slot win for a Python int exponent of any size (K181), or should dispatch try the (float, float) slot when the int does not fit i32? The same question applies to (int, u32) vs (int, int64) for integer bases with exponents `>= 2**32` (py-ops-dispatch-promotion).
- isclose: should integer and bool inputs compute in f64, as numpy does? Today the tolerances round to f32, e.g. `isclose(u8[1], u8[2], rtol=0, atol=0.99999999)` -> True (py-ops-cmp-bitwise-select; K175). py-ops-arith-wrappers guessed the rounding concern might apply to the Rust ApproxEq API; it does not, because core takes typed tolerances.
- `bitwise_rotate_left` / `_right` accept only bool / u8 / u16 / u32 amounts, so the most common numpy integer arrays (`np.arange` -> int64) raise TypeError, while the shifts accept them. This is documented; should signed amounts with a range check be accepted (py-ops-cmp-bitwise-select)?
- Cosmetic: for a Python int amount, rotate's error message shows the pre-cast dtype, e.g. "does not support operands with dtypes [bool, u32]" for `rotl(True, 1)` (py-ops-cmp-bitwise-select).

From batch7-5, "where() semantics":

- Should a 0-d jix bool Array get the same scalar shortcut as a 0-d numpy array or `np.bool_` (that means reading its one element), or should the docstring say that only Python / numpy scalars short-circuit (py-ops-dispatch-promotion, py-ops-arith-wrappers, py-core-dtype-surface, py-core-read-path; K179)?
- `where(cond, x, x)` never reads `cond`, so an integer error inside `cond` is skipped, while `where(cond, x, x + 0)` raises. Should the docstring say that the unread branch or condition is never evaluated (py-core-dtype-surface)?

From batch7-5, "NumPy protocol surface":

- `np.sum(a)`, `np.mean(a)`, `np.max(a)` etc. on a `jix.Array`: NumPy's `_wrapreduction` calls `a.sum(axis=None, out=None)` via getattr, which probably raises TypeError on the unknown `axis` keyword. Not tested. Now that `__array__` exists, users may expect these to work (py-core-read-path).

From batch7-6, "NumPy protocol surface":

- How should K183 be fixed? (dynamic-python-differential)
  - Option 1: accept NumPy keyword aliases (`axis`, `out=None`, `keepdims`, `dtype=None`) on the reduction methods. `_wrapreduction` prefers `getattr(obj, 'sum')` for any non-ndarray. So accepting the aliases, or renaming the methods, is the only way to keep both the jix signatures and `np.sum` working.
  - Option 2: define `__array_function__`, converting via `__array__` or raising a clear error.
- `reversed(z)` raises TypeError "jix.Array is not a sequence", while `list(z)` iterates through `__getitem__`. `x in z` on n-d raises ValueError, which K172's notes already mention. Both have the same root cause: no `__iter__` / `__contains__`. Should the K172 fix also add `__reversed__` (or `__iter__`) so the sequence protocols agree? (dynamic-python-differential)
- Some 0-d calls raise where numpy returns a result:
  - `jix.argmax` / `argmin(a0)` and `jix.roll(a0, k)` raise "dim must be specified for arrays with ndim != 1".
  - `jix.tile(a0, n)` raises.
  - Probably intended (batch 6's argmax dim=None question), but nothing documents it for roll / tile. (concept-degenerate-shapes)
- var / std with `ddof >= n` return NaN where NumPy returns inf for a nonzero spread, e.g. `jix.var([1.0, 3.0], ddof=2)`. The Rust docs state it (`variance.rs:73`, `:301`), but the Python docstrings do not. Is that gap intentional? (dynamic-python-differential)

From the #106 / #107 scan, "Python error classes and messages":

- **Core error kinds without their own mapping become RuntimeError** (`jix-py/src/util.rs:90-102`), for example InvalidShapeOperation and InvalidArgument. So these raise RuntimeError where numpy raises ValueError:
  - `jix.flip(x, dim=(0, 0))`;
  - concatenate / stack with mismatched shapes;
  - duplicate dims in permute_dims / sum;
  - squeeze of a non-1 dim;
  - reshape to a wrong size.

  Is that intended after #106's "better errors"? (shape-ops-join)
- **Concatenate's error messages** say "cannot stack arrays of different shapes/dtypes" (concatenate.rs:103, :110). Cosmetic. (shape-ops-join)

### Tooling, test infrastructure and review gaps

From batch7-2, "Test infrastructure":

- i686 debug builds panic in `NdCopier::scalar_fn::<u64>` (the nd_copy.rs:117 debug_assert: align_of is 4, the dtype alignment is 8) on every push copy of an 8-byte, 8-aligned dtype, e.g. a Plain f64 read through `permute_dims`. This is K8 (pilot F008 point 4) and still open. Until it is fixed, i686 experiments need `--release` or 4-byte dtypes. Raised by storage-core-leaf-backends; array-frontend-compaction hit it too.

From batch7-4, "Tooling":

- Under `cfg(miri)` the lib emits unused-import warnings (context.rs:1 `UnsafeCell`, codec/mod.rs:11). They are harmless, but a future Miri CI step with `-D warnings` would fail (select-transmute-subdtype).

From batch7-5, "Bookkeeping and stubs":

- K76 no longer reproduces after #105 (`rotl(64, 1)` -> int64 128, `rotl(64, u32_arr)` -> int64 [128, 256]; pinned by test_bitwise.py:248-265), but `KNOWN.md` and `todo.md:17` still list it as open (py-ops-dispatch-promotion, py-ops-cmp-bitwise-select).
- `remainder` (div_rem.rs:163) and `Array.remainder` (array.rs:989) use a string annotation in `#[pyo3(signature = ...)]`, which CLAUDE.md says to avoid. It probably resolves only because divide's `RoundingMode` INPUT_TYPE already imports `typing.Literal` into the stub. Check the generated `jix.pyi` (py-ops-arith-wrappers, py-core-dtype-surface).

From the #106 / #107 scan, "Review gaps":

- **jix-py GIL detach by element cost:** the read entry points run without the GIL and write into numpy buffers, and read-paths did not review that. py-core-archive-params checked that the detached paths touch no Python objects and that the heuristic cannot overflow. A dedicated look at the read-path side is still open. (array-frontend-read-paths)

### From the second scan of #106 / #107

ReadTiling and the single-pass read (K192, K197, K198):
- ReadTiling answers two questions with one level: how a parent may fuse a node, and whether a top-level read of the node needs tiling. Select is the only storage where the two differ (K192). Should K192 be fixed by classifying Select as Tiled unless condition and y are ZeroCopy, or by giving `Select::read_data` a pipeline path for typed element types, as Op2 / Clamp / Map have?
- The concept-read-contract audit answered the follow-up question of whether other storages pull their inputs while classified `fused()`: Select is the only one.
- K198: should the identity Slice / Flip / Roll and `&dyn ArrayStorage` forward `read_as_elementwise_pipeline`, or demote their tiling with `forwarded()`? The hunter prefers forwarding the pipeline, since an identity op should cost nothing.
- Not measured: `ident.view() + ident.view()`. The identity Slice opens its own cache scope (slice.rs:146), which hides the parent's `arrays_to_cache`. The inner op is then probably computed twice, with two region-sized temporaries (3x memory). An allocation count would settle it.
- K197: should jix-py's `should_detach` see the tiling from before the ArrayAny erasure, for example through a core accessor, or should the Fusable arm be dropped?

32-bit region checks (K193, K194):
- K193 and K194 share a root: `check_region_size` runs on pull reads that return zero-stride or existing-memory views, which allocate nothing. Should those pulls skip the check on 32-bit, which would fix both? Or should the single-pass decision (read.rs:458) and the stepped-Slice box check account for the input itemsizes, for example by carrying a maximum leaf itemsize in the spec?
- Still unexecuted: K193 through the x86 / armv7 Python wheels, and the batch7-5 32-bit slice over-rejection ("any slice on a dim longer than 2**31-1 raises IndexError"). Both need an i686 or armv7 CPython build of the extension.

Python index and shape arguments (K195, K196, K200):
- K195, K196 and the batch7-5 32-bit slice question could share one fix: read start / stop / step from the slice object as Python ints (or Option<i64>), instead of relying on `PySlice_Unpack` and the `Py_ssize_t::try_from(length)` guard. The maintainers should decide whether strict bounds or clamping is the contract for bounds beyond Py_ssize_t.
- Should shape arguments follow the #93 "no bool" policy that Dim follows? That covers reshape / broadcast, and maybe also the `read_size` / `block_shape` params, which take u64 / u32 and accept bool too. numpy rejects bool in shapes but accepts it for repeats / shift.

numpy dtype metadata (K199 and where):
- Suggested fix for K199: `broadcast_operand` carries a numpy dtype only if one is already cached, through a peek that does not compute it, and otherwise keeps the pre-#107 `Operand::Array`. Alternatively, `Operand::Array` could carry an `Option<Py<PyArrayDescr>>` instead of allocating a new Python Array object per broadcast operand.
- where / stack / concatenate of plain numpy arrays, and `jix.asarray` without `dtype=`, drop numpy dtype metadata and struct titles. jix arrays made with `asarray(dtype=)` keep them through the same ops. This is consistent across paths, but users lose titles silently. Is it intended?
- `where()` takes the result's numpy dtype from x only:
  - `where(c, 0, xm)` and `where(c, plain_i4_arr, xm)` drop y's metadata.
  - `where(c, xm, 0)` and the scalar shortcut `where(False, 0, xm)` keep it.

  #107 added `test_where_broadcast_keeps_numpy_dtype`. Should the dtype be merged from both branches, or kept only when x and y agree?

Python boundary, other:
- #107's `advise_huge_pages` is reached only from `Array::to_ndarray_sub` (read.rs:226). jix-py's numpy() allocates through numpy, so Python never gets the huge-page advice. Is that intended?
- The `LazyLock` around `advise_huge_pages` calls `std::env::var` (page_size.rs:17) on whatever thread first reads 4 MiB or more. That races with a concurrent C-level `setenv` in the same process. Rust API only; Python cannot reach it today.
- Free-threaded builds should be checked on 3.14t, together with K152. The point to confirm is that no jix-py internal state races: the frozen PyArray with its Mutex cache, the LazyLock dispatch tables and the PyOnceLock init.

Verification baselines:
- The pre-#106 baselines of K192, K194 and K198 are argued from the `a7b0656` source (`with_cleared_flags`, and the `plain_read`-only single read), not from a rebuilt binary. Only K193 was confirmed against a build of `a7b0656` (i686, 4/4 pass).

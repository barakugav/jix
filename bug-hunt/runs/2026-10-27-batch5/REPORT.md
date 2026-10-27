# jix bug hunt - run 2026-10-27-batch5

- Date: 2026-10-27. The code is at commit `ddbf4c3` ("Bug hunt: sync with main after #91"). #91 fixed, among others, K66 (inner-shape `numpy()`), K70 (sealed `ElementType`), K74 (complex mean) and K79 (`SliceSpec` panic).
- Scope: 12 module hunter units, listed in section 1. No concept or dynamic units ran.
- Process: Hunt, then Dedup, then Verify, then Report. Each finding got one adversarial verifier, which first tried to refute it and then built a repro. Severity follows the table in `bug-hunt/PRIMER.md`.
- What is in this directory:
  - `findings/<ID>.md`: one write-up for each of the 23 findings.
  - `repros/` holds:
    - 13 Rust repros, `repro_2026_10_27_batch5__fNNN.rs`, for F001, F002, F003, F007, F010, F011 and F016-F022.
    - 16 Python repros: `F004.py`, `F005.py`, `F006.py`, `F007.py`, `F008.py`, `F010.py`, `F012.py`-`F017.py` and `F020.py`-`F023.py`.
    - `F009_internal.rs`, a crate-internal test (see "Re-running a repro").
    - `F002_i686_release.log`, the output of the F002 repro on i686 in release.
  - `scratch/` holds 49 Python probes from hunters and verifiers, a log file, and three small scratch crates without build output:
    - `crate-F009/`: a copy of jix with the F009 repro already appended.
    - `crate-util_nd_iter_unordered/`: the 60k-case iterator check (module `hunt_batch5_tests` in `src/util/nd_iter_unordered.rs`).
    - `py_ops_operand_broadcast_arrmap/`: the Miri harness for `array_try_from_fn` / `array_try_map`.
    - `reduction_driver_h2_release.log`.
- Paths in the write-ups that are now out of date:
  - The write-ups cite Rust repros as `jix/tests/repro_2026_10_27_batch5__fNNN.rs`. They have moved to `repros/` under the same names.
  - The 25 hunter scratch tests (`jix/tests/hunt_2026_10_27_batch5__*.rs`) were deleted. They include every Rust experiment cited in the coverage notes, and the third F019 variant (`hunt_2026_10_27_batch5__shape_ops_forwarders_1.rs`, test `warmed_pool_pull_result_with_len1_dim_reused_as_out`).
  - F009's write-up says to append `F009_internal.rs` to the scratch crate. It is already appended; do not append it again.
  - This run's build output was deleted, about 7.3 GB in all:
    - test binaries, `.d` files, incremental caches and fingerprints under `jix/target` (debug, release and the Miri target) and under `target-i686/i686-unknown-linux-musl` (debug and release);
    - `scratch/target-F009` (4.1 GB).

    The first re-run of a repro rebuilds its test binary.

## 1. Summary

| verdict | count |
|---------|------:|
| confirmed | 23 |
| plausible | 0 |
| refuted | 0 |
| intended | 0 |
| unverified (not sent to a verifier) | 0 |

| severity (confirmed + plausible) | count | IDs |
|----------|------:|-----|
| critical | 0 | - |
| high | 1 | F001 |
| medium | 5 | F002, F005, F006, F007, F008 |
| low | 17 | F003, F004, F009, F010, F011, F012, F013, F014, F015, F016, F017, F018, F019, F020, F021, F022, F023 |

By category (confirmed): questionable-semantics 8, panic 6, wrong-result 3, error-handling 3, overflow 1, hang 1, latent-hazard 1.

Hunters run: 12. Hunters that failed: `[]`.

| hunter | raw reports | findings |
|--------|------------:|----------|
| util-nd-iter-unordered | 2 | F001, F002 (both in reduction.rs, found by following the reduction caller) |
| util-nd-iter-ordered | 1 | F009 |
| reduction-driver | 2 | F001, F002 |
| reduction-scalar-traits | 1 | F010 |
| pipeline-nodes | 2 | F003, F011 |
| py-ops-operand-broadcast | 2 | F004, F012 |
| py-core-read-path | 5 | F005, F006, F013, F014, F015 |
| codec-archive-array-format-dtype | 3 | F016, F017, F018 |
| shape-ops-forwarders | 3 | F007, F011, F019 |
| shape-ops-join | 1 | F020 |
| elementwise-ops-kernel-framework | 1 | F021 |
| elementwise-ops-op1-unary-classify-complex | 3 | F008, F022, F023 |

Dedup merged 26 raw reports into 23 findings:
- util-nd-iter-unordered and reduction-driver both reported F001 and F002.
- pipeline-nodes and shape-ops-forwarders both reported F011.
- No report was dropped as a duplicate of a `KNOWN.md` entry.
- Two hunters raised open questions that other hunters reported as findings:
  - util-nd-iter-ordered asked about the u64 -> u32 block-shape truncation, which is F018.
  - pipeline-nodes asked about the NaN element cost, which is F020.

Links to earlier fixes and entries:
- Regressions from recent fixes:
  - F005: #91's fix for K66 made `numpy()` build shapes with up to 12 dims, but the allocation helpers still hold only 8.
  - F015: the same #91 change was not carried over into `plain()`.
  - F017: the K13 fix (#86) applied its zeros-as-1 overflow guard to the archive reader's padded shape.
  - F004: the K2 fix (#86) made `Cast::new` fallible, but jix-py still calls the unwrapping `Array::cast`.
  - F021: the 1 GiB block cap added with the K18 fix (#87) rejects block shapes that a widening view inherits.
- Incomplete fixes: F019 is a K58 trigger that the K58 fix did not cover, because `check_out_buf` still has no aliasing check.
- The same kind of bug as an earlier entry, at a new site:
  - F011: the "more than 8 entries panics" bug of K12 / K32 / K79.
  - F003: a fallible constructor that panics on invalid input. It is rated low, like K79.
  - F016: unchecked arithmetic on untrusted input before validation, which panics only in debug builds (K55, K83).
- Answers to earlier open questions:
  - F010 answers batch4's question about `std` taking the sqrt after casting to the output type.
  - F013 is the self-overlapping `out` question that trial2 and batch3 left open.
  - F001 and F002 are the reduction-driver 32-bit suspicions that batch4 did not examine.

Reachability:
- Reachable from Python (17): F004, F005, F006, F007, F008, F010, F012, F013, F014, F015, F016, F017, F018, F020, F021, F022, F023.
  - F014 and F016 panic only in the debug extension (`maturin develop`). A release wheel returns `out` unchanged for F014. For F016 it returns an error, which in one trigger has the wrong kind.
  - F007 reaches Python only through `concatenate`, with a zero-length broadcast member between non-empty members.
- Rust only (4): F003, F009 (crate-internal), F011, F019.
- 32-bit only (2): F001 and F002. Both were shown on i686 through the public Rust API. The x86 and armv7 Python wheels probably reach them too; this was not run.
- Memory safety:
  - F001 is the only finding that corrupts memory: a heap overflow from safe code, on 32-bit.
  - F013 breaks the no-alias precondition of an unsafe constructor from Python, but no UB results today.

Headlines:
- **F001 (high): heap overflow in var/std on 32-bit.** On i686 or armv7, `var` / `std` over a region of more than 2^32 / 24 output cells wraps the size of the reduction's state buffer. States are then written up to 4 GB past the buffer. Release builds crash with SIGSEGV from a plain `compact_with` or `read_data`.
- **F002 (medium): on 32-bit, reductions over more than 2^32 items silently lose data.** `sum` of 2^32 + 16 ones gives 16, the mean gives 3.7e-9, and argmax indices wrap.
- **Python read path after #91 (F005, F015).** `numpy()`, indexing and `plain()` panic when the array's ndim plus the dtype's inner ndim exceeds 8. `plain()` of an inner-shape dtype returns an array with a different shape and dtype.
- **F006 (medium): permanent hang.** A per-Array Mutex stays locked while Python code runs inside `dtype_to_numpy`. In GIL builds the hang needs a Python `__import__` hook and an aligned struct dtype.
- **F007 (medium): empty Broadcast reads evaluate the inner row.** A zero-element read can raise a spurious divide-by-zero, cost a full inner read, or abort on a huge lazy row.
- **F008 (medium): `jix.log2` / `jix.log10` are not exact on exact powers.** `log10(1000)` is 2.9999999999999996, so `floor` or `int` of the result is off by one.
- **Validation gaps:**
  - on `out` buffers: F013, F014, F019;
  - on archive headers: F016, F017, F018.

## 2. Findings

All findings are confirmed. They are sorted by severity, with ties in ID order.

| ID | severity | verdict | category | title | location | write-up |
|----|----------|---------|----------|-------|----------|----------|
| F001 | high | confirmed | overflow | 32-bit: the reduction's temp state buffer size `out_nitems * size_of::<State>()` wraps, so var/std states are written far past a few-byte pool buffer (heap overflow, SIGSEGV from safe `read_data` / `compact_with`) | `jix/src/ops/reduction.rs:516` | [F001](findings/F001.md) |
| F002 | medium | confirmed | wrong-result | 32-bit: reduction driver computes reduced-stream positions in usize; a reduction over 2^32+ items silently drops everything before the last tile at position k*2^32 (all kernels), and argmax indices are truncated mod 2^32 | `jix/src/ops/reduction.rs:588` | [F002](findings/F002.md) |
| F005 | medium | confirmed | panic | `Array.numpy()` / `__getitem__` / `plain()` panic (ArrayVec capacity) when the array's ndim plus its dtype's inner ndim exceeds 8 (`item()` is not affected) | `jix-py/src/util.rs:103` | [F005](findings/F005.md) |
| F006 | medium | confirmed | hang | `Array.numpy_dtype` holds the per-Array `std::sync::Mutex` while `dtype_to_numpy` runs Python code (`py.import` -> `builtins.__import__`). If that code yields the GIL to a thread using the same Array, or uses the same Array itself, the process hangs forever | `jix-py/src/array.rs:301` | [F006](findings/F006.md) |
| F007 | medium | confirmed | error-handling | Broadcast reads and evaluates its inner row for an EMPTY request (broadcast dim 0..0 becomes inner 0..1), so zero-element reads raise spurious arithmetic errors, do O(row) work, or abort | `jix/src/ops/shape_ops/broadcast.rs:137` | [F007](findings/F007.md) |
| F008 | medium | confirmed | wrong-result | `jix.log(x, base)`, `jix.log2` and `jix.log10` round three times: exact powers come out 1 ulp low (log10(1000) = 2.9999999999999996, f16 log2(32) = 4.996), up to 2.7 ulp error in f64 | `jix-py/src/ops/op1.rs:362` | [F008](findings/F008.md) |
| F003 | low | confirmed | panic | `MapMultiple::new` panics (index out of bounds) on an empty array sequence instead of returning Err like `Stack::new` / `Concatenate::new` | `jix/src/ops/map.rs:256` | [F003](findings/F003.md) |
| F004 | low | confirmed | panic | Widening cast of a huge lazy array raises PanicException instead of RuntimeError: jix-py `cast_impl` calls the panicking convenience `Array::cast` | `jix-py/src/ops/mod.rs:116` | [F004](findings/F004.md) |
| F009 | low | confirmed | latent-hazard (stale internal doc contract) | `NdIterExtBlockOffsetSize` doc prescribes `block_end = end.div_ceil(b)`, which visits a phantom size-0 block for an empty range with an unaligned begin; every caller uses `calc_block_end` instead (latent, crate-internal) | `jix/src/util/iter/block.rs:12` | [F009](findings/F009.md) |
| F010 | low | confirmed | questionable-semantics | std of f16/f32/complex64 casts the f64 variance to the narrow output dtype before sqrt, giving inf, 0, or results tens of ulps off where the correctly rounded std is representable | `jix/src/ops/reduction.rs:2853` | [F010](findings/F010.md) |
| F011 | low | confirmed | error-handling | `RemoveDim::new` / `new_array` panics (`dim_arr` NDIM_MAX assert) on a DimsArg with more than 8 entries instead of returning Err | `jix/src/ops/shape_ops/remove_dim.rs:58` | [F011](findings/F011.md) |
| F012 | low | confirmed | questionable-semantics | asarray / compact / cast with `dtype=` silently wrap out-of-range Python ints (`asarray(300, dtype=uint8)` -> 44, `compact([-1, 5], dtype=uint32)` -> [4294967295, 5]), while every jix op bounds-checks Python ints and raises OverflowError | `jix-py/src/ops/as_array.rs:60` | [F012](findings/F012.md) |
| F013 | low | confirmed | wrong-result | `numpy(out=)` accepts self-overlapping destinations (positive strides that alias), breaking `StridedBuf::from_raw_parts_mut`'s no-alias precondition; reductions folded into such an out leave values that belong to no output cell | `jix-py/src/array.rs:266` | [F013](findings/F013.md) |
| F014 | low | confirmed | panic | `numpy(out=)` with an empty out that has a negative stride on an extent>1 dim (or a huge stride) panics in debug builds: the stride rebuild runs before the nitems==0 early return (debug_assert at array.rs:607, multiply overflow at array.rs:609) | `jix-py/src/array.rs:607` | [F014](findings/F014.md) |
| F015 | low | confirmed | questionable-semantics | `Array.plain()` of a dtype with an inner shape returns a different array: the inner dims become array dims and the dtype becomes the base dtype, contradicting its docstring and `compact()` | `jix-py/src/array.rs:733` | [F015](findings/F015.md) |
| F016 | low | confirmed | panic | `read_from` rounds untrusted header dims up with unchecked `next_multiple_of`: a dim near u64::MAX panics in debug builds (Rust and dev Python) instead of returning InvalidArchive; release wraps and returns InvalidShapeOperation | `jix/src/archive/array.rs:490` | [F016](findings/F016.md) |
| F017 | low | confirmed | questionable-semantics | An empty array that jix writes cannot be read back when its padded shape product (zero dims counted as 1) overflows u64 - the archive reader is stricter than the writer | `jix/src/archive/array.rs:493` | [F017](findings/F017.md) |
| F018 | low | confirmed | questionable-semantics | Archive reader truncates u64 header fields to u32 before validating them, so a block dim or block_size of 2^32 + 2 is silently read as 2 | `jix/src/archive/array.rs:471` | [F018](findings/F018.md) |
| F019 | low | confirmed | questionable-semantics | K58 fix is incomplete: `check_out_buf` still accepts a writable `out` whose strides alias over the requested region (InsertDim pull results, any pool-backed pull reused for a wider region), so push reads return Ok with the region not in `out` | `jix/src/ops/shape_ops/insert_dim.rs:188` | [F019](findings/F019.md) |
| F020 | low | confirmed | panic | The f32 `element_cost` hint has no upper bound, so it becomes +inf past f32::MAX. A zero-length concatenate member then turns it into NaN (0 * inf), and the next multi-input constructor panics in `max_cost_layout_order` (`partial_cmp().unwrap()`) | `jix/src/storage/params.rs:912` | [F020](findings/F020.md) |
| F021 | low | confirmed | questionable-semantics | A widening element-wise view keeps the source's explicit block shape (in elements, all fixed), so `compact()` / `write_to()` with default params fails with InvalidArgument once that block is over 1 GiB at the wider itemsize | `jix/src/ops/op1.rs:36` | [F021](findings/F021.md) |
| F022 | low | confirmed | questionable-semantics | f16 tan(+-177.5) rounds to +-inf with no Overflow report, because Tan's error set (op1.rs:674) has no overflow condition while the f16 kernel rounds the f32 result to f16 (Exp covers the same case) | `jix/src/ops/op1.rs:674` | [F022](findings/F022.md) |
| F023 | low | confirmed | error-handling | `jix.log` silently accepts an invalid base (1, 0, -0.0, negative): every element becomes inf/NaN/signed zero with no error, despite the documented "positive and not 1" precondition | `jix-py/src/ops/op1.rs:378` | [F023](findings/F023.md) |

### Bugs

**F001** (high; Rust on 32-bit targets, probably also the x86 / armv7 Python wheels).
Var/Std states take 24-32 bytes and never fit the 2-8 byte output. `read_data_impl` therefore allocates a temporary state buffer of `out_nitems * size_of::<State>()` bytes, using an unchecked usize multiply. On 32-bit, `check_region_size` bounds the region using only the output itemsize. So for `var` of a broadcast f16 array over 178956971 cells, the buffer size wraps to 8 bytes, and the fold writes states up to about 4.3 GB past the buffer. In release builds both `read_data` and `compact_with` crash with SIGSEGV. Debug builds panic at reduction.rs:516.
Fix: bound the multiply by isize::MAX with a checked multiply, e.g. `check_region_size` with `max(output itemsize, state size)`. Alternatively, allocate the state buffer per bulk.

**F002** (medium; Rust on 32-bit targets).
The driver computes stream positions in usize: the logical strides at reduction.rs:486-497 and the tile base at reduction.rs:588-593. The fold treats position 0 as "first visit", so a tile that starts at position k*2^32 re-initializes the cell. On i686 in release, `broadcast([2^32+16]).sum(0)` gives 16, the mean gives 3.7e-9, and argmax indices come back mod 2^32. Debug builds panic. There is no UB.
Fix: use u64 positions or an explicit first-tile flag, or return an Err when the reduced count exceeds usize::MAX.

**F005** (medium; Python).
Since #91, `numpy()` allocates with the outer shape followed by the dtype's inner dims, up to 12 entries. `numpy_empty` and `numpy_empty_ordered` collect that shape into an ArrayVec of capacity 8. So `numpy()`, `a[...]` and `plain()` raise PanicException for any array whose ndim plus inner ndim exceeds 8, e.g. 6 + 3. This happens in release builds too. `numpy(out=)` works.
Fix: size those buffers for `NDIM_MAX + DTYPE_MAX_NDIM`.

**F006** (medium; Python).
`numpy_dtype` keeps the per-Array std Mutex locked across the call to `dtype_to_numpy`. For aligned struct dtypes, that function calls `py.import`, which runs `builtins.__import__`. Python code running there can yield the GIL to another thread that uses the same Array, or read the Array itself. Either way the process hangs forever; gdb shows the GIL and the Mutex taken in opposite orders. The free-threaded stop-the-world deadlock is argued from the code only.
Fix: compute the descr without holding the lock, then store it if the slot is still empty. Wherever the Mutex is taken while attached, use `lock_py_attached`.

**F007** (medium; Rust and Python).
`Broadcast::read_data` maps every broadcast dim to the inner range `0..1`, even when the request is empty on that dim, and then reads the inner region. So a zero-element read evaluates the whole inner row. The consequences:
- Spurious arithmetic errors. In Python, `jix.concatenate([o, jix.broadcast(q, (0, 3)), o])` raises FloatingPointError.
- O(row) work for an empty read.
- An allocation abort when the inner row is a huge lazy array.

Meanwhile `.compact()` of the same array succeeds.
Fix: return the materialized empty `out` before reading the inner region, as InsertDim does.

**F008** (medium; Python).
`jix.log(x, base)` multiplies the rounded `ln(x)` by a rounded `1/ln(base)`, which is cast to T for f16/f32. `log2` and `log10` are aliases for it. As a result:
- Exact powers come out 1 ulp low: `log10(1000) = 2.9999999999999996`, f16 `log2(32) = 4.996`. So `floor` or `int` of the result is off by one.
- f64 `log10` is off by up to 2.7 ulp; numpy stays within 0.5.

Fix: add native Log2/Log10 kernels in core. For a general base, compute in a wider type and round once.

**F003** (low; Rust only).
For an empty sequence, `MapMultiple::new` skips its checks (`if narrays > 0`) and then indexes `inputs[0]` in `combine_hints` (params.rs:895). So it panics, in release too, instead of returning Err. `Stack::new` and `Concatenate::new` return Err for the same input.
Fix: `ensure!(narrays > 0, InvalidShapeOperation, ..)`.

**F004** (low; Python).
jix-py's `cast_impl` calls core's unwrapping `Array::cast`. Since the K2 fix, `Cast::new` returns Err when the widened byte size overflows u64. So widening a valid lazy array of about 2^61 or more elements raises PanicException instead of RuntimeError. `except Exception` does not catch PanicException. This applies to `jix.cast`, `asarray(dtype=)`, `big + 1.5`, and where/concatenate/stack promotion.
Fix: use `Cast::new_array(..).into_py_result()?` in `cast_typed!`.

**F009** (low; crate-internal doc).
The `# Usage` doc of `NdIterExtBlockOffsetSize` gives the block end as `end.div_ceil(b)`. With that bound, an empty range with an unaligned begin visits a phantom block of size 0. All six callers use the correct `calc_block_end` instead, so no caller is affected today. A new caller that followed the doc would decode one block for nothing.
Fix: correct the doc, or compute the block-space bounds inside `with_block_offset_size_ext`.

**F011** (low; Rust only).
`RemoveDim::new` collects the dims into an 8-capacity `DimArray` before validating them. A `&[usize]` or `Vec<usize>` with 9 or more entries therefore panics (`ndim <= NDIM_MAX`) instead of returning Err. Flip and InsertDim return Err for the same input. Python is protected by `normalize_dims`.
Fix: validate while iterating `dim.get(i)`, or check `dim.len()` first.

**F013** (low; Python).
`check_output_array` checks each `out` stride on its own, so self-overlapping writeable views pass. Examples: `sliding_window_view(writeable=True)`, `as_strided` with equal strides, or a stride smaller than the itemsize. Such an `out` breaks the no-alias precondition of `StridedBuf::from_raw_parts_mut`.
- Reductions fold into the shared slots, which end up holding values that belong to no cell: `sum` gives [14 34 58 70 55 36] where numpy gives [14 20 26 32 34 36].
- Copies are last-write-wins, as in numpy.
- No UB today.

Fix: check for internal overlap (sort the dims of extent > 1 by stride), or read into a temporary and copy.

**F014** (low; Python, debug builds only).
`check_output_array` accepts any empty `out` before it looks at the strides, and `numpy()` rebuilds the strides before its `nitems == 0` early return. Two empty-`out` triggers, built with `as_strided` or `np.ndarray(strides=...)`:
- A negative stride on a dim of extent > 1 fails the `debug_assert!` at array.rs:607.
- A huge stride overflows the multiply at array.rs:609.

Release builds return `out` unchanged (argued from the code, not run).
Fix: move the empty early return before the stride rebuild.

**F016** (low; Rust and Python; the panic is debug-only).
The archive reader pads untrusted header dims with an unchecked `next_multiple_of`, before anything bounds them by i64::MAX. A dim within `b - 1` of u64::MAX panics in debug builds, including the dev extension. Release builds wrap the dim and reject the file. In one trigger the error kind is InvalidShapeOperation instead of InvalidArchive.
Fix: reject header dims > i64::MAX with InvalidArchive before padding, or use `checked_next_multiple_of`.

**F020** (low; Rust and Python).
The f32 `element_cost` hint is never saturated. About 125 rounds of `x = x + x` make it +inf, even on an empty array; so do nested huge reductions. A zero-length concatenate member then turns the cost into NaN (0 * inf). The next join panics in `partial_cmp().unwrap()` at params.rs:912. Debug builds panic earlier, on the `is_finite` debug_assert at array.rs:1662.
Fix: saturate costs at f32::MAX, skip zero-weight inputs in `combine_hints`, and compare with `total_cmp`.

**F023** (low; Python).
`jix.log` never validates `base`. A base of 1, 0, -0.0 or a negative value silently turns every element into inf, NaN or a signed zero. The docstring says the base must be positive and not 1, but nothing checks it. The multiply runs inside an `Array::map`, which has no status channel, so errstate never sees it. For f16, a valid base within about 1.5e-5 of 1 overflows the multiplier to inf.
Fix: raise ValueError for an invalid base, and multiply in a wider type for f16.

### Questionable semantics

**F010** (low; Rust and Python).
`std` takes the sqrt of a variance that `Variance::finalize` has already cast to the narrow output dtype, which throws away the f64 accumulation. For f16 / f32 / complex64, `std` then gives inf (f16 [-300, 300]), 0 (f16 values near 0.01), or a result 26 ulps off (f16 near 1.0), even though the correctly rounded std is representable. NumPy gives the same values, hence low.
Fix: take the sqrt in f64 and cast once.

**F012** (low; Python).
With `dtype=`, `asarray` / `compact` / `cast` convert Python ints without the target dtype: a scalar becomes an i64, and a list becomes int64 through `np.asarray`. A wrapping cast follows. So `asarray(300, dtype=uint8)` is 44 and `compact([-1, 5], dtype=uint32)` is [4294967295, 5]. jix ops and numpy 2 raise OverflowError for out-of-range Python ints. In the other direction, `asarray(2**63, dtype=uint64)` raises although the value fits.
Fix: pass the target dtype down for inputs that have no dtype of their own, and keep the wrapping cast for typed sources.

**F015** (low; Python).
`plain()` is `asarray(self.numpy(index))`. For an inner-shape dtype it therefore returns the base dtype, with the inner dims appended to the shape. That contradicts its docstring ("same dtype as self") and differs from `compact()`. For example, `f[..., 1]` and `f.plain()[..., 1]` select different elements. For an ndim-8 array whose dtype has an inner dim, `plain()` can never work through numpy.
Fix: build the Plain storage directly from the numpy allocation with the jix dtype, or document the expansion.

**F017** (low; Rust and Python).
The reader checks the padded shape with `try_shape_product`, which counts zero dims as 1. That is stricter than the writer's `check_shape_overflow`. So an empty array such as `[0, 3, 0x5555555555555555]`, with block dim 2 on the 3, is written fine and then rejected as InvalidArchive. This is a regression from the K13 fix (#86).
Fix: validate the stored nitems with an ordinary checked product.

**F018** (low; Rust and Python).
The reader casts `block_shape[d]` and `block_size`, which are u64 on the wire, to u32 before validating them. So 2^32 + 2 is accepted as 2 under Strict validation, and writing the array back emits a different header. The data stays self-consistent and correct.
Fix: use `BlockSize::try_from` and return InvalidArchive on failure.

**F019** (low; Rust only).
K58 made pull results read-only when they alias, but only relative to their own region. `check_out_buf` checks stride count, writability and extent, but not aliasing. So a writable pull result passed back as `out` makes the push read return Ok while elements overwrite each other. Two such pull results: InsertDim's inserted dims with stride 0, and packed strides reused for a wider region. The real defect is in `check_out_buf` (buf.rs:307). No UB.
Fix: add an aliasing check to `check_out_buf`, or make every pull result read-only.

**F021** (low; Rust and Python).
Op1 and Op2 copy the input's block shape unchanged, even when the output itemsize grows. The block shape is counted in elements, and all its dims are fixed if it was set explicitly. Example: an array with a 64 MiB u8 block, cast to complex128. Its `compact()` / `write_to()` with default params fails with InvalidArgument, because the block exceeds 1 GiB. The error is clean.
Fix: treat an inherited fixed block as a hint, and rescale or un-fix it when it exceeds the cap.

**F022** (low; Rust and Python).
The f16 `float_op1!` kernel checks error conditions on the f16-rounded result, and Tan's error set has no Overflow condition. So f16 tan(177.5) = -inf and tan(-177.5) = +inf go unreported, even with `over='raise'`. f16 Exp reports the same kind of rounding overflow, and numpy warns. An exhaustive scan found only these two inputs.
Fix: add `Overflow: r.is_infinite() && x.is_finite()` to Tan's error set.

### Re-running a repro

The Rust repros are in `repros/`. To run one:
1. Copy it back to `jix/tests/`.
2. From the repo root, run `cargo test --manifest-path jix/Cargo.toml --all-features --test <name>`, where `<name>` is the file name without `.rs`.
3. Delete the copy afterwards. `jix/tests/repro_*.rs` is gitignored, so a leftover copy would not show up in `git status`.

Example:

    cp bug-hunt/runs/2026-10-27-batch5/repros/repro_2026_10_27_batch5__f007.rs jix/tests/
    cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_27_batch5__f007 -- --nocapture --test-threads=1
    rm jix/tests/repro_2026_10_27_batch5__f007.rs

Every repro asserts the correct behavior, so it fails while its bug is present. Control tests pass.

Per-repro notes:
- **f001, f002**: gated on `target_pointer_width = "32"`; on x86_64 they compile to empty test binaries.
  - Run them on i686 with `bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test <name> -- --nocapture --test-threads 1`. This needs the i686 part of `bash bug-hunt/setup.sh`.
  - f001: in debug both tests fail (multiply overflow at reduction.rs:516). With `--release`, each test crashes the process with SIGSEGV, so run them one at a time.
  - f002: with `--release`, all 5 tests fail with wrong values after about a minute (log: `repros/F002_i686_release.log`). In debug all 5 panic.
- **f003**: 3 of 4 tests fail; the Stack/Concatenate control passes.
- **f007**: use `--test-threads=1`. 4 tests fail and 2 pass, including the baseline control. The ignored test aborts the process (SIGABRT); run it alone with `-- --ignored empty_broadcast_of_huge`.
- **f010**: all 7 tests fail.
- **f011**: 2 of 3 tests fail; the Flip/InsertDim control passes.
- **f016**:
  - In debug, 2 of 3 tests fail (add overflow) and `rewrite_sanity` passes.
  - With `--release`, the first test passes (InvalidArchive) and the second fails on the error kind (InvalidShapeOperation).
- **f017**: both tests fail at the read step.
- **f018**: use `--test-threads 1`. 2 of 3 tests fail; the sanity test passes.
- **f019**: use `--test-threads 1`. Both tests fail.
- **f020**: both tests fail. In debug they panic at array.rs:1662 (the `is_finite` debug_assert); with `--release`, at params.rs:912.
- **f021**: use `--test-threads 1`. Both tests fail.
- **f022**: both tests fail.
- **F009** (crate-internal):
  - The items under test are `pub(crate)`, so the repro lives in the scratch copy of the crate, `scratch/crate-F009`. Its `src/util/iter/block.rs` already contains `repros/F009_internal.rs`.
  - Run `CARGO_TARGET_DIR=bug-hunt/runs/2026-10-27-batch5/scratch/target-F009 cargo test --manifest-path bug-hunt/runs/2026-10-27-batch5/scratch/crate-F009/Cargo.toml --all-features --lib f009_repro -- --nocapture`. All 3 tests fail.
  - This is a full build of about 4 GB. Delete the target dir afterwards.

To run a Python repro:
- First build the debug extension into `.venv`, either with `maturin develop` in `jix-py/` or with `bash bug-hunt/setup.sh`.
- Then run the script, e.g. `.venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F005.py`. Each script exits non-zero while its bug is present.
- F004: set `RUST_BACKTRACE=0` to keep the panic output short.
- F006: each scenario runs in a child process with a 15 s timeout, so the script takes about 45 s.
- F014, F016 and F020 show the debug extension's PanicExceptions. Expected with a release wheel, though none of this was run:
  - F014 should pass.
  - F016 should raise an error instead of panicking.
  - F020 should panic later, at the second `jix.concatenate`.

## 3. Unverified findings

None. Every deduplicated finding was sent to a verifier.

## 4. Refuted and intended

No finding was refuted, judged intended, or dropped as already known.

### Verifier corrections to hunter claims

- F003:
  - `Stack::new` and `Concatenate::new` reject an empty sequence with InvalidShapeOperation, not InvalidArgument.
  - Severity lowered from medium to low: the input is invalid and only Rust reaches it (precedent K79).
- F004:
  - `negative` on unsigned arrays casts to the same width and cannot overflow; the repro shows `-big_u8` works.
  - `cast_impl` is the only unwrapping call in jix-py ops that changes the itemsize.
  - Severity lowered from medium to low, because only the exception type is wrong.
- F005: `item()` is not affected. It requires ndim 0, so the numpy shape has at most 4 dims.
- F006: the free-threaded deadlock is argued only. In a GIL build the trigger needs a Python `__import__` hook and an aligned struct dtype whose numpy dtype is not cached yet.
- F007: Python `numpy()` returns early for an empty array, so `jix.broadcast(q, (0, 3)).numpy()` alone works. Python reaches the bug through `concatenate`.
- F009: the impact is smaller than the hunter implied:
  - No caller follows the doc, and callers return early on empty regions.
  - The phantom block is inside the grid and has size 0.
  - The "bounds not tied to the range" part is an API design remark, not a separate defect.
- F012:
  - Wrapping is correct for typed sources; it matches `np.asarray(np.int64(300), dtype=np.uint8)`.
  - Only literal Python ints and lists of them are affected.
  - The verifier also found the reverse case: `asarray(2**63, dtype=uint64)` raises.
- F013: no UB was found. Every core write to `out` goes through one pointer at a time.
- F014: the verifier found a second debug panic with the same root cause, the multiply overflow at array.rs:609.
- F015: "`p[1]` and `f[1]` select differently" is wrong; indices without an ellipsis give identical results. The silent divergence comes from end-anchored indexing (`[..., k]`).
- F016: the trigger is a dim within `b - 1` of u64::MAX, not any dim > i64::MAX. For example, 2^63 with b = 2 does not overflow.
- F019: the true defect is in `check_out_buf`, not in insert_dim.rs. The packed-strides variant shows that making only InsertDim's pull read-only would not fix it.
- F020: the verifier found a simpler trigger than the hunter's nested reductions: about 125 rounds of `x = x + x` on an empty f32 array.
- F021:
  - The hunter's second point, that blocks grow k times in bytes below the limit, is not counted as a defect. The `block_shape` docs say a fixed shape is preserved as-is.
  - The hunter only argued Python reachability; `F021.py` shows it.
  - The hunter's remark that Rust has no real -> complex cast is wrong; the repro uses `cast::<Complex<f64>>()` on u8.
- F023: base=NaN (all NaN) and base=inf (all 0) are not bugs; both match `math.log`.

### Stale context-doc suspicions settled by hunters (update the docs)

- util-nd:
  - nd.rs:61 / `remaining()` product overflow: does not hold. Every NdIter walk is a sub-region or block grid of a shape that passed `check_shape_overflow`. Tested, e.g. with a `[2^31, 2^31, 0]` u8 array.
  - block.rs:10-16 `div_ceil` doc: holds, but only as the latent doc issue F009.
- pipeline:
  - elementwise_pipeline.rs:891/:869 overflow and truncation: refuted. They are bounded by `check_shape_overflow` on 64-bit and by `check_region_size` on 32-bit, and were tested on x86_64 and i686.
  - :845-856 bool validity: obsolete, since `bool` is no longer `Dtyped` (bool8, #89).
  - pipeline.md has the `to_buf_impl` failure modes swapped. Yielding fewer operands than `N_OPERANDS` panics on an unwrap; yielding more is only debug_asserted.
- pipeline / elementwise-ops: the `assert_unchecked_eq` in op2.rs:181 and into_type.rs is gone. It is now a `debug_assert_eq` (K25), and `ArrayStorage` is sealed (K50).
- shape-ops:
  - "PermuteDims/RemoveDim only check ndim": no longer true, because `check_out_buf` now runs `check_get_range` first (K61).
  - buf.rs:282-297, "check_out_buf never checks extent": fixed by K14. Aliasing is still unchecked (F019).
  - concatenate.rs:111 unchecked border sum: fixed with `checked_add` (K23).
- reduction:
  - Empty sum gives -0.0: fixed (K62).
  - Complex mean: fixed (K74, #91).
  - Complex Sum/Product seeds are not identities: true, but they match NumPy bit for bit, so not reported (see the open questions).
- codec-archive / types:
  - S1 (struct field layout): fixed by #88 (K42).
  - S7 / S8: fixed (K53 / K52).
  - conversions.rs:58 (bool8 read from files): by design since #89.
- py-ops / py-core:
  - Byte order ignored (dtype.rs:136/138): fixed (K20); every path tested rejects it.
  - A user numpy dtype attached verbatim (ops/mod.rs:96, as_array.rs:63): refuted, because `dtype_from_numpy` rejects it first.
  - operand.rs:185 bool bytes: fixed (#89).
  - operand.rs:176 negative stride on extent <= 1 dims: fixed (K80).
- py-core:
  - array.rs:551 inner-shape strides: fixed by #91. F005 is a new problem in the new code.
  - array.rs:529 / util.rs:86 32-bit casts: not a bug, because numpy bounds the shape first. Reasoned only; no 32-bit CPython was available.
  - shape_ops.rs:314 and :245, `normalize_dims`, and the array.rs:489 docstring: fixed (K21, K33, K32, K40).
- elementwise-ops:
  - `power(x, 0.5)` -> Sqrt (jix-py op2.rs:681-688): refuted. numpy's scalar-exponent fast path gives the same nan and -0.0.
  - `power(x, 2)` -> Square dropping Overflow: this is K60, which is intended.

### Hypotheses the hunters checked and ruled out

- util-nd-iter-unordered (checked with a 60k-case randomized test):
  - Dim coalescing is an exact identity for any strides.
  - `dims_sort_by` always yields a permutation.
  - The CONTIG flags match the strides on every sentinel.
  - On 64-bit, each reduction cell is first visited at stream position 0.
- util-nd-iter-ordered:
  - The advance carry is correct for ndim 0-8, and no `Dim<N>` with N > 8 can exist.
  - Block border math cannot underflow.
  - On 32-bit, `check_region_size` bounds the `from_u64` truncation.
- reduction-driver:
  - Bulks partition the kept ranges exactly.
  - Empty reductions and `out_nitems == 0` are handled.
  - In-place finalize is correct for every kernel whose state lives in `out`.
  - An inf `element_cost` is harmless inside the driver itself.
- reduction-scalar-traits:
  - The Welford bulk update equals the scalar update for real and complex inputs.
  - SIMD i64/u64 mul and converts match scalar on every level.
  - ArgMax/ArgMin handle NaN, ties, +-0 and bool8 correctly.
  - Integer sum/mean wrapping is intentional (#81).
- pipeline-nodes:
  - `pick_output_layout` always gives a packed bijection, with 0 strides exactly on the collapsed dims (2932-case differential).
  - `N_OPERANDS` is consistent for all 11 implementors.
  - The Vec/slice scratch is sound under Miri.
  - The dtype is checked before every `transmute_copy`.
  - The `MaybeCompact::as_compact` unwrap always succeeds.
- py-ops-operand-broadcast:
  - Every `Plain::new` precondition holds for numpy inputs: lifetime, strides, bool8 bytes, byte order and NULL data.
  - `array_try_from_fn` / `array_try_map` are sound under Miri, with panics and drops.
  - Broadcasting and unusual numpy layouts match numpy with 0 mismatches.
- py-core-read-path:
  - Over about 70 op chains, push equals pull, with no holes and no writes outside `dst`.
  - Huge and unaligned strides on length-1 `out` dims are safe; the K36 / K78 fixes hold.
  - Fuzzing the compact params gave no panics.
  - The struct dtype check on `out` is strict.
- codec-archive-array-format-dtype:
  - Read params never change the values read (K43).
  - The `builder_init` unwrap is unreachable.
  - `to_proto` / `from_proto` is lossless for nested, packed and array-field structs.
  - Unknown enum and oneof values give an error.
- shape-ops-forwarders:
  - The lifetime transmute at buf.rs:413 is sound.
  - The stride closures are exact inverses (200 random sub-regions per chain, pull and push).
  - The PLAIN_READ fast paths are correct.
  - Specs always give valid permutations.
- shape-ops-join:
  - The member borders are monotone.
  - The `inner_contiguous` and `in_place` fast paths are exact.
  - Every element of the region is written.
  - An ElementType mismatch is unreachable (K50, K70).
- elementwise-ops-kernel-framework:
  - Op1/Op2 `read_bulk` forwarding and `transmute_copy` are sound.
  - Every Op1/Op2 user goes through the zero-itemsize and shape-overflow checks.
  - Swapped comparisons give correct values and error messages.
- elementwise-ops-op1-unary-classify-complex:
  - SIMD and scalar agree on every x86 level, including signed zero, NaN and integer MIN.
  - The T-precision checks in `float_op1!` agree with the f64 re-test.
  - The Sqrt status for -0.0, NaN and -inf matches scalar.
  - The AVX2 bool-output lane mapping is correct.

### Noticed but not reported (judgment calls)

- `MAX_ITEMSIZE` for `[Array<S>; N]` / tuple sequences ignores N. Overflowing the stack in a debug build takes N = 512 f64 arrays, which is unrealistic. (pipeline-nodes)
- Doc nits (pipeline-nodes):
  - `Array::map`'s `# Panics` section names a dtype mismatch that cannot happen.
  - The `map_multiple` panic list is incomplete.
  - The Operand doc still says the cursor "advances".
- A 4th `NdIter` builder extension would compile and then panic at runtime. No caller uses more than two, so this is a latent API wart. (util-nd-iter-ordered)
- Complex reduction seeds are not identities; NumPy behaves the same. (reduction-scalar-traits)
- i686 debug builds trip the `nd_copy.rs:93` debug_assert when compacting f64. This is the K8 side effect already noted in batch3. (util-nd-iter-unordered)

## 5. Coverage

- **util-nd-iter-unordered**:
  - Read `nd_iter_unordered.rs` and `nd_iter_unordered_dyn.rs` line by line, with every caller (pipeline, where_op, reduction).
  - Ran a 60k-case randomized check in a scratch crate copy, plus 30k reduction-order cases and 200k comparator sorts. All passed.
  - Found no defect in the unit itself; its two findings are in reduction.rs.
- **util-nd-iter-ordered**:
  - Read `iter/mod.rs`, `nd.rs`, `block.rs`, `strides.rs` and `calc_block_end` in full.
  - Checked the carry for ndim 0-8, the block border math and every caller's bounds.
  - One finding (F009).
- **reduction-driver**:
  - Read reduction.rs 1-660 and 1137-1279 (driver, tiling, state buffer, finalize) and the fold loops the driver relies on.
  - Confirmed both 32-bit suspicions on i686 (F001, F002).
- **reduction-scalar-traits**:
  - Read reduction.rs 1378-2069 (Sum, Product, ArgMax/ArgMin, Mean, Welford Var), and the kernels and folds that use them.
  - Checked fearless_simd's i64/u64 mul and converts at each SIMD level.
  - Ran three Rust test files and Python probes. One finding (F010).
- **pipeline-nodes**:
  - Read elementwise_pipeline.rs 734-993, map.rs, into_type.rs and maybe_compact.rs.
  - Ran a 2932-case pull/push differential, overflow probes on x86_64 and i686, and Miri on the Vec/slice scratch.
  - Findings F003, F011.
- **py-ops-operand-broadcast**:
  - Read operand.rs, broadcast.rs, as_array.rs and the util.rs helpers in full, and checked every unsafe precondition.
  - Ran Miri on `array_try_*`, and 14 Python probe scripts, mostly differential against numpy.
  - Findings F004, F012.
- **py-core-read-path**:
  - Read the jix-py Array read path (array.rs 1-313, 405-760 and 1615-1770; util.rs) and every unsafe block in it.
  - Read about 70 op chains into 4 `out` layouts with sentinels, and fuzzed the compact params.
  - Findings F005, F006, F013, F014, F015.
- **codec-archive-array-format-dtype**:
  - Read archive/array.rs 1-559 and the schema conversions in full, with the dtype, params and writer contracts.
  - Tested round trips of 0-d, empty and struct dtypes, and probed bool8 bytes.
  - Findings F016, F017, F018.
- **shape-ops-forwarders**:
  - Read shape_ops/mod.rs, permute_dims.rs, insert_dim.rs, remove_dim.rs, broadcast.rs and storage/buf.rs in full.
  - Ran random sub-region sweeps of forwarder chains, pull and push.
  - Findings F007, F011, F019.
- **shape-ops-join**:
  - Read concatenate.rs, stack.rs and the arr_sequence impls they use.
  - Ran a pull/push differential over 300 random sub-regions per case, and a Python differential.
  - One finding (F020).
- **elementwise-ops-kernel-framework**:
  - Read op1.rs 1-268 and op2.rs 1-327, with the pipeline driver and every Op1/Op2 consumer.
  - Checked each unsafe precondition and all 50 `define_op1!` / `define_op2!` invocations.
  - One finding (F021).
- **elementwise-ops-op1-unary-classify-complex**:
  - Read op1.rs 269-1261, logical1.rs, complex.rs, traits_util.rs and simd.rs.
  - Checked fearless_simd at each SIMD level.
  - Ran bit-exact edge-value tests at the AVX-512 and SSE2 levels.
  - Findings F008, F022, F023.

Skipped or only skimmed:
- util-nd-iter-unordered:
  - The where_op and pipeline kernels beyond their use of the iterator, the SIMD bodies, and the reduction kernels.
  - block.rs, map.rs and arr_sequence; this hunter did not settle the suspicions there.
- util-nd-iter-ordered:
  - The in-file tests were only skimmed.
  - The util-nd suspicions at nd_copy.rs:90, map.rs:238-247 and the arr_sequence.rs `set_len` are outside its files.
  - No Miri or i686 run.
- reduction-driver:
  - Per-kernel scalar math and the SIMD fold internals.
  - Python reachability of the 32-bit findings; no 32-bit Python was available.
- reduction-scalar-traits: the AVX2, SSE4.2, NEON and WASM bodies were checked by reading only, because the host runs AVX-512.
- pipeline-nodes:
  - The `REQUIRE_ALIGNED` (armv7) paths were reasoned about only.
  - The Python bindings were not exercised.
  - The map tests were only skimmed.
- py-ops-operand-broadcast:
  - Only skimmed: the dispatch tables, the `numpy()` stride rebuild, `check_output_array`, and the `dtype_to_numpy` struct round trips.
  - Free-threaded concurrent mutation of wrapped numpy buffers was not tested.
- py-core-read-path:
  - No free-threaded, 32-bit, release jix-py or Miri runs, so the free-threaded deadlock and the TOCTOU are argued only.
  - plain.rs was only skimmed, and test_getitem.py / test_slice.py were only grepped.
  - array.rs 313-405 was skipped.
- codec-archive-array-format-dtype:
  - Only skimmed: the archive/array.rs tests (560-1300), the BlockTableStorageRead/Mmap and common.rs reader internals, and the codec pipeline.
  - The writer's unchecked `nitems = nblocks * block_size` (archive/block.rs:137) was not checked.
  - No 32-bit run.
- shape-ops-forwarders:
  - The pipeline and reduction internals.
  - The other shape ops (slice, reshape, flip, roll, repeat, tile, concatenate, stack).
  - No i686 or Miri run.
- shape-ops-join:
  - The ElementwisePipelineTuple / `read_bulk_as_iter` parts of arr_sequence.rs.
  - No Miri or i686 run.
- elementwise-ops-kernel-framework:
  - Per-kernel arithmetic, SIMD-vs-scalar agreement of individual kernels, the `map_masks_to_bools` lane order, the driver's staging and alignment paths, and jix-py dispatch/promotion.
  - A 15/17-deep Add chain ran out of memory at compile time, so the dyn path was tested instead.
- elementwise-ops-op1-unary-classify-complex:
  - The NEON and AVX-512 bool-mask paths, which are scalar by design.
  - The op1 proptests beyond `simd_bodies`, the jix-py dispatch/promotion machinery, and error.rs.

## 6. Open questions from hunters

### Answered during this run
- Does archive/array.rs:471 accept a block dim of 2^32 + k as k? Yes, see F018. (util-nd-iter-ordered)
- Can a zero-weight concatenate member turn an inf `element_cost` into a NaN that panics `max_cost_layout_order`? Yes, see F020. (pipeline-nodes)
- Is F021 reachable from Python? Yes, `F021.py` shows it. (elementwise-ops-kernel-framework)

### 32-bit and huge sizes
- F001 reachability: can the 32-bit Python wheels reach it, e.g. with `jix.compact` or `write_array` and a large `block_shape`, on a var/std of f16 or complex64? Not run; no 32-bit Python was available. (util-nd-iter-unordered)
- F001 audit: do other ops also size internal scratch buffers from an item width larger than the output itemsize that `check_region_size` uses? (util-nd-iter-unordered)
- F001 fix: bound `out_nitems * size_of::<State>()` with a usize check in `ReductionOp::new` and `read_data_impl` (like `check_region_size`, but with the state size)? Or split huge regions into several state passes? (reduction-driver)
- F002:
  - Argmax/argmin over a single dim longer than 2^32 on 32-bit should also return wrapped indices and drop the prefix. This was argued from the code only; the repro covers argmax only through a concatenation. (util-nd-iter-unordered)
  - Even with u64 stream positions, the in-tile logical offsets (NdIterUnordered operand 2, in usize) and `FoldOneCellCtx::item_idx` can still wrap for tiles whose logical span exceeds 2^32. Check the fold loops after the driver is fixed. (reduction-driver)
  - Alternative fix: an explicit first-visit flag, plus an Err for reductions longer than usize::MAX. (util-nd-iter-unordered)
- Possible debug panic on i686 for empty pull reads through shape ops (shape-ops-forwarders; not run):
  - InsertDim's short-circuit and Slice, Flip, Repeat, Tile, Concatenate and Reshape call `materialize_out_buf` with per-dim `as usize` extents.
  - `strides_for_layout_order` then multiplies the nonzero extents unchecked. Example: region `[0..0, 0..2^20, 0..2^20]` of a u8 broadcast with an inserted dim.
  - Release builds would wrap harmlessly.
- Release builds reject archive header dims > i64::MAX with InvalidShapeOperation instead of InvalidArchive (see F016). A caller that matches on the error kind to detect corrupt files would miss these. (codec-archive-array-format-dtype)

### Aliasing and the out= contract
- Should `check_out_buf` reject aliasing outs in general? (shape-ops-forwarders)
  - One approach: sort the dims of extent > 1 by stride, and require each stride to be at least the span of the previous dims plus the itemsize.
  - Approach (b) of the K58 fix (read-only pull results) cannot cover reuse for a different region (F019).
  - Python rejects 0 strides, but not equal nonzero strides such as `[4, 4]` over `[2, 2]`.
- F013 fix: reject internal overlap, as numpy's `solve_may_have_internal_overlap` does, or copy through a temporary like numpy? Should `from_raw_parts_mut` also get a debug-mode overlap check? (py-core-read-path)
- If a push-mode read fails partway, states have already been folded into the caller's `out`. Is it documented anywhere that `out=` contents are unspecified after an error? (reduction-driver)
- Free-threaded TOCTOU (py-core-read-path):
  - `check_output_array` reads `out`'s shape, strides and data pointer. `numpy()` then reads the strides and pointer again from the live ndarray (array.rs:589, 630), and the write runs detached.
  - On 3.14t another thread can change `out.shape` or `out.strides` in between, so the write would use strides that were never validated.
  - Should `check_output_array` return the snapshot it validated?

### Python boundary
- F006 likelihood: on GIL builds, can anything inside `dtype_to_numpy` run bytecode, GC or finalizers without a user import hook, e.g. `PyErr_CheckSignals` or a numpy descriptor path? On 3.14t, which critical sections inside it are safe points? The fix is cheap either way. (py-core-read-path)
- F012 design: should `asarray(value, dtype)` pass the dtype to `numpy.asarray` for non-array inputs, or bounds-check untyped Python scalars through `Operand::cast`? (py-ops-operand-broadcast)
- Core `Array::cast` (cast.rs:477-486) panics on a widened shape overflow but has no `# Panics` section, unlike `map`, `sub_dtype` and `where_op` (see F004). (py-ops-operand-broadcast)
- Non-canonical bools (py-core-read-path, codec-archive-array-format-dtype):
  - A bool8 array can hold bytes other than 0/1, e.g. from a zero-copy `asarray` of a u8 view, or from a .jix file. `numpy()` copies them byte for byte into a numpy bool array.
  - NumPy formally treats such bools as invalid, although the probed ops read them as True.
  - Should `numpy()` normalize them to 0/1, or is the pass-through intended? Could any numpy path or C extension misbehave on them?
- Struct dtypes that go through `asarray` lose numpy field titles and round-trip the align flag inconsistently. Cosmetic? (py-ops-operand-broadcast)
  - With a titled struct `z`, `jix.asarray(z).numpy(out=np.empty_like(z))` raises ValueError.
  - A packed (u1, u1) struct comes back with align=True, while (i4, i4) stays unaligned.
- Python `concatenate` / `stack` take the output numpy dtype from the first input (jix-py shape_ops.rs:907, 984). If the inputs' jix dtypes are equal but their numpy descrs differ, e.g. an aligned and a packed struct, the output silently gets the first descr. Not tested. (shape-ops-join)
- `Scalar::from_any` with `only_scalar=true`, used for the isclose `rtol` / `atol`, rejects 0-d ndarrays, while the ordinary operand path accepts them. Probably intended, given the docstring. (py-ops-operand-broadcast)

### Numerics
- Complex reduction seeds are not identities: `Sum::init` is (-0.0, +0.0) and `Product::init` is (1, 0). (reduction-scalar-traits)
  - A length-1 or `dim=()` reduction turns an imaginary -0.0 into +0.0, and an inf component makes the other component NaN. NumPy does the same.
  - If jix wants "a reduction over one item equals the item", it could seed with `cast(x)` in `init_state(Some(x))`.
- The multi-cell fold accumulates each cell sequentially with a scalar `update`, with no pairwise tree. So an f32 `sum(0)` of a C-order (n, k) array of ones with n > 2^24 may stall at 16777216, as NumPy's non-contiguous reduction does. Not verified. Is that the intended trade-off? (reduction-scalar-traits)
- Should float Square, Mul and Add report Overflow? f16 `square(300)` and `multiply(300, 300)` return inf silently, while numpy warns for both. The jix behavior is consistent, and K60 was triaged as intended. (elementwise-ops-op1-unary-classify-complex)
- `jix.sqrt` / `jix.exp` of int8 inputs promote to float32, where numpy gives float16. Is that a deliberate choice of the Safe cast tables? (elementwise-ops-op1-unary-classify-complex)
- F008 fix, two options: (elementwise-ops-op1-unary-classify-complex)
  - native Log2 / Log10 kernels in core, with f16 computed in f32 and rounded once;
  - or compute `ln(x) / ln(b)` in f64 in jix-py for f16/f32 inputs. This is cheaper, but still not correctly rounded for f64.

### Spec hints and block layout
- F021: is it intended that `block_shape` (counted in elements) and `block_shape_fixed_dims` pass unchanged through ops that change the itemsize (Cast, Map, Map2, MapMultiple, reductions)? If so, should `tune` rescale or un-fix an inherited block that exceeds `MAX_BLOCK_SIZE_BYTES`, instead of erroring? In the other direction, a narrowing op silently gives blocks k times smaller than configured. (elementwise-ops-kernel-framework)
- F020 fix: two options, and the write-up suggests doing both: (shape-ops-join)
  - saturate `element_cost` at a finite maximum where it is produced: reduction.rs:177, and the Op2 / where / map sums, which can double through shared Arcs;
  - or have `max_cost_layout_order` use `total_cmp` and `combine_hints` skip zero-weight inputs.

  The debug_assert at array.rs:1662 already makes debug builds, including the dev extension, panic on any lazy view with an inf cost.
- Should `MAX_ITEMSIZE` for `[Array<S>; N]` / tuple sequences account for N * itemsize, as its doc ("every intermediate value") suggests? Stack use currently scales with N * LANES. (pipeline-nodes)

### API design and unproven correctness
- `DimsArg` is a public, unsealed, safe trait. Should the dims be snapshotted once? (shape-ops-forwarders)
  - `InsertDim::new` computes `orig_ndim + dim.len()` unchecked.
  - InsertDim and RemoveDim trust `len()` / `get()` to stay consistent across calls.
  - A hostile impl only gets panics or Errs; no UB was found.
- reshape.rs:296-314 matches new dims to original dims by equal logical stride, in a left-to-right scan. Nobody has proved that an unmatched new dim can never contribute an offset to a matched original dim. Worth a check by the reshape unit. (util-nd-iter-ordered)
- Archive forward compatibility: is a format or minimum-reader version check intended? (codec-archive-array-format-dtype)
  - `FileMetadata.lib_version_semver` is written but never read, and prost silently ignores unknown fields.
  - A future writer could add a field that changes how data is interpreted, such as a layout flag in ArrayHeader or a zstd parameter. Today's readers would then read such files silently wrong.

### Platform coverage
- Which level does fearless_simd's `Level::new()` pick on the CI machines, AVX-512 or AVX2? The AVX2-only `update_bulk_simd` and `cast_bulk` branches (the u32 -> u64 and f32 -> f64 widens) are verified only by reading the code. (reduction-scalar-traits)

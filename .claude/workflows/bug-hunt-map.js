export const meta = {
  name: 'bug-hunt-map',
  description: 'Regenerate the bug-hunt context docs (bug-hunt/context/*.md): one mapping agent per jix subsystem',
  whenToUse: 'Before a bug hunt when the code has changed since the context docs were generated. args: {only?: [subsystem ids], since?: <commit of the last scan>}. Review the resulting diff before hunting.',
  phases: [{ title: 'Map', detail: 'one agent per subsystem writes bug-hunt/context/<id>.md' }],
}

// The docs this writes are what every hunter trusts about contracts and intended behavior, so
// review the diff (git diff bug-hunt/context) before a run, and fix anything wrong by hand.
// Each agent also proposes review units; update MODULE_UNITS in bug-hunt.js if the code moved.

const A = args || {}

const SUBSYSTEMS = [
  {
    id: 'storage-core',
    title: 'the ArrayStorage contract (read_data with Index / OutBuf, the impls for &S and &mut S), StridedBuf, Index, and the Plain / Scalar / any backends',
    files: 'jix/src/storage/core_trait.rs, jix/src/storage/buf.rs, jix/src/storage/index.rs, jix/src/storage/mod.rs, jix/src/storage/any.rs, jix/src/storage/info.rs, jix/src/storage/scalar.rs, jix/src/storage/plain.rs (non-test parts)',
    extra: 'Cover the pull / push contract of read_data with OutBuf, what an Index guarantees (non-empty sizes: who checks that a region is non-empty before a storage sees it?), the forwarding impls for &S / &mut S, and every unsafe constructor of StridedBuf.',
  },
  {
    id: 'array-frontend',
    title: 'the Array<S> front end (construction, read entry points, the request-splitting heuristic, compaction and write paths) and the ReadContext with its read-result cache for arrays used more than once in an expression',
    files: 'jix/src/array/mod.rs, jix/src/array/read.rs, jix/src/array/construct.rs, jix/src/array/compact.rs, jix/src/context.rs, jix/src/storage/cache_id.rs (non-test parts). Consult jix/src/storage/compact.rs (read_cached), jix/src/storage/params.rs (arrays_to_cache in the spec), jix/src/ops/mod.rs and how ops forward cache ids.',
    extra: 'Cover the entry-point checks (each read is checked once at its entry point and storages below never see an empty region - which paths bypass the entry points?), pull vs push (to_buf_strided / to_buf_strided_into / to_buf_slice), the splitting heuristic, and the read cache: cache keys and scopes, when a cached buffer is returned for a different region / dtype / out, buffers lent out while the cache holds them, nested and reentrant reads, Clone of cached arrays. There is no previous doc for this subsystem: the Array front end was covered by the old storage-core doc, so read `git show HEAD:bug-hunt/context/storage-core.md` and reuse what is still accurate.',
  },
  {
    id: 'storage-compact',
    title: 'block-compressed storage (Compact / CompactMmap), block grid bookkeeping, and ArrayParams / ArraySpec (block-shape auto selection, read-shape heuristics)',
    files: 'jix/src/storage/compact.rs, jix/src/storage/block.rs, jix/src/storage/params.rs (consult jix/src/codec/mod.rs for the decode side and jix/src/util/mod.rs for scale_read_shape / calc_block_end)',
    extra: 'Also cover partial edge blocks, block shapes that do not divide the array shape, and mmap\'d files that are malformed or truncated.',
  },
  {
    id: 'codec-archive',
    title: 'the codec pipeline (byte-shuffle / bit-shuffle filters, zstd, Encoder / Decoder) and the .jix archive format (write, read, protobuf schema conversions, versioning)',
    files: 'jix/src/codec/ (all), jix/src/archive/ (all except the generated jix/src/archive/schema/_proto_gen/, which, like the .proto sources in jix/schema/proto/jix/v1/, is reference only). The codec swaps zstd for memcpy under cfg(miri).',
    extra: 'Treat archive files as potentially untrusted / corrupted input: which malformed files cause panics, unbounded allocation, out-of-bounds access or UB, versus clean errors? Spell out which archive fields are validated on load (per ArchiveValidation mode) and which are trusted.',
  },
  {
    id: 'pipeline',
    title: 'the element-wise pipeline engine that fuses chains of element-wise ops into one SIMD loop, the generic op plumbing, and the SIMD scalar layer',
    files: 'jix/src/ops/elementwise/pipeline.rs (non-test part), jix/src/ops/elementwise/op1.rs, jix/src/ops/elementwise/op2.rs, jix/src/ops/elementwise/mod.rs, jix/src/ops/elementwise/map.rs, jix/src/ops/mod.rs, jix/src/ops/dims_arg.rs, jix/src/ops/maybe_compact.rs, jix/src/ops/into_type.rs, jix/src/scalar/ (all). Consult jix/src/util/nd_iter_unordered*.rs, jix/src/util/nd_copy.rs, jix/src/storage/buf.rs.',
    extra: 'Cover the safety contract of read_bulk, what an op node must guarantee, what CONTIGUOUS / REQUIRE_ALIGNED / N_OPERANDS / MAX_ITEMSIZE mean and where they are checked, the static-N vs dyn paths, loop tails, operands of different itemsizes, and how output-vs-input overlap is handled.',
  },
  {
    id: 'elementwise-ops',
    title: 'the concrete element-wise operations and kernels: arith, div_rem (divide / remainder with rounding_mode), exp_log, trig, rounding, min_max, cmp, bitwise, bit_count, float_special, complex, select, cast, transmute, sub_dtype',
    files: 'jix/src/ops/elementwise/{arith,div_rem,exp_log,trig,rounding,min_max,cmp,bitwise,bit_count,float_special,complex,select,cast}.rs, jix/src/ops/transmute.rs, jix/src/ops/sub_dtype.rs (non-test parts). Consult jix/src/ops/elementwise/pipeline.rs, op1.rs, op2.rs, jix/src/error.rs, jix/src/scalar/.',
    extra: 'Kernel semantics matter: integer div / rem by zero and MIN / -1 (integer errors now always raise, float errors never do), rounding modes, overflow, shifts by >= bit width, float <-> int casts (same-size signed <-> unsigned now by transmute), bool8 inputs, f16 / complex, NaN in min / max / cmp, signed zero, isclose; SIMD paths vs scalar tails must agree. State what makes transmute / sub_dtype sound.',
  },
  {
    id: 'reduction',
    title: 'all reductions (sum, prod, mean, var / std, min / max, argmin / argmax, any / all, ...), their accumulators and kernels, multi-axis / keepdims handling, and how they tile their reads',
    files: 'everything under jix/src/ops/reduction/ (non-test parts; read common.rs in full). Consult jix/src/ops/elementwise/pipeline.rs, jix/src/storage/buf.rs, jix/src/util/mod.rs, jix/src/storage/params.rs.',
    extra: 'Cover empty reductions (identities; min / max / argmin of empty), NaN propagation, accumulator dtypes and overflow, numerical accuracy (variance algorithm, ddof), argmin / argmax index correctness across tiles and ties, unsorted / duplicate / negative dims, keepdims, partial tiles, 0-d inputs.',
  },
  {
    id: 'shape-ops',
    title: 'lazy shape operations that remap index ranges / strides: slice, reshape, broadcast, permute_dims, insert_dim, remove_dim, flip, roll, repeat, tile, stack, concatenate',
    files: 'everything under jix/src/ops/shape_ops/ (non-test parts). Consult jix/src/storage/buf.rs, jix/src/storage/index.rs, jix/src/storage/core_trait.rs, jix/src/storage/params.rs (ArraySpec propagation).',
    extra: 'Cover both read modes per op (pull returns a re-strided view; push re-strides the caller\'s out and forwards it - is that legal for each op?), ops that split a request into several inner reads (roll, tile, repeat, concatenate, stack) and the new transparent / forwarding fast paths (full-range Slice, no-op Flip / Roll, a Concatenate read inside one member), empty requests and the non-empty Index invariant, ArraySpec / flags / cache-id propagation, NDIM_MAX when ops add dims.',
  },
  {
    id: 'util-memory',
    title: 'low-level memory utilities: AlignedBytes / RawAlignedBytes, ArrayVec / DimArray, BufferPool / PoolBuf, DimBitmap, CPU cache detection, hints, and the helpers in util/mod.rs',
    files: 'jix/src/util/aligned_vec/ (all), jix/src/util/arrayvec.rs, jix/src/util/buf_pool.rs, jix/src/util/bitmap.rs, jix/src/util/cpu_cache.rs, jix/src/util/hint.rs, jix/src/util/mod.rs (non-test part). Grep callers across jix/src: callers decide whether a helper\'s precondition holds.',
    extra: 'Cover allocation size arithmetic and overflow, Layout construction, zero-capacity allocations, grow paths, set_len over uninitialized bytes, ArrayVec bounds, PoolBuf slices larger than requested (stale bytes), BufferPool reentrancy through UnsafeCell, Send / Sync impls, the noalias claims of PtrNoalias / PtrMutNoalias, AlternatingBuffers\' raw-pointer reborrows.',
  },
  {
    id: 'util-nd',
    title: 'n-dimensional iteration and copy machinery: NdCopier, unordered n-d iterators (static and dyn), the iter/ module, arr_ext and arr_sequence',
    files: 'jix/src/util/nd_copy.rs, nd_iter_unordered.rs, nd_iter_unordered_dyn.rs, iter/ (all), arr_ext.rs, arr_sequence.rs (non-test parts). Consult jix/src/util/mod.rs and grep callers.',
    extra: 'Cover dim merging / coalescing validity (0-stride and length-1 dims), dim reordering by stride (ties, 0 strides), span computations used to build PtrNoalias slices, loop bounds and remainders, empty shapes and 0-d, NDIM_MAX, partial edge blocks, overlapping src / dst, itemsize-specialized copy kernels including odd sizes (3, 12), u64 -> usize on 32-bit targets.',
  },
  {
    id: 'types',
    title: 'the runtime dtype system (Dtype, struct dtypes, inner shapes, Alignment / Itemsize, the unsafe Dtyped trait), #[derive(Dtyped)], dimensions, ElementType, the error type, and the crate root',
    files: 'jix/src/dtype.rs (non-test part), jix-macros/src/lib.rs, jix/src/dimension.rs, jix/src/element_type.rs, jix/src/error.rs, jix/src/lib.rs',
    extra: 'Cover what an `unsafe impl Dtyped` promises and whether the derive macro can break it (repr(C) check, generics, padding, non-Dtyped fields, enums, ZSTs, packed structs); dtype equality checks that gate unsafe reinterpretation (can two dtypes compare equal with different layouts?); itemsize / alignment arithmetic for structs and inner shapes; dtype parsing from strings / archive metadata.',
  },
  {
    id: 'py-core',
    title: 'the PyO3 bindings core: the Python Array type, numpy conversion (zero-copy or copy, __array__), indexing, numpy() / out=, the archive Python API, dtype mapping, GIL release / UnsafeSend utilities, type hints, and the Python package',
    files: 'jix-py/src/lib.rs, array.rs, archive.rs, dtype.rs, util.rs, type_hint.rs (non-test parts), jix-py/python/jix/__init__.py. Consult jix/src/storage/plain.rs and jix/src/array/.',
    extra: 'Cover numpy memory lifetime under zero-copy wrapping, mutation / free of that memory while jix reads it (GIL released; free-threaded builds are shipped), read-only and aliasing out= arrays, non-contiguous / NEGATIVE-stride / byte-swapped / unaligned numpy arrays, structured dtypes vs jix struct dtypes, 0-d arrays and numpy scalars, Python int overflow, negative indices / negative-step slices, mmap\'d archives whose file goes away, the UnsafeSend justification.',
  },
  {
    id: 'py-ops',
    title: 'the Python op layer: runtime dtype dispatch, dtype promotion, Python-side broadcasting, operand conversion, and the wrappers for every op family',
    files: 'everything under jix-py/src/ops/. Consult jix-py/src/array.rs, jix-py/src/dtype.rs, the Rust core ops under jix/src/ops/, and jix-py/python/tests/ (test_type_promotion.py etc. document the intended promotion rules).',
    extra: 'Cover dispatch arms mapped to the wrong type or op, promotion of Python scalars (too large, negative with unsigned, float with int, bool), mixed signed / unsigned, promotion in where / stack / concatenate, broadcasting shape computation, reduction axis normalization, shape-op argument normalization (negative indices, negative-step slices, reshape -1, invalid permutations, zero repeat / tile counts), out= checks and aliasing, Python int -> u64 / usize conversions. List what the core asserts or panics on that this layer must validate first.',
  },
]

const prompt = s => `You are preparing context for a bug hunt on the jix library (this checkout). Later, many sub-agents will each review one small slice of the code for HARD bugs: logic bugs, wrong results, memory-safety / UB in unsafe code, panics or overflow on valid input, unsoundness of safe APIs. They will not have time to learn the whole library, so your job is to write the context document they will read for your subsystem. You are NOT the bug hunter: do not spend time proving bugs. But you must understand the code well enough to state its contracts precisely.

Subsystem: ${s.id} - ${s.title}
Files: ${s.files}
${s.extra}
${SINCE}
If bug-hunt/context/${s.id}.md already exists, read it first: keep what is still accurate (re-check it against the code), fix what is stale, and keep every note marked "(maintainer)" verbatim unless the code it describes is gone. Start the file with the line: <!-- Generated by the bug-hunt-map workflow at commit <git rev-parse --short HEAD> (<today's date>). Line numbers refer to that commit; re-check them against the current code. Maintainer corrections welcome - mark them "(maintainer)" so a regeneration keeps them. -->

Write bug-hunt/context/${s.id}.md (ASCII only - repo CI rejects any non-ASCII byte; use '-' not em-dashes or arrows; aim for 150-300 lines) with these sections:

## Purpose
3-8 lines: what this subsystem does, its main types / functions.

## Contracts
The invariants this code PROVIDES to callers and RELIES ON from other modules, stated precisely with file:line refs. Include the preconditions of every pub / pub(crate) unsafe fn and who is responsible for establishing them.

## Unsafe inventory
Each unsafe block / fn / impl (group trivial ones): file:line, what it assumes, where that assumption is established (caller, local check, debug_assert only, or NOT established / unclear). Flag the ones whose justification depends on non-local reasoning.

## Edge-case surface
How the code handles (or where it would need to handle): zero-length dims / empty ranges, 0-d arrays (shape []), length-1 dims, broadcast 0-stride dims (in inputs AND in out buffers), non-packed / permuted strides, NDIM_MAX=8 boundary, unaligned buffers, struct dtypes / inner shapes / itemsize != alignment, bool / f16 / complex, very large sizes (u64 -> usize on 32-bit targets: x86 and armv7 wheels are shipped), integer edge values, NaN / inf, aliasing between inputs and the out buffer. Only what is relevant.

## Intentional behaviors
Behaviors that are deliberate (documented in comments, docs, README, or asserted by tests), especially deviations from NumPy, so hunters do not report them as bugs. Cite where each is documented.

## Hot spots
Where a bug hunter should look hardest, and why: complex index arithmetic, subtle unsafe, special-case fast paths, duplicated logic that must agree, recent churn (git log --oneline -15 -- <files>, and skim recent diffs). Phrase as questions / hypotheses. If something looks actually wrong, list it under a sub-heading "Suspicions (unverified)" with file:line and one line of reasoning.

## Review units
Propose how to split this subsystem into hunter units of roughly 600-1500 non-test lines each (tests live in a #[cfg(test)] mod tests at the bottom of each file; do not count them). For each unit: id (kebab-case, prefixed with "${s.id}-"), files with line ranges, the other files a hunter must consult, and 3-8 specific focus questions.

Ground rules:
- Read the actual code; be precise; cite file:line. Wrong context is worse than missing context.
- Every hunter already gets CLAUDE.md; do not repeat its architecture overview.
- Do not modify any file other than the one you write. Do not run builds or tests; reading code and git history is enough.
- Final reply: ONLY a JSON object: {"doc": "bug-hunt/context/${s.id}.md", "units": [{"id": "...", "title": "...", "files": ["path:start-end"], "consult": ["path"], "focus": ["..."]}], "suspicions": ["file:line - one line"]}`

const UNITS_SCHEMA = {
  type: 'object',
  properties: {
    doc: { type: 'string' },
    units: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          files: { type: 'array', items: { type: 'string' } },
          consult: { type: 'array', items: { type: 'string' } },
          focus: { type: 'array', items: { type: 'string' } },
        },
        required: ['id', 'title', 'files', 'consult', 'focus'],
      },
    },
    suspicions: { type: 'array', items: { type: 'string' } },
  },
  required: ['doc', 'units', 'suspicions'],
}

const SINCE = A.since
  ? `This regeneration follows code changes since commit ${A.since}, the commit the last bug hunt scanned; files may have moved or been split (git diff -M --stat ${A.since} HEAD). Rewrite every file:line to the current layout. Fold any "Changed in <commit>" subsections of the old doc into the main sections, keeping the facts that are still true (especially intentional behaviors) and dropping the per-commit headings. In Hot spots, mark which code is new or changed since ${A.since} (read git diff -M ${A.since} HEAD for your files): the next hunt concentrates on it. In each proposed unit, add a focus question about the new code it contains.\n`
  : ''

const selected = SUBSYSTEMS.filter(s => !A.only || A.only.includes(s.id))
phase('Map')
const out = await parallel(selected.map(s => () => agent(prompt(s), { label: `map:${s.id}`, phase: 'Map', schema: UNITS_SCHEMA })))
const failed = selected.filter((s, i) => !out[i]).map(s => s.id)
if (failed.length) log(`Mapping failed for: ${failed.join(', ')}`)
return {
  docs: out.filter(Boolean).map(o => o.doc),
  failed,
  proposedUnits: out.filter(Boolean).flatMap(o => o.units),
  suspicions: out.filter(Boolean).flatMap(o => o.suspicions),
}

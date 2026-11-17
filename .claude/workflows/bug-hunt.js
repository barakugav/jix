export const meta = {
  name: 'bug-hunt',
  description: 'Hunt hard bugs in jix (unsafe/UB, wrong results, panics): module + concept + dynamic hunters, dedup, adversarial verification with repros, report',
  whenToUse: 'Run a full or partial bug hunt over jix. args: {runId, units?, skip?, kinds?, verify?, focus?, minConfidence?}. Afterwards write the returned reportMarkdown to bug-hunt/runs/<runId>/REPORT.md. See bug-hunt/README.md.',
  phases: [
    { title: 'Hunt', detail: 'one agent per module unit, cross-cutting concept, and dynamic tester' },
    { title: 'Dedup', detail: 'merge findings that share a root cause, drop known issues' },
    { title: 'Verify', detail: 'adversarial verifier per finding: refute first, then reproduce' },
    { title: 'Report', detail: 'write bug-hunt/runs/<runId>/REPORT.md' },
  ],
}

// ---------------------------------------------------------------------------------------------
// Arguments
//   runId          output dir name under bug-hunt/runs/ (default "run"); use a date, e.g. "2026-10-21"
//   units          list of unit-id prefixes to run, e.g. ["reduction", "concept-alignment"]; default all
//   skip           list of unit-id prefixes to leave out (applied after units), e.g. units already hunted
//   kinds          subset of ["module", "concept", "dynamic"]; default all
//   verify         false to skip verification (raw hunter output goes to the report); default true
//   focus          extra instructions appended to every hunter prompt
//   minConfidence  "low" | "medium" | "high": only verify findings at or above it; default "low"
// ---------------------------------------------------------------------------------------------
const A = args || {}
const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '_')
const RUN = A.runId || 'run'
const RUN_DIR = `bug-hunt/runs/${RUN}`
// Scratch / repro integration tests carry the run id so that concurrent runs never collide.
const RS = slug(RUN)
const huntTest = u => `hunt_${RS}__${slug(u.id)}_<n>`
const reproTest = f => `repro_${RS}__${f.id.toLowerCase()}`
const KINDS = A.kinds || ['module', 'concept', 'dynamic']
const FOCUS = A.focus ? `\nAdditional focus for this run: ${A.focus}\n` : ''
const CONF_RANK = { low: 0, medium: 1, high: 2 }
const MIN_CONF = CONF_RANK[A.minConfidence || 'low']
const SEV_RANK = { critical: 0, high: 1, medium: 2, low: 3 }

// ---------------------------------------------------------------------------------------------
// Units. `ctx` names docs in bug-hunt/context/. Module units review a slice of files in depth;
// concept units trace one invariant across the codebase; dynamic units write and run tests.
// ---------------------------------------------------------------------------------------------

const MODULE_UNITS = [
  {
    "id": "storage-core-buf-contract",
    "ctx": [
      "storage-core"
    ],
    "title": "ArrayStorage / Index / OutBuf / StridedBuf contract, check_out_buf, forwarders (&S, &mut S, Box/Rc/Arc, &dyn, Any)",
    "files": [
      "jix/src/storage/core_trait.rs:1-200",
      "jix/src/storage/buf.rs:1-555",
      "jix/src/storage/index.rs:1-113",
      "jix/src/storage/mod.rs:1-267",
      "jix/src/storage/any.rs:1-110",
      "jix/src/storage/info.rs:1-74"
    ],
    "consult": [
      "jix/src/util/nd_copy.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/mod.rs",
      "jix/src/context.rs",
      "jix/src/error.rs",
      "jix/src/array/read.rs",
      "jix/src/ops/shape_ops/permute_dims.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix/src/ops/shape_ops/remove_dim.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/ops/into_type.rs",
      "jix/src/ops/elementwise/pipeline.rs"
    ],
    "focus": [
      "NEW Index (index.rs): can an empty (size 0) or start/size length-mismatched index reach a storage through any Index::new_unchecked caller or the Array front doors (split_ranges after check_ranges)? A mismatch is UB via assert_unchecked (index.rs:93), a 0 size underflows Plain's span math.",
      "NEW OutBuf pull slot protocol: does every impl fill the slot exactly once on Ok (write_out_buf_slot assert, read_data_pull_in expect), and is a slot left filled by an Err ever reused?",
      "NEW push identity: can any push path assign *out, call set_strides/make_readonly/with_offset on the caller's out, or write elsewhere, leaving the caller's buffer (to_ndarray_sub assume_init, uninit compaction chunk) unwritten?",
      "NEW OutBuf::set / OutBuf::materialize are safe fns whose soundness needs check_out_buf to have run; NEW Shared variant + into_shared/from_shared (read cache): writable-to-read-only conversion, offsets vs pool capacity, buffer returned to the pool while a Shared view is alive.",
      "check_out_buf_impl (buf.rs:384-431): fast reverse-order path vs sorted path, stride ties, zero strides, len-1 dims, saturation, PoolBuf capacity used as len.",
      "NEW read_data_and_map_strides (no transmute): pull relabels a possibly writable buffer in place without make_readonly - do PermuteDims/InsertDim/RemoveDim maps keep it non-aliasing over the outer region? Push inner_out strides address the same bytes?",
      "NEW &S/&mut S forwarders (replacing View), Box/Rc/Arc, &dyn (default read_as_elementwise_pipeline) and ArrayStorageAny cached metadata: any default that changes results or cache identity?",
      "alloc_out_buf: unchecked usize product (no isize::MAX bound on 64-bit) and non-permutation layout orders giving a writable aliasing buffer."
    ]
  },
  {
    "id": "storage-core-leaf-backends",
    "ctx": [
      "storage-core"
    ],
    "title": "Plain and Scalar leaf storages: constructors, read_data pull/push, Constant pipeline",
    "files": [
      "jix/src/storage/plain.rs:1-525",
      "jix/src/storage/scalar.rs:1-182"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/mod.rs",
      "jix/src/storage/params.rs",
      "jix/src/element_type.rs",
      "jix/src/dtype.rs",
      "jix/src/error.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/array/read.rs"
    ],
    "focus": [
      "NEW Plain::read_data loop (plain.rs:463-475, empty-region special case removed): offset/span math with len-1 dims carrying arbitrary user strides (plain_ndarray_ptr), 0 strides, 32-bit `start as usize`; only debug_assert!(size != 0) guards (size - 1).",
      "NEW &[u64] constructors with check_ndim (Plain::new, Scalar::new): is every shape validated before DimDyn::from_slice / into_spec, and is Plain's derived read_layout_order always a permutation?",
      "strides_from_ndarray / plain_ndarray_with / plain_ndarray_ref_impl: into_raw_vec_and_offset offsets, empty arrays, negative strides on len <= 1 dims, owned vs borrowed lifetimes.",
      "Scalar push: contiguous aligned fast fill vs strided copy_from - alignment assert, region_len vs the check_out_buf span, padding bytes of derived structs.",
      "Scalar Constant pipeline (N_OPERANDS = Some(0), ignores offset) vs Scalar::read_data: same values for bool8, structs, inner shapes; zero-operand pipelines.",
      "Plain element_type_change and the unsafe constructor contracts vs what read_data actually does (K111 is open: do not re-report span aliasing)."
    ]
  },
  {
    "id": "array-frontend-read-paths",
    "ctx": [
      "array-frontend"
    ],
    "title": "Array<S> read entry points (to_ndarray*, to_buf_slice, to_buf_strided pull, to_buf_strided_into push) and the request-splitting slow path",
    "files": [
      "jix/src/array/mod.rs:1-373",
      "jix/src/array/read.rs:1-609"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/error.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/iter/block.rs",
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/plain.rs",
      "jix/src/context.rs"
    ],
    "focus": [
      "NEW (#103/#104): is every Index::new_unchecked (read.rs:256, 321, 389, 454, 571-572) reached only with a non-empty, in-shape, <= NDIM_MAX region, and is every usize computation preceded by the 32-bit region check (to_buf_slice computes strides at read.rs:316 before any check)?",
      "NEW (#104): pull to_buf_strided vs push to_buf_strided_into - lifetimes of returned buffers (pool, read-only cache Shared, borrowed Plain view), out.view_mut() forwarding, and each empty-region path (alloc_empty, dims-only check).",
      "Slow path read.rs:527-576: are pieces disjoint and covering for unaligned starts, read_shape snapped to array_shape and partial last pieces, and is out_offset right for permuted/padded/offset outs, in both pull (materialize) and push?",
      "to_ndarray_sub read.rs:197-268: inv_order/permuted_axes for non-C read_layout_order, ndarray strides of len-1 dims, and what could leave holes under assume_init (storage push paths, cache-hit copy).",
      "small_read / plain_read bypass of splitting (read.rs:476-487): can a small read be arbitrarily large or skip a check the slow path relies on?",
      "NEW (#100): one cache scope per top-level read (read.rs:481), shared by all slow-path pieces; Array::dtype() vs storage dtype in buffer sizing."
    ]
  },
  {
    "id": "array-frontend-compaction",
    "ctx": [
      "array-frontend"
    ],
    "title": "Compaction (compact/compact_with/compact_into_builder) and constructors (compact_ndarray*, compact_ndarray_ptr, compact_fn* / FnStorage)",
    "files": [
      "jix/src/array/compact.rs:1-339",
      "jix/src/array/construct.rs:1-312"
    ],
    "consult": [
      "jix/src/storage/block.rs",
      "jix/src/storage/compact.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/aligned_vec/mod.rs",
      "jix/src/codec/mod.rs",
      "jix/src/archive/array.rs",
      "jix/src/storage/plain.rs",
      "jix/src/util/iter/strides.rs",
      "jix/src/context.rs"
    ],
    "focus": [
      "NEW (#103): chunk sizes are cast to usize for the Index and buffer (compact.rs:266-273) but stay u64 for the block loop (287-332); on 32-bit with a large user read_size and a dim >= 2^32, does get_unchecked(src_byte_offset..) read past chunk_buf?",
      "Does block_index (compact.rs:282-302) match the C-order block grid, is every block written exactly once, are partial blocks zero-padded and full blocks fully copied?",
      "Uninitialized chunk_buf (set_len at :270) filled only by the push contract, uninit compressed buffer + encode bound (:336-342): any path that compresses or keeps uninit bytes?",
      "compact_with encodes with params tuned in compact_into_builder and then re-tunes in into_spec via ArrayBlockTableStorageBase::new (only debug_asserts compare geometry) - is tune idempotent so the spec matches the blocks written?",
      "FnStorage (construct.rs:247-286): aligned write decision vs align_of::<T>(), NdIter offsets relative to begin, pull layout, user f panicking mid-read.",
      "NEW (#100): compaction's cache scope (compact.rs:180, entered before tune) with push chunk reads; 0-d and empty arrays; compact_ndarray_ptr safety contract forwarding and double tune in compact_ndarray_with."
    ]
  },
  {
    "id": "array-frontend-read-cache",
    "ctx": [
      "array-frontend"
    ],
    "title": "ReadContext read cache (read_cached, cache scopes), ArrayCacheIdMap, cache-id propagation, OutBuf/PoolBufShared glue",
    "files": [
      "jix/src/context.rs:1-314",
      "jix/src/storage/cache_id.rs:1-230",
      "jix/src/storage/params.rs:614-619",
      "jix/src/storage/params.rs:821-909",
      "jix/src/storage/buf.rs:337-348",
      "jix/src/storage/buf.rs:449-515",
      "jix/src/util/buf_pool.rs:58-63",
      "jix/src/util/buf_pool.rs:133-165"
    ],
    "consult": [
      "jix/src/storage/compact.rs",
      "jix/src/ops/elementwise/op1.rs",
      "jix/src/ops/elementwise/op2.rs",
      "jix/src/ops/elementwise/map.rs",
      "jix/src/ops/elementwise/select.rs",
      "jix/src/ops/elementwise/min_max.rs",
      "jix/src/ops/reduction/common.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix/src/ops/shape_ops/concatenate.rs",
      "jix/src/ops/shape_ops/reshape.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix/src/ops/transmute.rs",
      "jix/src/storage/any.rs",
      "jix/src/storage/mod.rs"
    ],
    "focus": [
      "NEW: can a cache hit (key = id + exact start/size, context.rs:190-191) return data for another region, dtype, itemsize or storage - e.g. ids shared by element_type_change, as_compact, forwarders, ArrayAny clones?",
      "NEW: the hit path skips Compact's and ReductionOp's own check_out_buf (they check inside the cached closure); is every hit preceded by a validated uncached read of the same storage at that index, and is the push-hit check_out_buf_impl + OutBuf::set copy sound (aliasing, itemsize, strides)?",
      "NEW: the pull-miss path converts the result with into_shared and hands back a read-only Shared view - does any consumer (ops, pipeline, jix-py) need a writable pull result, and are Slice/SliceMut correctly left uncached?",
      "NEW: eviction (pull only, context.rs:205-209) and scope exit (282-291) while views are lent: return_shared refcounts, the stale same-id entry inserted by a pass-through inner read (241-250).",
      "NEW: RefCell/Cell discipline under nested scopes, reentrant top-level reads on the same context from a Map closure capturing &ReadContext, errors with a filled slot, and panics; the unwrap at context.rs:245 and the is_cache_enabled invariant.",
      "NEW: ArrayCacheIdMap MaybeUninit invariant (values[i] init iff keys[i] is Some) on insert-spill, remove, clear_with, Drop, Clone and panic paths.",
      "NEW: params.rs:867-951 id merging (add_cache_id, add_cache_ids_from_impl, with_cache_scope) and scope placement - does a wrong arrays_to_cache cost only speed, or can it cost correctness?"
    ]
  },
  {
    "id": "storage-compact-read-path",
    "ctx": [
      "storage-compact"
    ],
    "title": "Compact nd read path (fast paths, direct decode, slow path, read-cache wrapper) and BlockTable / mmap block access",
    "files": [
      "jix/src/storage/compact.rs:1-522",
      "jix/src/storage/block.rs:1-458"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/context.rs",
      "jix/src/error.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/iter/block.rs",
      "jix/src/util/iter/nd.rs",
      "jix/src/util/mod.rs",
      "jix/src/codec/mod.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/archive/array.rs",
      "jix/src/archive/block.rs",
      "jix/src/array/compact.rs",
      "jix/src/array/read.rs",
      "jix/src/element_type.rs"
    ],
    "focus": [
      "NEW: does the read-cache wrapper (compact.rs:243-256) ever serve a hit that an uncached read would have rejected or computed differently (hit skips check_get_range/check_region_size; cache_id shared via as_compact / element_type_change / &S)?",
      "NEW: with the nitems==0 early return replaced by a debug_assert (compact.rs:312-325), is every Index reaching Compact non-empty and in-shape, and is ndim 0 handled entirely by fast path 1?",
      "Is the direct-decode condition (read_into_out + full block, compact.rs:459-467, 494) sufficient for every block/out shape (size-1 dims before/after lead_dim, all-ones block, ndim 5-8, sub-tile outs)?",
      "Can out_start/active_start exceed the slices given to get_unchecked(_mut) (compact.rs:441, 507-508) or the NdCopier spans, for any valid Index and any out accepted by check_out_buf?",
      "Do fast path 1, the single-block path and the general path agree on partial edge blocks of block shapes that do not divide the shape, unaligned starts, size-1 dims and ndim 5-8?",
      "Is prod(spec.block_shape) == blocks.block_size (and the same shape) guaranteed for every constructor, given that compact_with tunes 2-3 times and only debug_asserts geometry (compact.rs:210-221)?",
      "Malformed / truncated mmap archives: any location, section or header value that panics, aborts or reads out of bounds instead of erroring (block.rs:157, 163-174, MmapData::as_ref 452-457)? 32-bit truncation at block.rs:157, 199, 288, 318?"
    ]
  },
  {
    "id": "storage-compact-params",
    "ctx": [
      "storage-compact"
    ],
    "title": "ArrayParams::tune / block-shape heuristics, ArraySpec read hints and cache-id sets, scale_read_shape",
    "files": [
      "jix/src/storage/params.rs:1-1050",
      "jix/src/util/mod.rs:400-522"
    ],
    "consult": [
      "jix/src/array/compact.rs",
      "jix/src/array/construct.rs",
      "jix/src/array/read.rs",
      "jix/src/ops/reduction/common.rs",
      "jix/src/ops/shape_ops/",
      "jix/src/ops/elementwise/",
      "jix/src/util/bitmap.rs",
      "jix/src/util/cpu_cache.rs",
      "jix/src/archive/array.rs",
      "jix/src/storage/compact.rs",
      "jix/src/context.rs",
      "jix/src/storage/cache_id.rs",
      "jix/src/error.rs"
    ],
    "focus": [
      "NEW: with block dims now clamped to max(shape,1) (params.rs:370-373) before the fixed-dim check and scaling, is tune idempotent for every mix of block_shape / fixed dims / block_size (incl. 0 and < itemsize) / read_size / ndim 0, and can the clamp change a block shape some caller assumes unchanged (archive read, compact_with's second tune in into_spec)?",
      "NEW: consuming builder API (params.rs:126-231) and inherit_unset_from_storage (262-290): any path that drops, overrides (e.g. the 1 GiB fixed-dims reset at 280-283) or length-mismatches user-set fields?",
      "NEW: cache-id set merging (add_cache_id / add_cache_ids_from_impl / with_cache_scope, params.rs:867-950): can a missing or extra id break correctness rather than speed, and do the debug_assert-only invariants (909, 915) hold for every op that calls these helpers?",
      "Which inputs make tune / into_spec panic instead of returning Err (params.rs:435, 516, 540-548)?",
      "Prove 1 <= block_len <= min(dim_len, 2^30) in block_len_heuristic (479-517) for every input, including max_volume < inner_block_volume and dim_len near i64::MAX.",
      "scale_read_shape (util/mod.rs:479-537): tile invariants, unchecked u64 volume products, max_shape[d] == 0, an order that is not a permutation; do all callers clip snapped tiles?",
      "Cost / weight combiners (params.rs:982-1025) with NaN (f32::clamp(NaN) stays NaN, partial_cmp().unwrap() at 1021), inf, zero cost, an empty input list, or inputs of different ndim."
    ]
  },
  {
    "id": "codec-archive-codec-pipeline",
    "ctx": [
      "codec-archive"
    ],
    "title": "Codec pipeline: Encoder/Decoder, byte shuffle (SIMD), bit shuffle",
    "files": [
      "jix/src/codec/mod.rs:1-310",
      "jix/src/codec/filter.rs:1-52",
      "jix/src/codec/byte_shuffle.rs:1-224",
      "jix/src/codec/bit_shuffle.rs:1-357"
    ],
    "consult": [
      "jix/src/context.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/aligned_vec/mod.rs",
      "jix/src/storage/block.rs",
      "jix/src/storage/compact.rs",
      "jix/src/array/compact.rs"
    ],
    "focus": [
      "New in #98: Decoder now reads context.decompressor / context.buffer_pool directly, and ReadContext gained a region cache that also hands out PoolBufShared buffers from the same BufferPool - can a cached buffer alias the Decoder or bit-shuffle tmp buffers, or can decode run re-entrantly while the decompressor &mut (codec/mod.rs:300) is live?",
      "Decoder buffer parity for 0..4 filters (codec/mod.rs:278-296, only debug_assert at :294): does the last filter always land in dst, and what does dst hold on every error path given the compact fast path passes the caller's out buffer as dst (storage/compact.rs:347-357)?",
      "Is the nbytes == dst_len check (codec/mod.rs:310-314) the only guard before filters read the pooled tmp, and is every byte of the Encoder's set_len tmp buffers (:157,:166) written before it is read?",
      "SIMD byte-shuffle index math per lanes 16/32/64 x itemsize 2/4/8/16 in both the interleave and deinterleave branches (byte_shuffle.rs:52-112,160-200): correct only if fearless_simd zips full vectors; does the scalar tail start exactly at n_elements_done?",
      "get_unchecked bounds for tiny/empty inputs and for callers that might bypass the entry asserts now that the module files moved (byte_shuffle.rs:28,136,82,107,128-129,176,195,216-217)",
      "Bit-shuffle inverse (bit_shuffle.rs:277-333) vs forward (:206-271) for odd/non-power-of-two typesize and n % 8 != 0; tail copy for lengths not a multiple of typesize",
      "Under cfg(miri), do all encode/decode callers pass dst.len() equal to the input length (codec/mod.rs:186,307)?"
    ]
  },
  {
    "id": "codec-archive-archive-io",
    "ctx": [
      "codec-archive"
    ],
    "title": "Archive I/O: reader/writer, sections, block table (de)serialization, Owned and Mmap storage, version",
    "files": [
      "jix/src/archive/common.rs:1-276",
      "jix/src/archive/block.rs:1-515",
      "jix/src/archive/version.rs:1-129",
      "jix/src/archive/mod.rs:1-31"
    ],
    "consult": [
      "jix/src/storage/block.rs",
      "jix/src/util/mod.rs",
      "jix/src/storage/params.rs",
      "jix-py/src/archive.rs",
      "jix/schema/proto/jix/v1/block.proto"
    ],
    "focus": [
      "Section bounds: check_section_bounds (common.rs:264-274) plus the Mmap mapping check (block.rs:497-503) - can any crafted Section still yield an MmapData slice outside the mapping or a usize truncation on 32-bit (block.rs:495,504)?",
      "Owned::read_section chunked growth and Vec::from_raw_parts (block.rs:452-469): any path where uninitialized or partially-read elements become a Vec<T>?",
      "End-of-archive seek (block.rs:359-367 -> common.rs:255-262) with len=None and saturated i64::MAX offsets through Take/BufReader/PyReader seek_relative: panic or wrong position?",
      "Strict vs Minimal: with Minimal, do all consumers of block locations (block_bytes, write_content's copy path at block.rs:84) error cleanly on out-of-range or overlapping locations?",
      "Writer bookkeeping: unchecked nitems/nblocks arithmetic and vec![..; nblocks as usize] in BlockArchiveWriter::start (block.rs:133,172,184), block_index as usize (:205), duplicate/missing block indices, append-mode detection (:238-244)",
      "Does the #94 doc change in archive/mod.rs (Strict checks every block's (offset, len) lies in the data section) match what read_content actually validates in each mode?",
      "uleb128 and message reads (common.rs:157-225): 10-byte cap, read_byte on Interrupted, error kinds (Io vs InvalidArchive) for truncation and protobuf decode errors"
    ]
  },
  {
    "id": "codec-archive-array-format-dtype",
    "ctx": [
      "codec-archive"
    ],
    "title": "Array archive format: header validation, write_to/read_from entry points, dtype proto conversions",
    "files": [
      "jix/src/archive/array.rs:1-571",
      "jix/src/archive/schema/conversions.rs:1-174",
      "jix/src/archive/schema/mod.rs:1-8"
    ],
    "consult": [
      "jix/src/dtype.rs",
      "jix/src/storage/compact.rs",
      "jix/src/storage/params.rs",
      "jix/src/error.rs",
      "jix/src/array/compact.rs",
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/ops/shape_ops/roll.rs",
      "jix/src/ops/shape_ops/flip.rs",
      "jix/src/ops/maybe_compact.rs",
      "jix-py/src/archive.rs"
    ],
    "focus": [
      "New in #95: write_to's compact copy path (array.rs:421-430) is now reached through identity Slice/Roll/Flip views forwarding as_compact - is is_identity exactly 'same shape and element order' in each op, so the header shape (self.shape()) always matches the inner block geometry? Can MaybeCompact::as_compact's unwrap (maybe_compact.rs:102-108) now fail?",
      "Changed in #94: write_to(writer, params, ctx) merged with write_to_with and write_to_file now passes ArrayParams::default() + self.read_ctx() (array.rs:333-344,411-452); read_from_file drops params - any behavior drift for compact vs lazy sources or for the Python callers?",
      "Header validation arithmetic (array.rs:479-564): i64::MAX dim check, b <= BlockSize::MAX before cast, try_shape_product with zero dims, unchecked nblocks product - any panic or wrong acceptance in debug or release?",
      "Caller params flowing into the spec on read (array.rs:566-581): block_shape forced, fixed dims replaced, but block_size and read_size kept - does any consumer assume spec.block_size() == prod(block_shape) * itemsize?",
      "Dtype::from_proto (schema/conversions.rs:9-128): shape ndim checked against NDIM_MAX then DTYPE_NDIM_MAX, u16 products, struct path new_struct + set_shape idempotence, zero-size fields and packed layouts - can a file produce a Dtype whose offsets/itemsize violate what consumers (SubDtype, nd copy) assume?",
      "to_proto/from_proto round trip for every dtype form (scalar with inner shape, nested/packed struct, bool8, f16, complex) and the nesting limit interplay with prost's recursion limit (writer cap block.rs:134-138, test array.rs:622)"
    ]
  },
  {
    "id": "pipeline-driver",
    "ctx": [
      "pipeline"
    ],
    "title": "Fused loop driver, staging, inner loops and lane math",
    "files": [
      "jix/src/ops/elementwise/pipeline.rs:1-727"
    ],
    "consult": [
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/util/nd_iter_unordered_dyn.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/mod.rs",
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/error.rs"
    ],
    "focus": [
      "[new #103/#104] With the empty-region early return gone and `|| len == 0` dropped from the collapse test, is every shape/stride path still right for 0-d, all-size-1 and fully collapsed regions, and does to_buf leave the pull slot sane (writable aliasing buffer, make_readonly skipped) when to_buf_fn returns Err?",
      "Can inputs_contiguous / output_contiguous be true for an operand whose cursor stride != itemsize (static vs dyn flags, the all-size-1 NdIter case, staged output)?",
      "Staging fill cache: is the stride-0 `n = usize::MAX` shortcut and the `prev_n >= n` reuse always correct, and are staging buffers always >= chunk_len elements?",
      "REQUIRE_ALIGNED = true paths (armv7/wasm): staged-output scatter, base aligned but coalesced strides unaligned, aligned [T; LANES] stores.",
      "Are to_buf_type_erased and to_buf_type_erased_dyn (and NdIterUnordered vs NdIterUnorderedDyn) equivalent in every case?",
      "Pointer add / PtrMutNoalias spans for every stride combination, including chunked inner runs longer than 8192.",
      "Lane-count const math (contiguous vs strided, MAX_ITEMSIZE vs S::u8s::LEN) and the `len % LANES` tail loop.",
      "Status accumulation per chunk and the single status.into_result() at the end (#95): any early return that skips it or a path that loses a flag?"
    ]
  },
  {
    "id": "pipeline-nodes",
    "ctx": [
      "pipeline"
    ],
    "title": "Operands, pull output layout, Op1/Op2 node plumbing and read cache wrapping",
    "files": [
      "jix/src/ops/elementwise/pipeline.rs:729-987",
      "jix/src/ops/elementwise/op1.rs:1-278",
      "jix/src/ops/elementwise/op2.rs:1-362",
      "jix/src/ops/elementwise/mod.rs:1-68"
    ],
    "consult": [
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/scalar.rs",
      "jix/src/storage/buf.rs",
      "jix/src/context.rs",
      "jix/src/storage/cache_id.rs",
      "jix/src/ops/elementwise/cmp.rs",
      "jix/src/ops/elementwise/select.rs",
      "jix/src/ops/elementwise/min_max.rs",
      "jix/src/dtype.rs"
    ],
    "focus": [
      "[new #104] materialize_pipeline_out_buf / write_out_buf_slot / make_readonly_if_aliasing with the OutBuf slot: can a writable buffer with aliasing 0 strides escape, can a slot be reused, is a push out ever replaced?",
      "[new #100] read_cached wrapping in Op1/Op2 read_data: cached pull buffers with 0 strides, push cache hits copied via OutBuf::set, cache ids registered in new() (incl. swapped Op2 in cmp.rs:351).",
      "Does pick_output_layout always produce a non-aliasing layout that fits nitems * itemsize and agrees with the driver's degenerate-dim collapse rule (Constant-only pipelines, operands disagreeing on order)?",
      "N_OPERANDS / MAX_ITEMSIZE / operands() consistency of every node (Op1, Op2, Select, Clamp, Constant): any undercount reaching the null-cursor read in release?",
      "Is a dtype check (check_dtype T vs Output) always done before every transmute_copy, including through IntoType/&S forwarding?",
      "Operand::new_output borrow juggling (raw base_ptr + shared reborrow for strides) and OperandTyped::read_bulk contiguous vs strided address math."
    ]
  },
  {
    "id": "pipeline-plumbing",
    "ctx": [
      "pipeline"
    ],
    "title": "MapMultiple, IntoType, MaybeCompact, DimsArg and the ops module",
    "files": [
      "jix/src/ops/elementwise/map.rs:1-442",
      "jix/src/ops/into_type.rs:1-209",
      "jix/src/ops/maybe_compact.rs:1-162",
      "jix/src/ops/mod.rs:1-156",
      "jix/src/ops/dims_arg.rs:1-143"
    ],
    "consult": [
      "jix/src/util/arr_sequence.rs",
      "jix/src/storage/mod.rs",
      "jix/src/storage/params.rs",
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/element_type.rs",
      "jix/src/storage/compact.rs",
      "jix/src/context.rs"
    ],
    "focus": [
      "[new #100/#102] MapMultiple cache id + read_cached, and &S / Box / Rc / Arc forwarding of read_as_elementwise_pipeline through IntoType / MaybeCompact: anything that breaks when the same storage is reached twice in one pipeline?",
      "MapMultiple Vec/slice scratch (UnsafeCell, spare_capacity_mut, set_len) and the drop-before-next-call contract of read_bulk_as_iter.",
      "MAX_ITEMSIZE of fixed-size [Array<S>; N] sequences ignores N while read_bulk_as_iter holds [[T; M]; N] on the stack: stack overflow for large N (see Suspicions)?",
      "IntoType with TypeDyn vs Ty<T>: is the inner dtype always checked before a typed pipeline or typed read is built (only debug_assert in read_data / read_as_elementwise_pipeline)?",
      "MaybeCompact::as_compact unwrap (maybe_compact.rs:107) vs every storage whose spec carries the compact flag (identity Slice/Roll/Flip, &S, IntoType, element_type_change results).",
      "DimsArg::get out-of-range behavior per impl and constructors panicking on degenerate inputs (zero arrays, one array, zero-size output)."
    ]
  },
  {
    "id": "pipeline-simd",
    "ctx": [
      "pipeline"
    ],
    "title": "Scalar trait macros and SIMD helper layer",
    "files": [
      "jix/src/scalar/simd.rs:1-381",
      "jix/src/scalar/traits_util.rs:1-348",
      "jix/src/scalar/mod.rs:1-28",
      "jix/src/scalar/bool.rs:1-101"
    ],
    "consult": [
      "jix/src/ops/elementwise/op1.rs",
      "jix/src/ops/elementwise/op2.rs",
      "jix/src/ops/elementwise/rounding.rs",
      "jix/src/ops/elementwise/exp_log.rs",
      "jix/src/ops/elementwise/trig.rs",
      "jix/src/ops/elementwise/cast.rs",
      "jix/src/ops/elementwise/cmp.rs",
      "jix/src/util/test_util.rs"
    ],
    "focus": [
      "[new] float_op1! (traits_util.rs:309-344): does `x.$f()` always resolve to an inherent f32/f64 method (else infinite recursion), is f16-through-f32 rounding right, and do SIMD floor/ceil/sqrt bodies match std bit-for-bit at every fearless_simd level?",
      "[changed #95] Unary traits lost their status parameter and map_vectors_status was removed: does any status-taking trait now drop flags, and do map_vectors2_status scalar vs SIMD flags agree?",
      "map_masks_to_bools / mask8s_from_mask lane order for every element width (narrow concatenation order) on AVX2.",
      "map_vectors_different_width chunk / KT / KU index arithmetic and the const assert.",
      "Scalar fallback when N % lanes != 0 and its agreement with the SIMD body (NaN, -0.0, overflow, saturation).",
      "Macro arm selection in impl_scalar_op1! / impl_scalar_op2! (bool-output arms, missing bulk fn): can an impl silently pick the wrong arm?",
      "dispatch_baseline! vs runtime dispatch! level coverage on x86 / x86_64 / aarch64 / others, and for_each_unrolled debug/release equivalence.",
      "bool8 ops (Not, BitAnd/Or/Xor, Eq/Ord/Hash) normalize non-0/1 bytes everywhere."
    ]
  },
  {
    "id": "elementwise-ops-div-rem",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Integer and float divide / remainder kernels (trunc, floor, euclid), scalar and SIMD",
    "files": [
      "jix/src/ops/elementwise/div_rem.rs:1-831"
    ],
    "consult": [
      "jix/src/ops/elementwise/op2.rs",
      "jix/src/scalar/traits_util.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/error.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix-py/src/ops/elementwise/div_rem.rs"
    ],
    "focus": [
      "[changed #95] int_quotient / int_remainder (293-331): is MIN / -1 the only non-zero-divisor None of checked_div / checked_rem / checked_rem_euclid, now wrapping with no flag, and is DivideByZero set per element in both the SIMD bulk and the N=1 tail?",
      "Can the signed floor / euclid / remainder adjustments (237, 254, 276) overflow (debug panic) for any (a, b)?",
      "Do div_rem_trunc (401-410) and div_rem_trunc_v (465-477) agree bit for bit, including the sign of zero, for b = +-0, a = +-inf, NaN, subnormals and quotients that overflow to inf?",
      "Past EXACT_LIMIT, div_floor / div_euclid use the inexact residual while rem_trunc / rem_floor use fmod (fmod_past_limit 493-502): can q*b + r == a or the sign rules break between the pair?",
      "[changed #95] float Div and the div family lost their error classification but still run through map_vectors2_status with an unused status: any leftover behavior difference between SIMD and scalar?",
      "Is the f16 path (f32 exact quotient / fmod, one rounding, 596-613) correct for every finite f16 pair?"
    ]
  },
  {
    "id": "elementwise-ops-arith-pow-unary",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Arithmetic, Pow, Sign/Abs/Square/Neg and the unary float ops (exp/log, trig, rounding)",
    "files": [
      "jix/src/ops/elementwise/arith.rs:1-657",
      "jix/src/ops/elementwise/exp_log.rs:1-207",
      "jix/src/ops/elementwise/trig.rs:1-254",
      "jix/src/ops/elementwise/rounding.rs:1-133"
    ],
    "consult": [
      "jix/src/scalar/traits_util.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/ops/elementwise/op1.rs",
      "jix/src/ops/elementwise/op2.rs",
      "jix/src/error.rs",
      "fearless_simd 1.1.0 src/generated/sse2.rs"
    ],
    "focus": [
      "[changed #95] Pow int x signed (335-356): NegativePower flag and 0 fallback for every width, and the square-and-multiply loop for i64 exponents above u32::MAX (344-352).",
      "Float Pow match arms (0 -> 1, 1 -> a, 2 -> a*a, i32 via f64; 357-401) vs powf for NaN / inf / -0 bases; f16 ** i32 rounds f64 -> f32 -> f16 (389, 392).",
      "SIMD bodies vs scalar: Abs / Neg / Square integer wrap at MIN on every level, Floor / Ceil / Sqrt sign of zero and NaN (SSE2 floor/ceil are per-lane scalar fallbacks).",
      "Sign for f16, NaN and -0.0 (46-56); unsigned Sign.",
      "[changed #95] float_op1! is now plain f16::from_f32(x.to_f32().f()) with no status: does any f16 unary kernel deviate from the documented f32 semantics?",
      "f16 add/sub/mul/square via f32 vs native aarch64 fp16 paths: identical results?"
    ]
  },
  {
    "id": "elementwise-ops-cmp-minmax-classify",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Comparisons (incl. swapped Greater, u64/i64), ApproxEq, Maximum/Minimum/Clamp, float classification",
    "files": [
      "jix/src/ops/elementwise/cmp.rs:1-720",
      "jix/src/ops/elementwise/min_max.rs:1-403",
      "jix/src/ops/elementwise/float_special.rs:1-171"
    ],
    "consult": [
      "jix/src/ops/elementwise/op2.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/scalar/bool.rs",
      "jix/src/context.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix-py/src/ops/elementwise/cmp.rs"
    ],
    "focus": [
      "Mixed u64/i64 comparison formulas (cmp.rs:98-157) and the swapped Greater / GreaterEqual (Op2::new(b, a, kernel, true) at 351): operand order, kernel selection, error message and spec base (402-409).",
      "Maximum / Minimum NaN check (simd_ne unordered) SIMD vs scalar, canonical NaN bits, signed zero, and bool8 non-canonical operands returned by std::cmp::max/min.",
      "[new #100] Clamp: check_out_buf before read_cached, own cache id, and the cache id across element_type_change (IntoType).",
      "ApproxEq (228-234): inf / NaN handling, a - b overflow (MAX vs -MAX), f16 tolerance rounding, Complex per-component tolerances.",
      "Bool-output SIMD on AVX2 (simd.rs:96-138, 325-359): mask narrowing order for 16/32/64-bit lanes, used by cmp and is_nan / is_finite / is_infinite."
    ]
  },
  {
    "id": "elementwise-ops-bitwise-cast",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Bitwise, shifts/rotates, bit counts and the Cast kernels (scalar tables and SIMD step chains)",
    "files": [
      "jix/src/ops/elementwise/bitwise.rs:1-471",
      "jix/src/ops/elementwise/bit_count.rs:1-321",
      "jix/src/ops/elementwise/cast.rs:1-485"
    ],
    "consult": [
      "jix/src/scalar/simd.rs",
      "jix/src/scalar/traits_util.rs",
      "jix/src/ops/elementwise/op1.rs",
      "jix/src/scalar/bool.rs",
      "fearless_simd 1.1.0 (narrow/widen/truncate_from_precise/float_from)",
      "jix-py/src/ops/elementwise/casting.rs"
    ],
    "focus": [
      "Shr SIMD (bitwise.rs:110-125): i32 rhs bitcast + min(31), u32 rhs select - does every (lhs, rhs) signedness pair and negative count match unbounded_shr(shift_count(b))?",
      "ShiftCount (179-197) for every count type vs every lhs width, for Shl and Shr.",
      "Cast step chains (cast.rs:196-283): KT/KU counts, widen order (346-373), sign extension, saturating chains for f32 -> i8/u8 and f64 -> i8..u32 at mid-range values (200.0, 40000.0, -1.0, 2^31), and the only_on level lists.",
      "[changed #95, cross] jix-py casts same-size int <-> int by Transmute (casting.rs:68-72): bit-identical to Cast (as), bool8 and f16 excluded?",
      "X -> bool8 SIMD (simd_ne(zero), 134-148) vs scalar for -0.0 and NaN; bool8 -> X is 0/1 for non-canonical bytes.",
      "CountZeros SIMD for i32 / u32 (19-32) vs count_zeros."
    ]
  },
  {
    "id": "elementwise-ops-select-transmute-subdtype",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Select (where), Transmute, SubDtype and Real/Imaginary storages",
    "files": [
      "jix/src/ops/elementwise/select.rs:1-454",
      "jix/src/ops/transmute.rs:1-141",
      "jix/src/ops/sub_dtype.rs:1-187",
      "jix/src/ops/elementwise/complex.rs:1-161"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/context.rs",
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/index.rs",
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/dtype.rs"
    ],
    "focus": [
      "[new #100/#104] Select::read_data_uncached (130-311): pull cond and y, materialize out, push x, mix in place - strides of three buffers, 0-stride y/cond, caller-strided push out, the typed-loop table (185-194) vs struct dtypes with itemsize != alignment.",
      "[new #100] Select checks check_out_buf inside the read_cached closure (136 vs 329): can a cache hit serve a region or push out that the checks would reject? Can the x push or a cache eviction invalidate live y/cond views when the operands share storage?",
      "[changed #104] SubDtype pull: with_offset of the parent buffer (Slice / PoolBufMut / Shared) at the parent strides - valid for every consumer (later push out, REQUIRE_ALIGNED pipeline leaf)?",
      "[changed #104] SubDtype push: get_unchecked(offset..) (105) and NdCopier rely on a non-empty Index and the struct invariant - any path where either fails?",
      "[changed #104] Transmute forwards OutBuf (87-95): the inner checks a push out with its own (equal) itemsize; pull bytes reinterpreted at a possibly larger alignment rely on pipeline staging.",
      "With bool replaced by bool8, what (if anything) still makes Transmute's unsafe contract non-vacuous (padding / uninitialized source bytes)?"
    ]
  },
  {
    "id": "reduction-driver",
    "ctx": [
      "reduction"
    ],
    "title": "Reduction driver: ReductionOp::new, read_data with read cache, bulk/tile loop, state buffer choice, finalize",
    "files": [
      "jix/src/ops/reduction/common.rs:1-19",
      "jix/src/ops/reduction/common.rs:116-606",
      "jix/src/ops/reduction/common.rs:1121-1263"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/context.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/iter/nd.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/error.rs",
      "jix/src/array/read.rs",
      "jix/src/storage/core_trait.rs"
    ],
    "focus": [
      "NEW (#103): tile_size[d] as usize (566) feeds Index::new_unchecked (567-568, debug_assert only); on 32-bit with a large user read_size, can a reduced-dim tile exceed u32 and truncate to 0 or a smaller size, breaking NonZero, dropping items, and making a later tile assume_init_read an uninitialized state?",
      "NEW (#104): after out.materialize fills the pull slot (359), check_region_size (501) and the inner read (569) can still fail; does any caller (read_cached_impl context.rs:219, read_data_pull_in, shape ops) inspect or reuse a filled slot after an Err?",
      "NEW (#100): read_data goes through context.read_cached and enters self.cache_scope for the inner reads; can a cached (shared, read-only) reduction result or a per-tile cached inner buffer reach a consumer expecting a writable/exclusive buffer, or be recycled by return_shared while still borrowed?",
      "NEW (#103): the empty-output early return became debug_assert!(out_nitems != 0); is every path into read_data guaranteed a non-empty Index (inner indices derived by shape ops included)?",
      "state_in_out_buf (485-491) aliases states with the output: is every kernel passing the size/align test (Mean<f64>, Mean<i8..u32/bool8> with i64/u64 state, Mean<Complex<f64>>) safe to finalize in place, and does the early return at 1155-1161 ever skip a needed write?",
      "Is every output cell initialized by the first tile of its bulk (tile grid starts at reduced coords 0, non-reduced dims one tile per bulk), so finalize never reads uninitialized pool memory?",
      "state_offset / tile_base_reduced_idx (571-576) and get_unchecked_mut (579): is tile containment (debug_assert 558-564) guaranteed when scale_read_shape snaps a kept dim to array_shape and the index starts mid-block?"
    ]
  },
  {
    "id": "reduction-fold-loops",
    "ctx": [
      "reduction"
    ],
    "title": "Reduction fold loops: kernel contract, reduce_tile_impl, one-cell pairwise/lanes and multi-cell folds, op macros",
    "files": [
      "jix/src/ops/reduction/common.rs:20-114",
      "jix/src/ops/reduction/common.rs:608-1119",
      "jix/src/ops/reduction/common.rs:1269-1421"
    ],
    "consult": [
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/util/mod.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/ops/reduction/min_max.rs",
      "jix/src/storage/buf.rs"
    ],
    "focus": [
      "NEW-adjacent: reduce_tile_impl now receives the tile shape as usize from the #103 Index (566) and casts inner_len/offsets back with as usize (737-744); are these casts lossless on 32-bit for every tile the driver can produce?",
      "Init vs merge is keyed only on base_item_idx (637 multi-cell, 808 one-cell): can any layout (len-1 or broadcast reduced dims, reverse memory order, permuted out strides, single-element tile) give a cell's first visit a position > 0 or a later visit position 0?",
      "One-cell vs multi-cell is state_inner_stride == 0 (717): can the coalesced inner run ever mix reduced and kept dims, or be a kept dim with state stride 0 (relies on check_out_buf_impl rejecting 0 strides on len>1 dims)?",
      "ArgMax/ArgMin indices: lane init indices and idx_step (939-966), tail (873-880), pairwise halves (912-918), cross-tile merge (808-811): is the first max / first NaN always reported?",
      "Lane count (825-840) and dispatch_lanes (842-850): can lanes exceed PAIRWISE_SPLIT_THRESHOLD/2 or violate fold_pairwise's left_len >= LANES assumption (914); are the [T; LANES] reads (995-1011) always within the run?",
      "Contiguous vs strided paths must agree: multi-cell contiguous synthesizes strides (1080) from the flags at 729; aligned reads under REQUIRE_ALIGNED rely on items_aligned (704-706) computed over coalesced strides only.",
      "merge_states lane collapse (1017-1036) with MaybeUninit and a panicking Reduce closure: any double read, leak or partially written state visible afterwards?"
    ]
  },
  {
    "id": "reduction-arith-minmax",
    "ctx": [
      "reduction"
    ],
    "title": "Sum/Product/Mean and Max/Min/ArgMax/ArgMin scalar traits and kernels",
    "files": [
      "jix/src/ops/reduction/arith.rs:1-614",
      "jix/src/ops/reduction/min_max.rs:1-447"
    ],
    "consult": [
      "jix/src/ops/reduction/common.rs",
      "jix/src/ops/elementwise/arith.rs",
      "jix/src/ops/elementwise/cast.rs",
      "jix/src/ops/elementwise/min_max.rs",
      "jix/src/scalar/bool.rs",
      "jix/src/scalar/simd.rs"
    ],
    "focus": [
      "New since 6dbeb15: only the #97 move from reduction.rs (no logic change); do these kernels still satisfy what the reworked driver assumes (init_state(None) only via finalize_states_empty_inner_loop, nitems = full_reduction_size, NEEDS_FINALIZE=false only when Output == State bit-identical)?",
      "update_bulk_simd (arith.rs:16-46): does the SIMD path (scalar Cast then add_bulk/mul_bulk) give the same integer results as update for every type, including bool8 -> u64 (must count non-zero bytes) and the Sse2/Neon wide-cast fallback?",
      "Mean (arith.rs:262-320): accumulator types and wrap/precision (i8..i32 in i64, u8..u32/bool8 in u64, i64/u64 in f64, f16 in f32), finalize cast chain to the output, IS_PRECISE_SUM driving PREFER_TREE_MERGE.",
      "Seeds: Sum init -0.0 vs empty +0.0 (arith.rs:104-110, 382-387); Product init 1; confirm real-valued seeds are exact (complex seeds are intentional, NumPy-matching).",
      "ArgMax/ArgMin update and merge_states (min_max.rs:56-85): ties keep the lower index, first NaN wins, both orders of merge; bool8 compares by truthiness; f16 path.",
      "Max/Min NaN canonicalization via Maximum/Minimum bulk vs scalar; bool8 max/min return raw input bytes (intentional open question, do not report as a bug unless it breaks a contract)."
    ]
  },
  {
    "id": "reduction-variance-bools-custom",
    "ctx": [
      "reduction"
    ],
    "title": "Variance/StandardDeviation (Welford), All/Any, Reduce (reduce_unordered)",
    "files": [
      "jix/src/ops/reduction/variance.rs:1-513",
      "jix/src/ops/reduction/bools.rs:1-155",
      "jix/src/ops/reduction/custom.rs:1-336",
      "jix/src/ops/reduction/mod.rs:1-34"
    ],
    "consult": [
      "jix/src/ops/reduction/common.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/ops/elementwise/cast.rs",
      "jix/src/scalar/bool.rs",
      "jix-py/src/ops/reduction.rs"
    ],
    "focus": [
      "New since 6dbeb15: only the #97 move (custom.rs changed in tests only); with the #103/#104 driver, is the Reduce closure still invoked exactly n-1 times per cell and never on an uninitialized state?",
      "welford_update_bulk (variance.rs:190-214) vs scalar update (95-103): same mean/m2 up to rounding for real and complex (re/im passes sharing m2, 173-187), same_count fast path, lanes with differing counts?",
      "merge_states (variance.rs:128-153): count-0 shortcuts, complex delta * Complex(r, 0) producing NaN from inf*0, m2 cross term overflow.",
      "finalize (variance.rs:155-161): uses nitems not state.count; m2 < 0 -> inf, NaN stays NaN (K108 open, known); ddof NaN/negative/inf; sqrt before narrowing to f16/f32.",
      "All/Any (bools.rs): normalize to 0/1 for any input byte, empty identities true/false, NEEDS_FINALIZE=false with State = Output = bool8.",
      "Reduce (custom.rs:72-140): ReduceKernel claims NEEDS_FINALIZE=false and PREFER_TREE_MERGE=true; wide/struct T lane caps (K72 debug-build stack still open); empty dims error in new."
    ]
  },
  {
    "id": "shape-ops-forwarders",
    "ctx": [
      "shape-ops"
    ],
    "title": "Stride-remapping forwarders (PermuteDims, InsertDim, RemoveDim, Broadcast) and the OutBuf / check_out_buf / read_data_and_map_strides plumbing",
    "files": [
      "jix/src/ops/shape_ops/permute_dims.rs:1-203",
      "jix/src/ops/shape_ops/insert_dim.rs:1-247",
      "jix/src/ops/shape_ops/remove_dim.rs:1-241",
      "jix/src/ops/shape_ops/broadcast.rs:1-221",
      "jix/src/storage/buf.rs:351-555"
    ],
    "consult": [
      "jix/src/storage/index.rs",
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/params.rs",
      "jix/src/context.rs",
      "jix/src/storage/plain.rs",
      "jix/src/storage/scalar.rs"
    ],
    "focus": [
      "[new] read_data_and_map_strides now relabels the caller's pull slot in place with set_strides (buf.rs:540-546): can the relabel ever leave a WRITABLE inner buffer with aliasing strides (0 must only go to extent-1 dims)?",
      "Are the inner2outer / outer2inner closures exact inverses for unsorted, adjacent, leading and trailing inserted/removed dims, and is the re-strided push out still extent-correct for the callee's check_out_buf?",
      "[new] Cache scopes: Broadcast never scopes and InsertDim skips the scope when all inserted dims are in front (insert_dim.rs:124-128). Does anything depend on a scope for values, or can a guard leave the context's cache in a wrong state?",
      "Broadcast: pull calls make_readonly before setting 0 strides (broadcast.rs:163-164); push copies from a pull of the collapsed region. Any path that writes through or forwards a 0-stride buffer?",
      "check_out_buf_impl (buf.rs:383-431): is every overlapping or too-short out rejected (sorted-stride disjointness, length-1 dims with arbitrary strides, padded / permuted outs)?",
      "Spec derivation: layout-order permutation validity (InsertDim splice via read_layout_order_insert_dont_care_dim, RemoveDim renumbering, PermuteDims inv_dims), per-dim array lengths, PLAIN_READ kept / IS_COMPACT cleared."
    ]
  },
  {
    "id": "shape-ops-slice-reshape",
    "ctx": [
      "shape-ops"
    ],
    "title": "Slice (identity, step-1 forward, plain box fast path, stepped general path, SliceSpec/DimSlice) and Reshape",
    "files": [
      "jix/src/ops/shape_ops/slice.rs:1-450",
      "jix/src/ops/shape_ops/reshape.rs:1-426"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/error.rs",
      "jix/src/archive/array.rs",
      "jix-py/src/ops/shape_ops.rs"
    ],
    "focus": [
      "[new] Identity Slice forwards the inner's spec (flags, cache ids) and as_compact() (slice.rs:122-124, 223-241): is is_identity exactly 'start 0, step 1, full length', and is write_to (archive/array.rs:421-430, inner blocks under the outer shape) then correct?",
      "[new] Plain pull fast path rewritten on the caller's slot (slice.rs:151-174): box start/size math, stride*step vs itemsize for output dims of length <= 1, and the 32-bit usize overflow of the box size for a Scalar inner with dims > usize::MAX.",
      "Stepped general path (slice.rs:176-212): per-item sub-buffer offsets with permuted / padded / push out, inner_size on step-1 dims; is every output element written exactly once?",
      "DimSlice::resolve and the Rust range -> SliceItem conversions (slice.rs:335-432): negative, inclusive (..=-1), saturated bounds, start == len, zero-length dims, steps not dividing the length.",
      "Reshape read (reshape.rs:281-372): matched-prefix invariant, the flat decomposition that omits matched dims, orig_strides with dummy 0 on unmatched orig dims, read_size, for 0-d, length-1 and trailing-1 shapes.",
      "Reshape::new: try_shape_product().unwrap() on the inner shape (50), the constructor scan (61-73) that can map two new dims to one orig dim, and the one_to_one layout-order guard (106-130)."
    ]
  },
  {
    "id": "shape-ops-copy-ops",
    "ctx": [
      "shape-ops"
    ],
    "title": "Read-then-copy and split ops: Flip, Roll, Repeat, Tile",
    "files": [
      "jix/src/ops/shape_ops/flip.rs:1-231",
      "jix/src/ops/shape_ops/roll.rs:1-209",
      "jix/src/ops/shape_ops/repeat.rs:1-286",
      "jix/src/ops/shape_ops/tile.rs:1-349"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/index.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/mod.rs",
      "jix/src/array/read.rs"
    ],
    "focus": [
      "[new] Flip's size<=1 skip (flip.rs:59-61) runs before the duplicate bookkeeping, and the identity path (94-96) bypasses check_out_buf and the cache scope while forwarding spec/as_compact: any wrong acceptance or mismatch?",
      "[new] Roll identity (shift 0) and the non-wrap forward of out (roll.rs:107-112): j_start when e <= shift; wrap split lengths and dst offsets (117-145).",
      "Every get_unchecked / get_unchecked_mut offset (flip.rs:157-158, repeat.rs:180-190, tile.rs:216-290): provably <= the slice length for all requests, including 0-stride inner dims and length-1 out dims with huge strides?",
      "Repeat head/middle/tail (repeat.rs:204-228) cover each output position exactly once for every s mod n and e mod n; (ndim+1)-d copy shapes/strides and the g_len <= 1 stride special case.",
      "Tile case A/B/C boundaries (s_in + total <= l, total <= l), case C head/middle/tail extents vs out_shape, the stride-0 synthesized dim with dst stride l * out_strides[k]; inputs with ndim == NDIM_MAX - 1.",
      "[new] With empty-request returns removed, are all sub-Index sizes (roll len1/len2, tile len1/len2/l, repeat g_end - g_start) non-zero and untruncated before Index::new_unchecked?"
    ]
  },
  {
    "id": "shape-ops-join",
    "ctx": [
      "shape-ops"
    ],
    "title": "Concatenate and Stack (array sequences, forward-to-one-member path, packed fast paths)",
    "files": [
      "jix/src/ops/shape_ops/concatenate.rs:1-349",
      "jix/src/ops/shape_ops/stack.rs:1-270",
      "jix/src/util/arr_sequence.rs:53-197",
      "jix/src/util/arr_sequence.rs:592-655"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/params.rs",
      "jix/src/context.rs",
      "jix/src/ops/into_type.rs",
      "jix-py/src/ops/shape_ops.rs"
    ],
    "focus": [
      "[new] Forward-to-one-member path (concatenate.rs:221-246): first_arr with zero-length members at the start / middle / end, requests ending on a border, more than 32 members (partition_point), local start req_start - arr_start.",
      "[new] Zero-length member skip (concatenate.rs:275-277): is there any other way the general loop builds a zero-size sub-Index?",
      "inner_contiguous (concatenate.rs:260-265, request shape) and in_place (stack.rs:202-203, array shape): is the packed slot exactly the member's bytes in out?",
      "Per-member sub-index and buffer offset with permuted / padded out; stack dim at 0, middle and ndim; 0-d stacks; request starting mid-stack.",
      "[new] Cache-id merge: Concatenate uses repeats_across_inners = false, Stack true (concatenate.rs:156-162, stack.rs:139): values unaffected, scopes balanced?",
      "ElementType soundness: can a sequence reach Stack/Concatenate with mismatched runtime dtypes but a typed ElementType (Vec<ArrayAny>, IntoType wrappers, tuples)?"
    ]
  },
  {
    "id": "util-memory-alloc-pool",
    "ctx": [
      "util-memory"
    ],
    "title": "AlignedBytes / RawAlignedBytes allocation and BufferPool / PoolBuf / PoolBufShared (incl. the new read-cache sharing)",
    "files": [
      "jix/src/util/aligned_vec/raw.rs:1-258",
      "jix/src/util/aligned_vec/mod.rs:1-445",
      "jix/src/util/buf_pool.rs:1-165"
    ],
    "consult": [
      "jix/src/context.rs",
      "jix/src/storage/buf.rs",
      "jix/src/codec/mod.rs:100-310",
      "jix/src/array/compact.rs:235-335",
      "jix/src/storage/compact.rs:405-445",
      "jix/src/ops/elementwise/pipeline.rs:495-585,870-890",
      "jix/src/ops/reduction/common.rs:480-600",
      "jix/src/dtype.rs:40-120"
    ],
    "focus": [
      "NEW (2d5e279): PoolBufShared / return_shared / PoolBuf::into_inner and the read cache (context.rs:104-290, storage/buf.rs:337-348): can a buffer be pooled, reused or written while any Arc clone, StridedBuf view or raw pointer into it is alive; is the read-only promise airtight (into_shared of a PoolBufMut, set_strides / with_offset on a Shared view)?",
      "Every alloc/realloc/dealloc: size > 0, align a non-zero power of two, the same Layout on alloc and dealloc on every path (grow from cap 0, shrink, clone, pool reuse, Arc dropped outside the pool)?",
      "Does any consumer of PoolBuf::as_*slice / PoolBufShared::as_slice or a pool-backed StridedBuf::data() treat the full capacity as the logical length?",
      "Is every PoolBuf / AlignedBytes::set_len byte written before it is read (filter tails, partial blocks, zstd bound, chunk read_data_push)?",
      "get_pool's raw *mut Vec across align_other.insert; reentrancy via guard drops or return_shared under read_cache.borrow_mut(); !Sync really holding for ReadContext and Encoder?",
      "Can the size passed to BufferPool::get wrap (32-bit product * itemsize, state buffer size) while the strides still describe the full shape?"
    ]
  },
  {
    "id": "util-memory-arrayvec-bitmap-cache",
    "ctx": [
      "util-memory"
    ],
    "title": "ArrayVec / DimArray, DimBitmap, CPU cache detection and branch hints",
    "files": [
      "jix/src/util/arrayvec.rs:1-607",
      "jix/src/util/bitmap.rs:1-131",
      "jix/src/util/hint.rs:1-23",
      "jix/src/util/cpu_cache.rs:1-326"
    ],
    "consult": [
      "jix/src/dimension.rs:480-515",
      "jix/src/dtype.rs:125-135,1205-1221",
      "jix/src/storage/params.rs:285-410",
      "jix/src/ops/elementwise/pipeline.rs:889-930",
      "jix/src/ops/shape_ops/stack.rs:100-140",
      "jix/src/storage/info.rs:60-80"
    ],
    "focus": [
      "CHANGED (doc-only, a7b0656): do the rewritten docs still match the code (ArrayVec::insert/truncate panics, DimBitmap::insert, CacheSizes 'takes its DEFAULT, raised to the next level down')?",
      "Can any path store len > CAP or count an uninitialized slot (set_len, extend ScopeExitGuard on panic, Clone guard, IntoIter next/next_back interplay), given len() feeds assert_unchecked?",
      "insert shift arithmetic at index == len and when full; insert_first_const in const eval for [T; N] dtypes.",
      "ZST handling in extend_from_iter / Clone / IntoIter; release-mode truncation in extend_from_slice.",
      "DimBitmap: insert at pos == len and len 7 -> 8, high bits staying zero (derived Eq/Hash), all() for len 0 and 8.",
      "Cache detection: 0 / garbage / duplicate sysfs values, parse_size overflow on 32-bit, and what storage/params.rs does with l1_data == 0 or a tiny l2.",
      "Anything assuming CACHE_LINE_SIZE >= a dtype alignment (32 on armv7, 64 on x86)?"
    ]
  },
  {
    "id": "util-memory-helpers",
    "ctx": [
      "util-memory"
    ],
    "title": "util/mod.rs helpers: span/stride math, casts, AlternatingBuffers, scale_read_shape, noalias wrappers, aligned access, POD I/O",
    "files": [
      "jix/src/util/mod.rs:1-833"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/plain.rs:440-475",
      "jix/src/ops/elementwise/pipeline.rs:240-300,395-420,495-650,835-865",
      "jix/src/util/nd_copy.rs:400-450",
      "jix/src/ops/elementwise/select.rs:120-280",
      "jix/src/ops/reduction/common.rs:480-600,690-740",
      "jix/src/archive/block.rs:340-520",
      "jix/src/archive/common.rs:40-130",
      "jix/src/storage/params.rs:740-800,915-930",
      "jix/src/codec/mod.rs:139-310",
      "jix/src/storage/index.rs:1-100"
    ],
    "focus": [
      "NEW (6274df1): strided_span_bytes_from_iter (no external caller since c28c62a) and the inline span math that replaced it in Plain::read_data (storage/plain.rs:449-460): right for 0-d, length-1 dims and 0 strides; does any caller pass mismatched shape/stride lengths?",
      "Is every span given to PtrNoalias::new / PtrMutNoalias::new / from_slice at least what the kernel touches (LANES-wide tails, n == 0, bytes_offset past the span)?",
      "Any call site where a PtrMutNoalias range overlaps memory read through another pointer during its lifetime (reduction state in the out buffer, select x-in-out, pipeline dst vs operand cursors, cached Shared inputs)?",
      "strides_for_layout_order with a non-permutation layout_order; default_strides / Idx::from_u64 truncation on 32-bit.",
      "AlternatingBuffers parity and buffer lengths in Encoder/Decoder (CHANGED docs: do they match edit()?); soundness of the raw-pointer move in edit().",
      "cast_slice / cast_slice_mut / value_as_bytes / value_from_io with invalid-bit-pattern T, typed references over uninitialized buffers.",
      "scale_read_shape: product overflow, zero dims, duplicate or > 8 order entries, budget math; calc_block_end callers with block size 0.",
      "On REQUIRE_ALIGNED targets (armv7, ppc64le), is every aligned read_maybe_aligned / write_maybe_aligned guarded by a real (not debug_assert) check of base and strides?"
    ]
  },
  {
    "id": "util-nd-copier",
    "ctx": [
      "util-nd"
    ],
    "title": "NdCopier dtype-specialized n-d copy, its new long-dim pre-pass, fill_contiguous and the noalias pointer helpers",
    "files": [
      "jix/src/util/nd_copy.rs:1-471",
      "jix/src/util/mod.rs:103-123",
      "jix/src/util/mod.rs:653-804"
    ],
    "consult": [
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/dtype.rs",
      "jix/src/storage/buf.rs",
      "jix/src/storage/compact.rs",
      "jix/src/array/compact.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/ops/shape_ops/tile.rs",
      "jix/src/ops/shape_ops/flip.rs",
      "jix/src/ops/shape_ops/repeat.rs",
      "jix/src/ops/sub_dtype.rs",
      "bug-hunt/KNOWN.md"
    ],
    "focus": [
      "NEW (c28c62a) pre-pass nd_copy.rs:66-92: for 0 or 1 long dims does it reproduce exactly what NdIterUnordered did (unit dims dropped whatever their stride, empty early return before any pointer math, 0-d, shape slices longer than 8, stride slices longer or shorter than shape)?",
      "Both 1-d paths (121-148, 399-413) now take every single-long-dim copy from Flip/Tile/Repeat/compact reads: with dst stride 0, both strides 0, stride < itemsize or not a multiple of align_of::<T>(), does every pick_inner_loop route satisfy the assert_unchecked at 250-251 and 308?",
      "Aligned kernel selection: are the base pointers and every stride actually used checked against align_of::<T>() on both paths, including i686 where align_of::<u64>() == 4 (K8 debug_assert at 117)?",
      "copy_untyped routing (388-397) and the doubling fill (366-375) for itemsize 0, 3, 12, 24 on both the 1-d and iterator paths: any out-of-span byte or overlapping copy_nonoverlapping?",
      "Does every yielded bytes_offset (mod.rs:710, 764) stay <= span - itemsize, and does each caller's get_unchecked(off..) sub-slice (compact.rs:438, 504-505; repeat.rs:182-191; flip.rs:157-158; tile.rs:218, 258, 281; sub_dtype.rs:105; array/compact.rs:316) cover the strided span?",
      "fill_contiguous (440-470) has no Safety doc: does its only caller Staging::fill_from (pipeline.rs:528-545) guarantee n >= 1, a non-null dst of n*itemsize bytes and no overlap?"
    ]
  },
  {
    "id": "util-nd-iter-unordered",
    "ctx": [
      "util-nd"
    ],
    "title": "NdIterUnordered / NdIterUnorderedDyn dim sort, coalescing, sentinels and their consumers",
    "files": [
      "jix/src/util/nd_iter_unordered.rs:1-332",
      "jix/src/util/nd_iter_unordered_dyn.rs:1-247"
    ],
    "consult": [
      "jix/src/util/iter/nd.rs",
      "jix/src/util/iter/strides.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/ops/elementwise/select.rs",
      "jix/src/ops/reduction/common.rs",
      "jix/src/util/nd_copy.rs"
    ],
    "focus": [
      "Coalescing (131-152, _dyn 94-116): can 0 strides, equal strides, an excluded operand (new_with) or a wrapped unchecked s[d]*shape[d] merge two dims whose bijection fails for some operand? Is the group stride always the innermost member's?",
      "dims_sort_by (312-331) with the None-transparent arm: always a permutation; any consumer whose correctness (not just speed) depends on the resulting order?",
      "Sentinels: all-unit/0-d gives FAKED strides = size and flags true, empty gives one inner call with len 0; do pipeline (step_by at pipeline.rs:247/378), select.rs:201-212, the reduction fold (common.rs:674-760) and finalize/empty_run (1158-1181) tolerate both?",
      "Static vs dyn parity and the 2-d manual loop (276-290) vs the NdIter path (291-304), with I = u64 in the reduction on 32-bit targets.",
      "Which consumers are order-sensitive (argmax/argmin ties, float folds) or walk aliasing buffers (finalize with state == out)?",
      "New code: none (doc only since 6dbeb15). With NdCopier now handling <= 1 long dim itself, are the single-dim and sentinel branches still correct for the remaining consumers?"
    ]
  },
  {
    "id": "util-nd-iter-ordered",
    "ctx": [
      "util-nd"
    ],
    "title": "NdIter C-order walker, stride-offset / logical-index / block-border extensions, calc_block_end",
    "files": [
      "jix/src/util/iter/mod.rs:1-5",
      "jix/src/util/iter/nd.rs:1-520",
      "jix/src/util/iter/block.rs:1-142",
      "jix/src/util/iter/strides.rs:1-173",
      "jix/src/util/mod.rs:620-629"
    ],
    "consult": [
      "jix/src/storage/compact.rs",
      "jix/src/array/read.rs",
      "jix/src/array/compact.rs",
      "jix/src/array/construct.rs",
      "jix/src/ops/reduction/common.rs",
      "jix/src/ops/shape_ops/flip.rs",
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/ops/shape_ops/reshape.rs",
      "jix/src/ops/shape_ops/repeat.rs"
    ],
    "focus": [
      "advance/advance_dim (73-120): carry and wrap notifications for ndim 0-8, both the fixed-D carry ladder and the DimDyn loop; does the right-to-left protocol hold for every extension combination?",
      "Block border math (block.rs:55-81, 102-125) for every begin/end/b relation (same block, end % b == 0, begin == end aligned and unaligned, b larger than the range): any wrong (offset, size) or underflow?",
      "Does every caller build NdIter bounds exactly as [begin/b, calc_block_end(begin, end, b)) with b >= 1 (compact.rs:466-481, read.rs:525-537, array/compact.rs:247-286, reduction/common.rs:443-546)?",
      "32-bit: S::from_u64 truncation in the stride-offset extensions with S = usize (construct.rs FnStorage, slice.rs, flip.rs) - is the walked range always bounded to usize first?",
      "nitems / remaining() products and size_hint: still bounded for every caller (settled in batch5 via check_shape_overflow)?",
      "New code: none (doc only since 6dbeb15, 3-extension cap now documented). Does any caller chain a 4th extension or pass bounds that differ from the documented ones?"
    ]
  },
  {
    "id": "util-nd-arr-sequence",
    "ctx": [
      "util-nd"
    ],
    "title": "ArraySequence adapters (arrays, Vec, slices, tuples), pipeline tuples and inline array helpers",
    "files": [
      "jix/src/util/arr_sequence.rs:1-727",
      "jix/src/util/arr_ext.rs:1-258"
    ],
    "consult": [
      "jix/src/ops/elementwise/map.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/ops/shape_ops/stack.rs",
      "jix/src/ops/shape_ops/concatenate.rs",
      "jix/src/storage/core_trait.rs"
    ],
    "focus": [
      "NEW (c28c62a/7ebac4c): read_data now forwards &Index and OutBuf<'_, 'a> and returns Result<()> (62-68, 163-171, 261-269, 358-366, 480-488, 604-615); can the 'a lifetime let a pull view outlive what it borrows, and do stack/concatenate pass the right sub-index?",
      "N_OPERANDS exactness: do 219, 316, 420, 542 and 678 equal what operands() yields for every pipeline type, given pipeline.rs:136-140 unwraps (too few) and only debug_asserts leftovers (too many)?",
      "Vec/slice scratch (444-459, 566-581): is every one of narrays*M slots written before set_len, and can the user closure in map.rs:349-366 re-enter while the UnsafeCell borrow is live?",
      "Empty sequences (narrays() == 0 for Vec, slices, [_; 0]): is every consumer guarded (map.rs:237, stack.rs:66, concatenate.rs:73)?",
      "arr_ext drop guards (27-55, 75-97, 135-163): exactly-once drop on Err and on panic, including the needs_drop const branches?",
      "Index validity of arr across all impls, and tuple out_of_bounds_array_index (723-727)."
    ]
  },
  {
    "id": "types-dtype-core",
    "ctx": [
      "types"
    ],
    "title": "Runtime Dtype: Alignment, struct layout validation, set_shape, PartialEq",
    "files": [
      "jix/src/dtype.rs:1-924"
    ],
    "consult": [
      "jix/src/util/mod.rs",
      "jix/src/util/arrayvec.rs",
      "jix/src/archive/schema/conversions.rs",
      "jix-py/src/dtype.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix/src/ops/transmute.rs"
    ],
    "focus": [
      "Does every runtime constructor (from_fields 476-504, new_struct 569-622, set_shape 832-857) guarantee that fields tile [0, element_itemsize) exactly with no padding, unique names and nesting <= 32, using checked u16 arithmetic? SubDtype (sub_dtype.rs:98,105) reads field bytes with no re-check.",
      "Can two dtypes compare equal under PartialEq (897-924) but differ in layout? Only is_aligned is ignored; check the derive's packed is_aligned=false vs the runtime true for all-align-1 fields.",
      "new_struct uses element_itemsize = itemsize / shape_prod for the aligned check while from_fields uses itemsize: does any shape/alignment combination pass with fields that do not tile one element?",
      "set_shape 849-851 does an unchecked product::<Itemsize>() of the current shape plus an assert!: is it reachable with a dtype whose shape product overflows u16 or is zero?",
      "from_fields sorts with a stable sort and uses saturating_add before check_fields: can zero-sized fields at equal offsets, or a saturated itemsize, slip a bad layout through?",
      "NEW since 6dbeb15: dtype.rs only changed docs (new_struct args 515-523, Dtyped Safety 1153-1170). Do the rewritten docs promise anything the code does not enforce, now that Transmute is safe (#105) and any safe-built Dtype can label arbitrary bytes?"
    ]
  },
  {
    "id": "types-dtyped-derive",
    "ctx": [
      "types"
    ],
    "title": "Dtyped trait, built-in impls, #[derive(Dtyped)] and ElementType",
    "files": [
      "jix/src/dtype.rs:925-1223",
      "jix-macros/src/lib.rs:1-246",
      "jix/src/element_type.rs:1-98"
    ],
    "consult": [
      "jix-macros/tests/derive_external.rs",
      "jix/src/storage/scalar.rs",
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/block.rs",
      "jix/src/ops/transmute.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/scalar/bool.rs"
    ],
    "focus": [
      "Can the derive emit a DTYPE that breaks D1-D6 (itemsize == size_of, alignment == align_of, offsets == Rust layout, every bit pattern valid, no padding, accepted by new_struct) without user-written unsafe? Consider generics, ZST fields, packed, transparent, and repr spellings via cfg_attr or attribute macros.",
      "The derive passes align_of::<Self>() and is_aligned = !is_packed to new_struct_borrowed_unchecked (jix-macros 169-176) and never checks field DTYPE alignments: besides K8 (i686), can this give an 'aligned' struct whose offsets are not multiples of the field dtype alignment?",
      "Hygiene: the generated code uses ::core paths and a block-local const, but a call-site `let fields_size` (158). Can any user item in scope change the computed DTYPE rather than only cause a compile error?",
      "scalar.rs:133-136 does transmute_copy::<T, T2> after check_dtype(T2::DTYPE, T::DTYPE): is equal DTYPE always enough for same size and valid bit patterns across two distinct Dtyped types (transparent newtypes, two derived structs)?",
      "[T; N] (1205-1222): it checks size_of::<T>() > 0 but multiplies the DTYPE itemsize; is insert_first_const (util/arrayvec.rs:159-170) sound in const eval? Are all 4-dim/65535 limits enforced at compile time?",
      "NEW since 6dbeb15: TypeDyn::new (element_type.rs:41-46) is used by the now-safe transmute_elements_dyn (ops/transmute.rs:137-139). Does any consumer of a TypeDyn or Ty dtype trust more than itemsize equality, e.g. buffer alignment == dtype.alignment()?",
      "block.rs:229 assert_unchecked_eq!(element_type.dtype(), decoder_config.dtype): is equality established on every construction and element_type_change path (126, 238) and never broken afterwards?"
    ]
  },
  {
    "id": "types-dims-errors-root",
    "ctx": [
      "types"
    ],
    "title": "Dimension helpers, error/validation helpers, ArithmeticStatus, crate root",
    "files": [
      "jix/src/dimension.rs:1-514",
      "jix/src/error.rs:1-405",
      "jix/src/lib.rs:1-441"
    ],
    "consult": [
      "jix/src/storage/index.rs",
      "jix/src/array/read.rs",
      "jix/src/storage/buf.rs",
      "jix/src/util/arrayvec.rs",
      "jix/src/util/mod.rs",
      "jix/src/ops/elementwise/pipeline.rs",
      "jix/src/ops/elementwise/div_rem.rs",
      "jix/src/ops/elementwise/arith.rs",
      "jix/src/ops/reduction/common.rs"
    ],
    "focus": [
      "NEW (#103): check_ranges (error.rs:227-246) is the only validation before split_ranges (`as usize`) and Index::new_unchecked at array/read.rs:232, 299, 369, 436. Does every path establish all Index invariants (non-zero sizes, no start+size overflow, len <= 8)? Does anything use the unchecked product, e.g. default_strides_slice at read.rs:294 before check_get_buffer_size at 295?",
      "CHANGED (#103): check_get_range (248-259) now trusts Index and checks only end <= dim. Find Index::new_unchecked callers (e.g. read.rs:551-554) whose derived index could violate start + size <= dim or the Index invariants (debug_assert only, index.rs:46).",
      "CHANGED (#103): check_get_buffer_size (269-295) compares buf.len() with a checked u64 product of usize sizes. Is it still the exact guard StridedBuf::from_slice_mut at read.rs:301 needs, including 0-d, empty and 32-bit regions?",
      "REWRITTEN (#95): ArithmeticStatus/ArithmeticErrorKind (314-404). Are the flags disjoint bits and correct in into_result? Can a kernel status be lost before into_result (pipeline.rs:310, 441), e.g. on early returns, chunked or threaded paths?",
      "check_region_size (297-312) runs only on 32-bit and counts zero dims as 1: do callers pass the right itemsize (reduction/common.rs:495 passes the state size)? What bounds the region on 64-bit?",
      "dimension.rs: assert_unchecked(len <= NDIM_MAX) at 151/157 relies on the ArrayVec invariant; DimDyn/Dim to ndarray conversions use try_into().unwrap() (193, 288). Are they reachable with dims >= 2^32 on 32-bit, or with more than 8 dims?",
      "NEW (#94/#98): Dimension/Dim/DimDyn/IntoDimension became pub(crate); lib.rs adds the context module and the CompactArray/PlainArray/PlainArrayRef aliases. Does `pub use util::*`, `pub use dimension::*` or `__private` (434-441) expose anything that lets safe user code reach unsafe internals?"
    ]
  },
  {
    "id": "py-core-read-path",
    "ctx": [
      "py-core"
    ],
    "title": "Array.numpy()/out=/__getitem__/__array__/plain, index parsing, numpy allocation helpers",
    "files": [
      "jix-py/src/array.rs:122-329",
      "jix-py/src/array.rs:437-783",
      "jix-py/src/array.rs:1857-1872",
      "jix-py/src/util.rs:1-202",
      "jix-py/src/util.rs:318-329",
      "jix-py/src/ops/shape_ops.rs:222-393"
    ],
    "consult": [
      "jix/src/array/read.rs",
      "jix/src/storage/buf.rs",
      "jix/src/storage/plain.rs",
      "jix/src/error.rs",
      "jix/src/storage/params.rs",
      "jix-py/src/ops/common/operand.rs",
      "jix-py/src/dtype.rs",
      "jix-py/python/tests/test_numpy_out.py",
      "jix-py/python/tests/test_getitem.py",
      "jix-py/python/tests/test_array_protocol.py"
    ],
    "focus": [
      "NEW __array__ (array.rs:759-783): can any dtype/copy combination (positional dtype, copy=True, unsupported dtype) return something other than a fresh ndarray, and is the 'same shape as self' promise broken for inner-shape dtypes?",
      "CHANGED (#104) numpy() now writes via to_buf_strided_into (array.rs:641-656); core's check_out_buf_impl only compares the region span with the slice jix-py built from the same strides. Can the &mut [u8] at array.rs:632-637 ever exceed the real numpy memory (stride rebuild 597-626: dropped dims, length-1 dims with negative or huge strides, inner dims, transposed fresh allocation)?",
      "Is check_output_array's overlap/alignment/inner-dim validation (array.rs:213-309) equivalent to core's disjoint_span (buf.rs:398-416) for ties, length-1 dims and inner dims, and is every check skipped for empty outs actually unnecessary?",
      "CHANGED (#95) into_py_result (util.rs:64-100) maps Arithmetic to ArithmeticError: does a PyErr nested more than one source() level deep lose its type, and do all read paths go through it?",
      "32-bit: is read_shape 'as usize' (array.rs:557) always guarded before use (numpy_empty npy_intp check util.rs:121, out shape compare, core check_ranges), and can nitems/data_span overflow?",
      "Detached and free-threaded reads: what Python-side mutation between check_output_array and the write (K152), or during py.detach, invalidates the &mut [u8] over out or the &[u8] over Plain sources?",
      "parse_basic_index (shape_ops.rs:229-378): zero-length dims, start == len, -0, ints beyond i64 (TypeError vs IndexError), __index__ side effects before check_output_array, 8 items plus Ellipsis.",
      "read_ctx/ContextGuard (array.rs:161-174, 1857-1872) and read_layout_order use (570-580): re-entrancy, mutex poisoning, a non-permutation order (panic at drop_dims[dim])."
    ]
  },
  {
    "id": "py-core-dtype-surface",
    "ctx": [
      "py-core"
    ],
    "title": "numpy <-> jix dtype mapping, cached numpy descr, Array dunders and forwarders, module/package",
    "files": [
      "jix-py/src/dtype.rs:1-227",
      "jix-py/src/array.rs:1-121",
      "jix-py/src/array.rs:330-436",
      "jix-py/src/array.rs:785-1609",
      "jix-py/src/array.rs:1833-1855",
      "jix-py/src/lib.rs:1-96",
      "jix-py/python/jix/__init__.py:1-60"
    ],
    "consult": [
      "jix/src/dtype.rs",
      "jix-py/src/ops/as_array.rs",
      "jix-py/src/ops/elementwise/casting.rs",
      "jix-py/src/ops/elementwise/div_rem.rs",
      "jix-py/src/ops/common/operand.rs",
      "jix-py/python/tests/test_struct_dtype.py",
      "jix-py/python/tests/test_sub_dtype.py",
      "jix-py/python/tests/test_bad_input.py",
      "jix-py/python/tests/test_as_array.py"
    ],
    "focus": [
      "CHANGED (#96) divide(rounding_mode)/remainder(rounding_mode) and the //, % dunders (array.rs:936-1014): is every reflected variant's operand order and rounding mode right?",
      "NEW __all__ with aliases and log2/log10 wrappers (__init__.py:8-60) and changed lib.rs exports (byteswap, isfinite/isinf/isnan, unified divide/remainder): any name exported twice, missing, or shadowed incorrectly?",
      "dtype_from_numpy (dtype.rs:104-227): which numpy descrs map to a jix dtype whose itemsize/alignment/inner shape differ from numpy's in release builds (layout checks at 221-225 are debug_assert only; K8 known for i686)?",
      "The cached user descr (from_core_with_np_dtype in as_array.rs:80, select.rs:92, shape ops) is used by numpy()/check_output_array: can a descr with titles, metadata, aligned flag, field order different from offsets, or subarrays produce an allocation or is_equiv_to result inconsistent with the jix dtype?",
      "dtype_to_numpy (dtype.rs:11-84) for untrusted archive dtypes: field names, nested structs, inner shapes, aligned vs packed - any panic instead of an error?",
      "Empty struct np.dtype([]) (itemsize 0): does it reach code that divides by or trusts itemsize before core rejects it?",
      "binary_op_builtin (array.rs:1833-1855) and __pos__ (1063-1076): NotImplemented logic for np.generic, bool, subclasses; any path that returns self where a new Array is expected?"
    ]
  },
  {
    "id": "py-core-archive-params",
    "ctx": [
      "py-core"
    ],
    "title": "read_array/write_array, Python file-like adapters, mmap, compact/resolve_array_params, type hints, util helpers",
    "files": [
      "jix-py/src/archive.rs:1-540",
      "jix-py/src/array.rs:1611-1831",
      "jix-py/src/type_hint.rs:1-247",
      "jix-py/src/util.rs:204-316",
      "jix-py/src/util.rs:331-418"
    ],
    "consult": [
      "jix/src/archive/array.rs",
      "jix/src/archive/block.rs",
      "jix/src/archive/mod.rs",
      "jix/src/storage/params.rs",
      "jix/src/array/compact.rs",
      "jix-py/python/tests/test_archive.py",
      "jix-py/python/tests/test_bad_input.py",
      "jix-py/python/tests/test_read_size.py"
    ],
    "focus": [
      "CHANGED (#94/#95) archive paths now call read_from_file_with / read_from / write_to without error-mode plumbing (archive.rs:102-148, 241-297): what is borrowed across py.detach, and is cleanup on failure (set_len / remove_file, archive.rs:259-272) correct incl. the BufWriter drop order and the exists/create TOCTOU?",
      "MOVED ArchiveValidation (archive.rs:152-177): default (None -> core Strict) and error types; does any read path bypass validation?",
      "PyReader (archive.rs:317-395): a non-standard tell()/seek() (lying, negative, None) - does BufReader::stream_position at archive.rs:144 panic, and can a seek with SeekFrom::Start > i64::MAX wrap?",
      "PyWriter::write memoryview (archive.rs:441-492): beyond K109 (open), any other way a view of Rust memory escapes write(), or a write() return value that corrupts BufWriter state?",
      "mmap reads (archive.rs:120-123): file truncated/unlinked/appended while mapped, append-mode writes to the mapped file, and the K5 bounds fix.",
      "CHANGED (#94) resolve_array_params builder style (array.rs:1735-1831): any silently dropped or swallowed param, read_size arity vs into_dim_array, block_shape/fixed_dims length > 8.",
      "MOVED ItemOrTuple (type_hint.rs:141-247) and Dim (94-109): only tuples are sequences, OverflowError passthrough, NDIM_MAX cap; normalize_dim* (util.rs:219-278) for i32 extremes.",
      "array_try_map / array_try_from_fn (util.rs:331-418): panic or error in f mid-way, T needs drop but U does not (double drop or leak)?"
    ]
  },
  {
    "id": "py-ops-dispatch-promotion",
    "ctx": [
      "py-ops"
    ],
    "title": "Runtime dtype dispatch, cast rules, where/stack/concatenate promotion, operand broadcasting",
    "files": [
      "jix-py/src/ops/common/dispatch.rs:1-322",
      "jix-py/src/ops/common/dtype_promote.rs:1-246",
      "jix-py/src/ops/common/macros.rs:1-201",
      "jix-py/src/ops/common/broadcast.rs:1-105",
      "jix-py/src/ops/common/mod.rs:1-14",
      "jix-py/src/util.rs:296-418"
    ],
    "consult": [
      "jix-py/src/ops/common/operand.rs",
      "jix/src/dtype.rs",
      "jix-py/src/ops/shape_ops.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix-py/python/tests/test_type_promotion.py",
      "jix-py/python/tests/test_broadcasting.py"
    ],
    "focus": [
      "NEW in #105: define_op2_desc now threads a per-operand (a_cast, b_cast) pair through its @push recursion (macros.rs:95-146). Is the pair applied in (T1, T2) order for both (a, b) items and single-type items, and does the single-kind form still expand to (k, k)?",
      "is_cast_allowed (dtype_promote.rs:61-141): with the untyped Int->UInt remap, the P8 short-circuit for int->float, and Unsafe, is there an input combination where the chosen dtype silently loses value with no OverflowError?",
      "dispatch_args (dispatch.rs:196-258): is the order all-untyped P8 default, then cast_untyped_float_to_f64, then f16 skip right for every mix of typed and untyped operands? Is returning a cast error instead of falling through to the next descriptor intended in every table?",
      "promotion_target (dtype_promote.rs:168-222, moved from dispatch.rs in #97): the equal-dtype shortcut applies only when every operand is an array. Do struct plus scalar dtypes always give TypeError? Untyped floats get no f64 rule here; can that pick the wrong dtype?",
      "array_try_map / array_try_from_fn (util.rs:331-418): exactly-once drop of every Operand on Ok, Err, and panic in f?",
      "broadcast_shapes / broadcast_operand: zero-length dims, 0-d operands, 8-dim results, Scalar shape replacement, and a PyArray losing its cached numpy dtype after broadcasting."
    ]
  },
  {
    "id": "py-ops-operand-cast",
    "ctx": [
      "py-ops"
    ],
    "title": "Python/numpy -> Operand conversion, scalar range checks and casts, cast_impl, asarray",
    "files": [
      "jix-py/src/ops/common/operand.rs:1-521",
      "jix-py/src/ops/elementwise/casting.rs:1-138",
      "jix-py/src/ops/as_array.rs:1-95",
      "jix-py/src/ops/sub_dtype.rs:1-57",
      "jix-py/src/dtype.rs:86-227"
    ],
    "consult": [
      "jix/src/storage/plain.rs",
      "jix/src/storage/scalar.rs",
      "jix/src/ops/elementwise/cast.rs",
      "jix/src/ops/transmute.rs",
      "jix/src/ops/into_type.rs",
      "jix-py/src/array.rs",
      "jix-py/python/tests/test_as_array.py",
      "jix-py/python/tests/test_cast.py",
      "bug-hunt/KNOWN.md"
    ],
    "focus": [
      "NEW in #95, made safe in #105: cast_impl transmutes between integer dtypes of the same itemsize (casting.rs:63-66). Does it preserve values mod 2^n for every pair, with bool8 and f16 never taking this path?",
      "Python/numpy scalar conversion (operand.rs:33-153): bool checked before int, np.generic detection, 0-d ndarrays turned into typed Scalars through .item(). Is any kind extracted lossily or with the wrong precision (f16 via f32, complex64 parts, u64)?",
      "check_int_in_bounds / check_float_in_bounds (operand.rs:385-446): complete for every target kind, including NaN, inf, subnormals, -0.0 and complex parts? Is any unreachable!() arm in into_array (219, 245, 254) reachable?",
      "Do the rebuilt Scalar and every cast_impl result always have exactly the dtype the descriptor expects? Otherwise the into_typed().unwrap() calls in dispatch.rs panic.",
      "plain_from_numpy (operand.rs:449-520): strides of empty, length-1 and zero-stride dims, unaligned bases, inner-shape contiguity (reachable only from Array.plain), and K152 races on free-threaded builds.",
      "asarray (as_array.rs:105-142): the user's numpy dtype is attached verbatim (135-138). Can it disagree with the jix dtype, e.g. a sub-array dtype or a struct with titles? np.generic inputs skip numpy.asarray(dtype=)."
    ]
  },
  {
    "id": "py-ops-arith-wrappers",
    "ctx": [
      "py-ops"
    ],
    "title": "Arithmetic, division/remainder, exp/log, rounding, trig, float-special and complex op wrappers",
    "files": [
      "jix-py/src/ops/elementwise/arith.rs:1-558",
      "jix-py/src/ops/elementwise/div_rem.rs:1-239",
      "jix-py/src/ops/elementwise/exp_log.rs:1-171",
      "jix-py/src/ops/elementwise/rounding.rs:1-110",
      "jix-py/src/ops/elementwise/trig.rs:1-188",
      "jix-py/src/ops/elementwise/float_special.rs:1-85",
      "jix-py/src/ops/elementwise/complex.rs:1-74",
      "jix-py/src/ops/elementwise/mod.rs:1-38"
    ],
    "consult": [
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/src/ops/common/operand.rs",
      "jix/src/ops/elementwise/arith.rs",
      "jix/src/ops/elementwise/div_rem.rs",
      "jix/src/ops/elementwise/exp_log.rs",
      "jix/src/ops/elementwise/rounding.rs",
      "jix/src/ops/elementwise/complex.rs",
      "jix-py/python/tests/test_arith.py",
      "jix-py/python/tests/test_div_rem.py",
      "jix-py/python/tests/test_type_promotion.py"
    ],
    "focus": [
      "NEW in #96: divide(rounding_mode) and remainder(rounding_mode) replace six functions (div_rem.rs:77-191). Does each mode map to the right core op (Trunc/Floor/Euclid x Divide/Remainder) with the old tables? Do the two parse paths (FromPyObject vs &str + parse) agree?",
      "MOVED in #97 from op1.rs/op2.rs: compare every table with git show 6dbeb15:jix-py/src/ops/op1.rs and op2.rs. Order is semantics.",
      "power (arith.rs:463-558): are the SQUARE/SQRT shortcut tables identical to the Pow pairs? The value-based switch also fires on typed scalars (np.uint64(2), np.float16(2.0)), and the table is chosen before broadcasting. Same dtype and value as the general path?",
      "multiply(a, a) -> square (arith.rs:385-393), negative's unsigned None arms before the signed Safe arms, and the absolute/floor/ceil/round/real fast paths: same dtype and values as the general path for typed and untyped scalars? absolute's Complex -> Float keeps P4/P8.",
      "log (exp_log.rs:121-171): base validation (NaN and inf pass), exact matches for E, 2 and 10, and the generic path computed in f64 then cast back to T for f16/f32."
    ]
  },
  {
    "id": "py-ops-cmp-bitwise-select",
    "ctx": [
      "py-ops"
    ],
    "title": "Comparison, isclose, min/max/clamp, bitwise/logical/shift/rotate, bit-count/byteswap, where",
    "files": [
      "jix-py/src/ops/elementwise/cmp.rs:1-433",
      "jix-py/src/ops/elementwise/min_max.rs:1-196",
      "jix-py/src/ops/elementwise/bitwise.rs:1-490",
      "jix-py/src/ops/elementwise/bit_count.rs:1-223",
      "jix-py/src/ops/elementwise/select.rs:1-94"
    ],
    "consult": [
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/src/ops/common/dtype_promote.rs",
      "jix-py/src/ops/common/operand.rs",
      "jix/src/ops/elementwise/cmp.rs",
      "jix/src/ops/elementwise/min_max.rs",
      "jix/src/ops/elementwise/bitwise.rs",
      "jix/src/ops/elementwise/bit_count.rs",
      "jix/src/ops/elementwise/select.rs",
      "jix-py/python/tests/test_cmp.py",
      "jix-py/python/tests/test_bitwise.py",
      "jix-py/python/tests/test_select.py",
      "jix-py/python/tests/test_min_max.py",
      "bug-hunt/KNOWN.md"
    ],
    "focus": [
      "NEW in #105: the bitwise_rotate rewrite (bitwise.rs:402-490). rotate_amount pre-casts only untyped Int amounts to u32, and the table is (None, Safe) with i64 first. Does K76 still reproduce? Check typed bool/u8/u16 amounts, a Python int value with a typed amount, and 0-d numpy amounts.",
      "NEW in #95: where shortcuts (select.rs:68-83). A scalar-bool condition returns before the condition is typed, and `x is y` returns after it. Do non-bool scalar conditions still raise TypeError? Are the returned dtype and shape right after broadcasting and promotion? The result's numpy dtype comes from x.",
      "NEW in #95: byteswap returns 1-byte ints and bool unchanged (bit_count.rs:178-183). Correct for typed scalars and for untyped Python ints, which dispatch to i64?",
      "Comparison tables: the (u64,i64)/(i64,u64) pairs come after the u64/i64 slots, so Python ints next to u64 raise. cast_untyped_float_to_f64 applies only when no float operand is typed. bool8 is listed last and is unreachable. Can any ordering give a wrong result or dtype?",
      "clamp/maximum/minimum list bool8 first; one-bound clamp delegates to maximum/minimum; isclose's cast_tol has no range check (f16 overflow) and handles a complex atol. Logical ops use Unsafe -> bool8 for NaN and complex inputs."
    ]
  },
  {
    "id": "py-ops-shape-ops",
    "ctx": [
      "py-ops"
    ],
    "title": "Shape-op wrappers, basic index parsing, dim/shape argument normalization",
    "files": [
      "jix-py/src/ops/shape_ops.rs:1-1222",
      "jix-py/src/util.rs:16-62",
      "jix-py/src/util.rs:219-278",
      "jix-py/src/type_hint.rs:94-109",
      "jix-py/src/type_hint.rs:141-247"
    ],
    "consult": [
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/ops/shape_ops/reshape.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix/src/ops/shape_ops/permute_dims.rs",
      "jix/src/ops/shape_ops/remove_dim.rs",
      "jix/src/ops/shape_ops/stack.rs",
      "jix/src/ops/shape_ops/concatenate.rs",
      "jix/src/ops/shape_ops/repeat.rs",
      "jix/src/ops/shape_ops/tile.rs",
      "jix/src/ops/shape_ops/roll.rs",
      "jix/src/ops/shape_ops/flip.rs",
      "jix/src/ops/dims_arg.rs",
      "jix/src/error.rs",
      "jix-py/src/array.rs",
      "jix-py/python/tests/test_shape_ops.py",
      "jix-py/python/tests/test_slice.py",
      "jix-py/python/tests/test_getitem.py"
    ],
    "focus": [
      "parse_basic_index (shape_ops.rs:228-378): negative ints, start == len, zero-length dims, the PY_SSIZE_T_MAX clamp of explicit huge bounds (util.rs:39-45), the 32-bit length guard, and NDIM_MAX items plus an Ellipsis.",
      "reshape -1 inference and checked product (711-769); broadcast with -1 and leading dims (55-95); broadcast_array_impl, shared with operand broadcasting.",
      "insert_dim output-position normalization against ndim+len(dim); squeeze default dims; permute_dims with Vec<Dim> (any sequence); swapdims.",
      "concatenate/stack: promotion before the ndim check, numpy dtype taken from the first post-promotion array, single-array fast paths, stack's normalize_dim(dim, ndim + 1).",
      "repeat/tile/roll: the no-op early returns run before core validation (repeats == 1 on 8-d arrays). Python bool is accepted for repeats/shift/shape entries but rejected for Dim.",
      "Logic unchanged since 6dbeb15. Only ItemOrTuple moved from util.rs to type_hint.rs in #96; did its extraction or the NDIM_MAX bound change behavior?"
    ]
  },
  {
    "id": "py-ops-reductions",
    "ctx": [
      "py-ops"
    ],
    "title": "Reduction wrappers: dim normalization, keepdim, dispatch tables, extra args",
    "files": [
      "jix-py/src/ops/reduction.rs:1-527",
      "jix-py/src/util.rs:219-278",
      "jix-py/src/type_hint.rs:94-247"
    ],
    "consult": [
      "jix/src/ops/reduction/common.rs",
      "jix/src/ops/reduction/arith.rs",
      "jix/src/ops/reduction/variance.rs",
      "jix/src/ops/reduction/min_max.rs",
      "jix/src/ops/reduction/bools.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/src/ops/common/macros.rs",
      "jix-py/python/tests/test_reduction.py"
    ],
    "focus": [
      "normalize_dims_optional leaves duplicates to core (RuntimeError). keepdim re-inserts with InsertDim at the normalized, possibly unsorted, positions. Is the output shape right for every dim tuple?",
      "single_dim variant (argmax/argmin): dim=None allowed only for 1-D; keepdim re-insertion; normalization for 0-d inputs.",
      "Tables (None casts): argmax's different order, product without bool, all/any Unsafe -> bool8 (NaN and complex). Are struct dtypes rejected cleanly?",
      "Extra args (ddof) passed through the define_op1_desc accessor list in the right order; Python scalar / 0-d inputs through asarray; dim=() reduces nothing.",
      "Logic unchanged since 6dbeb15, but the macro now uses crate::type_hint::{Dim, ItemOrTuple} and define_op1_desc moved to macros.rs in #97. Did the move change argument order or extraction?"
    ]
  }
]

const CONCEPT_UNITS = [
  {
    id: 'concept-push-pull',
    title: 'read_data push/pull contract across every storage and op',
    ctx: ['storage-core', 'shape-ops', 'pipeline'],
    start: ['grep -n "fn read_data<" -r jix/src', 'grep -n "fn read_as_elementwise_pipeline" -r jix/src', 'jix/src/storage/buf.rs (OutBuf, check_out_buf, read_data_and_map_strides)'],
    focus: [
      'In push mode, does every impl write exactly the requested region at dst\'s own strides (not assuming packed, row-major, or zero offset) and return a view of dst - never a pooled buffer or its own memory?',
      'Does any consumer assume a pull-mode result is packed, aligned, writable, or free of 0-stride dims?',
      'Forwarders that re-stride out and push it down (read_data_and_map_strides and similar): is the remapped stride set valid for the inner region in every case, e.g. when the op duplicates, reverses, or drops elements?',
      'Ops that split one request into several inner reads (concatenate, roll, tile, repeat, stack, reductions, Array::to_buf_strided / to_buf_strided_into): does each piece land at the right byte offset of out with the right strides, and are the pieces disjoint and covering?',
      'out with 0-length dims, with strides larger than needed, or as a strided sub-view of a bigger buffer.',
    ],
  },
  {
    id: 'concept-aliasing-lifetimes',
    title: 'Lifetimes, aliasing and noalias claims',
    ctx: ['util-memory', 'storage-core', 'pipeline', 'py-core'],
    start: ['grep -n "transmute" -r jix/src jix-py/src', 'grep -n "PtrNoalias\\|PtrMutNoalias" -r jix/src', 'grep -n "from_raw_parts" -r jix/src jix-py/src', 'jix/src/util/buf_pool.rs'],
    focus: [
      'Every lifetime extension (transmute of lifetimes, &*ptr / &mut *ptr, from_raw_parts with a caller-chosen lifetime): what keeps the memory alive and unmutated for the extended lifetime?',
      'PtrNoalias / PtrMutNoalias / &mut [u8] built over regions that could overlap: an input vs out (a user out buffer that views the same memory as a Plain input; Python out= over the numpy array an input wraps), two operands of one pipeline, src and dst of NdCopier.',
      'Pooled buffers: can a view into a PoolBuf outlive it (returned to the pool and handed out again while still referenced)? Can BufferPool::get be reentered while a &mut to its free list is live?',
      'Reads of stale bytes from a reused PoolBuf that the current request never wrote.',
      'Plain / CompactMmap over external memory: what if the backing memory is mutated or unmapped while an Array exists?',
    ],
  },
  {
    id: 'concept-degenerate-shapes',
    title: 'Empty, zero-length, 0-d and length-1 shapes through every op',
    ctx: ['storage-core', 'shape-ops', 'reduction', 'pipeline', 'util-nd'],
    start: ['grep -nE "\\- 1\\b|div_ceil|ilog2|\\.last\\(\\)\\.unwrap|\\[0\\]" over jix/src (shape / stride code)'],
    focus: [
      'Which `len - 1`, `/ x`, `% x`, div_ceil, ilog2, `.last().unwrap()`, `[0]` on shape / stride slices can see 0 or an empty slice?',
      'product() of an empty shape is 1: code that conflates ndim == 0 with an empty array, or vice versa.',
      'Arrays with a 0-length dim through reshape, broadcast, tile / repeat with count 0, concatenate / stack of empties, reductions over empty dims (identity or error), argmin / argmax of empty, slices to empty ranges with unaligned starts (calc_block_end special case).',
      '0-d arrays through every op, including insert_dim / remove_dim / permute_dims on shape [] and reductions over all dims.',
      'Length-1 dims in dim-merging and stride-sorting logic (their stride is arbitrary and may be anything).',
    ],
  },
  {
    id: 'concept-integer-width',
    title: 'Integer width, overflow and truncation in size and index arithmetic',
    ctx: ['util-memory', 'util-nd', 'storage-compact', 'codec-archive', 'storage-core'],
    start: ['grep -nE " as (usize|u32|i32|i64|u64|isize)\\b" -r jix/src jix-py/src', 'grep -n "product::<usize>\\|product::<u64>" -r jix/src'],
    focus: [
      'Every `as usize` / `as u32` / `as i64` / `as u64` on a size, index, offset or stride: can it truncate on a 32-bit target (x86 and armv7 wheels ship) or flip sign?',
      'Unchecked products shape * stride * itemsize, block counts, block byte sizes, allocation sizes: which inputs make them wrap in release, and does a wrapped value then size an allocation or a from_raw_parts?',
      'Where are byte sizes validated against isize::MAX - at construction, archive load, reshape / broadcast / tile / repeat (which multiply sizes)?',
      'Signed / unsigned mixing in slicing (negative steps, start > end), roll shifts, Python-provided ints.',
    ],
  },
  {
    id: 'concept-dtype-layout',
    title: 'Dtype layout and bit validity: bool, struct, inner shape, f16, complex',
    ctx: ['types', 'elementwise-ops', 'codec-archive', 'py-core'],
    start: ['jix/src/dtype.rs', 'jix/src/scalar/bool.rs (bool8, #89)', 'jix/src/ops/transmute.rs', 'jix/src/ops/sub_dtype.rs', 'grep -n "check_dtype\\|into_typed" -r jix/src jix-py/src', 'grep -n "bool8\\|as bool\\|: bool" -r jix/src'],
    focus: [
      'Where are bytes from an untrusted or unvalidated source (archive file, numpy buffer, transmute, sub_dtype view, stale pool memory) read as a Rust bool or another type with invalid bit patterns? And since #89 made the element type bool8 (any byte valid, non-zero = true): which kernels still assume a bool8 byte is 0 or 1 (bitwise ops, sums, casts, SIMD masks, comparisons)?',
      'Dtype equality / compatibility checks that gate unsafe reinterpretation (into_typed, check_dtype, transmute, sub_dtype, Python dispatch): can two dtypes pass but differ in layout (field order, offsets, padding, inner shape, nested structs, alignment)?',
      'Struct dtypes with itemsize != alignment, padding, or odd itemsizes (3, 6, 12) through copy kernels, shuffles, casts and reductions.',
      'Inner-shape dtypes through shape ops and element-wise ops: is the inner shape treated as part of the element everywhere?',
      'f16 / complex code behind cargo features: do both feature states keep dispatch tables consistent (a missing arm, an arm mapped to the wrong type)?',
    ],
  },
  {
    id: 'concept-alignment',
    title: 'Alignment of typed accesses',
    ctx: ['pipeline', 'util-memory', 'util-nd', 'reduction', 'storage-core'],
    start: ['grep -n "REQUIRE_ALIGNED\\|read_maybe_aligned\\|write_maybe_aligned\\|cast_slice" -r jix/src', 'grep -nE "\\.read\\(\\)|\\.write\\(|from_raw_parts" -r jix/src'],
    focus: [
      'Every typed read / write through a pointer derived from StridedBuf, PoolBuf, mmap or file bytes: aligned or unaligned? If aligned, what proves the alignment, including when strides are not multiples of the alignment and on targets where REQUIRE_ALIGNED is true (armv7, ppc64le)?',
      'cast_slice / cast_slice_mut assert alignment (panic, not UB): are they reachable with unaligned data from a public API?',
      'SIMD loads / stores: aligned variants on possibly unaligned data?',
      'Struct dtypes whose alignment exceeds what BufferPool / AlignedBytes / mmap provide.',
    ],
  },
  {
    id: 'concept-arith-semantics',
    title: 'Arithmetic semantics and error modes of kernels',
    ctx: ['elementwise-ops', 'reduction', 'pipeline', 'py-ops'],
    start: ['jix/src/error.rs (ArithmeticErrorKind and its status set)', 'jix/src/ops/elementwise/op2.rs', 'jix/src/ops/elementwise/op1.rs', 'jix/src/ops/elementwise/div_rem.rs (divide / remainder with rounding_mode)', 'jix/src/ops/elementwise/cast.rs'],
    focus: [
      'Integer division / remainder by zero and MIN / -1, floor-div / mod sign conventions, overflow in add / sub / mul / neg / abs / pow, pow with negative int exponents, shifts by >= bit width or negative counts, float -> int casts of NaN / inf / out-of-range values, int -> float rounding, bool arithmetic.',
      'Does every kernel report the same ArithmeticErrorKind with the same fallback value for the same condition - SIMD body and scalar tail, element-wise ops and reductions, Rust and Python?',
      'Since #95 there are no error modes: integer errors (DivideByZero, NegativePower) always fail the read, float errors are never reported, MIN / -1 wraps. Is every integer error raised on every path (SIMD body, scalar tail, pipelines, reductions, Python), and is an Err raised after a caller-visible out (to_buf_strided_into, to_buf_slice, compaction) was partially written observable?',
      'NaN handling in min / max / argmin / argmax / clip and comparisons; signed zeros; isclose with inf / nan.',
      'Behaviors that silently lose information in a way a user would call a bug (questionable-semantics), even where they differ from NumPy on purpose.',
    ],
  },
  {
    id: 'concept-tiling-coverage',
    title: 'Request splitting and tiling: pieces must exactly cover the request',
    ctx: ['storage-core', 'storage-compact', 'reduction', 'util-nd'],
    start: ['jix/src/array/read.rs (to_buf_strided / to_buf_strided_into and the splitting heuristic), jix/src/array/compact.rs (compaction / write paths)', 'jix/src/util/mod.rs (scale_read_shape, calc_block_end)', 'jix/src/storage/params.rs', 'jix/src/util/iter/block.rs'],
    focus: [
      'Array::to_buf_strided / to_buf_strided_into and the write paths cut a request into cache-sized pieces: are the pieces disjoint, covering, correctly offset into out, and correct at array edges and for unaligned starts?',
      'scale_read_shape and the read-shape heuristics: can they produce a 0-sized tile, a tile larger than the request, or a tile shape the caller does not expect?',
      'Compact reads: block index math for partial edge blocks and ranges that start / end mid-block; the calc_block_end empty-range special case.',
      'Reductions that tile the reduced dims: is accumulator state (sum, var, argmin index offsets) carried correctly across tiles and blocks?',
    ],
  },
  {
    id: 'concept-untrusted-archives',
    title: 'Malformed and adversarial .jix files',
    ctx: ['codec-archive', 'storage-compact', 'types'],
    start: ['jix/src/archive/', 'jix/src/archive/schema/conversions.rs', 'jix-py/src/archive.rs'],
    focus: [
      'From every public read entry point (Rust and Python, bytes / file / mmap): what is validated in ArchiveValidation::Minimal and Strict, and what is trusted?',
      'For each trusted field (shape, block shape, dtype, itemsize, filters, block offsets / lengths, decompressed sizes, byte offsets of arrays in a multi-array file): what happens if it lies - panic, huge allocation, OOB read, UB? Skipping a check in Minimal mode is acceptable only if the consequence stays memory-safe (an Err or a panic), unless documented otherwise.',
      'Decompressed block size that does not match the expected block byte size; filters run on lengths that are not a multiple of itemsize or 8.',
      'Truncated files and truncated mmaps.',
      'Build repros by writing a valid array and then patching bytes of the file.',
    ],
  },
  {
    id: 'concept-python-boundary',
    title: 'Python boundary: numpy memory, GIL, threads, out=',
    ctx: ['py-core', 'py-ops', 'storage-core'],
    start: ['jix-py/src/util.rs', 'jix-py/src/array.rs', 'grep -n "allow_threads\\|detach\\|unsafe" -r jix-py/src'],
    focus: [
      'Zero-copy wrapping of numpy memory: is the numpy object kept alive as long as any jix array or lazy view points into it? Can Python resize, free or mutate it meanwhile?',
      'GIL release while reading or writing numpy memory, and free-threaded CPython builds: data races on numpy buffers or on jix-py internal state.',
      'out= arrays: read-only, non-contiguous, negative-stride, unaligned, byte-swapped, wrong dtype or shape, and out overlapping an input.',
      'Negative-stride and byte-swapped numpy inputs (jix strides are usize; jix is little-endian only).',
      'Python int / slice / index conversions to u64 / usize / i64: overflow, negative values, huge values.',
      'A Rust panic inside a call from Python: does it leave shared state (pools, caches, contexts) inconsistent for later calls?',
    ],
  },
  {
    id: 'concept-threads-send-sync',
    title: 'Send / Sync, interior mutability and global state',
    ctx: ['util-memory', 'storage-core', 'py-core'],
    start: ['grep -n "unsafe impl" -r jix/src jix-py/src', 'grep -n "UnsafeCell\\|RefCell\\|Cell<\\|static \\|OnceLock\\|thread_local" -r jix/src jix-py/src'],
    focus: [
      'Every `unsafe impl Send / Sync`, and every Send / Sync type containing UnsafeCell / Cell / RefCell / raw pointers: is concurrent use actually safe?',
      'ReadContext and BufferPool: can they be shared across threads, or reentered (a read path that calls back into the same context while holding a borrow)?',
      'ArrayAny / Arc<dyn ArrayStorage>: what bounds are required, and can a !Sync storage end up shared across threads?',
      'Global state: CPU cache-size detection, statics, lazy-init races.',
    ],
  },
  {
    id: 'concept-composition',
    title: 'Op composition: code paths that must agree',
    ctx: ['pipeline', 'shape-ops', 'storage-core', 'elementwise-ops'],
    start: ['grep -n "fn read_as_elementwise_pipeline" -r jix/src', 'jix/src/storage/params.rs (ArraySpec, flags)', 'jix/src/ops/maybe_compact.rs'],
    focus: [
      'For each op overriding read_as_elementwise_pipeline: does it compute exactly what its read_data computes (index mapping, broadcasting, dtype)?',
      'ArraySpec / flags / layout propagation: can a flag a wrapper claims (contiguity, no-broadcast, layout order, scale weights) be false for its actual output, so a consumer takes an invalid fast path?',
      'Shape ops over element-wise ops over shape ops (e.g. flip(add(broadcast(a), permute_dims(b)))) with a push-mode out at odd strides: is the forwarded out valid at every level?',
      'Fast paths that depend on the inner storage type (as_compact, maybe_compact): do they agree with the generic path?',
    ],
  },
  {
    id: 'concept-read-contract',
    title: 'The read_data contract after #102-#104: Index (non-empty sizes), OutBuf, and ArrayStorage for &S / &mut S',
    ctx: ['storage-core', 'array-frontend', 'shape-ops', 'pipeline'],
    start: ['jix/src/storage/index.rs', 'jix/src/storage/buf.rs (OutBuf, check_out_buf, read_data_and_map_strides)', 'jix/src/storage/core_trait.rs (read_data, the &S / &mut S impls)', 'grep -n "Index::new\\|Index {\\|split_ranges" -r jix/src', 'jix/src/array/read.rs (the entry points)'],
    focus: [
      'Storages below the Array entry points may now assume a non-empty, in-bounds region. Find every place an Index is built from computed ranges (shape ops mapping an outer region to an inner one, concatenate / stack / tile / repeat / roll pieces, reductions tiling their reads, select, the pipeline, Python wrappers): can any of them produce an empty or out-of-bounds region, and what does the callee then do (unsafe indexing, `size - 1`, NonZeroUsize assumptions)?',
      'Which public paths reach read_data without going through an entry-point check (ArrayStorage::read_data called directly by users, &S / &mut S impls, ArrayAny, Python)? Is Index::new the only constructor, and does it validate against the shape?',
      'OutBuf: in push mode, does every impl write exactly the region at the caller\'s strides and leave the result in the caller\'s buffer; in pull mode, who owns the returned buffer and how long may it be borrowed? Are the new transparent fast paths (full-range Slice, no-op Flip / Roll, a Concatenate read inside one member, select returning one operand) correct in both modes?',
      'ArrayStorage for &mut S: can a &mut S storage be read while another reference to S is alive, or let a read write through it?',
      'Code changed since 6dbeb15 (git diff -M 6dbeb15 HEAD) that converts between Range and start / size, or between u64 and usize sizes: off-by-one, 32-bit truncation, NonZeroUsize built from 0.',
    ],
  },
  {
    id: 'concept-read-cache',
    title: 'The read-result cache for arrays used more than once in one expression (#100)',
    ctx: ['array-frontend', 'storage-compact', 'pipeline', 'shape-ops'],
    start: ['jix/src/context.rs (read_cached, cache_scope)', 'jix/src/storage/cache_id.rs', 'grep -n "cache_id\\|arrays_to_cache\\|read_cached\\|cache_scope" -r jix/src'],
    focus: [
      'Cache key: is a hit only returned for the same array, region, dtype and read mode? What if the same array is read twice with different regions, a different out (strides, offset) or once pull and once push?',
      'Cached buffers: a pull read may lend a buffer while the cache keeps it; can a later read push into, reuse or return to the pool a buffer that is still lent? Is a cached pull result ever handed out as a writable out?',
      'Identity: which ops keep, forward or mint cache ids (into_type, transmute, sub_dtype, cast-by-transmute, Clone, into_any, Python wrappers)? Can two arrays that hold different bytes share an id, so one gets the other\'s cached data?',
      'Scopes and reentrancy: nested top-level reads on the same ReadContext, reads from inside a reduction callback or map closure, panics while a CacheScopeGuard is alive, a ReadContext shared or cloned across threads.',
      'ArrayCacheIdMap / ArrayCacheIdSet (custom map with its own Drop and Clone): memory safety and correctness, growth without bound, FxHasher collisions.',
    ],
  },
]

const DYNAMIC_UNITS = [
  {
    id: 'dynamic-rust-differential',
    title: 'Differential testing of op chains against an ndarray reference (Rust)',
    ctx: ['storage-core', 'shape-ops', 'pipeline', 'reduction'],
    task: [
      'Write integration tests (jix/tests/hunt_dynamic_rust_<n>.rs) that build random op chains - element-wise ops, shape ops (slice with steps, permute, broadcast, flip, roll, tile, repeat, concatenate, stack, reshape), reductions, casts - over small random arrays, and compare against a reference computed with plain ndarray / loops.',
      'Vary the storage: Compact with tiny explicit block shapes (so reads straddle partial blocks), Plain over strided / permuted ndarray views, broadcast inputs, 0-d and 0-length dims.',
      'Vary the materialization: to_ndarray, to_ndarray_sub over random sub-regions, to_buf_slice and to_buf_strided_into into out buffers with permuted / padded strides (and check bytes outside the region are untouched), compact(), and a write_to / read round trip.',
      'Use proptest or a seeded enumeration; keep each run under a few minutes (debug build).',
      'For every mismatch: shrink it, then trace it to a root cause in the code before reporting. A mismatch you cannot explain is reported with low confidence and the minimal failing case.',
    ],
  },
  {
    id: 'dynamic-python-differential',
    title: 'Differential testing against NumPy (Python)',
    ctx: ['py-core', 'py-ops'],
    task: [
      'Write hypothesis or seeded-random scripts (bug-hunt/runs/<run>/scratch/dynamic_py_<n>.py, run with .venv/bin/python) that apply random op chains to numpy arrays and to jix arrays built from them, and compare results.',
      'Cover dtype promotion with Python scalars and mixed dtypes, broadcasting, indexing / slicing with negative steps, reductions with axis / keepdims, shape ops, out= with non-contiguous / offset / transposed numpy outputs, and save / load round trips.',
      'Check the context docs (py-core, py-ops "Intentional behaviors") and jix-py/python/tests before calling a NumPy difference a bug. Report deliberate-but-unreasonable behavior as questionable-semantics.',
      'For every mismatch: shrink it, then trace it to a root cause in the code before reporting.',
    ],
  },
]

// ---------------------------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------------------------
const FINDING = {
  type: 'object',
  properties: {
    title: { type: 'string', description: 'one line' },
    severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    category: {
      type: 'string',
      enum: ['memory-safety', 'unsound-api', 'wrong-result', 'panic', 'overflow', 'resource', 'error-handling', 'questionable-semantics', 'other'],
    },
    file: { type: 'string', description: 'repo-relative path of the defect' },
    line: { type: 'integer' },
    related: { type: 'array', items: { type: 'string' }, description: 'other file:line locations involved' },
    description: { type: 'string', description: 'the defect and its consequence, with the code path' },
    trigger: { type: 'string', description: 'concrete input / state / op chain' },
    expected_vs_actual: { type: 'string' },
    reachability: { type: 'string', enum: ['public-rust-api', 'python-api', 'crate-internal-only', 'unknown'] },
    evidence: { type: 'string', description: 'what you ran and what it printed; empty if nothing was run' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['title', 'severity', 'category', 'file', 'line', 'description', 'trigger', 'expected_vs_actual', 'reachability', 'confidence'],
}
const HUNT_SCHEMA = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: FINDING },
    coverage: { type: 'string', description: 'what was reviewed thoroughly, what was only skimmed or skipped, and why' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['findings', 'coverage'],
}
const DEDUP_SCHEMA = {
  type: 'object',
  properties: {
    groups: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          indices: { type: 'array', items: { type: 'integer' } },
          canonical: { type: 'integer', description: 'the index whose write-up is clearest' },
          known: { type: 'string', description: 'KNOWN.md id if this is an already-known issue with no new trigger, else ""' },
        },
        required: ['indices', 'canonical', 'known'],
      },
    },
  },
  required: ['groups'],
}
const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'plausible', 'refuted', 'intended'] },
    severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    category: { type: 'string' },
    title: { type: 'string', description: 'corrected one-line title' },
    reasoning: { type: 'string', description: 'why this verdict; for refuted / intended, the specific reason with file:line' },
    repro_path: { type: 'string', description: 'path of the repro file, or ""' },
    repro_command: { type: 'string' },
    repro_output: { type: 'string', description: 'the decisive lines of output' },
    writeup_path: { type: 'string', description: 'findings/<ID>.md path if written, else ""' },
    fix_sketch: { type: 'string' },
  },
  required: ['verdict', 'severity', 'title', 'reasoning', 'repro_path', 'writeup_path'],
}

// ---------------------------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------------------------
const bullets = xs => xs.map(x => `- ${x}`).join('\n')
const ctxDocs = u => u.ctx.map(c => `bug-hunt/context/${c}.md`).join(', ')
const READ_FIRST = u => `Read first, in full:
1. bug-hunt/PRIMER.md - mission, severity, key contracts, checklist, how to run experiments, the bar for a finding, and the rules (never modify tracked files; only create new files at the paths it names).
2. ${ctxDocs(u)} - subsystem contracts, unsafe inventory, edge cases, INTENTIONAL behaviors (do not report those as bugs), hot spots.
3. bug-hunt/KNOWN.md - known issues; do not re-report them.`

const RETURN_NOTE = `Return findings (possibly empty), coverage (what you reviewed thoroughly, what you only skimmed or skipped and why - be honest, a silent gap reads as "covered"), and open_questions (what you could not resolve and someone should look at).`

function huntPrompt(u) {
  const scratch = `jix/tests/${huntTest(u)}.rs or ${RUN_DIR}/scratch/${slug(u.id)}_<n>.py`
  if (u.kind === 'module') {
    return `You are a bug hunter on the jix library, one of many hunters working in parallel in this checkout. Run id: ${RUN}.

${READ_FIRST(u)}

Your unit: ${u.id} - ${u.title}

Review in depth (read every line):
${bullets(u.files)}

Consult for contracts (read what you need, follow callers with grep):
${bullets(u.consult)}

Focus questions - starting points, not limits:
${bullets(u.focus)}
${FOCUS}
Method:
- Read the whole unit before forming opinions. For every unsafe block, state its precondition and find where it is established - follow the callers across files.
- Walk the PRIMER checklist for every function. Real bugs are usually an edge case meeting a fast path, or two paths that should agree but do not.
- Try to break each hypothesis yourself before reporting it: is the precondition established elsewhere? is the input reachable? is it documented as intended?
- Confirm promising hypotheses with a small experiment (PRIMER section 4; files: ${scratch}). A few minutes per experiment at most; a well-argued but unconfirmed finding is still worth reporting at medium or low confidence.
- Stay focused on your unit, but follow contracts wherever they lead; report bugs you find elsewhere along the way too.
- The context docs end their "Hot spots" with "Suspicions (unverified)" noted while mapping the code. Settle every one that falls inside your files: report it as a finding if it holds (with your own trigger and evidence), or name it in coverage with the reason it does not.

${RETURN_NOTE}`
  }
  if (u.kind === 'concept') {
    return `You are a bug hunter on the jix library, one of many hunters working in parallel in this checkout. Run id: ${RUN}. Other hunters review each module in depth; you are a CROSS-CUTTING hunter: trace one invariant across the whole codebase (jix/src and jix-py/src), through every place it matters, including the seams between modules that module hunters miss.

${READ_FIRST(u)}
Other context docs in bug-hunt/context/ are available if your trail leads into their subsystems.

Your concept: ${u.id} - ${u.title}

Starting points:
${bullets(u.start)}

Questions to answer for every relevant site:
${bullets(u.focus)}
${FOCUS}
Method:
- First enumerate the sites (grep), then examine each one. Keep a list; in coverage, say which sites you checked and which you did not get to.
- For each site, establish whether the invariant holds and why (cite the line that establishes it). Sites where it holds only by accident of current callers are latent hazards - report those at low severity.
- Confirm promising hypotheses with a small experiment (PRIMER section 4; files: ${scratch}). A well-argued but unconfirmed finding is still worth reporting at medium or low confidence.

${RETURN_NOTE}`
  }
  // dynamic
  return `You are a bug hunter on the jix library, one of many hunters working in parallel in this checkout. Run id: ${RUN}. Other hunters read code; you find bugs by TESTING: write and run differential tests, then explain every failure from the code.

${READ_FIRST(u)}

Your task: ${u.id} - ${u.title}
${bullets(u.task)}
${FOCUS}
Rules: create only new files (Rust: jix/tests/${huntTest(u)}.rs, run with \`cargo test --manifest-path jix/Cargo.toml --all-features --test <name>\`; Python: ${RUN_DIR}/scratch/, run with .venv/bin/python). Never modify tracked files. Do not use Miri for this task. Budget your time: aim for broad coverage over many op combinations rather than exhaustive enumeration of one.

In each finding's evidence, include the minimal failing case and the observed vs expected output. In coverage, list which ops / storages / materialization paths your tests exercised and which they did not.

${RETURN_NOTE}`
}

function verifyPrompt(f, prior) {
  const rt = reproTest(f)
  const second = prior
    ? `\nA first verifier concluded "${prior.verdict}": ${prior.reasoning}\nYou are an independent second verifier. Do not trust that conclusion - redo the analysis. If it was "plausible", focus on producing the decisive repro or the decisive refutation. If it was "refuted", check whether the refutation really closes every trigger the hunters described.\n`
    : ''
  return `You are an adversarial verifier on a bug hunt of the jix library. Run id: ${RUN}. Finding id: ${f.id}.

Read bug-hunt/PRIMER.md (sections 4 and 5 especially) and the context doc(s) for the area: ${f.ctx.map(c => `bug-hunt/context/${c}.md`).join(', ')}.

The finding, as reported by hunter(s) ${f.sources.join(', ')}:
${JSON.stringify(f.finding, null, 2)}
${f.others.length ? `\nThe same root cause was also reported as:\n${f.others.map(o => `- [${o.unit}] ${o.title} (${o.file}:${o.line}): ${o.trigger}`).join('\n')}\n` : ''}${second}
Hunters are often wrong: misread code, a precondition established somewhere they did not look, an unreachable input, intended behavior. Your job is to decide, with evidence.
1. Re-read the code at the cited locations and trace the claimed path end to end.
2. Try to REFUTE it: a check or invariant elsewhere that prevents the trigger (grep the callers), documented intended behavior (context doc "Intentional behaviors", comments, tests), or an unreachable input.
3. If it survives, build the smallest repro: jix/tests/${rt}.rs (public Rust API; \`cargo test --manifest-path jix/Cargo.toml --all-features --test ${rt} -- --nocapture\`) and/or ${RUN_DIR}/repros/${f.id}.py (run with .venv/bin/python). For a memory-safety claim that a debug build cannot show, you may run Miri ONCE on a tiny version: \`python jix/run_miri.py --all-features --test ${rt}\` (it is very slow - keep the test to a handful of elements). For a claim that depends on 32-bit usize or i686 alignment, run the same test natively as 32-bit: \`bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test ${rt}\`. Follow the PRIMER repro convention: the repro asserts the CORRECT behavior, so it fails while the bug exists.
4. Re-assess the severity with the PRIMER table; it may go up or down.
5. If the verdict is confirmed or plausible, write ${RUN_DIR}/findings/${f.id}.md with the PRIMER template.
Never modify tracked files and do not fix the bug.`
}

// ---------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------
const ALL = [
  ...MODULE_UNITS.map(u => ({ ...u, kind: 'module' })),
  ...CONCEPT_UNITS.map(u => ({ ...u, kind: 'concept' })),
  ...DYNAMIC_UNITS.map(u => ({ ...u, kind: 'dynamic' })),
]
const selected = ALL.filter(u => KINDS.includes(u.kind) && (!A.units || A.units.some(p => u.id.startsWith(p))) && !(A.skip || []).some(p => u.id.startsWith(p)))
if (!selected.length) {
  log(`No units match units=${JSON.stringify(A.units)} kinds=${JSON.stringify(KINDS)}. Unit ids: ${ALL.map(u => u.id).join(', ')}`)
  return { runId: RUN, error: 'no units selected' }
}
log(`Run ${RUN}: ${selected.length} hunters (${selected.filter(u => u.kind === 'module').length} module, ${selected.filter(u => u.kind === 'concept').length} concept, ${selected.filter(u => u.kind === 'dynamic').length} dynamic)`)

phase('Hunt')
const hunts = await parallel(selected.map(u => () =>
  agent(huntPrompt(u), { label: `hunt:${u.id}`, phase: 'Hunt', schema: HUNT_SCHEMA, effort: A.effort })
    .then(r => r && { unit: u, ...r })))
const done = hunts.filter(Boolean)
const failedUnits = selected.filter((u, i) => !hunts[i]).map(u => u.id)
if (failedUnits.length) log(`Hunters that returned nothing (rerun them with units=[...]): ${failedUnits.join(', ')}`)

const raw = done.flatMap(h => h.findings.map(f => ({ unit: h.unit.id, ctx: h.unit.ctx, finding: f })))
log(`${raw.length} raw findings from ${done.length} hunters`)

let verifyList = []
if (raw.length) {
  phase('Dedup')
  const listing = raw.map((r, i) => JSON.stringify({
    i, unit: r.unit, title: r.finding.title, loc: `${r.finding.file}:${r.finding.line}`, related: r.finding.related || [],
    category: r.finding.category, severity: r.finding.severity, description: r.finding.description, trigger: r.finding.trigger,
  })).join('\n')
  const dd = await agent(`You deduplicate bug reports produced by parallel hunters on the jix library. Read bug-hunt/KNOWN.md first.

Below is one JSON finding per line (field "i" is its index). Group findings that describe the SAME root defect - the same faulty code, even if reported through different triggers, symptoms or locations. Different defects in the same function are different groups. When unsure whether two share a root cause, open the code and check. Set "known" to the KNOWN.md id when a group is an already-known issue and adds no new trigger, else "". Every index must appear in exactly one group.

${listing}`, { label: 'dedup', phase: 'Dedup', schema: DEDUP_SCHEMA })

  const groups = (dd && dd.groups) || []
  const seen = new Set()
  const clean = []
  for (const g of groups) {
    const idx = [...new Set(g.indices)].filter(i => i >= 0 && i < raw.length && !seen.has(i))
    if (!idx.length) continue
    idx.forEach(i => seen.add(i))
    clean.push({ indices: idx, canonical: idx.includes(g.canonical) ? g.canonical : idx[0], known: g.known || '' })
  }
  raw.forEach((_, i) => { if (!seen.has(i)) clean.push({ indices: [i], canonical: i, known: '' }) })
  const knownDropped = clean.filter(g => g.known)
  if (knownDropped.length) log(`Dropped ${knownDropped.length} groups matching KNOWN.md: ${knownDropped.map(g => `${g.known} (${raw[g.canonical].finding.title})`).join('; ')}`)

  const canon = clean.filter(g => !g.known).map(g => {
    const best = raw[g.canonical]
    const members = g.indices.map(i => raw[i])
    // A group is as severe / confident as its strongest member.
    const severity = members.map(m => m.finding.severity).sort((a, b) => SEV_RANK[a] - SEV_RANK[b])[0]
    const confidence = members.map(m => m.finding.confidence).sort((a, b) => CONF_RANK[b] - CONF_RANK[a])[0]
    return {
      finding: { ...best.finding, severity, confidence },
      sources: [...new Set(members.map(m => m.unit))],
      ctx: [...new Set(members.flatMap(m => m.ctx))],
      others: g.indices.filter(i => i !== g.canonical).map(i => ({ unit: raw[i].unit, ...raw[i].finding })),
    }
  })
  canon.sort((a, b) => SEV_RANK[a.finding.severity] - SEV_RANK[b.finding.severity] || CONF_RANK[b.finding.confidence] - CONF_RANK[a.finding.confidence])
  canon.forEach((c, i) => { c.id = `F${String(i + 1).padStart(3, '0')}` })
  verifyList = canon
  log(`${canon.length} distinct findings after dedup`)
}

let results = []
const toVerify = A.verify === false ? [] : verifyList.filter(f => CONF_RANK[f.finding.confidence] >= MIN_CONF)
const skipped = verifyList.filter(f => !toVerify.includes(f))
if (skipped.length) log(`Not verified (verify=false or below minConfidence): ${skipped.length} findings; they are listed unverified in the report`)

if (toVerify.length) {
  phase('Verify')
  results = await pipeline(
    toVerify,
    f => agent(verifyPrompt(f), { label: `verify:${f.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: A.effort }),
    (v1, f) => {
      if (!v1) return { f, v: null }
      const serious = SEV_RANK[f.finding.severity] <= SEV_RANK.high
      const needSecond = serious && (v1.verdict === 'plausible' || (v1.verdict === 'refuted' && f.finding.confidence === 'high'))
      if (!needSecond) return { f, v: v1 }
      return agent(verifyPrompt(f, v1), { label: `verify2:${f.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: A.effort })
        .then(v2 => ({ f, v: v2 || v1, first: v1 }))
    },
  )
  results = results.filter(Boolean)
}

phase('Report')
const summary = results.map(({ f, v, first }) => ({
  id: f.id,
  verdict: v ? v.verdict : 'verifier-failed',
  severity: v ? v.severity : f.finding.severity,
  category: (v && v.category) || f.finding.category,
  title: v ? v.title : f.finding.title,
  location: `${f.finding.file}:${f.finding.line}`,
  sources: f.sources,
  reasoning: v ? v.reasoning : '',
  first_opinion: first ? `${first.verdict}: ${first.reasoning}` : '',
  repro: v ? v.repro_path : '',
  writeup: v ? v.writeup_path : '',
}))
const unverified = skipped.map(f => ({ id: f.id, severity: f.finding.severity, confidence: f.finding.confidence, title: f.finding.title, location: `${f.finding.file}:${f.finding.line}`, sources: f.sources, description: f.finding.description, trigger: f.finding.trigger }))
const coverage = done.map(h => ({ unit: h.unit.id, coverage: h.coverage, open_questions: h.open_questions || [], findings: h.findings.length }))

// Subagents may not write report files, so the report agent RETURNS the markdown; the session that
// ran the workflow writes it to `report` from this workflow's return value (`reportMarkdown`).
const report = await agent(`Write the final report of jix bug-hunt run "${RUN}" for the maintainers, as markdown (ASCII only - the repo CI rejects non-ASCII bytes). Do NOT write it to a file: your final reply must be exactly the report markdown and nothing else - the caller saves it as ${RUN_DIR}/REPORT.md. Read findings/*.md and the repros under ${RUN_DIR}/ as needed to summarize accurately.

Sections:
1. Summary: counts by verdict and by severity; hunters run; hunters that failed: ${JSON.stringify(failedUnits)}.
2. Findings: a table of confirmed + plausible findings sorted by severity, then confirmed before plausible (ID, severity, verdict, category, title, location, link to findings/<ID>.md when it exists). Then, per finding, a 2-3 line summary. Put questionable-semantics findings in their own sub-section after the bugs.
3. Unverified findings (if any): one line each with severity / confidence / location.
4. Refuted and intended: one line each with the reason - this tunes the next run, keep it.
5. Coverage: one line per hunter from its coverage note; then a list of everything hunters said they skipped or only skimmed.
6. Open questions from hunters, grouped by theme.

Before writing the report, clean up this run's test files in jix/tests/ so the crate's test suite stays clean: mkdir -p ${RUN_DIR}/repros, MOVE every jix/tests/repro_${RS}__*.rs into ${RUN_DIR}/repros/ (keep the file names), and delete jix/tests/hunt_${RS}__*.rs (hunter scratch). Touch no other file in jix/tests/ - other runs may be using them. Also free the disk these tests used: delete jix/target/debug/deps/{repro,hunt}_${RS}__* and target-i686/i686-unknown-linux-musl/debug/deps/{repro,hunt}_${RS}__* (test binaries), the matching .fingerprint/{repro,hunt}_${RS}__* dirs next to deps/, and any scratch/target-* dirs under ${RUN_DIR}/scratch/. Python repros are already in ${RUN_DIR}/repros/. In the report, say how to re-run a repro: copy it back to jix/tests/ and run \`cargo test --manifest-path jix/Cargo.toml --all-features --test <name>\`. Do not modify tracked files.

Verified findings:
${JSON.stringify(summary, null, 1)}

Unverified findings:
${JSON.stringify(unverified, null, 1)}

Hunter coverage:
${JSON.stringify(coverage, null, 1)}`, { label: 'report', phase: 'Report' })

const count = k => summary.filter(s => s.verdict === k).length
return {
  runId: RUN,
  report: `${RUN_DIR}/REPORT.md`,
  note: 'Write reportMarkdown to the report path (subagents cannot write it).',
  hunters: done.length,
  failedHunters: failedUnits,
  rawFindings: raw.length,
  distinct: verifyList.length,
  confirmed: count('confirmed'),
  plausible: count('plausible'),
  refuted: count('refuted'),
  intended: count('intended'),
  unverified: unverified.length,
  findings: summary.filter(s => s.verdict === 'confirmed' || s.verdict === 'plausible').map(s => `${s.id} [${s.severity}/${s.verdict}] ${s.title} (${s.location})`),
  reportMarkdown: typeof report === 'string' ? report.replace(/^```(?:markdown)?\n([\s\S]*)\n```\s*$/, '$1') : '',
}

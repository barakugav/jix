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
    "id": "types-dtype-layout",
    "ctx": [
      "types"
    ],
    "title": "Dtype representation, struct layout validation, set_shape, equality",
    "files": [
      "jix/src/dtype.rs:1-918"
    ],
    "consult": [
      "jix/src/util/mod.rs",
      "jix/src/util/arrayvec.rs",
      "jix/src/archive/schema/conversions.rs",
      "jix-py/src/dtype.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix/src/error.rs"
    ],
    "focus": [
      "Can any input to from_fields/new_struct (including a crafted archive via conversions.rs:84-114 or a numpy dtype) yield a struct dtype whose field extends past the base itemsize? The `expected_offset += itemsize` and `ceil_to_multiple` steps in dtype.rs:476-506 and 653-717 are unchecked u16 arithmetic, so they wrap in release builds.",
      "Does every constructor (new_struct with an inner shape, set_shape on struct dtypes, from_proto calling set_shape after new_struct) keep itemsize == base * prod(shape) and the field-in-bounds invariant? ops/sub_dtype.rs:124-160 relies on it without re-checking.",
      "Can two dtypes compare equal under PartialEq (dtype.rs:914-941) but differ in layout or in a property some consumer reads (is_aligned, field order, nested inner shape, zero-sized fields at the same offset)?",
      "from_fields vs new_struct: duplicate names (only from_fields rejects them), caller order vs sorted order, ambiguous packed/aligned layouts, alignment > itemsize, an empty struct with an inner shape. Does either path accept something that later gives wrong results?",
      "set_shape (dtype.rs:840-874): can a reachable dtype hit the unchecked current-shape product (857), the assert at 867, or a zero-element shape and panic?",
      "Alignment::new / as_usize (dtype.rs:65-96): is there any path where the transmute or the assert_unchecked sees a value that is not a valid power of two?"
    ]
  },
  {
    "id": "types-dtyped-derive",
    "ctx": [
      "types"
    ],
    "title": "Dtyped trait contract, scalar and array impls, derive macro, ElementType",
    "files": [
      "jix/src/dtype.rs:919-1219",
      "jix-macros/src/lib.rs:1-315",
      "jix/src/element_type.rs:1-90"
    ],
    "consult": [
      "jix-macros/tests/derive_external.rs",
      "jix/src/ops/into_type.rs",
      "jix/src/ops/transmute.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/storage/plain.rs",
      "jix/src/archive/schema/conversions.rs"
    ],
    "focus": [
      "Can safe code get a Dtyped impl from #[derive(Dtyped)] whose DTYPE does not match the real Rust layout? Candidates: bare repr(packed) field order, no per-field offset_of! check, call-site hygiene of unqualified size_of/align_of/assert!/std:: and block-local BASE_OFFSET/FIELDS, transparent generic wrappers.",
      "Do the built-in scalar impls give itemsize == size_of and alignment == align_of on every shipped target? On i686 Linux, u64/i64/f64/Complex<f64> have align_of 4 but the hardcoded dtype alignment is 8. What breaks: derive vs new_struct round-trip, the nd_copy debug_assert, the numpy layout check?",
      "bool validity: which safe paths can put a byte other than 0/1 into a Ty<bool> buffer or a struct with a bool field (archive decode + into_typed, numpy import, SubDtype, Plain)? ops/transmute.rs treats the same reinterpretation as unsafe.",
      "The [T; N] impl (dtype.rs:1186-1198): shape-product overflow for zero-sized T, more than 4 inner dims, alignment of [T; 0]. Do the resulting dtypes survive an archive write then read?",
      "ElementType is a public, safe, unsealed trait. Which unsafe code trusts ET::DTYPE, from_dtype or dtype() (into_type.rs:87,102 assert_unchecked_eq, sub_dtype.rs copies, Array::dtype at array.rs:560), and can a user impl make safe code UB?",
      "Does the derive always pass an is_aligned flag that matches what from_fields/new_struct would compute? Does any consumer depend on it (jix-py dtype_to_numpy align flag, archive)?"
    ]
  },
  {
    "id": "types-dims-errors",
    "ctx": [
      "types"
    ],
    "title": "Dimension types, IntoDimension conversions, validation helpers and arithmetic error modes",
    "files": [
      "jix/src/dimension.rs:1-546",
      "jix/src/error.rs:1-441",
      "jix/src/lib.rs:1-429"
    ],
    "consult": [
      "jix/src/util/arrayvec.rs",
      "jix/src/util/arr_ext.rs",
      "jix/src/array.rs",
      "jix/src/storage/buf.rs",
      "jix/src/codec/mod.rs",
      "jix-py/src/error.rs"
    ],
    "focus": [
      "check_get_buffer_size (error.rs:240-267) is the only safety justification for StridedBuf::from_slice_mut in the safe to_ndarray_slice (array.rs:840-853). On 32-bit targets, can the unchecked u64 product, the `as usize` truncation, or the `nitems * itemsize` overflow let a too-small buffer pass?",
      "check_get_range and check_shape_overflow bound sizes in u64, not usize. Is that enough for callers that then do `as usize` offset arithmetic on x86/armv7?",
      "DimDyn's assert_unchecked(len <= NDIM_MAX) at dimension.rs:173,179: does every DimArray construction path guarantee it, given that ArrayVec::set_len only has a debug_assert?",
      "IntoDimension conversions: ndarray Dim/IxDyn to Dim/DimDyn and back (try_into().unwrap() at dimension.rs:215,309 panics on 32-bit for dims >= 2^32), Dim<0>, Dim<7>/Dim<8> mapping to IxDyn, and the ndim checks in from_fn/vec.",
      "ArithmeticErrorModes / ArithmeticOpStatus (error.rs:269-419): are the modes applied consistently (which kind is raised first, warning dedup by message, flags set only when cond holds)? The defaults deviate from NumPy on purpose.",
      "dim_arr / try_dim_arr / assert_dim with ndim > NDIM_MAX or a mismatched static ndim: can a user-reachable path hit a panic where an Err is expected?"
    ]
  },
  {
    "id": "storage-compact-read-path",
    "ctx": [
      "storage-compact",
      "codec-archive",
      "util-nd"
    ],
    "title": "Compact nd read path (read_data / read_data_slow) and BlockTable decode/build/mmap backends",
    "files": [
      "jix/src/storage/compact.rs:1-508",
      "jix/src/storage/block.rs:1-453"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/iter/block.rs",
      "jix/src/util/iter/nd.rs",
      "jix/src/util/iter/strides.rs",
      "jix/src/util/mod.rs",
      "jix/src/codec/mod.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/archive/array.rs",
      "jix/src/archive/block.rs",
      "jix/src/archive/common.rs",
      "jix/src/array.rs",
      "jix/src/element_type.rs"
    ],
    "focus": [
      "Is the direct decode-into-out condition (compact.rs:429-437, 464) sufficient for every block/out shape: size-1 dims before and after lead_dim, all-ones block shape, ndim 5-8, partial edge blocks, out sub-tiles of a larger buffer (array.rs:1061)?",
      "Can out_start/active_start exceed the slices passed to get_unchecked(_mut) (compact.rs:411, 477-478) or the span NdCopier touches, for any valid index and any StridedBuf honoring its construction contract?",
      "Do fast path 1 (compact.rs:320-328), the single-block slow path (401-424) and the general path (454-491) agree on edge blocks, unaligned starts, size-1 dims, ndim 0 and ndim 5-8?",
      "Is prod(spec.block_shape) == blocks.block_size (and the same per-dim shape) guaranteed for every constructor: compact_with double tune, archive read_from with user ArrayParams, as_compact?",
      "What happens in release when a block decodes short or empty (block.rs:191-193 only debug_asserts; zstd returns Ok(0) for empty input)? Is uninitialized pool memory exposed, and is bool UB reachable?",
      "MmapData::as_ref (block.rs:442-446) soundness for truncated files, oversized caller len, and wrapping section bounds (archive/block.rs:469-500, archive/common.rs:265-272).",
      "32-bit truncation/overflow at compact.rs:281-282, 395-396, buf.rs:312, block.rs:180, 189, 306: can pull mode allocate fewer bytes than are written?",
      "Can any path build a BlockTable whose element_type dtype differs from decoder_config.dtype (assert_unchecked_eq at block.rs:216)?"
    ]
  },
  {
    "id": "storage-compact-params",
    "ctx": [
      "storage-compact"
    ],
    "title": "ArrayParams::tune, block-shape auto selection, ArraySpec read hints and scale_read_shape",
    "files": [
      "jix/src/storage/params.rs:1-983",
      "jix/src/util/mod.rs:411-532"
    ],
    "consult": [
      "jix/src/array.rs",
      "jix/src/ops/reduction.rs",
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/reshape.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix/src/ops/shape_ops/stack.rs",
      "jix/src/util/bitmap.rs",
      "jix/src/util/cpu_cache.rs",
      "jix/src/archive/array.rs",
      "jix/src/storage/compact.rs"
    ],
    "focus": [
      "Is tune idempotent (tune(tune(p)) == tune(p)) for every combination of block_shape, block_shape_fixed_dims, block_size and read_size? compact_with tunes at array.rs:1289 and again via into_spec (compact.rs:200).",
      "Which inputs make tune/into_spec panic instead of returning Err: params.rs:121, 140, 374, 400, 427 (division by zero with a 0 in a fixed dim), 448, 454, the into_spec asserts at 478-486, and block volumes > u32::MAX that later unwrap at array.rs:1208/1293?",
      "Prove 1 <= block_len <= min(dim_len, 2^30) in block_len_heuristic (params.rs:417-455) for any base length, including base > max_block_len.",
      "scale_read_shape (util/mod.rs:468-526): tile invariants (>= 1, <= max_shape or snapped to array_shape), u64 overflow in volume products, max_shape[d] == 0, an order that is not a permutation; do callers clip snapped tiles correctly?",
      "read_shape_heuristic / scale_order_from_weights / read_layout_order_insert_dont_care_dim: ndim consistency of block_shape, weights and layout order across all spec constructors.",
      "combine_select_hints / combine_elementwise_hints / max_cost_layout_order / combine_block_layout with NaN or inf costs, zero cost, or an empty input list (partial_cmp().unwrap() at params.rs:862, inputs[0] indexing).",
      "override_from_storage (params.rs:255-268) silently replaces user-set fixed dims when block_shape is unset; can inherited fixed dims have b > shape after shape-changing views, making compact() fail?"
    ]
  },
  {
    "id": "util-nd-copier",
    "ctx": [
      "util-nd",
      "util-memory"
    ],
    "title": "NdCopier strided n-d copy kernels, fill_contiguous, PtrNoalias and span helpers",
    "files": [
      "jix/src/util/nd_copy.rs:1-447",
      "jix/src/util/mod.rs:107-190",
      "jix/src/util/mod.rs:658-815"
    ],
    "consult": [
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/dtype.rs",
      "jix/src/storage/buf.rs",
      "jix/src/storage/compact.rs",
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/ops/shape_ops/tile.rs",
      "jix/src/ops/shape_ops/repeat.rs",
      "jix/src/ops/sub_dtype.rs"
    ],
    "focus": [
      "Does every route into inner_loop / inner_loop_broadcast satisfy the assert_unchecked(CONTIG == (stride == size_of::<T>())) at nd_copy.rs:223-224 and 281, including the 1-d fast path (94-121), the single-element sentinel, and src==0 && dst==0 strides?",
      "Is the ALIGNED kernel only chosen when both base pointers and all post-coalesce strides are multiples of align_of::<T>() (92, 108-110, 130-131)? What happens on 32-bit x86, where dtype alignment 8 maps to u64 with align_of 4 (debug_assert at 90)?",
      "Do inner_loop_contiguous_const_len reads of [T; LEN] always get len == LEN (only debug_asserted at 264)?",
      "inner_loop_untyped_broadcast doubling (341-348): bounds and non-overlap for all len/itemsize incl. itemsize 0, 3, 12, 24.",
      "copy_untyped (351-410): does pick_inner_loop classify consistently between the 1-d path and the iterator path; does it ever call the broadcast loop when dst stride != itemsize?",
      "fill_contiguous (413-443) has no Safety doc: does its only caller (elementwise_pipeline.rs:544) give a src valid for itemsize bytes and a dst of n*itemsize, non-overlapping, even when n == 0?",
      "What if copy() is passed a dtype whose (itemsize, alignment) differs from the one given to new() (scalar_fn uses size_of::<T>(), copy_untyped uses the copy() dtype)?",
      "PtrNoalias/PtrMutNoalias bytes_offset uses get_unchecked(offset..): can any NdIterUnordered offset exceed the wrapped slice length given the documented span contract?"
    ]
  },
  {
    "id": "util-nd-iter-unordered",
    "ctx": [
      "util-nd"
    ],
    "title": "NdIterUnordered / NdIterUnorderedDyn: dim drop, sort, coalesce, flags and walk",
    "files": [
      "jix/src/util/nd_iter_unordered.rs:1-326",
      "jix/src/util/nd_iter_unordered_dyn.rs:1-241"
    ],
    "consult": [
      "jix/src/util/iter/nd.rs",
      "jix/src/util/iter/strides.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/ops/where_op.rs",
      "jix/src/ops/reduction.rs"
    ],
    "focus": [
      "Coalescing (129-150 / dyn 91-107): can 0 strides, equal strides, or an excluded (affects_dim_order=false) operand make two dims merge when (i,j) -> merged index is not a bijection for some operand? Is the group stride always the innermost member's stride and the length the product?",
      "dims_sort_by (306-325): with the None-is-transparent comparator, is it always a permutation of the input (no lost or duplicated dims), and does copy_within shift correctly?",
      "Flags on the special shapes: 0-d / all-size-1 (84-94, faked strides = size), single remaining dim (96-101), empty sentinel (172-186) - are is_contiguous / is_strides_aligned / inner_strides consistent with what foreach_inner_1d passes?",
      "foreach_inner_1d calls the closure once with len 0 for an empty region: do all callers (reduction.rs:1159-1176, where_op.rs:234-245) tolerate it?",
      "Static vs dyn parity, and the 2-d manual loop (270-284) vs the NdIter outer walk for ndim >= 3 (285-297) incl. ndim 8: same offsets?",
      "Which callers depend on visit order (argmax/argmin 'tie keeps the earlier one' reduction.rs:1638, float accumulation) or on in-place aliasing (reduction.rs:1153 'may alias'), and is that safe given only per-dim increasing order is guaranteed?",
      "Overflow: s[d] * shape[d] (137) and group_len *= shape[d] (140) are unchecked - reachable with valid strides?"
    ]
  },
  {
    "id": "util-nd-iter-ordered",
    "ctx": [
      "util-nd"
    ],
    "title": "NdIter ordered walker, extension tuples, stride-offset and block-border extensions",
    "files": [
      "jix/src/util/iter/mod.rs:1-5",
      "jix/src/util/iter/nd.rs:1-520",
      "jix/src/util/iter/block.rs:1-142",
      "jix/src/util/iter/strides.rs:1-174",
      "jix/src/util/mod.rs:630-639"
    ],
    "consult": [
      "jix/src/storage/compact.rs",
      "jix/src/array.rs",
      "jix/src/ops/reduction.rs",
      "jix/src/ops/shape_ops/flip.rs",
      "jix/src/dimension.rs"
    ],
    "focus": [
      "advance/advance_dim (nd.rs:74-121): correct carry, wrap-to-begin and on_increase/on_decrease (dim, after, diff) for every ndim 0..=8, Dim<N> and DimDyn?",
      "Emptiness via shape.product::<u64>() (nd.rs:61) and remaining() (136-150): overflow in debug, wrap-to-0 in release - reachable from compact_fn / broadcast arrays with huge dims and one zero dim?",
      "NdIterExtBlockOffsetSize::new (block.rs:47-91): verify low/high border (offset, size) for begin==end, same-block ranges, end % b == 0, begin % b == 0, b larger than range; any underflow?",
      "Do all callers (compact.rs:446, array.rs:1040, 1353, 1383, reduction.rs:453, 545) bound NdIter by [begin/b, calc_block_end) as the extension requires, and never with block size 0?",
      "Stride-offset extensions (strides.rs) use Idx::from_u64 (truncating as cast): any caller where diff or offsets exceed usize on 32-bit (e.g. array.rs:435-460 FnStorage sizing its buffer with `as usize`)?",
      "nd_iter_ext_logical_global_index initial offset and the 3-extension cap (nd.rs:370-379): any builder chain that hits unimplemented!()?"
    ]
  },
  {
    "id": "util-nd-arr-sequence",
    "ctx": [
      "util-nd",
      "pipeline"
    ],
    "title": "ArraySequence adapters (arrays, Vec, slices, tuples) and fixed-array map helpers",
    "files": [
      "jix/src/util/arr_sequence.rs:1-737",
      "jix/src/util/arr_ext.rs:1-258"
    ],
    "consult": [
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/ops/map.rs",
      "jix/src/ops/shape_ops/stack.rs",
      "jix/src/ops/shape_ops/concatenate.rs",
      "jix/src/storage/params.rs"
    ],
    "focus": [
      "Does every ElementwisePipelineTuple::N_OPERANDS (arr_sequence.rs:225, 322, 426, 550, 688) equal operands().count() exactly? to_buf_impl only debug_asserts extra operands (elementwise_pipeline.rs:137-139), so an under-count leaves operands without a cursor in release.",
      "Vec/slice read_bulk_as_iter (435-469, 559-593): UnsafeCell scratch + Vec::set_len over uninitialized items - is every one of narrays*M slots written before the slices are handed out, and can the scratch be re-borrowed while a previous iterator is alive (map.rs:345-356, nested map_multiple)?",
      "Empty sequences: map_multiple (map.rs:238-247) does not reject narrays()==0 while stack/concatenate do - what panics and where?",
      "arr_ext drop/panic safety: array_try_from_fn_inline InitGuard, array_try_map_inline and array2_try_map_inline DropGuards - any double drop or leak-to-UB when f returns Err or panics mid-way, including when T has no Drop but U does?",
      "Index validity: read_data/shape/dtype/spec with arr >= narrays() for each impl (array index panic vs out_of_bounds_array_index).",
      "Tuple impl read_as_elementwise_pipeline: all arrays read with the same index - is shape equality always validated by the op that builds the sequence?"
    ]
  },
  {
    "id": "reduction-driver",
    "ctx": [
      "reduction",
      "util-nd"
    ],
    "title": "ReductionOp construction, bulk/tile streaming loop, state-buffer choice and finalize pass",
    "files": [
      "jix/src/ops/reduction.rs:1-630",
      "jix/src/ops/reduction.rs:1132-1274"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/iter/nd.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/error.rs"
    ],
    "focus": [
      "Do bulks and tiles cover every output cell's whole reduced stream exactly once, for index ranges that start mid-block and for tile_shape snapped past the requested extent (scale_read_shape), so no state is left uninitialized before finalize?",
      "state_in_out_buf (485-491): when the output bytes double as the state buffer (Mean<f64>, Mean<i8..u32> with i64/u64 state into f64 output, Sum/Product/Max/Min/All/Any/Reduce), is the in-place finalize (1145-1170) correct, and does the NEEDS_FINALIZE=false early return ever skip a needed write?",
      "state_offset / get_unchecked_mut at 568-582: is state_offset always <= state_buf.len() for push-mode out buffers with permuted or gapped strides and for the tmp buffer, whose size is out_nitems * kernel_state_sizeof (501, unchecked multiply)?",
      "Empty reductions: are 0 bulks guaranteed when a reduced extent is 0 (calc_block_end(0,0,1)), and does finalize_states' empty branch write every cell for arbitrary out strides? Check the out_nitems == 0 early return too.",
      "u64 -> usize casts (335, 395, 477, 501, 570, 575-580) on 32-bit targets: can reduced extents > 2^32 (lazy broadcast or compressible Compact) truncate stream positions or overflow logical strides?",
      "ReductionOp::new spec filtering (154-198): are block_shape, block_shape_fixed_dims, scale weights and read_layout_order (dim_orig2new) consistent for unsorted dims, all dims reduced (0-d output) and 0-d input?"
    ]
  },
  {
    "id": "reduction-fold-loops",
    "ctx": [
      "reduction",
      "util-nd"
    ],
    "title": "Kernel trait contract and the fold inner loops (one-cell, pairwise tree, multi-cell, SIMD vs strided)",
    "files": [
      "jix/src/ops/reduction.rs:18-118",
      "jix/src/ops/reduction.rs:632-1130"
    ],
    "consult": [
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/util/mod.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/storage/buf.rs"
    ],
    "focus": [
      "Init vs merge is keyed only on base_item_idx == 0 (640, 803). Can a cell's first visit have position > 0 (assume_init_read of uninitialized pool memory) or a later visit have position 0 (state overwritten)? Try length-1 reduced dims, broadcast reduced dims, reduced dims whose memory order is the reverse of their logical order, and the single-element special case in NdIterUnordered.",
      "One-cell vs multi-cell is chosen by state_inner_stride == 0 (712). Can coalescing mix a reduced and a kept dim, or can a kept len>1 dim get state stride 0? The debug_assert at 718-721 vanishes in release.",
      "Contiguous fast-path flags (715, 724) vs the unchecked assumptions (925, 1090-1091) and aligned reads (978, 988-993, 1099, 1104) on REQUIRE_ALIGNED targets (armv7) and with unaligned or 0-stride inputs.",
      "ArgMax/ArgMin positions: tile_base_reduced_idx + per-run base (735), lane indices and idx_step (929-955), and tail indices (865). Is the first max/NaN reported across lanes, pairwise halves, tails and tiles?",
      "fold_pairwise split (902-908) and leaf loop (946-957): is every item folded exactly once for every len and LANES (1..512), including len < LANES and len not a multiple of LANES?",
      "merge_states lane collapse (1007-1026) with MaybeUninit: is each slot read exactly once? Is the non-panic-safety (1012) harmless for Reduce with a panicking user closure?"
    ]
  },
  {
    "id": "reduction-scalar-traits",
    "ctx": [
      "reduction",
      "elementwise-ops"
    ],
    "title": "Scalar kernel traits: Sum, Product, ArgMax/ArgMin, Mean, Variance (Welford/Chan)",
    "files": [
      "jix/src/ops/reduction.rs:1393-2064"
    ],
    "consult": [
      "jix/src/ops/cast.rs",
      "jix/src/ops/op2.rs",
      "jix/src/ops/cmp.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/scalar/mod.rs"
    ],
    "focus": [
      "Variance (rewritten in adef94b; only concrete tests): does welford_update_bulk (1933-1957) match scalar update (1836-1844) for real and complex (re/im passes sharing m2, 1916-1930)? Check merge_states with count 0 (1870-1875) and that finalize uses nitems, not state.count.",
      "Seeds: init op x vs x. Sum::init = -0.0 (1472) and Product::init = 1 (1558). Is the result exact for every dtype, including complex with an infinite or -0.0 component, and correct for empty reductions?",
      "Complex mean finalize (1732-1734) divides Complex by Complex(n, 0). Check inf/NaN/overflow behavior against dividing by a real count.",
      "update_bulk_simd (1385-1415): do the SIMD cast+add/mul paths give the same wrapping integer results as the scalar paths for every dtype and SIMD level?",
      "ArgMax/ArgMin update/merge (1632-1655): NaN-first and lower-index tie-break when states hold NaN, +-0.0 or bool; is update's assumption that items arrive in stream order guaranteed?",
      "Mean accumulator choices (1740-1756): overflow of i64/u64 sums for i8..u32 inputs, precision of the f64 sum for i64/u64, f16 double rounding through f32."
    ]
  },
  {
    "id": "reduction-op-kernels",
    "ctx": [
      "reduction",
      "py-ops"
    ],
    "title": "Per-op kernel impls and flags, define_reduction_op macros, Reduce, Array methods",
    "files": [
      "jix/src/ops/reduction.rs:1276-1391",
      "jix/src/ops/reduction.rs:2066-3443"
    ],
    "consult": [
      "jix/src/ops/cmp.rs",
      "jix/src/ops/op1.rs",
      "jix/src/ops/common.rs",
      "jix/src/storage/mod.rs",
      "jix-py/src/ops/reduction.rs",
      "jix-py/src/util.rs"
    ],
    "focus": [
      "Every NEEDS_FINALIZE=false kernel (Max, Min, Sum, Product, All, Any, Reduce) must have Output layout == State and an identity finalize. Does any one break this unsafe promise?",
      "SUPPORTS_EMPTY / init_state(None) consistency: can a kernel with SUPPORTS_EMPTY=false ever reach init_state(None).unwrap() or Mean::finalize(..., 0).unwrap() (2615)?",
      "NEEDS_INDICES / ONE_CELL_LANES / PREFER_TREE_MERGE flags per kernel. ArgMax/ArgMin and Reduce keep the default NEEDS_INDICES=true. Do these flags match what each kernel's init/update actually read?",
      "StandardDeviation finalize (2816-2822): can the variance be negative or NaN in a way that trips debug_assert!(status.is_ok()) in debug builds?",
      "Reduce (3030-3091): soundness with arbitrary user closures (panics, non-commutative), and correct propagation of the Err/panic for empty dims.",
      "Python wrappers (jix-py/src/ops/reduction.rs): negative/duplicate/unsorted dims and keepdim via InsertDim positions; does dim=() or dim=None on 0-d arrays behave consistently with the Rust op?"
    ]
  },
  {
    "id": "util-memory-alloc-pool",
    "ctx": [
      "util-memory"
    ],
    "title": "Aligned byte allocation (AlignedBytes/RawAlignedBytes) and the BufferPool/PoolBuf scratch allocator",
    "files": [
      "jix/src/util/aligned_vec/raw.rs:1-258",
      "jix/src/util/aligned_vec/mod.rs:1-451",
      "jix/src/util/buf_pool.rs:1-128"
    ],
    "consult": [
      "jix/src/codec/mod.rs",
      "jix/src/storage/buf.rs",
      "jix/src/array.rs",
      "jix/src/storage/compact.rs",
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/codec/filter/bit_shuffle.rs",
      "jix/src/ops/reduction.rs",
      "jix/src/dtype.rs",
      "jix/src/util/cpu_cache.rs"
    ],
    "focus": [
      "Every alloc/realloc/dealloc: is size > 0, is align a non-zero power of two, and is the same Layout used on alloc and dealloc on every path (grow from cap 0, shrink, clone, pool reuse)?",
      "RuntimeAlign::new (aligned_vec/mod.rs:27-36) accepts align 0: can any path reach AlignedBytes/RawAlignedBytes with align 0 (NonNull::new_unchecked(null) at raw.rs:45)?",
      "RawAlignedBytes::shrink_to (raw.rs:144-159): realloc to size 0, unchecked null result, and the leak when capacity becomes 0.",
      "PoolBuf::as_slice/as_mut_slice cover the whole capacity, which can exceed the requested size (since commit 763a701). Does any consumer, including pool-backed StridedBuf::data()/data_mut(), treat the full length as the logical size?",
      "Is every byte of a PoolBuf or an AlignedBytes::set_len region written before it is read (short zstd decode, filter tails, partial blocks, out buffers reused as reduction state)?",
      "BufferPool UnsafeCell use: is the raw *mut Vec from get_pool still valid after align_other.insert? Is reentrancy through PoolBuf Drop possible? Does ReadContext really stay !Sync?",
      "Can the size passed to BufferPool::get wrap (product * itemsize, especially on 32-bit) while the strides still describe the real shape?"
    ]
  },
  {
    "id": "util-memory-arrayvec-bitmap-cache",
    "ctx": [
      "util-memory"
    ],
    "title": "ArrayVec/DimArray, DimBitmap, branch hints and CPU cache detection",
    "files": [
      "jix/src/util/arrayvec.rs:1-616",
      "jix/src/util/bitmap.rs:1-132",
      "jix/src/util/hint.rs:1-23",
      "jix/src/util/cpu_cache.rs:1-327"
    ],
    "consult": [
      "jix/src/dimension.rs",
      "jix/src/dtype.rs",
      "jix/src/storage/params.rs",
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/storage/info.rs",
      "jix/src/storage/plain.rs",
      "jix/src/codec/mod.rs"
    ],
    "focus": [
      "len() calls assert_unchecked(len <= CAP) (arrayvec.rs:57). Can any path store len > CAP or count an uninitialized slot (set_len's u8 truncation, the extend guard on panic, the Clone guard, next/next_back interplay in IntoIter)?",
      "insert shift arithmetic at index == len and when full; insert_first_const in const eval (dtype.rs:1194).",
      "ZST handling in extend_from_iter (an element that is not written is still dropped, arrayvec.rs:416), Clone and IntoIter.",
      "extend_from_slice silently truncates in release (arrayvec.rs:431-445). Do all callers pre-check the length?",
      "DimBitmap: insert at pos == len and growing len from 7 to 8, bits above len staying zero (derived Eq/Hash depend on it), all() for len 0 and len 8.",
      "Cache detection: zero, garbage or duplicate sysfs values, 32-bit overflow in parse_size. What does params.rs:328-381 do when l1_data is 0 or smaller than itemsize?",
      "Does anything assume CACHE_LINE_SIZE >= a dtype alignment, or assume CACHE_LINE_SIZE is 64/128 (it is 32 on armv7)?"
    ]
  },
  {
    "id": "util-memory-helpers",
    "ctx": [
      "util-memory",
      "pipeline"
    ],
    "title": "util/mod.rs helpers: strides/spans, cast_slice, AlternatingBuffers, scale_read_shape, SendSyncPtr, PtrNoalias/PtrMutNoalias, POD I/O, aligned access",
    "files": [
      "jix/src/util/mod.rs:1-843"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/ops/where_op.rs",
      "jix/src/ops/reduction.rs",
      "jix/src/archive/block.rs",
      "jix/src/archive/common.rs",
      "jix/src/storage/params.rs",
      "jix/src/array.rs",
      "jix/src/codec/mod.rs",
      "jix/src/util/iter/block.rs",
      "jix/src/storage/scalar.rs",
      "jix/src/storage/plain.rs"
    ],
    "focus": [
      "Is every span passed to PtrNoalias::new, PtrMutNoalias::new or from_slice at least as large as what the kernel touches? Check LANES-wide tails, n == 0 (Staging spans at elementwise_pipeline.rs:548,574), and mismatched shape/stride lengths in strided_span_bytes.",
      "Is there any call site where a PtrMutNoalias range overlaps memory read through another pointer while it is live (reduction state reusing the out buffer at reduction.rs:495/595/1153/1241, where_op x-in-out, pipeline dst vs operand cursors)?",
      "strides_for_layout_order (mod.rs:159) needs a permutation and nobody validates it. Can a producer of read_layout_order emit a non-permutation, which would cause OOB writes in materialize_out_buf? Also check truncation in default_strides / Idx::from_u64 on 32-bit.",
      "AlternatingBuffers: are parity and buffer lengths right in Encoder/Decoder? Is the raw-pointer move in edit() (mod.rs:379-385) sound?",
      "cast_slice / value_as_bytes / value_from_io: what happens with padded or invalid-bit-pattern T, and with uninitialized buffers handed to a user-supplied Read?",
      "scale_read_shape: product overflow, zero dims, duplicate or more than 8 order entries, budget math. Also check the callers of calc_block_end (block_size 0, begin > end).",
      "On REQUIRE_ALIGNED targets (armv7 is the only shipped one), is every read_maybe_aligned/write_maybe_aligned::<REQUIRE_ALIGNED> guarded by a real alignment check, not just a debug_assert?"
    ]
  },
  {
    "id": "pipeline-driver",
    "ctx": [
      "pipeline",
      "util-nd"
    ],
    "title": "Element-wise pipeline driver: to_buf, static/dyn type-erased loops, staging, inner loops",
    "files": [
      "jix/src/storage/elementwise_pipeline.rs:1-732"
    ],
    "consult": [
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/util/nd_iter_unordered_dyn.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/mod.rs",
      "jix/src/codec/mod.rs",
      "jix/src/storage/buf.rs",
      "jix/src/ops/common.rs"
    ],
    "focus": [
      "Can inputs_contiguous/output_contiguous (239-243 / 372-376) ever be true for an operand whose cursor stride != itemsize (stride-0 broadcast, single-element NdIter case, staged output)? A wrong flag makes OperandTyped read [T; N] out of bounds.",
      "Staging::fill_from cache (530-562): is the first fill of any src always a full chunk_len_max at pos == 0? Stride-0 entries are cached with n = usize::MAX over uninitialized pool memory.",
      "REQUIRE_ALIGNED=true paths (armv7; untested on x86 CI): staged-output scatter (295-306), alignment decision in compute_need_staging (480-482), aligned [T; LANES] writes and reads when dtype alignment vs align_of::<T> differ.",
      "Static (192-311) vs dyn (315-444) driver: any divergence in shape collapse, staging, flags or chunking?",
      "Degenerate-dim collapse (199-207), empty, 0-d and all-size-1 regions, plus u64->usize truncation at 200/326 on 32-bit targets.",
      "Pointer add / PtrMutNoalias span correctness for every stride combination, including a push-mode out with stride 0 and mixed itemsizes (casts).",
      "Lane-count const math (686-697), tail loop (638-649) and assert_unchecked at 608.",
      "ArithmeticOpStatus accumulation and check_op_status only after the whole region has been written."
    ]
  },
  {
    "id": "pipeline-nodes",
    "ctx": [
      "pipeline",
      "elementwise-ops"
    ],
    "title": "Pipeline operands, output layout, and generic op plumbing (Map/MapMultiple/IntoType/MaybeCompact)",
    "files": [
      "jix/src/storage/elementwise_pipeline.rs:734-993",
      "jix/src/ops/map.rs:1-440",
      "jix/src/ops/into_type.rs:1-154",
      "jix/src/ops/maybe_compact.rs:1-119",
      "jix/src/ops/common.rs:1-185",
      "jix/src/ops/mod.rs:1-182"
    ],
    "consult": [
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/scalar.rs",
      "jix/src/util/arr_sequence.rs",
      "jix/src/ops/op1.rs",
      "jix/src/ops/op2.rs",
      "jix/src/ops/where_op.rs",
      "jix/src/ops/cmp.rs",
      "jix/src/storage/params.rs",
      "jix/src/storage/buf.rs",
      "jix/src/dtype.rs"
    ],
    "focus": [
      "Does pick_output_layout (885-941) always produce a non-aliasing layout that fits nitems*itemsize, and agree with the driver's degenerate-dim rule (shape 0/1 dims, Constant-only pipelines, operands that disagree on order)? Can the product at 891 overflow?",
      "Does every node keep N_OPERANDS, MAX_ITEMSIZE and operands() consistent (Op1, Op2, Where, Clamp, MapMultiple, array/tuple/Vec sequences)? An undercount leaves cursors null; to_buf_impl only debug_asserts it (line 138).",
      "MapMultiple Vec/slice scratch (util/arr_sequence.rs:446-467, 570-591): set_len over uninitialized memory, UnsafeCell exclusivity, and the drop-before-next-call contract of read_bulk_as_iter.",
      "Is the dtype checked before every transmute_copy (Op1/Op2/Map/MapMultiple/Constant), including IntoType and TypeDyn forwarding paths?",
      "assert_unchecked_eq! in IntoType (88, 103) and Op2::shape (op2.rs:181): these trust that inner dtype() and shape() never change after construction, and trust read_data buffer coverage (core_trait.rs:133-134).",
      "Panics on degenerate inputs: MapMultiple::new with zero arrays, and MaybeCompact::as_compact unwrap.",
      "Operand::new_output borrow handling (756-763) and push-mode out handling via materialize_pipeline_out_buf."
    ]
  },
  {
    "id": "pipeline-simd",
    "ctx": [
      "pipeline",
      "elementwise-ops"
    ],
    "title": "SIMD scalar layer: map_vectors helpers, mask packing, scalar op trait macros",
    "files": [
      "jix/src/scalar/bool.rs:1-101",
      "jix/src/scalar/simd.rs:1-407",
      "jix/src/scalar/traits_util.rs:1-333",
      "jix/src/scalar/mod.rs:1-30"
    ],
    "consult": [
      "jix/src/ops/op1.rs",
      "jix/src/ops/op2.rs",
      "jix/src/ops/cast.rs",
      "jix/src/ops/cmp.rs",
      "jix/src/util/test_util.rs",
      "fearless_simd-1.1.0 source in ~/.cargo/registry (src/traits.rs, src/generated/fallback.rs)"
    ],
    "focus": [
      "Is the lane order of map_masks_to_bools / mask8s_from_mask (95-137, 350-384) correct for every element width, and is it gated to AVX2 only?",
      "Check the chunk arithmetic in map_vectors_different_width (KT/KU/chunk, 161-204) and the lane-equality const asserts in map_vectors2.",
      "When N % lanes != 0 the code falls back to the scalar fn, and tail elements are computed by the scalar fn too. Does every SIMD body agree with its scalar fn bit for bit (NaN, -0.0, overflow, saturation)?",
      "Status propagation in map_vectors_status / map_vectors2_status: do the scalar and SIMD flags match?",
      "impl_scalar_op1!/impl_scalar_op2! arm selection (bool-output arms, a missing bulk fn falling back to the default): can an impl silently pick the wrong arm?",
      "dispatch_baseline! (compile-time baseline, used on strided runs) vs the runtime dispatch! (used on contiguous runs): does any kernel give level-dependent results?",
      "for_each_unrolled release/debug equivalence."
    ]
  },
  {
    "id": "py-ops-dispatch-promotion",
    "ctx": [
      "py-ops"
    ],
    "title": "Runtime dtype dispatch, cast rules, promotion target, op macros, cast()",
    "files": [
      "jix-py/src/ops/common/dispatch.rs:1-400",
      "jix-py/src/ops/common/dtype_promote.rs:1-143",
      "jix-py/src/ops/common/mod.rs:1-154",
      "jix-py/src/ops/mod.rs:43-182"
    ],
    "consult": [
      "jix-py/src/ops/common/operand.rs",
      "jix/src/dtype.rs",
      "jix/src/ops/cast.rs",
      "jix-py/python/tests/test_type_promotion.py",
      "jix-py/src/util.rs"
    ],
    "focus": [
      "Can any input combination make dispatch pick a descriptor whose cast silently loses value (no OverflowError), e.g. untyped Scalar::UInt vs float slots, Unsafe casts of typed scalars, or the all-untyped P8 defaulting?",
      "Do the into_typed::<T>().unwrap() calls (dispatch.rs:74,101-102,141,171-172) always hold, i.e. does Operand::cast always yield exactly T::DTYPE for every CastKind/table?",
      "Is is_cast_allowed (dtype_promote.rs:62-142) consistent with the documented rules and with test_type_promotion.py for every Rank x Precision pair, including the python-int UInt remap and the dst==P8 short-circuit?",
      "Does promotion_target (dispatch.rs:260-314) match the documented rules for all array pairs and handle struct / inner-shape dtypes and all-scalar operand lists correctly?",
      "f16 skip logic (dispatch.rs:199-208, 332-336): can an f16-requiring descriptor be chosen without an f16 input, or an f16 input be mis-detected?",
      "cast_impl (mod.rs:104-180): every src/dst arm maps to the right Rust type; complex->real always rejected; no-op path returns correct dtype."
    ]
  },
  {
    "id": "py-ops-operand-broadcast",
    "ctx": [
      "py-ops",
      "py-core"
    ],
    "title": "Operand conversion (Python/numpy -> Scalar/Plain), scalar casts, broadcasting, asarray",
    "files": [
      "jix-py/src/ops/common/operand.rs:1-461",
      "jix-py/src/ops/common/broadcast.rs:1-106",
      "jix-py/src/ops/as_array.rs:1-84",
      "jix-py/src/util.rs:346-469"
    ],
    "consult": [
      "jix/src/storage/plain.rs",
      "jix/src/storage/scalar.rs",
      "jix-py/src/dtype.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix/src/ops/cast.rs"
    ],
    "focus": [
      "Plain::new safety (operand.rs:185-194): are lifetime, readable-region, stride, bool-validity and byte-order preconditions all established for every numpy input (views, frombuffer, structured fields, big-endian, as_strided)?",
      "Python scalar classification (operand.rs:59-90) and 0-d numpy extraction (101-163): any value whose precision/rank tag or extracted value is wrong (f16, complex64, uint64, huge ints, bool)?",
      "Scalar::cast / check_int_in_bounds / into_array (operand.rs:200-302, 375-460): does the rebuilt scalar always match the target dtype and value (rounding, f16 double rounding, wraparound)?",
      "broadcast_shapes/broadcast_operand: correct for 0-length dims, ndim mismatch, Scalar operands, and does it preserve PyArray identity only when shapes match?",
      "array_try_map / array_try_from_fn (util.rs:359-445): drop/leak/double-drop on Err or panic with [Operand; N]?",
      "asarray with dtype/params: params ignored for jix arrays, numpy dtype attached verbatim - any wrong-result path?"
    ]
  },
  {
    "id": "py-ops-arith-wrappers",
    "ctx": [
      "py-ops",
      "elementwise-ops"
    ],
    "title": "Unary and binary arithmetic wrappers incl. absolute, log, multiply fast path, power shortcuts",
    "files": [
      "jix-py/src/ops/op1.rs:1-707",
      "jix-py/src/ops/op2.rs:1-697"
    ],
    "consult": [
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/src/ops/common/operand.rs",
      "jix/src/ops/op1.rs",
      "jix/src/ops/op2.rs",
      "jix-py/python/tests/test_op1.py",
      "jix-py/python/tests/test_op2.py",
      "jix-py/python/tests/test_type_promotion.py"
    ],
    "focus": [
      "Does every table entry map to a core op whose semantics match the documented Python behavior (dtype, wrapping, NaN/inf, errstate)?",
      "power: are SQUARE_TABLE / SQRT_TABLE pairs exactly DISPATCH_TABLE's so dtypes never differ, and is the scalar match (op2.rs:682-689) right for typed scalars, 0-d arrays, -0.0, bool?",
      "multiply fast path (op2.rs:170-178): can square() give a different dtype or value than the general multiply for any self-multiplied input?",
      "absolute scalar path (op1.rs:618-664): precision kept when Complex becomes Float; i64::MIN; typed vs untyped results consistent with the array path?",
      "log with base (op1.rs:303-339): multiplier computed in f64 then cast to T (f16) - precision and edge bases (1, 0, negative, inf)?",
      "Unary None-cast tables with Python scalars: does the all-untyped P8 default always find an entry?"
    ]
  },
  {
    "id": "py-ops-cmp-bitwise-where",
    "ctx": [
      "py-ops",
      "elementwise-ops"
    ],
    "title": "Comparison, clamp, isclose, bitwise/logical, where, is_nan/real/imag, dtype_sub_field",
    "files": [
      "jix-py/src/ops/cmp.rs:1-625",
      "jix-py/src/ops/bitwise.rs:1-705",
      "jix-py/src/ops/where_op.rs:1-83",
      "jix-py/src/ops/logical1.rs:1-88",
      "jix-py/src/ops/complex.rs:1-88",
      "jix-py/src/ops/sub_dtype.rs:1-60"
    ],
    "consult": [
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/src/ops/common/broadcast.rs",
      "jix/src/ops/cmp.rs",
      "jix/src/ops/bitwise.rs",
      "jix/src/ops/where_op.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix-py/python/tests/test_cmp.py",
      "jix-py/python/tests/test_bitwise.py"
    ],
    "focus": [
      "clamp: bool listed last (cmp.rs:439) vs maximum/minimum bool first - dtype inconsistency; also min>max and NaN bounds via one-sided paths vs the three-operand path.",
      "isclose: rtol/atol conversion (cmp.rs:528-589) to f16/f32 and complex atol construction - any wrong tolerance or accepted invalid input?",
      "where: broadcast-then-promote order, condition into_typed::<bool>, result numpy dtype taken from x - correct for scalar/array/struct combinations?",
      "rotate tables (bitwise.rs:428-491): every (T, u32) pair reachable as intended; typed signed shift counts; Unsafe [bool] logical ops with complex/f16 inputs.",
      "dtype_sub_field: result numpy dtype for inner-shaped or nested fields; error on missing field?",
      "Do comparison tables with bool last ever pick a lossy common type (e.g. u64 vs i64 -> f64 equality false positives)?"
    ]
  },
  {
    "id": "py-ops-shape-ops",
    "ctx": [
      "py-ops",
      "shape-ops"
    ],
    "title": "Shape op wrappers and index parsing (also used by Array.__getitem__/numpy())",
    "files": [
      "jix-py/src/ops/shape_ops.rs:1-1262",
      "jix-py/src/util.rs:18-44",
      "jix-py/src/util.rs:172-329"
    ],
    "consult": [
      "jix/src/ops/shape_ops/",
      "jix-py/src/array.rs",
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/python/tests/test_slice.py",
      "jix-py/python/tests/test_getitem.py",
      "jix-py/python/tests/test_array_ops.py"
    ],
    "focus": [
      "parse_basic_index (shape_ops.rs:205-350): empty slices on 0-length dims and start==len (line 314), bool indices (245), huge stops clamped by slice_unpack, dims > i64::MAX cast as i64.",
      "permute_dims Vec<i32> with > NDIM_MAX entries -> normalize_dims ArrayVec collect panic (shape_ops.rs:606, util.rs:185-192); any other unbounded DimArray collect?",
      "reshape -1 inference (703-763): u64 overflow of known_size, zero-size arrays, flatten size as i64.",
      "stack/concatenate after promotion: single-array fast paths (872-876, 954-958), dim == ndim, 8-d inputs, python scalars in the list, numpy dtype of the result.",
      "insert_dim/remove_dim/squeeze/flip/roll/repeat/tile: negative dims, duplicates, empty tuples, repeats 0, dim=None rules - all errors raised as Python exceptions, never panics?"
    ]
  },
  {
    "id": "py-core-read-path",
    "ctx": [
      "py-core",
      "storage-core"
    ],
    "title": "Array.numpy / __getitem__ / out= / compact / read context and basic-index parsing",
    "files": [
      "jix-py/src/array.rs:1-302",
      "jix-py/src/array.rs:400-741",
      "jix-py/src/array.rs:1604-1751",
      "jix-py/src/util.rs:1-170",
      "jix-py/src/util.rs:368-379"
    ],
    "consult": [
      "jix-py/src/ops/shape_ops.rs",
      "jix/src/storage/buf.rs",
      "jix/src/array.rs",
      "jix/src/storage/params.rs",
      "jix/src/storage/plain.rs",
      "jix-py/src/error.rs",
      "jix-py/python/tests/test_numpy_out.py",
      "jix-py/python/tests/test_getitem.py",
      "jix-py/python/tests/test_slice.py"
    ],
    "focus": [
      "Can the allocating path give to_ndarray_buf a StridedBuf smaller than the index region? Check u64-to-usize/npy_intp casts on 32-bit (array.rs:529, util.rs:86) and numpy appending dims for subarray descrs. check_out_buf does not check the span.",
      "The stride rebuild (array.rs:563-592) assumes np_arr.ndim == out_shape.len(). Check it for dropped dims, extent-1 dims, negative extent-1 strides and transposed allocations (numpy_empty_ordered permute math).",
      "Is check_output_array (array.rs:216-286) enough for StridedBuf::from_raw_parts_mut's no-alias contract? Consider self-overlapping positive strides, stride < itemsize, and the TOCTOU between the check and the detached write.",
      "Push-mode reliance: the returned buffer is ignored at array.rs:624-626 and PyArray_Empty memory is uninitialized. Can any read path leave holes or return a pull view instead of writing to dst?",
      "Locking: a std Mutex is held across Python calls in numpy_dtype (array.rs:288-298), while read_ctx and ContextGuard::drop lock it from attached threads. Look for deadlocks or poisoning on free-threaded builds.",
      "compact/resolve_array_params: read_size parsing, error-kind mapping, and params being resolved twice (asarray plus compact)."
    ]
  },
  {
    "id": "py-core-numpy-wrap-dtype",
    "ctx": [
      "py-core",
      "py-ops",
      "types"
    ],
    "title": "numpy <-> jix dtype mapping, zero-copy Plain wrapping, numpy-dtype caching, type-hint extractors",
    "files": [
      "jix-py/src/dtype.rs:1-219",
      "jix-py/src/ops/common/operand.rs:18-199",
      "jix-py/src/ops/as_array.rs:1-84",
      "jix-py/src/ops/mod.rs:37-103",
      "jix-py/src/type_hint.rs:1-125"
    ],
    "consult": [
      "jix/src/storage/plain.rs",
      "jix/src/dtype.rs",
      "jix-py/src/array.rs",
      "jix-py/python/tests/test_sub_dtype.py",
      "jix-py/python/tests/test_bad_input.py"
    ],
    "focus": [
      "Byte order: dtype_from_numpy matches only (kind, itemsize) (dtype.rs:138). Trace a '>i4' input through asarray/compact, and asarray(dtype='>f8') or cast caching a non-native descr.",
      "Is the cached numpy descr (from_core_with_np_dtype) always identical in layout to arr.dtype()? Check user-supplied descrs, the align flag and subarray descrs.",
      "Struct classification: core aligned-vs-packed (jix/src/dtype.rs:653-717) vs numpy isalignedstruct and the debug_asserts at dtype.rs:206-213. Also round-trip dtype_to_numpy with align=True and the itemsize/alignment check.",
      "Subarray dtypes: dtype_from_numpy shape handling, dtype_to_numpy wrapping (dtype.rs:75-76), and what numpy() does with a top-level inner shape (e.g. from dtype_sub_field).",
      "Plain::new safety at operand.rs:165-197: keep-alive via Py ref, negative strides (including extent<=1), unaligned data, 0-d struct arrays, ndim > 8, zero-size arrays, and as_strided views past the buffer.",
      "Python-side mutation of a wrapped ndarray (resize refcheck=False, shape/strides setters, concurrent writers while detached) versus Plain's captured pointer, shape and strides."
    ]
  },
  {
    "id": "py-core-archive-errstate",
    "ctx": [
      "py-core",
      "codec-archive"
    ],
    "title": "read_array/write_array, PyReader/PyWriter adapters, mmap, errstate/warnings, module/package shim",
    "files": [
      "jix-py/src/archive.rs:1-501",
      "jix-py/src/error.rs:1-196",
      "jix-py/src/lib.rs:1-98",
      "jix-py/python/jix/__init__.py:1-46"
    ],
    "consult": [
      "jix/src/archive/array.rs",
      "jix/src/archive/mod.rs",
      "jix-py/src/array.rs",
      "jix-py/python/tests/test_archive.py",
      "jix-py/python/tests/test_errstate.py",
      "jix-py/python/tests/test_bad_input.py"
    ],
    "focus": [
      "PyWriter::write memoryview (archive.rs:409-437): can slices, casts or buffer exports created inside a user write() outlive buf? The release() error is ignored.",
      "PyReader/PyWriter: seek conversions (u64 as i64 at archive.rs:334/468), negative or bogus returns from write/read/tell, None semantics, partial writes, and BufWriter flushing during error unwinding.",
      "read_array: offset/len validation split between jix-py (archive.rs:105-114) and core, the mmap unsafe contract, and what runs inside py.detach.",
      "write_array: append semantics (seek End plus TOC patch), create_new, partial files on error, and writing to a file currently mmapped by a live array.",
      "errstate: modes apply at construction, ErrState reuse, nested or out-of-order exit, thread-local modes vs where the read runs, and warning emission (CString NUL, warnings-as-errors after out= was written).",
      "IntoPyResult kind mapping (util.rs:54-64) and the effect of the module defaulting to gil_used=false on all of the above."
    ]
  },
  {
    "id": "py-core-array-surface-util",
    "ctx": [
      "py-core",
      "py-ops"
    ],
    "title": "Array getters/dunder forwarding, __str__, and util helpers (dims normalization, ItemOrTuple, array_try_map, UnsafeSend)",
    "files": [
      "jix-py/src/array.rs:302-400",
      "jix-py/src/array.rs:742-1470",
      "jix-py/src/util.rs:172-366",
      "jix-py/src/util.rs:380-469"
    ],
    "consult": [
      "jix-py/src/ops/common/broadcast.rs",
      "jix-py/src/ops/common/dispatch.rs",
      "jix-py/src/ops/shape_ops.rs",
      "jix-py/src/ops/reduction.rs"
    ],
    "focus": [
      "normalize_dims (util.rs:185-192) collects an unbounded slice into ArrayVec<_,8>. Which callers (permute_dims Vec<i32>, insert_dim, remove_dim, flip, reductions) can pass more than 8 entries?",
      "array_try_from_fn / array_try_map: drop and panic safety, the needs_drop specializations, double-drop or leak when f fails at index i.",
      "IterExt::try_collect_array length-mismatch handling, and the unreachable! at util.rs:340.",
      "UnsafeSend (util.rs:308-321): confirm it is unused and that its unconditional Send impl cannot be reached by safe code.",
      "Reflected dunders: operand order (__rsub__, __rtruediv__, __rfloordiv__, __rmod__, __rpow__, __rlshift__, __rrshift__), __pos__ dtype gating, __pow__ modulo.",
      "size() u64 product overflow, __len__ cast, __str__ recursion over wrapper storages, and ItemOrTuple tuple-only extraction / error remapping."
    ]
  },
  {
    "id": "codec-archive-codec-pipeline",
    "ctx": [
      "codec-archive",
      "util-memory"
    ],
    "title": "Codec pipeline: Encoder/Decoder, ReadContext, byte-shuffle (SIMD) and bit-shuffle filters",
    "files": [
      "jix/src/codec/mod.rs:1-475",
      "jix/src/codec/filter/mod.rs:1-61",
      "jix/src/codec/filter/byte_shuffle.rs:1-232",
      "jix/src/codec/filter/bit_shuffle.rs:1-381"
    ],
    "consult": [
      "jix/src/util/mod.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/aligned_vec/mod.rs",
      "jix/src/storage/block.rs",
      "jix/src/array.rs"
    ],
    "focus": [
      "Decoder::decode (codec/mod.rs:270-321) runs filters over all of dst.len() and returns nbytes; the only nbytes check is debug_assert_eq at storage/block.rs:193. In release, what reaches the caller when zstd produces fewer bytes (block len 0, which passes Strict validation; a frame without a content-size field; a skippable frame)? Do the filters read uninitialized or stale PoolBuf bytes?",
      "Byte-shuffle SIMD (byte_shuffle.rs:52-112, 160-200): check the interleave/deinterleave index rotation for lanes 16/32/64 and ITEMSIZE 2/4/8/16, on both branches (use_deinterleave and is_sse2). Is it correct only if fearless_simd interleave/deinterleave act on the whole vector rather than per 128-bit lane? Does the scalar tail start exactly at n_elements_done?",
      "get_unchecked bounds in the SIMD loads/stores (82, 107, 176, 195) and the generic loops (128-129, 216-217): do they hold for nitems < lanes, nitems = 0, and lengths that are not a multiple of itemsize (only debug-asserted at 120/208)?",
      "Bit shuffle (bit_shuffle.rs:114-178, 206-333): is decode exactly the inverse of encode for non-power-of-two typesize (3, 5, 6, 12 from struct or inner-shape dtypes), for n < 8, and for n mod 8 != 0? Is the tail copy right when src.len() is not a multiple of typesize?",
      "Buffer routing in Encoder::encode (155-179) and Decoder::decode (282-318) with 1, 2, 3 and 4 filters, given the AlternatingBuffers::edit semantics (util/mod.rs:372-403): does the final result always land in dst / the returned slice?",
      "Encoder hands uninitialized AlignedBytes to filters as &mut [u8] (codec/mod.rs:161,170). Does every filter write every dst byte for every length? Is itemsize == 0 (division by zero in both filters) unreachable from every caller?",
      "cfg(miri) memcpy path (187-192, 308-312): which length assumptions differ from the zstd path?",
      "ReadContext's UnsafeCell decompressor (303) and BufferPool: is there any path where two Decoders, or nested reads on one context, re-enter decode while a &mut to the decompressor is live? Does every PoolBuf use slice to the requested size?"
    ]
  },
  {
    "id": "codec-archive-archive-io",
    "ctx": [
      "codec-archive",
      "storage-compact"
    ],
    "title": "Archive reader/writer core: sections, TOC, block-table (de)serialization for Owned and Mmap storage, version word",
    "files": [
      "jix/src/archive/common.rs:1-276",
      "jix/src/archive/block.rs:1-503",
      "jix/src/archive/version.rs:1-129",
      "jix/src/archive/mod.rs:1-48"
    ],
    "consult": [
      "jix/src/storage/block.rs",
      "jix/src/util/mod.rs",
      "jix/src/archive/array.rs",
      "jix-py/src/archive.rs",
      "jix/schema/proto/jix/v1/block.proto"
    ],
    "focus": [
      "check_section_bounds (common.rs:267) adds offset + size with a wrapping u64 add in release. With Mmap::read_section (block.rs:469-500), can a crafted Section yield an MmapData longer than the mapping or longer than isize::MAX bytes (from_raw_parts at storage/block.rs:445)? What happens next in Strict validation and in read_block?",
      "Mmap sections are checked against the caller's len, never against mmap.len(). What do read_from_file_mmap and the safe Python read_array(path, mmap=True, len=...) do when offset+len is past EOF or the file is truncated?",
      "Allocations sized from untrusted values before any validation: read_slice reserve (common.rs:161,209) from a uleb128 u64 length; Owned::read_section with len=None (total_size = u64::MAX, block.rs:452). Which give capacity-overflow panics, and which abort the process on OOM?",
      "read_uleb128 (common.rs:165-190): shift overflow from the 10th continuation byte on, an i32 loop counter, and `length as usize` truncation on 32-bit targets.",
      "Strict versus Minimal validation (block.rs:392-412): with a bad block (offset, len), is every later use (storage/block.rs:189, write_content block.rs:87) a clean panic or error for both Owned and Mmap, never UB?",
      "value_from_io, read_slice and Owned::read_section hand uninitialized memory to the caller's Read impl through the safe read_from_reader. Is the Vec<MaybeUninit<T>> to Vec<T> transmute (block.rs:459) sound?",
      "BlockArchiveWriter (block.rs:111-260): no check that every block index is written exactly once (an unwritten index stays (0,0)), `compressed.len() as u32` can truncate, and seek/offset arithmetic depends on a non-zero base_offset. Is the 8-byte padding on the absolute stream position consistent with mmap readers?",
      "On 32-bit: `section.size as usize` (450, 485) then the assert_eq in read_section_into (common.rs:243), `offset as usize` (488), and i64/u64 casts in seek_relative_to_base (256-263): panics or misreads where an error is due?"
    ]
  },
  {
    "id": "codec-archive-array-format-dtype",
    "ctx": [
      "codec-archive",
      "types",
      "storage-compact"
    ],
    "title": "Array-level archive API and header validation, protobuf dtype conversions",
    "files": [
      "jix/src/archive/array.rs:1-559",
      "jix/src/archive/schema/conversions.rs:1-174",
      "jix/src/archive/schema/mod.rs:1-8"
    ],
    "consult": [
      "jix/src/dtype.rs",
      "jix/src/storage/compact.rs",
      "jix/src/storage/params.rs",
      "jix/src/ops/sub_dtype.rs",
      "jix/src/array.rs",
      "jix-py/src/archive.rs",
      "jix/schema/proto/jix/v1/dtype.proto"
    ],
    "focus": [
      "read_from (array.rs:436-535): u64-to-u32 truncation of block_shape (468) happens before validation, ceil_to_multiple can overflow (487), and the nblocks product (503-507) is unchecked. Are 0-d shapes and zero-length dims handled the same way by the writer and the reader?",
      "Caller params reach the spec (array.rs:532). If block_shape_fixed_dims has any false entry (Rust ArrayParams or the Python params dict), tune rescales the block shape (storage/params.rs:346-359). Does spec.block_shape() then disagree with the on-disk block geometry (storage/compact.rs:193)? What do reads return, and what header does write_to_with (399-405) write?",
      "Dtype::from_proto with struct dtypes: field layout is validated with u16 arithmetic in dtype.rs (673, 682, 704) that wraps in release. Can a crafted file produce a struct dtype whose fields extend past itemsize, so SubDtype or field access (ops/sub_dtype.rs:135,149) reads out of bounds?",
      "Scalar dtypes: are the alignment, itemsize and inner-shape checks complete? For a bool dtype loaded from a file, is there any validation of byte values before into_typed::<bool>() and typed reads?",
      "to_proto then from_proto round trip for every dtype form (nested structs, inner shape on scalars and structs, packed vs aligned layout, alignment greater than itemsize, empty struct, to_proto `as u32` casts): lossless, and accepted on read?",
      "write_to_with: does builder_init's unwrap on the u32 block product (421) panic on reachable input? Does the compact fast path's header (spec.block_shape()) always match blocks.block_size and nitems?",
      "Version and forward compatibility: do unknown ScalarKind/Endianness/ArchiveType values and unknown codec/filter oneofs always turn into clean errors?"
    ]
  },
  {
    "id": "shape-ops-forwarders",
    "ctx": [
      "shape-ops",
      "storage-core"
    ],
    "title": "Stride-remapping forwarders: PermuteDims, InsertDim, RemoveDim, Broadcast, Array wrappers, read_data_and_map_strides",
    "files": [
      "jix/src/ops/shape_ops/mod.rs:1-192",
      "jix/src/ops/shape_ops/permute_dims.rs:1-176",
      "jix/src/ops/shape_ops/insert_dim.rs:1-236",
      "jix/src/ops/shape_ops/remove_dim.rs:1-221",
      "jix/src/ops/shape_ops/broadcast.rs:1-209",
      "jix/src/storage/buf.rs:370-418"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/core_trait.rs",
      "jix/src/storage/params.rs",
      "jix/src/storage/plain.rs",
      "jix/src/storage/scalar.rs",
      "jix/src/array.rs"
    ],
    "focus": [
      "Is the lifetime transmute at buf.rs:364 sound for every caller, and are each op's inner2outer/outer2inner stride closures exact inverses over the same dim correspondence?",
      "InsertDim/RemoveDim: are original_dims/dims_mapping and both stride maps correct for unsorted, adjacent, leading and trailing dims, in both pull and push mode (incl. the InsertDim empty-read short-circuit)?",
      "PermuteDims and RemoveDim check only ndim and rely on the inner's bounds check: can any inner skip it before touching memory?",
      "InsertDim/RemoveDim keep PLAIN_READ: do the consumers of that flag (slice.rs:146 fast path, array.rs:964 small_read) stay correct for every inner that can carry it?",
      "Broadcast pull relabels a possibly writable PoolBuf with 0 strides; push copies via copy_from. Can any path write through or forward such a buffer as an out?",
      "Spec derivation: is read_layout_order always a valid permutation and are block_shape/fixed dims/weights of length ndim with block entries in 1..=max(shape,1)?",
      "RemoveDim::new (remove_dim.rs:58): dim_arr(dim.len()) asserts <= NDIM_MAX before validation - panic instead of Err for more than 8 (e.g. repeated) dims?"
    ]
  },
  {
    "id": "shape-ops-slice-reshape",
    "ctx": [
      "shape-ops",
      "storage-core"
    ],
    "title": "Slice (steps, negative bounds, Plain fast path) and Reshape (dim matching and per-iteration reads)",
    "files": [
      "jix/src/ops/shape_ops/slice.rs:1-410",
      "jix/src/ops/shape_ops/reshape.rs:1-417"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/iter/nd.rs",
      "jix/src/storage/plain.rs",
      "jix-py/src/ops/shape_ops.rs"
    ],
    "focus": [
      "DimSlice::resolve and the From<range> conversions: negative, inclusive (..=-1 at slice.rs:320 gives end 0), and > i64::MAX bounds; dims longer than i64::MAX.",
      "Stepped slices: steps that do not divide the length, sub-ranges of the view, and the general path's per-item sub-buffers in push mode with permuted or padded out strides.",
      "Plain fast path (slice.rs:145-160): bounding-box math, unchecked stride*step overflow (157), and which inner storages can carry PLAIN_READ (InsertDim/RemoveDim over Plain/Scalar).",
      "Reshape read: is the matched/unmatched split and the flat decomposition (reshape.rs:355-360, which omits matched dims) correct for size-1 dims, 0-d shapes and orig dims skipped by the scan? Are orig_strides for the push sub-buffer right?",
      "Reshape::new: unchecked default_logical_strides (66) and try_product with zero-size dims (a leading 0 short-circuits the check); divergence between the constructor scan (67-79) and the read-time scan (290-308).",
      "Does every element of the region get written exactly once (to_ndarray_sub assume_init()s the destination)?"
    ]
  },
  {
    "id": "shape-ops-copy-ops",
    "ctx": [
      "shape-ops",
      "util-nd"
    ],
    "title": "Read-then-copy and split ops: Flip, Roll, Repeat, Tile",
    "files": [
      "jix/src/ops/shape_ops/flip.rs:1-190",
      "jix/src/ops/shape_ops/roll.rs:1-180",
      "jix/src/ops/shape_ops/repeat.rs:1-278",
      "jix/src/ops/shape_ops/tile.rs:1-340"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/util/mod.rs",
      "jix/src/array.rs",
      "jix/src/util/buf_pool.rs"
    ],
    "focus": [
      "Every get_unchecked/get_unchecked_mut offset feeding NdCopier: provably within the slice for all ranges, including 0-stride inner dims and caller-supplied out buffers?",
      "Flip: mirrored inner range [L-e, L-s) and dst_off = (len-1-idx)*stride for partial sub-ranges with several flipped dims.",
      "Roll: the wrap split at shift (roll.rs:93-141) - region lengths, dst offsets, and j_start in the non-wrap branch when e <= shift; rem_euclid with len > i64::MAX (roll.rs:54).",
      "Repeat copy_region: do head/middle/tail cover each output position exactly once for every s mod n and e mod n; are the (ndim+1)-d copy shapes/strides right; overflow of out_strides[k]*n when unused (repeat.rs:181)?",
      "Tile: exact case A/B/C boundaries, Case C head/middle/tail extents vs out_shape, and the synthesized stride-0 dim mapping (src_dim).",
      "32-bit targets: u64 extents/offsets cast with as usize versus a destination sized from truncated out_shape (tile.rs:205, repeat.rs:164): possible heap overflow."
    ]
  },
  {
    "id": "shape-ops-join",
    "ctx": [
      "shape-ops",
      "util-nd"
    ],
    "title": "Multi-array joins: Concatenate and Stack (with ArraySequence glue)",
    "files": [
      "jix/src/ops/shape_ops/concatenate.rs:1-321",
      "jix/src/ops/shape_ops/stack.rs:1-269",
      "jix/src/util/arr_sequence.rs:1-93",
      "jix/src/util/arr_sequence.rs:600-664"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/params.rs",
      "jix/src/ops/into_type.rs",
      "jix-py/src/ops/shape_ops.rs",
      "jix-py/src/ops/common/dispatch.rs"
    ],
    "focus": [
      "borders construction (unchecked u64 sum at concatenate.rs:111, before check_shape_overflow) and lookup (linear position vs partition_point), including zero-length members at the start, middle and end.",
      "inner_contiguous (concatenate) / in_place (stack) fast paths: is the packed slot exactly the sub-array's bytes in out, and are the conditions (output dims vs array dims before the join dim) sufficient?",
      "Per-member sub-index and buffer offset with permuted or padded out strides; stack dim at 0, middle and ndim; out with stride 0 on a length-1 dim.",
      "ElementType soundness: can a sequence reach Stack/Concatenate with mismatched runtime dtypes but a typed ElementType, given the tuple/Vec/array impls and the runtime dtype check?",
      "Spec combine (combine_block_layout, combine_select_hints, Stack layout-order shift + insert) always yields valid per-dim arrays and a permutation?",
      "Does every element of the requested region get written (pull buffers are uninitialized pool memory)?"
    ]
  },
  {
    "id": "storage-core-buf-contract",
    "ctx": [
      "storage-core",
      "shape-ops"
    ],
    "title": "ArrayStorage trait, StridedBuf and out-buffer helpers, forwarders (View/Box/Rc/Arc/&dyn), ArrayStorageAny, info",
    "files": [
      "jix/src/storage/core_trait.rs:1-179",
      "jix/src/storage/buf.rs:1-418",
      "jix/src/storage/mod.rs:1-302",
      "jix/src/storage/any.rs:1-114",
      "jix/src/storage/info.rs:1-74"
    ],
    "consult": [
      "jix/src/util/nd_copy.rs",
      "jix/src/util/buf_pool.rs",
      "jix/src/util/mod.rs",
      "jix/src/ops/shape_ops/permute_dims.rs",
      "jix/src/ops/shape_ops/insert_dim.rs",
      "jix/src/ops/shape_ops/remove_dim.rs",
      "jix/src/ops/shape_ops/broadcast.rs",
      "jix/src/ops/shape_ops/slice.rs",
      "jix/src/ops/into_type.rs",
      "jix/src/storage/elementwise_pipeline.rs"
    ],
    "focus": [
      "Can safe code make a push read write outside out's byte slice? check_out_buf (buf.rs:282) checks only stride count and writability, StridedBuf has no shape, NdCopier has no bounds checks, and pull-mode PoolBufs (buf.rs:316, elementwise_pipeline.rs:879) are writable, so they can be passed back as out for a larger region.",
      "read_data_and_map_strides (buf.rs:336-368): is the lifetime transmute at :364 sound for every caller? Does the push branch still hold if the inner storage returns a buffer other than inner_out?",
      "with_strides / from_pool / with_offset callers: which can produce a writable buffer with aliasing (0) strides (Broadcast pull, broadcast.rs:151) or an offset past the slice end?",
      "materialize_out_buf (buf.rs:300-319): what happens when product*itemsize overflows usize (32-bit) or the layout order is not a permutation?",
      "Are is_contiguous / is_contiguous_aligned (buf.rs:188-200) exact for len-0/len-1 dims and for itemsize not a multiple of alignment? Scalar indexes the slice directly based on them.",
      "Forwarders: does any method left at its default change behavior? The &dyn impl uses the default read_as_elementwise_pipeline. Can ArrayStorageAny's cached shape/dtype/spec (any.rs:105-112) disagree with the forwarded read_data? Is the assert_unchecked at any.rs:64 justified?",
      "Is the documented contract of pub unsafe StridedBuf::from_raw_parts(_mut) (buf.rs:64-93) sufficient? Consider a shape/strides length mismatch (silently shortens the span via zip), stride*(len-1) overflow, and later reads with a different shape."
    ]
  },
  {
    "id": "storage-core-leaf-backends",
    "ctx": [
      "storage-core",
      "types"
    ],
    "title": "Plain and Scalar backends, ndarray constructors, compact_ndarray / compact_nd_ptr / compact_fn (FnStorage)",
    "files": [
      "jix/src/storage/plain.rs:1-526",
      "jix/src/storage/scalar.rs:1-199",
      "jix/src/array.rs:177-517"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/nd_copy.rs",
      "jix/src/storage/params.rs",
      "jix/src/element_type.rs",
      "jix/src/dtype.rs",
      "jix/src/util/iter/strides.rs",
      "jix/src/error.rs"
    ],
    "focus": [
      "Negative ndarray strides (s![..;-1], invert_axis) into plain_ndarray (plain.rs:248 unwrap) and plain_ndarray_ref/_view/compact_ndarray (plain.rs:289 `s as usize * size_of`): panic, or wrapped strides that make plain.rs:461-471 call from_raw_parts/ptr.add with bogus values?",
      "Plain::read_data span and offset math (plain.rs:458-471) for empty ranges, starts at the end of a dim, len-1 dims with arbitrary strides, and 0-stride dims. Do the pull view and the push copy always produce the same bytes?",
      "Scalar push: when can is_contiguous_aligned be true while the region_len slice or the cast_slice_mut assert panics? Is the strided path with a 0-stride src correct? Does read_data agree with the Constant pipeline (scalar.rs:137-173), including for bool and struct T?",
      "FnStorage (array.rs:412-490): it has no index validation. Is the aligned typed write (446-459) proven, i.e. dtype.alignment() >= align_of::<T>() for every Dtyped T on every target? Are NdIter offsets relative to index.start?",
      "Padded #[derive(Dtyped)] structs: Scalar pull (scalar.rs:111) and Plain push expose uninit padding bytes as &[u8], NdCopier copies them as integers, and the encoder reads them. Is that a reachable uninit read?",
      "Is Plain::new's derived read_layout_order (plain.rs:147-181) always a permutation of 0..ndim for every stride pattern? to_ndarray_sub depends on it.",
      "Unsafe constructor contracts (Plain::new, plain_ndarray_ptr, compact_nd_ptr): what about a null pointer with an empty shape, and SendSyncPtr making Plain Send+Sync whenever its owner A is?"
    ]
  },
  {
    "id": "storage-core-array-read-paths",
    "ctx": [
      "storage-core",
      "storage-compact"
    ],
    "title": "Array front-end: to_ndarray / to_ndarray_sub / to_ndarray_slice / to_ndarray_buf + request splitting, compact_with / compact_into_builder, misc Array methods",
    "files": [
      "jix/src/array.rs:519-1735"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/storage/params.rs",
      "jix/src/util/mod.rs",
      "jix/src/util/iter/block.rs",
      "jix/src/util/iter/strides.rs",
      "jix/src/error.rs",
      "jix/src/storage/block.rs",
      "jix/src/storage/compact.rs",
      "jix/src/archive/array.rs",
      "jix/src/archive/block.rs"
    ],
    "focus": [
      "to_ndarray_buf_slow_dyn (array.rs:1010-1066): are the pieces disjoint, covering and correctly offset (out_offset 1057-1059) into permuted or padded out buffers? Check unaligned starts, a read_shape snapped to array_shape (util/mod.rs:521-525), and partial last pieces.",
      "to_ndarray_sub (743-801): is the layout_order to alloc_shape/inv_order mapping correct, and are buf_strides right? Which storage paths could leave elements unwritten before assume_init at :800?",
      "32-bit targets: product-as-usize and nitems*itemsize in check_get_buffer_size (error.rs:246-247) and Idx::from_u64 truncation in to_ndarray_slice; per-dim `as usize` in to_ndarray_sub (764). Can a cheap lazy array (broadcast/compact_fn) with a region >= 4 GiB pass the size check with a tiny buffer?",
      "compact_into_builder_dyn (1274-1442): is the block_index mapping (1378-1398) correct? Is zero-fill skipped only for blocks the copy fully covers? The chunk_buf at 1369 is uninit and relies on the push contract. Is the get_unchecked bound at 1420 sound? Check 0-d and empty arrays, and that chunk_shape_in_blocks is always >= 1.",
      "small_read (964): plain_read storages (Plain, Scalar, and InsertDim/RemoveDim/IntoType/View/Any, which keep the flag) are never split. Is that always safe and correct?",
      "Error paths: what partially written out or builder state is observable after an Err from a piece or chunk read?",
      "What does the unsafe into_table (block.rs:280, called at 1213) assume (every block written, dtype matches ET), and does compact_into_builder guarantee it for every shape?"
    ]
  },
  {
    "id": "elementwise-ops-kernel-framework",
    "ctx": [
      "elementwise-ops",
      "pipeline"
    ],
    "title": "Op1/Op2 storage wrappers and define_op1/define_op2 plumbing",
    "files": [
      "jix/src/ops/op1.rs:1-268",
      "jix/src/ops/op2.rs:1-327"
    ],
    "consult": [
      "jix/src/scalar/traits_util.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/storage/elementwise_pipeline.rs",
      "jix/src/error.rs",
      "jix/src/ops/common.rs",
      "jix/src/ops/cmp.rs"
    ],
    "focus": [
      "read_bulk forwarding in Op1/Op2 (op1.rs:98-113, op2.rs:155-168): same offset/status to every child; is check_dtype(T, K::Output) plus the const size assert enough to justify transmute_copy?",
      "Op2::new shape check with swapped_operand_order (op2.rs:33-85) vs the assert_unchecked_eq in Op2::shape (op2.rs:181): any path that builds Op2 without the check, or where shape() can differ later (e.g. a safe user ArrayStorage)?",
      "Op1/Op2 read_data vs read_as_elementwise_pipeline: identical results, dtypes and error statuses in pull and push mode?",
      "spec()/info()/element_type_change of Op1/Op2: correct operand order for swapped ops, consistent dtypes?",
      "Zero-itemsize output rejection and other constructor checks: any op that bypasses them?"
    ]
  },
  {
    "id": "elementwise-ops-op2-arith-div-pow",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Binary arithmetic kernels: add/sub/mul, int and float divide/remainder (trunc/floor/euclid), f16/complex div, pow and error classification",
    "files": [
      "jix/src/ops/op2.rs:328-1613"
    ],
    "consult": [
      "jix/src/scalar/traits_util.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/error.rs",
      "jix/src/ops/op2.rs (tests from line 1592)",
      "jix-py/python/tests/test_errstate.py"
    ],
    "focus": [
      "Integer divide/remainder family for every signedness and width: zero divisor and MIN/-1 in each op; can the non-wrapping `d + (...)` (op2.rs:655), `d - (...)` (672) or `rem + b` (694) ever overflow?",
      "div_rem_trunc special values (inf, 0, -0.0, subnormal, NaN) and behavior past EXACT_LIMIT (2^24 / 2^53): can divide_floor (adjusted from the inexact residual) and remainder_floor (fmod-based) disagree so that a == q*b + r breaks beyond rounding?",
      "Scalar vs SIMD error classification: quotient/float_div_error vs quotient_v, remainder vs remainder_v, for NaN divisors, infinite dividends, -0.0 divisors, and results that only become non-finite after `div - 1.0`",
      "f16 paths (op2.rs:765-776, 1097-1137, 1210-1229): are errors computed from the f16-rounded result? Does the aarch64+fp16 cfg branch behave the same as the f32 branch?",
      "Pow: negative int exponent, exponent > u32::MAX loop, `b as u32` / `b as i32` / `b as Self` conversions in the error tests; float_pow_error conditions (0^-y, overflow, NaN)",
      "impl_arith lists (op2.rs:576-637): each integer type gets a wrapping scalar body plus a matching SIMD body; i8 Mul is scalar-only",
      "Complex Div: naive formula with no status; overflow for large |b|"
    ]
  },
  {
    "id": "elementwise-ops-op1-unary-classify-complex",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Unary math kernels (sign/abs/square/neg/floor/ceil/round/sqrt/exp/ln/trig), float classification, complex real/imag",
    "files": [
      "jix/src/ops/op1.rs:269-1261",
      "jix/src/ops/logical1.rs:1-179",
      "jix/src/ops/complex.rs:1-158"
    ],
    "consult": [
      "jix/src/scalar/traits_util.rs",
      "jix/src/scalar/simd.rs",
      "jix/src/error.rs",
      "jix/src/ops/op1.rs:1-268",
      "jix-py/src/ops/op2.rs"
    ],
    "focus": [
      "float_op1! error conditions (op1.rs:473-558): per op (Ln at +-0, Asin/Acos |x|>1, Exp overflow only for finite x, trig of inf); the f16 variant tests the rounded result; the cold fn re-evaluates in f64. Can the T-precision test and the f64 test disagree?",
      "SIMD vs scalar: Abs/Neg on ints (wrapping MIN), Square ints (SIMD `x * x` vs wrapping_mul), Floor/Ceil/Sqrt on -0.0 and NaN. Signed zero is not checked by the tests",
      "Sqrt bulk status (`simd_lt(0.0).any_true()`) vs scalar for -0.0, NaN and -inf",
      "Sign for f16/f32/f64 at +-0.0 and NaN; Sign for unsigned types",
      "Square vs Mul vs Pow consistency (jix-py routes x*x and power(x, 2) to Square, and power(x, 0.5) to Sqrt; Square drops the Overflow flag Pow sets?)",
      "Real/Imaginary/Abs output dtype and the Dtyped bounds",
      "IsNan/IsFinite/IsInfinite bool-output SIMD path (AVX2 only) lane mapping"
    ]
  },
  {
    "id": "elementwise-ops-cmp-clamp-approx",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Comparisons, swapped Greater/GreaterEqual, Maximum/Minimum, Clamp, ApproxEq",
    "files": [
      "jix/src/ops/cmp.rs:1-1128"
    ],
    "consult": [
      "jix/src/ops/op2.rs:1-327",
      "jix/src/scalar/simd.rs:300-385",
      "jix/src/scalar/traits_util.rs:200-329"
    ],
    "focus": [
      "Maximum/Minimum: NaN propagation to the canonical NaN in the SIMD select vs scalar; +-0 handling; f16 and bool arms",
      "Greater/GreaterEqual built as Less/LessEqual with swapped operands (cmp.rs:419-563): operand order in the pipeline, info(), spec(), and error messages; the `S2::Item: Less<S1::Item>` bound",
      "Clamp (cmp.rs:699-897): shape checks, NaN in any operand, lo > hi, and the transmute_copy justification",
      "ApproxEq: tolerance formula, inf handling, overflow of a - b, negative or NaN tolerances, per-component complex rtol/atol, f16",
      "bool comparison formulas (!a & b, !a | b) and any path that could read a non-0/1 bool",
      "map_vectors2_to_bools for 64-bit types on AVX2 (8 masks narrowed into one mask8s)"
    ]
  },
  {
    "id": "elementwise-ops-bitwise-cast",
    "ctx": [
      "elementwise-ops"
    ],
    "title": "Bitwise/shift/rotate/count kernels and the Cast op with SIMD conversion chains",
    "files": [
      "jix/src/ops/bitwise.rs:1-786",
      "jix/src/ops/cast.rs:1-488"
    ],
    "consult": [
      "jix/src/scalar/simd.rs",
      "jix/src/scalar/traits_util.rs",
      "fearless_simd-1.1.0 source in ~/.cargo/registry (narrow, saturating_narrow, widen, truncate_from_precise, float_from)",
      "half crate f16 conversions"
    ],
    "focus": [
      "shift_count (bitwise.rs:295-313) for every (lhs, rhs) pair; unbounded_shl/shr for negative and oversized counts; whether the SIMD i32/u32 Shr bodies (bitwise.rs:110-125) match the scalar for every signedness pair",
      "CountZeros SIMD `(!x).count_ones()` plus bitcast vs scalar; SwapBytes/ReverseBits on all widths",
      "Scalar cast table (cast.rs:55-60): bool through `as i8`, f16 through f32 (double rounding from f64 sources - the test oracle at cast.rs:540-547 uses the same formula), NaN to int, float to bool, complex to bool",
      "SIMD cast chains (cast.rs:184-271): KT/KU vector counts, widen then bitcast for sign or zero extension, truncating vs saturating narrow for float to int, f64 to f32 narrow, and whether the only_on gating is consistent",
      "Cast to bool SIMD (simd_ne(zero)) on NaN and -0.0; the AVX2-only path",
      "Test gaps: no mid-range values (e.g. 200.0 to u8, 40000.0 to u16) in SimdTestValues; AVX-512 and NEON paths may never run in CI"
    ]
  },
  {
    "id": "elementwise-ops-where-transmute-subdtype",
    "ctx": [
      "elementwise-ops",
      "util-nd",
      "types"
    ],
    "title": "Where (unsafe strided select loops), Transmute (unsafe relabel), SubDtype (struct field views)",
    "files": [
      "jix/src/ops/where_op.rs:1-451",
      "jix/src/ops/transmute.rs:1-152",
      "jix/src/ops/sub_dtype.rs:1-197"
    ],
    "consult": [
      "jix/src/storage/buf.rs",
      "jix/src/util/nd_iter_unordered.rs",
      "jix/src/storage/elementwise_pipeline.rs:735-869",
      "jix/src/storage/core_trait.rs:63-141",
      "jix/src/dtype.rs:461-707",
      "jix/src/util/mod.rs:622-682"
    ],
    "focus": [
      "Where::read_data unsafe loops (where_op.rs:153-342): offsets and strides from NdIterUnordered over three buffers with different layouts (0-stride cond/y), the (itemsize, alignment) to typed-loop table, the `aligned` logic when REQUIRE_ALIGNED, contiguous chunking and tail",
      "Where in push mode (caller out with arbitrary strides): x is written into out and then overwritten; empty and 0-d regions; runtime dtype equality for TypeDyn x/y; element_type_change",
      "Where pipeline (where_op.rs:344-407) vs read_data equivalence; errors from the unselected branch still fail the read",
      "SubDtype pull mode (sub_dtype.rs:126-136): view shifted by the field offset with PARENT strides; packed-struct misalignment; empty read returns a parent-typed buffer; push-mode get_unchecked (149) and NdCopier sizes",
      "SubDtype::new: field lookup, nested struct fields, inner-shape fields, zero-size fields; element_type_change keeps the offset",
      "Transmute: itemsize-only check; bool and struct targets; source padding bytes; alignment increase; element_type_change keeps the transmuted dtype; confirm no path hands back the inner Compact with the old dtype (as_compact is not forwarded)"
    ]
  }
]

const CONCEPT_UNITS = [
  {
    id: 'concept-push-pull',
    title: 'read_data push/pull contract across every storage and op',
    ctx: ['storage-core', 'shape-ops', 'pipeline'],
    start: ['grep -n "fn read_data<" -r jix/src', 'grep -n "fn read_as_elementwise_pipeline" -r jix/src', 'jix/src/storage/buf.rs (read_data_and_map_strides, materialize_out_buf)'],
    focus: [
      'In push mode, does every impl write exactly the requested region at dst\'s own strides (not assuming packed, row-major, or zero offset) and return a view of dst - never a pooled buffer or its own memory?',
      'Does any consumer assume a pull-mode result is packed, aligned, writable, or free of 0-stride dims?',
      'Forwarders that re-stride out and push it down (read_data_and_map_strides and similar): is the remapped stride set valid for the inner region in every case, e.g. when the op duplicates, reverses, or drops elements?',
      'Ops that split one request into several inner reads (concatenate, roll, tile, repeat, stack, reductions, Array::to_ndarray_buf): does each piece land at the right byte offset of out with the right strides, and are the pieces disjoint and covering?',
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
    start: ['jix/src/error.rs (ArithmeticErrorKind, ArithmeticOpStatus, modes)', 'jix/src/ops/op2.rs', 'jix/src/ops/op1.rs', 'jix/src/ops/cast.rs'],
    focus: [
      'Integer division / remainder by zero and MIN / -1, floor-div / mod sign conventions, overflow in add / sub / mul / neg / abs / pow, pow with negative int exponents, shifts by >= bit width or negative counts, float -> int casts of NaN / inf / out-of-range values, int -> float rounding, bool arithmetic.',
      'Does every kernel report the same ArithmeticErrorKind with the same fallback value for the same condition - SIMD body and scalar tail, element-wise ops and reductions, Rust and Python?',
      'Error / warn / ignore modes: is an Err raised after out was partially written, and is that observable? Are warnings lost or duplicated across tiles / blocks?',
      'NaN handling in min / max / argmin / argmax / clip and comparisons; signed zeros; isclose with inf / nan.',
      'Behaviors that silently lose information in a way a user would call a bug (questionable-semantics), even where they differ from NumPy on purpose.',
    ],
  },
  {
    id: 'concept-tiling-coverage',
    title: 'Request splitting and tiling: pieces must exactly cover the request',
    ctx: ['storage-core', 'storage-compact', 'reduction', 'util-nd'],
    start: ['jix/src/array.rs (to_ndarray_buf and its splitting heuristic, write paths)', 'jix/src/util/mod.rs (scale_read_shape, calc_block_end)', 'jix/src/storage/params.rs', 'jix/src/util/iter/block.rs'],
    focus: [
      'Array::to_ndarray_buf and the write paths cut a request into cache-sized pieces: are the pieces disjoint, covering, correctly offset into out, and correct at array edges and for unaligned starts?',
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
]

const DYNAMIC_UNITS = [
  {
    id: 'dynamic-rust-differential',
    title: 'Differential testing of op chains against an ndarray reference (Rust)',
    ctx: ['storage-core', 'shape-ops', 'pipeline', 'reduction'],
    task: [
      'Write integration tests (jix/tests/hunt_dynamic_rust_<n>.rs) that build random op chains - element-wise ops, shape ops (slice with steps, permute, broadcast, flip, roll, tile, repeat, concatenate, stack, reshape), reductions, casts - over small random arrays, and compare against a reference computed with plain ndarray / loops.',
      'Vary the storage: Compact with tiny explicit block shapes (so reads straddle partial blocks), Plain over strided / permuted ndarray views, broadcast inputs, 0-d and 0-length dims.',
      'Vary the materialization: to_ndarray, to_ndarray_sub over random sub-regions, to_ndarray_buf into out buffers with permuted / padded strides (and check bytes outside the region are untouched), compact(), and a write_to / read round trip.',
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

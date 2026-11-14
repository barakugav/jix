# Read validation overhead: ideas

`ArrayStorage::read_data` validates its `index` (against the storage shape) and its `out` buffer
(writable, one stride per dim, no overlapping elements, large enough) at every level of a storage
chain, although inner levels receive an index and `out` derived from already valid ones. For tiny
reads this is a large part of the cost. These are the options considered for cutting it, kept here
to return to later.

## Where the cost is

Instructions per 1-element read (callgrind, `(Ir(2N) - Ir(N)) / N`), measured on top of the
`Index` change, by removing the nested checks (top level still checked):

| read                                  | today | no nested `out` checks | no nested checks at all |
|---------------------------------------|-------|------------------------|-------------------------|
| 6-op chain over two `Plain` (deep)    | 4540  | 4092 (-10%)            | 3661 (-19%)             |
| `neg` over `Plain`                    | 1766  | 1658 (-6%)             | 1560 (-12%)             |
| `Plain` `to_ndarray_sub`              | 1036  | 941 (-9%)              | 898 (-13%)              |
| `Plain` `to_buf_strided`              | 341   | 328 (-4%)              | 279 (-18%)              |
| `Compact` `to_ndarray_sub`            | 3778  | 3683 (-3%)             | 3637 (-4%)              |

Roughly half is the `out` checks, half the range checks. The `out` half can be removed safely by
types (options 4, 5). The range half can only be shrunk: each storage has to check that the index
is within its own shape.

The deep chain is
`(a.neg() + b.permute_dims([1, 0]).permute_dims([1, 0])).permute_dims([1, 0]).insert_dim(0).reshape([32, 600])`
over `[600, 32]` `i32` arrays.

## Bounds checks in index loops

Applied. Loops reading the index ran over the storage's own number of dims (`self.shape.ndim()`,
`shape.len()`), so the compiler could not relate the bound to the index length and kept a bounds
check on each `index[dim]`. Bounding them by `index.len()`, or iterating the index directly,
removes the checks. Instructions per read before -> after: owned `Plain` `to_buf_strided` 310 ->
261, `Plain` view `to_buf_strided` 341 -> 338, `Plain` `to_ndarray_sub` 1052 -> 1045, `neg` 1767
-> 1719, deep chain 4540 -> 4441, `stack` + `sum` 4714 -> 4634.

## Options, simplest first

### 1. `Index` holds only well-formed ranges

`Index::new` also checks `start <= end`. The per-level check shrinks to "same number of dims, and
`end <= dim_size`", and code can compute `end - start` without underflow concerns. Inner indices
built with `Index::new` re-check `start <= end`, so the saving shows where the same `Index` is
passed through (elementwise ops, forwarders, the read cache), unless combined with option 3.
The 32-bit "region fits in memory" check needs the itemsize, so it stays per level (it compiles
to nothing on 64-bit).

Status: folded into option 2, as a start and a size cannot describe a reversed range. Tried alone
on the `&[Range<u64>]` index (after the loop bounds change), with the entry points checking
`start <= end <= dim_size` in a single pass and derived indices built with a crate-internal
`unsafe fn Index::new_unchecked` (checks are a `debug_assert`), it cut every read: owned `Plain`
`to_buf_strided` 261 -> 249, `Plain` view `to_buf_strided` 338 -> 326, compact 3744 -> 3732, `neg`
1719 -> 1706, deep chain 4441 -> 4374. A first version that checked `start <= end` in `Index::new`
and then the bounds in a second pass was slower than without option 1 for shallow reads (view
`to_buf_strided` 341 -> 359): each level saves one compare per dim, the entry paid a whole loop.

Related: the `Array` entry points (`to_ndarray_sub`, `to_buf_slice`, `to_buf_strided`) check the
range twice, once themselves and again in `to_buf_strided_checked` through `check_out_buf`.

### 2. `Index` carries the region size

Every level recomputes `end - start` for the output shape, the element count, the region size
check and the `out` overlap check. Representing the index as `{ start: &[u64], size: &[u64] }`
makes the sizes available directly, and makes the comparison with a shaped `out` (option 4)
cheap.

Status: applied, with the sizes as `usize`: `Index { start: &[u64], size: &[usize] }`, used
everywhere internally (invariants: same length, at most `NDIM_MAX`, no `start + size` overflow).
Most levels use the size as a `usize` (buffer shapes, strides, copies); sizes derived from `u64`
values (block / tile iterators, `Tile` / `Roll` lengths) are cast, as each is bounded by the
requested region or a read-size heuristic. The public `&[Range<u64>]` reads check the ranges once
with `check_ranges` (also that each size fits in `usize`, 32-bit only) and split them with
`split_ranges`. `read_data` and the internal helpers take the `Index` by reference (`&Index`).

Measured first with the `Index` passed by value, instructions per read (`option 1`: option 1 alone
on the `&[Range<u64>]` index):

| read                                   | loop bounds | option 1 | `u64` sizes | `usize` sizes |
|----------------------------------------|-------------|----------|-------------|---------------|
| owned `Plain` `to_buf_strided` (bench) | 261         | 249      | 255         | 254           |
| `Plain` view `to_buf_strided`          | 338         | 326      | 344         | 337           |
| `Plain` `to_ndarray_sub`               | 1045        | 1038     | 1029        | 998           |
| `Compact` `to_ndarray_sub`             | 3744        | 3732     | 3753        | 3704          |
| `neg` over `Plain`                     | 1719        | 1706     | 1686        | 1653          |
| deep chain                             | 4441        | 4374     | 4393        | 4370          |
| `broadcast`                            | 1342        | 1318     | 1359        | 1330          |
| `flip`                                 | 1690        | 1677     | 1651        | 1512          |
| `tile`                                 | 1304        | 1286     | 1259        | 1197          |
| `stack` + `sum`                        | 4634        | 4599     | 4695        | 4415          |

The tiniest reads gained the least, and option 1 alone beat option 2 on them: the entry points split
the ranges into two arrays, and an `Index` of two slices is 4 words, which the Rust ABI passes by
reference to a fresh stack copy at every call, while a `&[Range<u64>]` index is 2 words passed in
registers. With `u64` sizes every level also converted the sizes to `usize`, which made that
version slower than option 1 alone. A first version whose `split_ranges` returned
`Result<(DimArray, DimArray)>` was much worse (483 instructions for the bench read) because of a
`memcpy` of the returned arrays; checking separately and splitting in an inlined, infallible
function fixed that.

Wall-clock by value did not follow, beyond code-placement noise (64-byte function and loop
alignment, pinned, median of 7 runs, ns per read, loop bounds -> option 1 alone -> option 2): owned
`Plain` `to_buf_strided` 13.6 -> 13.1 -> 18.8, view 15.5 -> 15.4 -> 22.7, `broadcast` 70.5 -> 68.8
-> 85.0, `roll` 59.1 -> 59.7 -> 67.3, `concat` 58.5 -> 59.7 -> 66.9, deep chain 275 -> 276 -> 297,
compact and `stack` + `sum` +3% to +4%, while `flip` 101 -> 98 -> 76, and `neg` and `Plain`
`to_ndarray_sub` were within noise. The likely cause was the 4-word `Index` copied to the stack: at
the entry points it is written with 8-byte stores and copied with 16-byte loads at the next call,
which cannot be store-forwarded. Making `to_buf_strided_checked` `#[inline(always)]` recovered 3 of
the 7 ns of the view read.

Passing `&Index`, so a forwarded index is passed as a pointer and never copied, confirms the cause:
the wall-clock regression is gone (same setup, loop bounds -> option 2 by value -> `&Index`): owned
`Plain` `to_buf_strided` 12.8 -> 18.6 -> 12.5, view 15.1 -> 21.8 -> 15.0, `broadcast` 70.7 -> 83.0
-> 66.1, `roll` 58.4 -> 65.8 -> 58.1, `concat` 58.2 -> 66.3 -> 57.7, deep chain 279 -> 295 -> 274,
`stack` + `sum` 271 -> 266 -> 261, `flip` 96.8 -> 75.2 -> 74.7, compact within noise. Instructions
per read drop too (by value -> `&Index`): owned `Plain` `to_buf_strided` 254 -> 242, view 337 ->
329, `neg` 1653 -> 1628, deep chain 4370 -> 4339, `broadcast` 1330 -> 1312, `stack` + `sum` 4415 ->
4391. The jix-py `.text` is 76 KB smaller, mostly in the per-op and per-dtype `read_data` of the
elementwise ops, which no longer copy the index for every inner call.

Checking the bounds checks in the asm: build a driver with `CARGO_PROFILE_RELEASE_DEBUG=2`,
disassemble with `objdump -d -l --inlines -C`, resolve the indirect `call *GOT(%rip)` targets
through `readelf -r` (R_X86_64_RELATIVE addends) and `nm`, and map each call to a panic function
(`panic_bounds_check`, `core::panicking::panic`, `slice_index_fail`, ...) to its source line.
Verified gone: bounds checks on `index.start()[d]` / `index.size()[d]` / `index.end(d)` in loops
over `index.ndim()`, the `ndim <= NDIM_MAX` assert of `dim_arr(index.ndim(), ..)`, the
`split_ranges` checks after `check_ranges`, and `Index::new` (absent from the release binary).
Left: indexing by a stored dim (`index.start()[k]`), loops that also index an op's own per-dim
array (`Broadcast`, `Slice`), and the `Dim<N>` assert of the per-ndim dispatch in `Compact`.

### 3. Methods deriving one `Index` from another

`index.permuted(dims)`, `index.with_inserted_dim(d)`, `index.sub(..)`, keeping the invariants of
options 1 and 2 without re-checking them. They cannot prove the result is within the inner
storage's shape (that depends on the op), so the per-level `end <= dim_size` check stays.

### 4. A validated destination type instead of `Option<&mut StridedBuf>`

An `OutBuf<'a>` holding `&mut [u8]` (writable by type), the region sizes and the strides, with the
invariant that the region does not overlap itself and fits in the slice. `OutBuf::new` runs
today's expensive check once. Safe methods derive new destinations without re-checking:

- `permuted` / `insert_dim` / `remove_dim` - for `PermuteDims`, `InsertDim`, `RemoveDim`
  (today `read_data_and_map_strides`).
- `sub_view(offset, size)` - the `&mut out_buf[offset..]` pattern in `Tile`, `Slice`,
  `Concatenate`, `Stack`, `Roll` and the slow-path tiling in `to_buf_strided`.

Each level then compares `out`'s sizes with the index sizes instead of the overlap check. Also
removes the `unsafe` around `StridedBuf::from_slice_mut` in the ops and makes
`read_data_and_map_strides` safe. `Reshape`'s push path needs the checked constructor or a
dedicated method. Changes the type of `read_data`'s `out` parameter (public API);
`Array::to_buf_strided` keeps taking a `StridedBuf` and converts once.

### 5. One parameter for the index and the destination

A `ReadTarget` holding the `Index` and an optional `OutBuf`, with the invariant that they describe
the same region: `read_data(&self, target, ctx)`. Derive methods transform both together
(`target.permuted(dims)`, `target.sub(sub_index)`), pull vs push is `target.out()` being `None` or
`Some`. Each level is left with only the check that depends on the storage: the index within
`self.shape()`. Where options 2, 3 and 4 converge.

### 6. A richer return type

Return a buffer that also carries the region sizes, so the re-striding of pulled buffers
(`with_strides` in `Broadcast`, `Slice`, `Flip`) becomes safe methods. Push mode could return
nothing instead of a view of `dst` that callers drop. Mostly removes `unsafe` and recomputation
rather than validation.

## The rest of the small-read path

With the index passed by reference, most of a 1-element read was outside the index checks
(profiled per inlined function, with `addr2line -i` over callgrind's per-instruction counts): the
copy machinery set up for one element, the destination checked twice, three passes over the dims
in `Plain::read_data`, and an empty-read branch at every level. The changes, in order:

1. `StridedBuf` takes its strides as a `DimArray`, moved or cloned instead of rebuilt.
2. `Plain::read_data` computes the read offset and span in one pass.
3. `to_buf_strided_checked` skips the read size heuristic for plain storages.
4. `NdCopier::copy` copies a region with at most one dim longer than 1 as a 1-dim one, without an
   `NdIterUnordered`.
5. Each `Array` read is checked once, at its entry point; the storages still check what they get.
6. `Index` is never empty: the entry points answer empty reads, the levels below drop their
   empty-read branches.

Instructions per read:

| read                                   | `&Index` | 1    | 2    | 3    | 4    | 5    | 6    |
|----------------------------------------|----------|------|------|------|------|------|------|
| owned `Plain` `to_buf_strided` (bench) | 242      | 232  | 217  | 208  | 208  | 161  | 160  |
| `Plain` view `to_buf_strided`          | 329      | 320  | 281  | 265  | 265  | 201  | 200  |
| `Plain` view `to_buf_slice`            | 725      | 725  | 686  | 669  | 556  | 420  | 410  |
| `Plain` `to_ndarray_sub`               | 988      | 988  | 949  | 932  | 818  | 683  | 687  |
| `Compact` `to_ndarray_sub`             | 3695     | 3692 | 3692 | 3691 | 3578 | 3459 | 3445 |
| `neg` over `Plain`                     | 1628     | 1617 | 1590 | 1590 | 1595 | 1477 | 1443 |
| deep chain                             | 4339     | 4327 | 4244 | 4244 | 4248 | 4129 | 3988 |
| `broadcast`                            | 1312     | 1303 | 1277 | 1277 | 1164 | 1040 | 1008 |
| `flip`                                 | 1502     | 1477 | 1426 | 1424 | 1315 | 1195 | 1197 |
| `roll`                                 | 1189     | 1190 | 1162 | 1161 | 1048 | 928  | 894  |
| `tile`                                 | 1184     | 1185 | 1158 | 1158 | 1044 | 925  | 887  |
| `concat`                               | 1179     | 1184 | 1154 | 1154 | 1043 | 926  | 897  |
| `stack` + `sum`                        | 4391     | 4357 | 4308 | 4308 | 4090 | 3973 | 3914 |

Wall-clock (64-byte function and loop alignment, pinned, median of 9 runs, ns per read,
`&Index` -> 6, in a run slower overall than the ones above): owned `Plain` `to_buf_strided`
15.7 -> 12.7, view 18.7 -> 14.1, `to_buf_slice` 42.5 -> 23.5, `Plain` `to_ndarray_sub` 65.9 -> 51.7,
compact -9%, the op chains -7% to -16%.

Notes:

- 1: a `DimArray` clone copies all 8 slots branch-free, where rebuilding from a slice loops over
  the dims, so `concat`, which clones the strides per input, gets 5 instructions slower.
- 4: the check is folded into the existing empty-region pass over the dims; a separate pass cost
  the pipeline, which already passes 1-dim regions, up to 28 instructions.
- 6: `Index` holds the sizes as `&[NonZeroUsize]`, transmuted from and back to `&[usize]` at its API
  (`NonZeroUsize` has the layout of `usize`), so the code computing sizes keeps plain `usize`. The
  type states the invariant; it changes no instruction count, as `size()` hands out `usize`s and the
  levels no longer test for zero. An invariant that an `unsafe` constructor relies on is easy to
  break in a derived index: `Concatenate` built empty sub-indices for inputs empty along the concat
  dim, caught by the `debug_assert` in `Index::new_unchecked` running under the tests. Two first
  versions of the entry points were slower: checking the destination against a size slice instead of
  `&Index` made `check_out_buf` save 3 more registers (+10 per level), and computing the packed
  strides from the sizes the `Index` borrows kept them in memory (+64 on `to_buf_slice`).

Left on the `Plain` `to_ndarray_sub` read (687): about 210 to allocate the ndarray (`malloc` and
`ndarray::Array::uninit`), 136 in `to_ndarray_sub` itself (the layout-order check 21, `?` 17,
`check_ranges` 16, the allocation size 13), 106 to copy the element, 66 in `Plain::read_data`,
and 95 for `Plain`'s own index and destination checks (option 4).

## Inlining the checks

`check_out_buf` was neither generic nor `#[inline]`, so the crates instantiating the reads (jix-py,
the benches; jix-py has no LTO) could only call its one copy. Variants, instructions per read
(`Plain` view `to_buf_strided` / `Plain` `to_ndarray_sub` / deep chain / `broadcast`):

| variant                                                   | view | nd  | deep | bcast |
|-----------------------------------------------------------|------|-----|------|-------|
| `check_out_buf` a call                                    | 201  | 687 | 3993 | 1022  |
| `#[inline]` or `#[inline(always)]` (same code)            | 206  | 694 | 3846 | 966   |
| range loop out of line, dim count compare inlined         | 218  | 708 | 4000 | 1017  |
| `check_get_range` out of line, then `assert_unchecked`    | 196  | 684 | 3951 | 1005  |
| inlined, `Plain::read_data` `#[inline(always)]` (applied) | 187  | 675 | 3837 | 933   |

Inlined, `Plain::read_data` outgrew LLVM's threshold in the forwarding `&Plain` read, hence the
`#[inline(always)]`; jix-py reaches `Plain` only through `dyn`, so it does not grow there. The
destination check (`check_out_buf_impl`) stays a call, `#[inline(never)]`; an `assert_unchecked`
of the destination's stride count after it changed nothing. The gain is the call itself, not
bounds checks (657 panic sites in the bench driver's read paths -> 651): the checks left index an
op's own per-dim arrays (`is_broadcast`, `PermuteDims`'s `inv_dims`), which no length compare
relates to the index. jix-py `.text` +66 KB (+0.6%).

## Not recommended

- `Index` storing the shape it was checked against: inner storages have different shapes, so
  comparing costs about as much as the bounds check.
- A private "trusted" flag in `Index` that skips the bounds check: the same trust model as an
  unchecked read method, hidden in the type.
- Proving `end <= dim_size` across levels at compile time (branded lifetimes): too heavy.
- `Index` requiring `end <= i64::MAX` (array dims never exceed it): it would make signed index
  arithmetic provably safe, but every derived index would have to keep it, and it is easy to get
  wrong.

## Tried and dropped: an unchecked read method

`read_data` as a provided method that validates once and calls a hidden
`unsafe fn read_data_unchecked`, which storages implement and call for their inner storages
(`read_as_elementwise_pipeline` unsafe with the same contract). Dropped because it adds a second
read method and `unsafe` at every nested call. Result, instructions per read vs before `Index`:
plain `to_buf_strided` 367 -> 259, plain `to_ndarray_sub` 1056 -> 754, `neg` 1783 -> 1423, deep
chain 4564 -> 3532, compact 3767 -> 3506; wall-clock -8% to -20% on tiny plain and op-chain reads,
-3.5% on compact.

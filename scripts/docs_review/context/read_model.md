# Context: storage and the read path

For units that touch `jix/src/storage/`, `jix/src/array/`, `jix/src/util/` or the read context.
Read `general.md` first. This is orientation written from the item signatures; the code decides.

## `read_data`

```rust
fn read_data<'a>(
    &'a self,
    index: &Index<'_>,
    context: &'a ReadContext,
    out: OutBuf<'_, 'a>,
) -> Result<()>;
```

`Index` (`storage/index.rs`) is the region to read: a start and a non-zero size per dim.
`StridedBuf` (`storage/buf.rs`) is a block of bytes plus one byte stride per dim. `OutBuf` is the
destination, and picks one of two modes:

- **pull** - the storage puts the cheapest buffer it can in the slot of `out` and picks the
  strides: a borrowed view into its own memory, possibly with stride-0 (broadcast) dims, or a
  freshly materialized buffer.
- **push** - the region is written into the buffer of `out` at that buffer's strides.

The array-level entry points take `&[Range<u64>]`: `Array::to_buf_strided` pulls,
`Array::to_buf_strided_into` pushes, and `Array::to_buf_slice` is the shorthand for a packed
row-major destination. `to_ndarray` and `to_ndarray_sub` are pushes into a new ndarray.

## Pieces

- `ReadContext` (`context.rs`) - reusable state for reads: scratch buffers, a decoder instance and
  a read cache. `!Sync`, one per thread. `Array::read_ctx()` creates one from the array's params.
- `Compact<ET>` / `CompactMmap<ET>` (`storage/compact.rs`) - block-compressed storage over an owned
  or memory-mapped block table (`storage/block.rs`). `CompactBorrowed` is the borrowed form.
- `Plain<A, ET>` (`storage/plain.rs`) - uncompressed storage over an in-memory buffer, contiguous
  or strided. `PlainArray<T>` and `PlainArrayRef<'a, T>` are the aliases.
- `ArrayParams` (`storage/params.rs`) - the layout and codec settings of a compact array: block
  shape, block size, codec, level, filters, read size. Set with builder-style methods
  (`ArrayParams::new().block_shape(&[64, 64])?`).
- `ArraySpec`, `Scalar` and several `ArrayStorage` methods (`spec`, `info`, `as_compact`,
  `element_type_change`, `read_as_elementwise_pipeline`) are `#[doc(hidden)]`: public for
  technical reasons, not part of the documented API. Their docs are for maintainers.

## Internal vs public docs

Much of `storage/` and all of `util/` is crate-internal. Internal docs are in scope, with the same
rules, but they speak to a maintainer: implementation detail, invariants and safety contracts
belong there and are not "too much detail". For an `unsafe fn` or an `unsafe` block's contract
(`# Safety`), precision beats brevity - do not trim a condition.

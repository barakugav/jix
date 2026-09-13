> **DRAFT with placeholder numbers.** See `report.md` for the tagging convention.

# README snippet

The block destined for the top-level `README.md`, between the feature description and the
"When should I use this library?" section. Four numbers, each a link into the full page.

---

## Benchmarks

Measured single-threaded on an Apple M2 Pro against NumPy 2.3, Blosc2 4.9 and Zarr 3.0 at matched
codec settings (zstd level 3, byte-shuffle). Full methodology and the cases where jix loses are in
[the benchmark report](https://jix.readthedocs.io/en/stable/benchmarks/).

- **Random reads from a compressed array: 38x faster than Blosc2** `[fake]`, at the same
  compression ratio. Reading a 16x70 region out of a 130k x 70 array takes 1.9 us, against 72 us
  for Blosc2 and 401 us for Zarr.
- **Operation chains: 3.7x faster than NumPy and 3x less peak memory** `[fake]`. An eight-step
  elementwise chain over 104 MB runs in 7.1 ms against NumPy's 26.0 ms, in 135 MB of RSS against
  430 MB - NumPy allocates a full intermediate per step, jix allocates none.
- **Integer reductions: 3.2x faster than NumPy** `[fake]`, on ordinary uncompressed arrays. Plain
  elementwise ops match NumPy to within 1%.
- **Compressed arrays stay workable.** Operating directly on a 55x-compressed array runs a full
  reduction at 2.4x NumPy's time using 1/55th the memory `[fake]`.

jix is single-threaded, has no cache of decompressed blocks, and is slower than NumPy on `std` and
on chains dominated by transcendental math. Those are in the report too.

---

## Notes

- Four bullets is the cap. A fifth turns a highlight into a table and nobody reads it.
- Each bullet gives a ratio *and* the two absolute numbers behind it. A bare "3.7x" invites the
  reader to assume it is cherry-picked, which it is - that is what a headline is - so showing the
  working costs nothing and buys trust.
- The closing paragraph is not modesty, it is positioning. It says what jix is not, so readers
  with a threaded throughput workload leave rather than trying it and being disappointed.
- `[fake]` tags come out before this ships. Every number is regenerated from the benchmark JSON.

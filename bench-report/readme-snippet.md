> **DRAFT with placeholder numbers.** See `report.md` for the tagging convention.

# README snippet

The block destined for the top-level `README.md`, between the feature description and
"When should I use this library?". Plots first, numbers second - same rule as the report.

---

## Benchmarks

[![reads and chains](bench-report/plots/readme_summary.png)](https://jix.readthedocs.io/en/stable/benchmarks/)

*Left: time to read a random region from a compressed array. Right: an eight-step elementwise
chain, time and peak memory. Single-threaded, macOS arm64; two more platforms in the full report.*

Measured against NumPy 2.3, Blosc2 4.9 and Zarr 3.0 at matched codec settings (zstd level 3,
byte-shuffle), and against `ndarray` in Rust. Full results, methodology, and the cases where jix
comes out behind are in **[the benchmark report](https://jix.readthedocs.io/en/stable/benchmarks/)**.

- **Random reads from a compressed array are ~38x faster than Blosc2** `[fake]` at the same
  compression ratio - 1.9 us against 72 us for a 16x70 region.
- **Operation chains: 3.7x faster than NumPy in 1/3 the peak memory** `[fake]`. Eight elementwise
  steps over 104 MB run in 7.1 ms against 26.0 ms, in 135 MB of RSS against 430 MB. NumPy allocates
  a full intermediate per step; jix allocates none.
- **Uncompressed arrays are NumPy's speed.** Elementwise ops on a plain in-memory buffer match
  NumPy to within 1%, so there is no cost to entering a jix pipeline.
- **Compressed arrays stay workable.** A full reduction over a 55x-compressed array costs 2.4x
  NumPy's time while holding the data in 1/55th the memory `[fake]`.

jix is single-threaded, keeps no cache of decompressed blocks, and is slower than NumPy on `std`
and on chains dominated by transcendental math. Those are in the report too.

---

## Notes

- One image, then four bullets. A reader who only looks at the picture should still leave with the
  right impression.
- `readme_summary.png` is a two-panel crop of plots that already exist in the report, not a
  separate benchmark. It regenerates from the same JSON.
- Each bullet gives a ratio *and* the two absolute numbers behind it. A bare "3.7x" reads as
  cherry-picked - which a headline is - so showing the working costs nothing and buys trust.
- **No integer-reduction bullet**, even though it is currently the biggest win in the suite
  (3.2x on `sum`). It is very likely an artifact of jix being tuned on arm64 against a NumPy tuned
  harder for x86, and a README claim that inverts on the most common platform is worse than no
  claim. It stays in the report, in a plot, per-platform.
- The closing paragraph is positioning, not modesty: readers with a threaded throughput workload
  should leave now rather than after being disappointed.
- `[fake]` tags come out before this ships. Every number regenerates from the benchmark JSON.

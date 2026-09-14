"""Standalone: decompress a whole array, jix against blosc2, shuffled and not.

Deliberately independent of `benches/` - no report, no plots, no JSON. Build an array, read all of
it back, print a table. For poking at decode throughput when a report number looks wrong.

    python bench-report/debug_whole_array_read.py
    python bench-report/debug_whole_array_read.py --distribution "4 unique" --rounds 7

The interesting column is MB/s of *original* array bytes, which is what decode throughput means
here, and it is not a function of the compression ratio - see the note this script was written for.
"""

import argparse
import time

import blosc2
import numpy as np

import jix

SHAPE = (130_000, 200)
BLOCK = (64, 200)
LEVEL = 3

blosc2.set_nthreads(1)
blosc2.nthreads = 1


def time_once(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def make_data(distribution, shape, seed=0):
    """`random`, `smooth`, or `<N> unique`. Kept here so the script stands alone."""
    rng = np.random.default_rng(seed)
    if distribution == "random":
        out = rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=shape)
    elif distribution == "smooth":
        n = int(np.prod(shape))
        t = np.arange(n) * (2.0 * np.pi / 1000)  # one period per 1000 elements, as benches/data.py
        out = ((np.sin(t) + 0.25 * np.linspace(0.0, 1.0, n)) * 1000.0).reshape(shape)
    elif distribution.endswith("unique"):
        out = rng.integers(0, int(distribution.split()[0]), size=shape)
    else:
        raise SystemExit(f"unknown distribution {distribution!r}")
    return np.ascontiguousarray(out, dtype=np.int32)


def build_jix(data, shuffle):
    filters = ["byte-shuffle"] if shuffle else []
    arr = jix.compact(
        data,
        params={"block_shape": BLOCK, "codec": "zstd", "compression_level": LEVEL, "filters": filters},
    )
    index = tuple(slice(0, dim) for dim in data.shape)
    return arr, (lambda: arr.numpy(index))


def build_blosc2(data, shuffle, match_chunks):
    arr = blosc2.asarray(
        data,
        blocks=BLOCK,
        chunks=BLOCK if match_chunks else None,
        cparams=blosc2.CParams(
            codec=blosc2.Codec.ZSTD,
            clevel=LEVEL,
            filters=[blosc2.Filter.SHUFFLE] if shuffle else [],
            nthreads=1,
        ),
        dparams=blosc2.DParams(nthreads=1),
    )
    return arr, (lambda: np.asarray(arr[:]))


IMPLS = {
    "jix": lambda d: build_jix(d, True),
    "jix-noshuffle": lambda d: build_jix(d, False),
    "blosc2": lambda d: build_blosc2(d, True, False),
    "blosc2-noshuffle": lambda d: build_blosc2(d, False, False),
    "blosc2-chunked": lambda d: build_blosc2(d, True, True),
    "blosc2-chunked-noshuffle": lambda d: build_blosc2(d, False, True),
}


def stored_bytes(name, arr):
    if name.startswith("jix"):
        import io

        buf = io.BytesIO()
        arr.write_to(buf)
        return buf.getbuffer().nbytes
    return int(arr.schunk.cbytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Whole-array read: jix vs blosc2, shuffled and not.")
    parser.add_argument("--distribution", default=None, help="default: run all of them")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--shape", type=int, nargs=2, default=SHAPE)
    args = parser.parse_args(argv)

    shape = tuple(args.shape)
    distributions = [args.distribution] if args.distribution else ["random", "smooth", "16 unique", "4 unique"]
    raw = int(np.prod(shape)) * 4

    for distribution in distributions:
        data = make_data(distribution, shape)
        print(f"\n{distribution}  -  int32 {shape} = {raw / 1e6:.0f} MB, block {BLOCK}, zstd level {LEVEL}")
        print(f"  {'impl':<26}{'ratio':>9}{'stored':>10}{'read':>11}{'MB/s':>10}   {'vs jix':>7}")
        baseline = None
        for name, build in IMPLS.items():
            arr, read = build(data)
            out = read()
            assert np.array_equal(out, data), f"{name} did not round-trip"
            best = min(time_once(read) for _ in range(args.rounds))
            baseline = baseline if baseline is not None else best
            stored = stored_bytes(name, arr)
            print(
                f"  {name:<26}{raw / stored:>8.1f}x{stored / 1e6:>9.1f}M"
                f"{best * 1e3:>10.1f}ms{raw / best / 1e6:>10.0f}   {best / baseline:>6.2f}x"
            )
            del arr


if __name__ == "__main__":
    main()

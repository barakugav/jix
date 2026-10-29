"""Run one chain in a fresh process and report the peak RSS held while it ran.

Run as a subprocess by `test_peak_rss.py`. A separate process per measurement is the only way to
get a clean number: a peak is a high-water mark, so anything measured in the pytest process would
carry every allocation every earlier test made.

The inputs and the output are allocated before the baseline is taken, so the number reported is
the engine's own overhead rather than the cost of the data. A one-step NumPy chain writes straight
into the output and should therefore come out at roughly zero.

Sampling the current RSS in a thread rather than reading `ru_maxrss` is deliberate. `ru_maxrss`
only ever increases, so it would report the transient of building the array - identical for every
library and large enough to bury the thing being measured (on a 2-step chain it hides the chain
entirely and the overhead comes out as zero). Sampling bounds the window to the chain itself.
"""

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put python/ on the path

import numpy as np
import psutil

from benches import report_spec
from benches.array_impls import ARRAY_IMPLS
from benches.conftest import DTYPES
from benches.data import make_data

SAMPLE_SECONDS = 0.001


PROCESS = psutil.Process()


def rss_bytes():
    """Current resident set size of this process, in bytes."""
    return PROCESS.memory_info().rss


class PeakSampler:
    """Context manager recording the highest RSS seen while the block runs."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, rss_bytes())
            time.sleep(SAMPLE_SECONDS)

    def __enter__(self):
        self.peak = rss_bytes()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak = max(self.peak, rss_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure peak RSS of one operation chain.")
    parser.add_argument("--library", required=True, choices=sorted(ARRAY_IMPLS))
    parser.add_argument("--steps", required=True, type=int, help="number of cheap elementwise steps")
    args = parser.parse_args(argv)

    impl = ARRAY_IMPLS[args.library]
    arrays = []
    for seed in (0, 1):
        data = make_data("smooth", report_spec.SHAPE, dtype=DTYPES["f32"], seed=seed)
        arrays.append(impl.from_numpy(data))
        del data  # the source ndarray is not part of what we are measuring

    # Allocate the destination up front and touch every page, so it is resident before the
    # baseline is taken. What is left to measure is then the memory the engine needs on top of the
    # arrays the caller was always going to hold - and a single operation writing straight into
    # `out` should need none, which is the zero this measurement is calibrated against.
    out = np.empty(report_spec.SHAPE, dtype=DTYPES["f32"])
    out.fill(0.0)  # np.zeros leaves the pages unfaulted, so they would land in the measurement
    gc.collect()
    baseline = rss_bytes()

    with PeakSampler() as sampler:
        arrays[0].chain_into(arrays[1], args.steps, out)
    overhead = max(0, sampler.peak - baseline)
    print(
        json.dumps(
            {
                "peak_rss_bytes": overhead,
                "baseline_rss_bytes": baseline,
                "absolute_peak_bytes": sampler.peak,
            }
        )
    )
    return overhead


if __name__ == "__main__":
    main()

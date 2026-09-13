"""Run one chain in a fresh process and report the peak RSS held while it ran.

Run as a subprocess by `test_peak_rss.py`. A separate process per measurement is the only way to
get a clean number: a peak is a high-water mark, so anything measured in the pytest process would
carry every allocation every earlier test made.

Sampling RSS in a thread rather than reading `ru_maxrss` is deliberate. `ru_maxrss` only ever
increases, so it would report the transient of building the array - identical for every library and
large enough to bury the thing being measured. Sampling bounds the window to the chain itself.
"""

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put python/ on the path

from benches import report_spec
from benches.array_impls import ARRAY_IMPLS
from benches.conftest import DTYPES
from benches.data import make_data

SAMPLE_SECONDS = 0.001


def rss_bytes():
    """Resident set size of this process, or None where it cannot be read cheaply."""
    try:
        # statm fields are in pages; the second is resident.
        resident = int(Path("/proc/self/statm").read_text().split()[1])
    except OSError:
        return None
    import resource

    return resident * resource.getpagesize()


class PeakSampler:
    """Context manager recording the highest RSS seen while the block runs."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            current = rss_bytes()
            if current is not None:
                self.peak = max(self.peak, current)
            time.sleep(SAMPLE_SECONDS)

    def __enter__(self):
        self.peak = rss_bytes() or 0
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        current = rss_bytes()
        if current is not None:
            self.peak = max(self.peak, current)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure peak RSS of one operation chain.")
    parser.add_argument("--library", required=True, choices=sorted(ARRAY_IMPLS))
    parser.add_argument("--steps", required=True, type=int, help="number of cheap elementwise steps")
    args = parser.parse_args(argv)

    data = make_data("smooth", report_spec.SHAPE, dtype=DTYPES["f32"], seed=0)
    arr = ARRAY_IMPLS[args.library].from_numpy(data)
    del data  # the source is not part of what we are measuring
    gc.collect()

    with PeakSampler() as sampler:
        out = arr.chain(args.steps)
        assert out is not None
    print(json.dumps({"peak_rss_bytes": sampler.peak}))
    return sampler.peak


if __name__ == "__main__":
    main()

"""Peak memory held while running an operation chain.

Shaped like `test_compress`: an ordinary pytest-benchmark test that records the metric it cares
about into `extra_info`, so the result lands in `python.json` with everything else. What is timed
here is irrelevant - `value_field` tells the report to read the recorded RSS instead.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benches import report_spec
from benches.conftest import record

WORKER = Path(__file__).parent / "peak_rss_worker.py"
LIBRARIES = report_spec.BY_KEY["chain_memory"]["libraries"]
STEPS = [case for case in report_spec.CHAIN_CASES if case != "exp/log"]


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("steps", STEPS, ids=lambda s: f"{s}ops")
def test_peak_rss(benchmark, library, steps):
    def run():
        out = subprocess.check_output(
            [sys.executable, str(WORKER), "--library", library, "--steps", str(steps)], text=True
        )
        return json.loads(out.splitlines()[-1])["peak_rss_bytes"]

    # One round: each is a fresh interpreter, and the number does not get better for repeating it.
    peak = benchmark.pedantic(run, rounds=1, iterations=1)
    record(
        benchmark,
        section="chain_memory",
        case=report_spec.chain_label(steps),
        library=library,
        peak_rss_bytes=peak,
        value_field="peak_rss_bytes",
    )
    assert peak > 0

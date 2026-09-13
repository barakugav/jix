"""Chains of operations: does the cost grow with the length of the chain?

NumPy evaluates eagerly and allocates a full intermediate per step, so its cost is linear in the
chain length. jix builds one lazy view and makes a single pass, so its cost is nearly flat. The
`exp/log` case is the counter-example, where one expensive kernel dominates and the intermediates
being saved stop mattering.
"""

import pytest

from benches import report_spec
from benches.conftest import build, record

SHAPE = report_spec.SHAPE
LIBRARIES = report_spec.BY_KEY["chain"]["libraries"]


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("steps", report_spec.CHAIN_CASES, ids=lambda s: str(s).replace("/", "_"))
def test_chain(benchmark, library, steps):
    arr = build(library, "smooth", "f32", SHAPE)
    run = arr.exp_log if steps == "exp/log" else (lambda: arr.chain(steps))
    record(benchmark, section="chain", case=report_spec.chain_label(steps), library=library)
    assert benchmark(run) is not None

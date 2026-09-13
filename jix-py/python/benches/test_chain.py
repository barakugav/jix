"""Chains of operations: what does fusing a chain actually save?

Every step is a binary op over two source arrays. NumPy re-reads an operand and writes a full
intermediate per step; jix reads both arrays once and keeps the running value in registers. The
`exp/log` case is the counter-example, where one expensive kernel decides the outcome and the
intermediates stop mattering.

No scalar constants anywhere - jix reads a 0-stride operand through the same strided loop as a real
array, so a chain of scalar ops measures that gap instead of this one.
"""

import pytest

from benches import report_spec
from benches.array_impls import CHAIN_READ_SIZE
from benches.conftest import build, record

SHAPE = report_spec.SHAPE
LIBRARIES = report_spec.BY_KEY["chain"]["libraries"]


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("steps", report_spec.CHAIN_CASES, ids=lambda s: str(s).replace("/", "_"))
def test_chain(benchmark, library, steps):
    # A cache-sized read region is what lets the intermediates stay resident; see CHAIN_READ_SIZE.
    # Compact keeps the default - a region smaller than a block decompresses the block anyway.
    read_size = CHAIN_READ_SIZE if library == "jix-plain" else None
    lhs = build(library, "smooth", "f32", SHAPE, None, 0, read_size)
    rhs = build(library, "smooth", "f32", SHAPE, None, 1, read_size)
    run = (lambda: lhs.exp_log(rhs)) if steps == "exp/log" else (lambda: lhs.chain(rhs, steps))
    record(benchmark, section="chain", case=report_spec.chain_label(steps), library=library)
    assert benchmark(run) is not None

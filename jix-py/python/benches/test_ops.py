"""Elementwise operations and reductions, against NumPy."""

import pytest

from benches import report_spec
from benches.conftest import build, record

SHAPE = report_spec.SHAPE


@pytest.mark.parametrize("library", report_spec.CORE)
@pytest.mark.parametrize("dtype", report_spec.DTYPES)
def test_negate(benchmark, library, dtype):
    arr = build(library, "smooth", dtype, SHAPE)
    record(benchmark, section="negate", case=dtype, library=library)
    assert benchmark(arr.negate) is not None


@pytest.mark.parametrize("library", report_spec.CORE)
@pytest.mark.parametrize("dtype", report_spec.DTYPES)
def test_add(benchmark, library, dtype):
    lhs = build(library, "smooth", dtype, SHAPE, None, 0)
    rhs = build(library, "smooth", dtype, SHAPE, None, 1)
    record(benchmark, section="add", case=dtype, library=library)
    assert benchmark(lambda: lhs.add(rhs)) is not None


@pytest.mark.parametrize("library", report_spec.CORE)
@pytest.mark.parametrize("distribution", report_spec.DISTRIBUTIONS)
def test_negate_by_distribution(benchmark, library, distribution):
    """The same operation over data that compresses differently - nothing else changes."""
    arr = build(library, distribution, "f32", SHAPE)
    record(benchmark, section="negate_dist", case=distribution, library=library)
    assert benchmark(arr.negate) is not None


@pytest.mark.parametrize("library", report_spec.CORE)
@pytest.mark.parametrize("case", report_spec.REDUCTION_CASES, ids=lambda c: f"{c[0]}_{c[1]}_axis{c[2]}")
def test_reduction(benchmark, library, case):
    op, dtype, axis = case
    arr = build(library, "smooth", dtype, SHAPE)
    record(benchmark, section="reduction", case=report_spec.reduction_label(*case), library=library)
    assert benchmark(lambda: arr.reduce(op, axis)) is not None

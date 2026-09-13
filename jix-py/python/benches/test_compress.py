"""Compressing the array: how long it takes, and how much it stores.

One run yields both metrics - the timed call produces the array whose stored size is then measured -
so this feeds the throughput plot and the compressed-size plot without compressing twice.
"""

import numpy as np
import pytest

from benches import report_spec
from benches.array_impls import ARRAY_IMPLS
from benches.conftest import DTYPES, record
from benches.data import make_data

SHAPE = report_spec.SHAPE
BLOCK = (64, 200)


@pytest.mark.parametrize("library", report_spec.COMPRESS_LIBRARIES)
@pytest.mark.parametrize("distribution", report_spec.DISTRIBUTIONS)
def test_compress(benchmark, library, distribution):
    data = make_data(distribution, SHAPE, dtype=DTYPES["i32"], seed=0)
    cls = ARRAY_IMPLS[library]
    result = benchmark(lambda: cls.from_numpy(data, block_shape=BLOCK))
    stored = result.stored_bytes()
    record(
        benchmark,
        section="compress",
        case=distribution,
        library=library,
        raw_bytes=int(data.nbytes),
        stored_bytes=stored,
        # The report plots the stored size as a fraction of the raw array, so record the usual
        # raw/stored ratio and let the renderer invert it.
        ratio=data.nbytes / stored,
        also={"compress_ratio": "ratio"},
    )
    assert np.prod(result.raw.shape) == np.prod(SHAPE)

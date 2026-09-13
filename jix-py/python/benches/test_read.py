import itertools

import numpy as np
import pytest

from benches.array_impls import ARRAY_IMPLS
from benches.conftest import record

NCOLS = 70


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize(
    "cfg",
    [
        ((16, NCOLS), (1, NCOLS)),
        ((16, NCOLS), (16, NCOLS)),
        ((64, NCOLS), (16, 16)),
    ],
    ids=lambda cfg: f"b{cfg[0][0]}x{cfg[0][1]}_r{cfg[1][0]}x{cfg[1][1]}",
)
@pytest.mark.parametrize("size", [1000, 5000, 20000, 60000, 130000], ids=lambda n: f"n{n}")
def test_read(benchmark, library, cfg, size):
    from benches.data import make_data

    block, read = cfg
    data = make_data("smooth", (size, NCOLS), dtype=np.int32, seed=0)
    arr = ARRAY_IMPLS[library].from_numpy(data, block_shape=block)
    regions = itertools.cycle(random_regions((size, NCOLS), read, 100_000, seed=1))
    case = f"read_b{block[0]}x{block[1]}_r{read[0]}x{read[1]}"
    record(benchmark, case=case, library=library, size=size)
    out = benchmark(lambda: arr.read(next(regions)))
    assert out is not None


def random_regions(shape, read_shape, count, seed):
    rng = np.random.default_rng(seed)
    sizes = [min(rs, dim) for dim, rs in zip(shape, read_shape)]
    starts = [rng.integers(0, dim - rs + 1, size=count) for dim, rs in zip(shape, sizes)]
    return [tuple(slice(int(s[i]), int(s[i]) + rs) for s, rs in zip(starts, sizes)) for i in range(count)]

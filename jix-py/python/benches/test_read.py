"""Reading a random region out of a compressed array - the data-loader pattern."""

import itertools

import numpy as np
import pytest

from benches import report_spec
from benches.conftest import build, record

SHAPE = report_spec.SHAPE

# Number of distinct regions cycled through during a run. Large enough that the working set
# exceeds cache - reading one fixed region over and over measures a warm read, not a random one.
NREGIONS = 1024


@pytest.mark.parametrize("library", report_spec.READ_LIBRARIES)
@pytest.mark.parametrize("case", report_spec.READ_CASES, ids=lambda c: report_spec.read_label(*c).replace("\n", " "))
def test_read(benchmark, library, case):
    block, read = case
    arr = build(library, "smooth", "i32", SHAPE, block)
    # Cycle through pre-generated regions so each timed call reads a different part of the array.
    # itertools.cycle is a C-level iterator; its per-call cost is negligible next to a read.
    regions = itertools.cycle(random_regions(SHAPE, read, NREGIONS, seed=1))
    record(benchmark, section="read", case=report_spec.read_label(*case), library=library, nregions=NREGIONS)
    out = benchmark(lambda: arr.read(next(regions)))
    assert out is not None


def random_regions(shape, read_shape, count, seed):
    """Return `count` random regions of `read_shape`, each a tuple of slices into `shape`.

    A `read_shape` of None means the whole array, so there is only one region to return.
    """
    if read_shape is None:
        return [tuple(slice(0, dim) for dim in shape)]
    rng = np.random.default_rng(seed)
    sizes = [min(want, dim) for dim, want in zip(shape, read_shape)]
    starts = [rng.integers(0, dim - size + 1, size=count) for dim, size in zip(shape, sizes)]
    return [
        tuple(slice(int(start[i]), int(start[i]) + size) for start, size in zip(starts, sizes)) for i in range(count)
    ]

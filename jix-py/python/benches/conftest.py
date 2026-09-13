import functools

import numpy as np

from benches.array_impls import ARRAY_IMPLS
from benches.data import make_data

DTYPES = {"f32": np.float32, "i32": np.int32}


def record(benchmark, *, section, case, library, **extra):
    """Attach the fields the report groups by to this benchmark's JSON entry.

    `section` and `case` come from `report_spec`, so a plot's x axis and the configuration that
    produced it cannot drift apart. Extra keyword fields are stashed alongside:

    - `value_field="<name>"` makes the report read that field instead of the measured time, for
      benchmarks whose point is something other than how long they took.
    - `also={"<section>": "<field>"}` emits an additional row under another section, for one run
      that yields two metrics - compression produces both a time and a stored size.
    """
    benchmark.extra_info.update(section=section, case=case, library=library, **extra)


@functools.lru_cache(maxsize=8)
def build(library, distribution, dtype, shape, block_shape=None, seed=0):
    """Build one library's array. Cached, because a single array usually feeds several cases.

    `shape` and `block_shape` must be tuples - this is an lru_cache key.
    """
    data = make_data(distribution, shape, dtype=DTYPES[dtype], seed=seed)
    return ARRAY_IMPLS[library].from_numpy(data, block_shape=block_shape)

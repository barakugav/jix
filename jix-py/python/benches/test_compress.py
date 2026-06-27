import io

import blosc2
import numpy as np
import pytest
import zarr

import jix
from benches.array_impls import ARRAY_IMPLS
from benches.conftest import record
from benches.data import PROFILES

NCOLS = 64
BLOCK = (32, NCOLS)


@pytest.mark.parametrize("library", [name for name in ARRAY_IMPLS if name not in ("numpy", "jix-plain")])
@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("size", [256, 1024, 4096], ids=lambda n: f"n{n}")
def test_compress(benchmark, library, profile, size):
    from benches.data import make_data

    data = make_data(profile, (size, NCOLS), dtype=np.int32, seed=0)
    cls = ARRAY_IMPLS[library]
    result = benchmark(lambda: cls.from_numpy(data, block_shape=BLOCK))
    stored = stored_bytes(result.raw)
    case = f"compress_{profile}_b{BLOCK[0]}x{BLOCK[1]}"
    record(
        benchmark,
        case=case,
        library=library,
        size=size,
        raw_bytes=int(data.nbytes),
        stored_bytes=stored,
        ratio=data.nbytes / stored,
    )
    assert result is not None


def stored_bytes(raw):
    match raw:
        case jix.Array():
            buf = io.BytesIO()
            raw.write_to(buf)
            return buf.getbuffer().nbytes
        case np.ndarray():
            return int(raw.nbytes)
        case blosc2.NDArray():
            return int(raw.schunk.cbytes)
        case zarr.Array():
            return int(raw.nbytes_stored())
        case _:
            raise ValueError(f"cannot measure stored bytes of {type(raw).__name__}")

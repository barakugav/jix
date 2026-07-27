import blosc2
import numpy as np
import pytest
import zarr

import jix
from benches.array_impls import ARRAY_IMPLS
from benches.data import make_data
from benches.test_compress import stored_bytes

BLOCK = (4, 8)
SHAPE = (16, 8)
BASE_LIBS = ["jix", "jix-plain", "numpy", "blosc2", "zarr"]


@pytest.mark.parametrize("name", BASE_LIBS)
def test_roundtrip_full(name):
    data = make_data("smooth", SHAPE, dtype=np.int32, seed=0)
    arr = ARRAY_IMPLS[name].from_numpy(data, block_shape=BLOCK)
    region = (slice(0, SHAPE[0]), slice(0, SHAPE[1]))
    out = arr.read(region)
    assert np.allclose(out, data)
    assert out.dtype == data.dtype


@pytest.mark.parametrize("name", BASE_LIBS)
def test_read_subregion(name):
    data = make_data("random", SHAPE, dtype=np.int32, seed=2)
    arr = ARRAY_IMPLS[name].from_numpy(data, block_shape=BLOCK)
    region = (slice(2, 6), slice(1, 5))
    assert np.allclose(arr.read(region), data[region])


@pytest.mark.parametrize("name", BASE_LIBS)
def test_stored_bytes_positive(name):
    data = make_data("smooth", SHAPE, seed=0)
    assert stored_bytes(ARRAY_IMPLS[name].from_numpy(data, block_shape=BLOCK).raw) > 0


@pytest.mark.parametrize("name", ["numpy"])
def test_uncompressed_ratio_is_one(name):
    # Only numpy has a type-derived stored size equal to raw; a plain jix array is the same
    # type as a compact one, so stored_bytes() cannot distinguish it and it is not measured.
    data = make_data("smooth", SHAPE, seed=0)
    assert stored_bytes(ARRAY_IMPLS[name].from_numpy(data, block_shape=BLOCK).raw) == data.nbytes


@pytest.mark.parametrize("name", ["jix", "blosc2", "zarr"])
def test_low_entropy_compresses(name):
    data = make_data("low_entropy", (256, 64), dtype=np.int32, seed=0)
    arr = ARRAY_IMPLS[name].from_numpy(data, block_shape=(32, 64))
    assert stored_bytes(arr.raw) < data.nbytes


def _run(op, arr_a, arr_b, axis):
    """Evaluate `op` on a built array, returning a NumPy array.

    Mirrors the per-engine expressions benchmarked in test_ops.py (dispatched by the raw array
    type) so this cross-checks that every backend agrees with NumPy on the same logical
    operation. `arr_b` is only used by `add`; `axis` only by `sum`/`std`.
    """
    a = arr_a.raw
    b = None if arr_b is None else arr_b.raw
    match op:
        case "negate":
            match a:
                case jix.Array():
                    return np.asarray((-a).numpy())
                case np.ndarray():
                    return -a
                case blosc2.NDArray():
                    return np.asarray((-a)[:])
                case zarr.Array():
                    return -np.asarray(a[:])
        case "add":
            match a:
                case jix.Array():
                    return np.asarray((a + b).numpy())
                case np.ndarray():
                    return a + b
                case blosc2.NDArray():
                    return np.asarray((a + b)[:])
                case zarr.Array():
                    return np.asarray(a[:]) + np.asarray(b[:])
        case "sum":
            match a:
                case jix.Array():
                    return np.asarray(a.sum(axis=axis).numpy())
                case np.ndarray():
                    return np.asarray(a.sum(axis=axis))
                case blosc2.NDArray():
                    return np.asarray(blosc2.sum(a, axis=axis))
                case zarr.Array():
                    return np.asarray(np.asarray(a[:]).sum(axis=axis))
        case "std":
            match a:
                case jix.Array():
                    return np.asarray(a.std(axis=axis).numpy())
                case np.ndarray():
                    return np.asarray(a.std(axis=axis))
                case blosc2.NDArray():
                    return np.asarray(blosc2.std(a, axis=axis))
                case zarr.Array():
                    return np.asarray(np.asarray(a[:]).std(axis=axis))
        case "pipeline_elementwise":
            match a:
                case jix.Array():
                    return np.asarray((a.exp() * 0.5 + 1.0).log().numpy())
                case np.ndarray():
                    return np.log(np.exp(a) * 0.5 + 1.0)
                case blosc2.NDArray():
                    return np.asarray(blosc2.log(blosc2.exp(a) * 0.5 + 1.0)[:])
                case zarr.Array():
                    z = np.asarray(a[:])
                    return np.log(np.exp(z) * 0.5 + 1.0)
        case "pipeline_reduction":
            match a:
                case jix.Array():
                    return np.asarray(a.exp().sum(axis=0).numpy())
                case np.ndarray():
                    return np.asarray(np.exp(a).sum(axis=0))
                case blosc2.NDArray():
                    return np.asarray(blosc2.sum(blosc2.exp(a), axis=0))
                case zarr.Array():
                    z = np.asarray(a[:])
                    return np.asarray(np.exp(z).sum(axis=0))
    raise ValueError(f"unhandled op {op!r} for array type {type(a).__name__}")


AGREE_OPS = [
    ("negate", None),
    ("add", None),
    ("pipeline_elementwise", None),
    ("pipeline_reduction", None),
    *[("sum", axis) for axis in (0, 1, None)],
    *[("std", axis) for axis in (0, 1, None)],
]


@pytest.mark.parametrize("op,axis", AGREE_OPS)
def test_ops_agree_across_libraries(op, axis):
    data_a = make_data("smooth", (32, 16), dtype=np.float64, seed=0)
    data_b = make_data("smooth", (32, 16), dtype=np.float64, seed=1)
    arrs_a = {n: ARRAY_IMPLS[n].from_numpy(data_a, block_shape=(8, 16)) for n in BASE_LIBS}
    arrs_b = {n: ARRAY_IMPLS[n].from_numpy(data_b, block_shape=(8, 16)) for n in BASE_LIBS}
    ref = _run(op, arrs_a["numpy"], arrs_b["numpy"], axis)
    for lib in ("jix", "jix-plain", "blosc2", "zarr"):
        got = _run(op, arrs_a[lib], arrs_b[lib], axis)
        assert np.allclose(got, ref), f"{lib} disagrees on op {op} (axis={axis})"

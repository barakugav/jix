import blosc2
import numpy as np
import pytest
import zarr

import jix
from benches.array_impls import ARRAY_IMPLS
from benches.conftest import record

COLS = 200
SIZES = [500, 2000, 8000, 30_000, 130_000]


def _build(library, size, dtype=np.float32, seed=0):
    """Build one library's array from smooth float64 data of shape (size, COLS)."""
    from benches.data import make_data

    data = make_data("smooth", (size, COLS), dtype=dtype, seed=seed).astype(dtype)
    return ARRAY_IMPLS[library].from_numpy(data)


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize("size", SIZES, ids=lambda n: f"n{n}")
def test_negate(benchmark, library, size):
    arr = _build(library, size)

    def run():
        """Unary negate `-a`, returning a NumPy array."""
        a = arr.raw
        match a:
            case jix.Array():
                return np.asarray((-a).numpy())
            case np.ndarray():
                return -a
            case blosc2.NDArray():
                return np.asarray((-a)[:])
            case zarr.Array():
                return -np.asarray(a[:])
            case _:
                raise ValueError(f"unknown library {library!r}")

    record(benchmark, case="negate", library=library, size=size)
    out = benchmark(run)
    assert out is not None


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize("size", SIZES, ids=lambda n: f"n{n}")
def test_add(benchmark, library, size):
    arr_a = _build(library, size, seed=0)
    arr_b = _build(library, size, seed=1)

    def run():
        """Binary elementwise add of two independent arrays, returning a NumPy array."""
        a, b = arr_a.raw, arr_b.raw
        match a:
            case jix.Array():
                return np.asarray((a + b).numpy())
            case np.ndarray():
                return a + b
            case blosc2.NDArray():
                return np.asarray((a + b)[:])
            case zarr.Array():
                return np.asarray(a[:]) + np.asarray(b[:])
            case _:
                raise ValueError(f"unknown library {library!r}")

    record(benchmark, case="add", library=library, size=size)
    out = benchmark(run)
    assert out is not None


SUM_DTYPES = [np.int32, np.float32]


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize("axis", [0, 1, None], ids=lambda a: "axisall" if a is None else f"axis{a}")
@pytest.mark.parametrize("dtype", SUM_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("size", SIZES, ids=lambda n: f"n{n}")
def test_sum(benchmark, library, dtype, axis, size):
    arr = _build(library, size, dtype=dtype)

    def run():
        """Sum reduction over `axis`, returning a NumPy array."""
        a = arr.raw
        match a:
            case jix.Array():
                return np.asarray(a.sum(axis=axis).numpy())
            case np.ndarray():
                return np.asarray(a.sum(axis=axis))
            case blosc2.NDArray():
                return np.asarray(blosc2.sum(a, axis=axis))
            case zarr.Array():
                return np.asarray(np.asarray(a[:]).sum(axis=axis))
            case _:
                raise ValueError(f"unknown library {library!r}")

    axis_str = "all" if axis is None else str(axis)
    dtype_str = np.dtype(dtype).name
    case = f"sum_{dtype_str}_axis{axis_str}"
    record(benchmark, case=case, library=library, size=size, dtype=dtype_str)
    out = benchmark(run)
    assert out is not None


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize("axis", [0, 1, None], ids=lambda a: "axisall" if a is None else f"axis{a}")
@pytest.mark.parametrize("size", SIZES, ids=lambda n: f"n{n}")
def test_std(benchmark, library, axis, size):
    arr = _build(library, size)

    def run():
        """Standard-deviation reduction over `axis`, returning a NumPy array."""
        a = arr.raw
        match a:
            case jix.Array():
                return np.asarray(a.std(axis=axis).numpy())
            case np.ndarray():
                return np.asarray(a.std(axis=axis))
            case blosc2.NDArray():
                return np.asarray(blosc2.std(a, axis=axis))
            case zarr.Array():
                return np.asarray(np.asarray(a[:]).std(axis=axis))
            case _:
                raise ValueError(f"unknown library {library!r}")

    axis_str = "all" if axis is None else str(axis)
    case = f"std_axis{axis_str}"
    record(benchmark, case=case, library=library, size=size)
    out = benchmark(run)
    assert out is not None


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize("size", SIZES, ids=lambda n: f"n{n}")
def test_elementwise_pipeline(benchmark, library, size):
    arr = _build(library, size)

    def run():
        """Multi-op elementwise chain `(exp(a) * 0.5 + 1).log()`, returning a NumPy array."""
        a = arr.raw
        match a:
            case jix.Array():
                return np.asarray((a.exp() * 0.5 + 1.0).log().numpy())
            case np.ndarray():
                return np.log(np.exp(a) * 0.5 + 1.0)
            case blosc2.NDArray():
                return np.asarray(blosc2.log(blosc2.exp(a) * 0.5 + 1.0)[:])
            case zarr.Array():
                a = np.asarray(a[:])
                return np.log(np.exp(a) * 0.5 + 1.0)
            case _:
                raise ValueError(f"unknown library {library!r}")

    record(benchmark, case="pipeline_elementwise", library=library, size=size)
    out = benchmark(run)
    assert out is not None


@pytest.mark.parametrize("library", ARRAY_IMPLS)
@pytest.mark.parametrize("size", SIZES, ids=lambda n: f"n{n}")
def test_reduction_pipeline(benchmark, library, size):
    arr = _build(library, size)

    def run():
        """Multi-op reduction chain `exp(a).sum(axis=0)`, returning a NumPy array."""
        a = arr.raw
        match a:
            case jix.Array():
                return np.asarray(a.exp().sum(axis=0).numpy())
            case np.ndarray():
                return np.asarray(np.exp(a).sum(axis=0))
            case blosc2.NDArray():
                return np.asarray(blosc2.sum(blosc2.exp(a), axis=0))
            case zarr.Array():
                a = np.asarray(a[:])
                return np.asarray(np.exp(a).sum(axis=0))
            case _:
                raise ValueError(f"unknown library {library!r}")

    record(benchmark, case="pipeline_reduction", library=library, size=size)
    out = benchmark(run)
    assert out is not None

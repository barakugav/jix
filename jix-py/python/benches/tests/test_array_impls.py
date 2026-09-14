import numpy as np
import pytest

from benches import report_spec
from benches.array_impls import ARRAY_IMPLS, chain_steps
from benches.data import make_data

SHAPE = (64, 32)
BLOCK = (16, 32)
# Every library the report plots, in any section.
LIBRARIES = sorted(set(report_spec.READ_LIBRARIES) | set(report_spec.COMPRESS_LIBRARIES) | set(report_spec.CORE))


def build(name, distribution="smooth", dtype=np.float32, seed=0):
    data = make_data(distribution, SHAPE, dtype=dtype, seed=seed)
    return data, ARRAY_IMPLS[name].from_numpy(data, block_shape=BLOCK)


@pytest.mark.parametrize("name", LIBRARIES)
def test_roundtrip_full(name):
    data, arr = build(name, dtype=np.int32)
    region = tuple(slice(0, dim) for dim in SHAPE)
    out = arr.read(region)
    assert np.array_equal(out, data)
    assert out.dtype == data.dtype


@pytest.mark.parametrize("name", LIBRARIES)
def test_read_subregion(name):
    data, arr = build(name, distribution="random", dtype=np.int32, seed=2)
    region = (slice(2, 20), slice(1, 5))
    assert np.array_equal(arr.read(region), data[region])


@pytest.mark.parametrize("name", LIBRARIES)
def test_stored_bytes_positive(name):
    _, arr = build(name, dtype=np.int32)
    assert arr.stored_bytes() > 0


@pytest.mark.parametrize("name", [n for n in LIBRARIES if n not in ("numpy", "jix-plain")])
def test_compressible_data_actually_compresses(name):
    data, arr = build(name, distribution="4 unique", dtype=np.int32)
    assert arr.stored_bytes() < data.nbytes


@pytest.mark.parametrize("name", ["numpy", "jix-plain"])
def test_uncompressed_stored_size_is_raw_size(name):
    data, arr = build(name, dtype=np.int32)
    assert arr.stored_bytes() == data.nbytes


# Every operation the report measures, as (method name, args). This is the cross-check that the
# libraries are being asked to do the same work - a benchmark where one arm quietly does less is
# worse than no benchmark.
UNARY_OPERATIONS = [
    ("negate", ()),
    *[("reduce", (op, axis)) for op in ("sum", "std") for axis in (0, 1, None)],
]
BINARY_OPERATIONS = [
    ("add", ()),
    *[("chain", (count,)) for count in (1, 2, 4, 8, 16)],
    ("exp_log", ()),
]


@pytest.mark.parametrize("method,args", UNARY_OPERATIONS, ids=lambda v: str(v))
@pytest.mark.parametrize("name", [n for n in LIBRARIES if n != "numpy"])
def test_operations_agree_with_numpy(name, method, args):
    _, arr = build(name, dtype=np.float32)
    _, reference = build("numpy", dtype=np.float32)
    got = getattr(arr, method)(*args)
    want = getattr(reference, method)(*args)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4), f"{name} disagrees with numpy on {method}{args}"


@pytest.mark.parametrize("method,args", BINARY_OPERATIONS, ids=lambda v: str(v))
@pytest.mark.parametrize("name", [n for n in LIBRARIES if n != "numpy"])
def test_binary_operations_agree_with_numpy(name, method, args):
    lhs, rhs = build(name, seed=0)[1], build(name, seed=1)[1]
    ref_l, ref_r = build("numpy", seed=0)[1], build("numpy", seed=1)[1]
    got = getattr(lhs, method)(rhs, *args)
    want = getattr(ref_l, method)(ref_r, *args)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4), f"{name} disagrees with numpy on {method}{args}"


def test_chain_steps_cycle():
    assert len(chain_steps(8)) == 8
    assert chain_steps(8)[:4] == chain_steps(4)
    assert chain_steps(8)[4:] == chain_steps(4)


def test_chain_uses_no_scalar_operands():
    """A scalar step would measure jix's 0-stride loop, not whether fusing a chain pays off."""
    a, b = np.full((4, 3), 2.0), np.full((4, 3), 3.0)
    for _, step, _, _ in chain_steps(len(chain_steps(4))):
        # Every step must accept two arrays and use them; a scalar-constant step would not.
        assert step(a, a, b).shape == a.shape

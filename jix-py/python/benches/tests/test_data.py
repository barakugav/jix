import numpy as np
import pytest

from benches.data import DISTRIBUTIONS, make_data


@pytest.mark.parametrize("distribution", list(DISTRIBUTIONS))
def test_deterministic_and_shaped(distribution):
    a = make_data(distribution, (16, 8), dtype=np.int32, seed=0)
    b = make_data(distribution, (16, 8), dtype=np.int32, seed=0)
    assert a.shape == (16, 8)
    assert a.dtype == np.int32
    assert np.array_equal(a, b)


@pytest.mark.parametrize("distribution", list(DISTRIBUTIONS))
def test_distinct_seeds_differ(distribution):
    a = make_data(distribution, (16, 8), seed=0)
    b = make_data(distribution, (16, 8), seed=1)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("count", [4, 16])
def test_unique_distributions_have_exactly_that_many_values(count):
    a = make_data(f"{count} unique", (64, 64), seed=0)
    assert np.unique(a).size == count


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        make_data("nonsense", (4, 4))

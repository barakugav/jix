import numpy as np
import pytest

from benches.data import PROFILES, make_data


@pytest.mark.parametrize("profile", list(PROFILES))
def test_deterministic_and_shaped(profile):
    a = make_data(profile, (16, 8), dtype=np.int32, seed=0)
    b = make_data(profile, (16, 8), dtype=np.int32, seed=0)
    assert a.shape == (16, 8)
    assert a.dtype == np.int32
    assert np.array_equal(a, b)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_distinct_seeds_differ(profile):
    a = make_data(profile, (16, 8), seed=0)
    b = make_data(profile, (16, 8), seed=1)
    assert not np.array_equal(a, b)


def test_low_entropy_has_few_values():
    a = make_data("low_entropy", (64, 64), seed=0)
    assert np.unique(a).size <= 8


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        make_data("nope", (4, 4))

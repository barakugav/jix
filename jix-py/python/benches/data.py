import numpy as np

# Elements per sine period for the `smooth` distribution.
SMOOTH_PERIOD = 1000

# The distributions the report varies, keyed by the label report_spec uses on the x axis.
DISTRIBUTIONS = {
    "random": "uniform over the whole dtype range; incompressible",
    "smooth": "sine plus gradient, one period per 1000 elements; moderately compressible",
    "16 unique": "16 distinct values; highly compressible",
    "4 unique": "4 distinct values; extremely compressible",
}


def make_data(distribution: str, shape: tuple[int, ...], dtype=np.int32, seed: int = 0) -> np.ndarray:
    """Return a deterministic synthetic array of the given distribution, shape and dtype.

    `N unique` distributions produce exactly N distinct values, which is what makes compression
    ratio a controlled variable rather than an accident of the generator.
    """
    rng = np.random.default_rng(seed)
    is_int = np.issubdtype(np.dtype(dtype), np.integer)
    if distribution == "random":
        if is_int:
            info = np.iinfo(np.int32)
            out = rng.integers(info.min, info.max, size=shape)
        else:
            out = rng.random(shape)
    elif distribution == "smooth":
        n = int(np.prod(shape))
        phase = rng.random() * 2.0 * np.pi
        # One period per SMOOTH_PERIOD elements, not per array. Scaling the period with the array
        # made compressibility a function of array size: at 26M elements the old generator fitted
        # four periods across the whole thing, so consecutive values were almost always equal after
        # rounding and it compressed 769x - runs, not the moderate redundancy this is meant to be.
        t = np.arange(n) * (2.0 * np.pi / SMOOTH_PERIOD)
        field = np.sin(t + phase) + 0.25 * np.linspace(0.0, 1.0, n)
        # scale to a moderate integer amplitude so it stays smooth but not near-constant
        out = (field * 1000.0 if is_int else field).reshape(shape)
    elif distribution.endswith("unique"):
        count = int(distribution.split()[0])
        out = rng.integers(0, count, size=shape)
        if not is_int:
            out = out.astype(np.float64) * 0.5
    else:
        raise ValueError(f"unknown distribution: {distribution!r}")
    return np.ascontiguousarray(out, dtype=dtype)

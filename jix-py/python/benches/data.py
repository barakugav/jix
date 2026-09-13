import numpy as np

# The distributions the report varies, keyed by the label report_spec uses on the x axis.
DISTRIBUTIONS = {
    "random": "uniform over the whole dtype range; incompressible",
    "smooth": "sine plus gradient; moderately compressible",
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
        t = np.linspace(0.0, 8.0 * np.pi, n)
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

import numpy as np

PROFILES = {
    "random": "uniform random; near-incompressible baseline",
    "smooth": "structured field (sine + gradient); moderately compressible",
    "low_entropy": "quantized/repetitive; highly compressible",
}


def make_data(profile: str, shape: tuple[int, ...], dtype=np.int32, seed: int = 0) -> np.ndarray:
    """Return a deterministic synthetic array of the given profile, shape, and dtype.

    The benchmarks use ``int32`` data; the generators produce integer-valued fields for
    integer dtypes. A float path is kept so float dtypes (used by the compute pipeline
    benchmark) still get sensible, safely-bounded values.
    """
    rng = np.random.default_rng(seed)
    is_int = np.issubdtype(dtype, np.integer)
    if profile == "random":
        if is_int:
            out = rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=shape)
        else:
            out = rng.random(shape)
    elif profile == "smooth":
        n = int(np.prod(shape))
        phase = rng.random() * 2.0 * np.pi
        t = np.linspace(0.0, 8.0 * np.pi, n)
        field = np.sin(t + phase) + 0.25 * np.linspace(0.0, 1.0, n)
        # scale to a moderate integer amplitude so it stays smooth but not near-constant
        out = (field * 1000.0 if is_int else field).reshape(shape)
    elif profile == "low_entropy":
        out = rng.integers(0, 4, size=shape)
        if not is_int:
            out = out.astype(np.float64) * 0.5
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return np.ascontiguousarray(out, dtype=dtype)

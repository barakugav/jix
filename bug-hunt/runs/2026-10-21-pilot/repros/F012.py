"""F012: block_shape / block_shape_fixed_dims with > 8 entries raise PanicException, not ValueError."""

import numpy as np

import jix

a = np.zeros((3, 5), np.float32)

cases = {
    "block_shape [1]*9": {"block_shape": [1] * 9},
    "block_shape_fixed_dims [True]*9": {"block_shape": [1, 1], "block_shape_fixed_dims": [True] * 9},
    "block_shape [1]*3 (contrast, wrong length <= 8)": {"block_shape": [1] * 3},
    "read_size 3 elems (contrast)": {"read_size": (1, 2, 3)},
}
for name, params in cases.items():
    try:
        jix.compact(a, params=params)
        print(f"{name}: no error")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: caught by `except Exception`: {type(e).__name__}: {e}")
    except BaseException as e:  # noqa: BLE001
        print(f"{name}: NOT an Exception subclass: {type(e).__module__}.{type(e).__name__}: {e}")

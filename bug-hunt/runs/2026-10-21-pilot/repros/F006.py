"""F006: ArrayParams::tune divides by zero for a 0 fixed block dim when another dim is auto-scaled."""

import numpy as np

import jix

CASES = [
    ("all fixed (reference)", (3, 5), {"block_shape": [0, 4]}),
    ("fixed [T, F]", (3, 5), {"block_shape": [0, 4], "block_shape_fixed_dims": [True, False]}),
    ("fixed [F, T]", (8, 4), {"block_shape": [4, 0], "block_shape_fixed_dims": [False, True]}),
    (
        "overflow [2^31, 2^31, 4, 1]",
        (3, 3, 3, 5),
        {"block_shape": [1 << 31, 1 << 31, 4, 1], "block_shape_fixed_dims": [True, True, True, False]},
    ),
]

for name, shape, params in CASES:
    x = np.zeros(shape, np.float32)
    try:
        jix.compact(x, params=params)
        print(f"{name}: Ok")
    except BaseException as e:  # PanicException derives from BaseException
        print(f"{name}: {type(e).__module__}.{type(e).__name__}: {e}")

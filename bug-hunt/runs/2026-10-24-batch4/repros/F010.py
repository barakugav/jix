"""F010: jix.clamp on an all-bool input with BOTH bounds returns uint8, while one bound,
jix.maximum/minimum and numpy.clip keep bool. Asserts the correct behavior (dtype bool), so it
fails while the bug exists."""

import jix
import numpy as np

a_np = np.array([True, False, True])
b = jix.compact(a_np)

failures = []


def check(label, got, expected):
    g = got.numpy()
    print(f"{label}: dtype={g.dtype} values={g.tolist()}  (expected dtype={expected.dtype} values={expected.tolist()})")
    if g.dtype != expected.dtype or not np.array_equal(g, expected):
        failures.append(label)


# Reference behaviors that already keep bool.
check("clamp(b, min=False)", jix.clamp(b, min=False), np.clip(a_np, False, None))
check("clamp(b, max=True)", jix.clamp(b, max=True), np.clip(a_np, None, True))
check("maximum(b, False)", jix.maximum(b, False), np.maximum(a_np, False))
check("minimum(maximum(b, False), True)", jix.minimum(jix.maximum(b, False), True), np.clip(a_np, False, True))

# The buggy path: both bounds.
check("clamp(b, min=False, max=True)", jix.clamp(b, min=False, max=True), np.clip(a_np, False, True))
lo = np.array([False, True, False])
hi = np.array([True, True, False])
check("clamp(b, min=bool_arr, max=bool_arr)", jix.clamp(b, min=lo, max=hi), np.clip(a_np, lo, hi))
check(
    "clamp(b, min=jix bool, max=jix bool)",
    jix.clamp(b, min=jix.compact(lo), max=jix.compact(hi)),
    np.clip(a_np, lo, hi),
)

print("FAILURES:", failures)
assert not failures, f"clamp(bool, min, max) does not keep dtype bool: {failures}"

"""F036: f32/f16 base ** int32 exponent loses the parity of odd exponents above 2**24.

The f32 and f16 kernels evaluate `powf(b as f32)`, and 2**24 + 1 rounds to 2**24 (even), so a
negative base gives a result with the wrong sign. The f64 ** int32 kernel is exact and serves as
the reference. Asserts the CORRECT behavior, so it fails while the bug exists.
"""

import numpy as np

import jix

N = 2**24 + 1  # odd; float32(N) == 2**24 (even)
base = np.array([-1.0, -0.5, -2.0, 2.0], dtype=np.float64)
exp = np.full(base.shape, N, dtype=np.int32)

with jix.errstate(all="ignore"):
    ref = jix.power(jix.compact(base), jix.compact(exp)).numpy()
print("jix f64 ** int32 (reference):", ref.tolist())
with np.errstate(all="ignore"):
    print("numpy f32 ** int32:          ", np.power(base.astype(np.float32), exp).tolist())

failed = []
for dt in (np.float32, np.float16):
    for label, e in (("int32 array", jix.compact(exp)), ("np.int32 scalar", np.int32(N))):
        with jix.errstate(all="ignore"):
            r = jix.power(jix.compact(base.astype(dt)), e)
            got = r.numpy()
        want = ref.astype(dt)
        print(f"jix {np.dtype(dt).name} ** {label}: dtype={r.dtype} got={got.tolist()} want={want.tolist()}")
        ok = np.array_equal(got, want) and np.array_equal(np.signbit(got), np.signbit(want))
        if not ok:
            failed.append(f"{np.dtype(dt).name} ** {label}")

assert not failed, f"wrong sign for odd exponent {N}: {failed}"
print("OK")

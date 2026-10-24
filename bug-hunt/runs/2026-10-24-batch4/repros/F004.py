"""F004 repro: jix.flatten / reshape of a split or merge issue one inner read per element.

Each inner read of a Compact array decompresses a whole block, so reading the view costs
N x block_bytes instead of one pass over the array. Asserts the correct (fast) behavior, so it
fails while the bug exists. Run: .venv/bin/python bug-hunt/runs/2026-10-24-batch4/repros/F004.py
"""

import time

import numpy as np

import jix


def best_ms(fn, n):
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


src = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
a = jix.compact(src)
flat = jix.flatten(a)
split = jix.reshape(jix.compact(src.reshape(-1)), (64, 64))

# Results are correct.
assert np.array_equal(flat.numpy(), src.reshape(-1))
assert np.array_equal(split.numpy(), src)

base = best_ms(lambda: a.numpy(), 5)
t_flat = best_ms(lambda: flat.numpy(), 2)
t_split = best_ms(lambda: split.numpy(), 2)
print(f"64x64 f32: numpy() {base:.2f} ms, flatten().numpy() {t_flat:.1f} ms ({t_flat / base:.0f}x), "
      f"reshape([4096]->(64,64)).numpy() {t_split:.1f} ms ({t_split / base:.0f}x)")
assert t_flat < 50 * base, f"flatten read is {t_flat / base:.0f}x slower than reading the array"
assert t_split < 50 * base, f"reshape read is {t_split / base:.0f}x slower than reading the array"
print("OK")

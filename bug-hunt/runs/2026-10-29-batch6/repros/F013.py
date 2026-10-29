# F013 repro (run 2026-10-29-batch6): a Python scalar or numpy operand overrides a compact
# operand's explicit block shape, so compact() of `a * 2.0` / `a + ndarray` / where / clamp
# re-blocks the user's column blocks [1024, 1] as row blocks [1, 1024].
#
# Python exposes no block-shape getter, so the layout is observed through the cost of reading one
# column: on a column-blocked array it touches 1 block; on a row-blocked one it decodes all 1024.
# Asserts the CORRECT behavior (the explicit layout survives, as it does for `a + a`), so it
# FAILS while the bug exists.
import time

import numpy as np

import jix

n = 1024
src = np.arange(n * n, dtype=np.float32).reshape(n, n)
a = jix.compact(src, params={"block_shape": [n, 1]})  # column blocks, explicit -> all fixed


def col_read_time(x, reps=20):
    np.asarray(x[:, 7])  # warm up
    t = time.perf_counter()
    for _ in range(reps):
        np.asarray(x[:, 7])
    return (time.perf_counter() - t) / reps


cases = {
    "a": a,
    "(a + a).compact()": (a + a).compact(),
    "(a * 2.0).compact()": (a * 2.0).compact(),
    "(a + ndarray).compact()": (a + np.ones((n, n), np.float32)).compact(),
    "jix.where(a > 5, a, 0.0).compact()": jix.where(a > 5, a, 0.0).compact(),
}
assert np.array_equal(cases["(a * 2.0).compact()"].numpy(), src * 2.0)

times = {name: col_read_time(x) for name, x in cases.items()}
for name, t in times.items():
    print("%-38s column read: %.6f s" % (name, t))

base = times["(a + a).compact()"]
bad = [name for name, t in times.items() if t > 20 * base]
assert not bad, "explicit column block layout lost (column read >20x slower than (a + a)): %s" % bad
print("OK")

"""F020 repro: a Python bool index is silently taken as integer 0/1.

Asserts the CORRECT behavior, so it fails while the bug exists: a bool scalar index (Python bool
or np.bool_) must not silently select row 0/1. Acceptable outcomes are either a TypeError/IndexError
(consistent with how np.bool_ is already treated) or NumPy's 0-d mask semantics.
"""

import sys

import numpy as np

import jix

a_np = np.arange(12, dtype=np.int32).reshape(3, 4)
a = jix.compact(a_np)


def outcome(fn):
    try:
        r = fn()
    except (TypeError, IndexError) as e:
        return ("error", type(e).__name__)
    if isinstance(r, jix.Array):
        r = r.numpy()
    return ("ok", np.asarray(r).shape, np.asarray(r).ravel().tolist())


cases = {
    "a[True]": (lambda: a[True], lambda: a_np[True]),
    "a[False]": (lambda: a[False], lambda: a_np[False]),
    "jix.slice(a, True)": (lambda: jix.slice(a, True), lambda: a_np[True]),
    "a.numpy((0, True))": (lambda: a.numpy((0, True)), lambda: a_np[0, True]),
    "a[1, False]": (lambda: a[1, False], lambda: a_np[1, False]),
    "a[np.True_]": (lambda: a[np.True_], lambda: a_np[np.True_]),
}

print("numpy", np.__version__)
failures = []
for name, (jfn, nfn) in cases.items():
    got = outcome(jfn)
    ref = outcome(nfn)
    print(f"{name:22s} jix={got}  numpy={ref}")
    # Correct = raises like np.bool_ does, or matches numpy's mask result.
    if got[0] == "ok" and got != ref:
        failures.append(f"{name}: silently returned {got}, numpy gives {ref}")

# Same boolean value, two spellings: must be handled the same way.
py_bool = outcome(lambda: a[True])
np_bool = outcome(lambda: a[np.True_])
if py_bool[0] != np_bool[0]:
    failures.append(f"a[True] -> {py_bool} but a[np.True_] -> {np_bool} (inconsistent)")

if failures:
    print("FAIL:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("PASS")

"""
K183 repro: NumPy's fromnumeric reductions (np.sum, np.mean, np.max, ...) raise TypeError on a
jix.Array although jix.Array implements `__array__`, while sibling functions (np.argmax,
np.median, np.nanmax, np.average, np.round) succeed through `__array__`; np.squeeze returns a
lazy jix.Array (fine: NumPy returns whatever the duck type's method returns) but
np.squeeze(z, axis=0) raises.

Asserts the CORRECT (NumPy-consistent) behavior: each np.<f>(z) succeeds and its value equals
np.<f>(np.asarray(z)). A lazy jix.Array result is accepted (compared via np.asarray), so the repro
passes after either fix (accepting NumPy keywords on the methods, or an __array_function__ that
converts through __array__). Fails while the bug exists. Run with .venv/bin/python.
"""

import sys

import numpy as np

import jix

ref = np.arange(6.0).reshape(2, 3)
z = jix.compact(ref)

funcs = [
    # (name, callable) - all go through fromnumeric's _wrapreduction / mean / std / var / squeeze
    ("np.sum", lambda a: np.sum(a)),
    ("np.prod", lambda a: np.prod(a)),
    ("np.max", lambda a: np.max(a)),
    ("np.amax", lambda a: np.amax(a)),
    ("np.min", lambda a: np.min(a)),
    ("np.amin", lambda a: np.amin(a)),
    ("np.mean", lambda a: np.mean(a)),
    ("np.std", lambda a: np.std(a)),
    ("np.var", lambda a: np.var(a)),
    ("np.all", lambda a: np.all(a)),
    ("np.any", lambda a: np.any(a)),
    ("np.sum(axis=0)", lambda a: np.sum(a, axis=0)),
    ("np.squeeze", lambda a: np.squeeze(a)),
    ("np.squeeze(axis=0) on (1,3)", None),
    # controls: these already work through __array__ (asarray or _wrapfunc's TypeError fallback)
    ("np.argmax", lambda a: np.argmax(a)),
    ("np.nanmax", lambda a: np.nanmax(a)),
    ("np.median", lambda a: np.median(a)),
    ("np.average", lambda a: np.average(a)),
    ("np.round", lambda a: np.round(a)),
]

failures = 0
for name, f in funcs:
    if f is None:
        zz = jix.compact(ref[:1])
        rr = ref[:1]
        f = lambda a: np.squeeze(a, axis=0)
    else:
        zz, rr = z, ref
    expected = f(rr)
    try:
        got = f(zz)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: raised {type(e).__name__}: {e}")
        failures += 1
        continue
    if isinstance(got, jix.Array):
        name += " (lazy jix.Array)"
        got = np.asarray(got)
    if np.shape(got) != np.shape(expected) or not np.array_equal(np.asarray(got), np.asarray(expected)):
        print(f"FAIL {name}: got {got!r}, expected {expected!r}")
        failures += 1
        continue
    print(f"ok   {name}: {got!r}")

print(f"numpy {np.__version__}: {failures} failure(s)")
sys.exit(1 if failures else 0)

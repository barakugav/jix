"""K179: where() raises for an integer error in an element it does not select, but only for some
spellings of a 0-d condition; numpy's guard idiom where(b != 0, a // b, 0) always raises.

Run with: .venv/bin/python bug-hunt/report/repros/K179.py

Asserts the CORRECT behavior, so it fails while the bug exists:
  A. every spelling of the same 0-d False condition gives the same outcome (value or error);
  B. the guard idiom works as in numpy, or where's docstring documents that integer errors in
     unselected elements raise.
"""

import warnings

import numpy as np

import jix


def outcome(f):
    try:
        return ("value", tuple(f().numpy().tolist()))
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__)


a_np = np.array([7, 8, 9], np.int32)
b_np = np.array([2, 0, 3], np.int32)
a = jix.compact(a_np)
b = jix.compact(b_np)
q = a // b  # lazy; reading it raises ArithmeticError (b[1] == 0)

spellings = {
    "False": False,
    "np.False_": np.False_,
    "np.array(False)": np.array(False),
    "jix.asarray(False)": jix.asarray(False),
    "jix.compact(np.array(False))": jix.compact(np.array(False)),
    "np.array([False])": np.array([False]),
    "jix.compact(np.zeros(3, bool))": jix.compact(np.zeros(3, bool)),
}
results = {}
for name, cond in spellings.items():
    results[name] = outcome(lambda cond=cond: jix.where(cond, q, a))
    print(f"where({name}, a // b, a) -> {results[name]}")

guard = outcome(lambda: jix.where(b != 0, q, 0))
print(f"where(b != 0, a // b, 0) -> {guard}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np_guard = np.where(b_np != 0, a_np // b_np, 0)
print(f"numpy where(b != 0, a // b, 0) -> {np_guard.tolist()} warnings={[str(x.message) for x in w]}")
workaround = outcome(lambda: a // jix.where(b != 0, b, 1))
print(f"a // where(b != 0, b, 1) -> {workaround}")

failed = False
distinct = set(results.values())
if len(distinct) != 1:
    failed = True
    print(f"A FAIL: the same all-False condition gives {len(distinct)} different outcomes: {distinct}")
else:
    print("A ok")

doc = jix.where.__doc__ or ""
documented = "ArithmeticError" in doc
if guard != ("value", tuple(np_guard.tolist())) and not documented:
    failed = True
    print("B FAIL: the guard idiom raises and where's docstring does not say unselected integer errors raise")
else:
    print("B ok")

if failed:
    raise SystemExit(1)

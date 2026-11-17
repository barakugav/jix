"""K199: since #107 (K180 fix) broadcast_operand computes the numpy dtype of every implicitly
broadcast jix Array operand (dtype_to_numpy), even when nothing uses it.

On i686 Linux dtype_to_numpy raises ValueError for every f8/i8/u8/c16 dtype (K8: jix alignment
8 vs numpy's 4), so there `x + y`, `x > y`, `jix.isclose`, `jix.clamp`, ... raise at op
construction whenever the jix operand x (no cached numpy dtype: jix.compact, jix.asarray without
dtype=, read_array, any op result such as `a.mean()`) has to be broadcast. Before #107
(a7b0656) the op built fine.

No i686 Python is available here, so this script SIMULATES the i686 failure on x86_64: the only
dtype_to_numpy branch that goes through Python is an align=True struct dtype, which calls
numpy.dtype(...). We replace numpy.dtype with a function that raises the i686 error message.
A struct operand is enough to show the code path (where accepts struct dtypes; element-wise ops
reach broadcast_operand before their dtype dispatch).

Repro convention: each case asserts the CORRECT (pre-#107) behavior, so the script fails
(exit 1) while the bug exists.
Run: .venv/bin/python bug-hunt/report/repros/K199.py
"""

import sys

import numpy as np

import jix

D = np.dtype([("a", "<i4"), ("b", "<f4")], align=True)
REAL_DTYPE = np.dtype
calls = []


def failing_dtype(*args, **kwargs):
    calls.append(args)
    raise ValueError("numpy dtype itemsize or alignment mismatch: (8, 4) != (8, 8)  [simulated i686]")


def under_simulated_i686(f):
    calls.clear()
    np.dtype = failing_dtype
    try:
        return f()
    finally:
        np.dtype = REAL_DTYPE


def case_where_no_broadcast():
    # Control: nothing is broadcast, only x's (cached) numpy dtype is used -> no numpy.dtype call.
    x = jix.asarray(np.zeros(3, D), dtype=D)  # carries a cached numpy dtype
    y = jix.compact(np.ones(3, D))  # no cached numpy dtype
    c = jix.asarray(np.array([True, False, True]))
    r = under_simulated_i686(lambda: jix.where(c, x, y))
    assert len(calls) == 0, calls
    got = r.numpy()
    assert got["a"].tolist() == [0, 1, 0], got


def case_where_broadcast():
    # y's numpy dtype is never used (the result takes x's), yet broadcasting y computes it.
    x = jix.asarray(np.zeros(3, D), dtype=D)
    y = jix.compact(np.ones(3, D))
    c = jix.asarray(np.array([[True, False, True], [False, True, False]]))
    r = under_simulated_i686(lambda: jix.where(c, x, y))  # correct: builds, as at a7b0656
    got = r.numpy()
    assert got["a"].tolist() == [[0, 1, 0], [1, 0, 1]], got


def case_elementwise_no_broadcast():
    # Control: same op, x not broadcast -> dispatch TypeError, no numpy.dtype call.
    x = jix.compact(np.zeros(3, D))
    other = np.zeros(3, D)
    try:
        under_simulated_i686(lambda: jix.equal(x, other))
    except TypeError as e:
        assert "does not support operands" in str(e), e
        assert len(calls) == 0, calls
    else:
        raise AssertionError("expected TypeError")


def case_elementwise_broadcast():
    # Element-wise ops broadcast BEFORE dtype dispatch. For a struct operand the correct answer
    # is the dispatch TypeError; the simulated i686 error proves dtype_to_numpy ran first.
    # (On i686 the same call with an f8 operand, which dispatch accepts, raises the ValueError.)
    x = jix.compact(np.zeros(3, D))
    other = np.zeros((2, 3), D)
    try:
        under_simulated_i686(lambda: jix.equal(x, other))
    except TypeError as e:
        assert "does not support operands" in str(e), e
    except ValueError as e:
        raise AssertionError(f"broadcast computed the operand's numpy dtype: {e}") from e
    else:
        raise AssertionError("expected TypeError")


ok = True
for name, case in [
    ("where, no broadcast (control)", case_where_no_broadcast),
    ("where, y broadcast", case_where_broadcast),
    ("equal, struct x, no broadcast (control)", case_elementwise_no_broadcast),
    ("equal, struct x broadcast", case_elementwise_broadcast),
]:
    try:
        case()
        print(f"[PASS] {name}  (numpy.dtype calls: {len(calls)})")
    except AssertionError as e:
        ok = False
        print(f"[FAIL] {name}  (numpy.dtype calls: {len(calls)}): {e}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}  (numpy.dtype calls: {len(calls)}): {type(e).__name__}: {e}")

sys.exit(0 if ok else 1)

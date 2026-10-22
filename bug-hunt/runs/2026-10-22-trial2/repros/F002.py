"""F002 repro: numpy bool arrays whose bytes are not 0/1 are wrapped zero-copy and read as Rust
`bool` (UB). Asserts the CORRECT behavior: each jix result matches NumPy truthiness (nonzero byte =
True) and every bool jix outputs is a 0/1 byte, or jix rejects the input with a regular Python
exception (ValueError/TypeError). Exits non-zero while the bug exists.

Run: .venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F002.py
"""

import sys

import numpy as np

import jix

failures = []


def short(a):
    a = np.asarray(a)
    return a.tolist() if a.size <= 8 else a.ravel()[:8].tolist() + ["..."]


def check(label, jix_fn, expected):
    try:
        got = np.asarray(jix_fn())
    except (ValueError, TypeError) as e:  # rejecting invalid bool bytes is also a correct fix
        print(f"ok   {label}: rejected ({type(e).__name__}: {e})")
        return
    exp = np.asarray(expected)
    raw = got.view(np.uint8) if got.dtype == np.bool_ else None
    got_cmp = (raw != 0) if raw is not None else got
    good = np.array_equal(got_cmp, exp) and (raw is None or bool(np.all(raw <= 1)))
    extra = f" raw_bytes={short(raw)}" if raw is not None else ""
    print(f"{'ok  ' if good else 'FAIL'} {label}: jix={short(got)}{extra} expected={short(exp)}")
    if not good:
        failures.append(label)


# 1-d, every byte nonzero (all truthy), created with the standard numpy idiom .view(bool)
raw = np.array([2, 1, 4, 1, 2, 2, 1, 8] * 5, dtype=np.uint8)
b = raw.view(np.bool_)
check("all(b)", lambda: jix.all(b).numpy(), np.all(b))
check("any(b)", lambda: jix.any(b).numpy(), np.any(b))
check("sum(b)", lambda: jix.sum(b).numpy(), np.sum(b))
check("argmax(b)", lambda: jix.argmax(b).numpy(), np.argmax(b))
check("min(b)", lambda: jix.min(b).numpy(), np.min(b))
check("mean(b)", lambda: jix.mean(b).numpy(), np.mean(b))

# frombuffer: bool bytes coming from a file / socket / foreign buffer
b2 = np.frombuffer(bytes([2, 4, 2, 4] * 10), dtype=np.bool_)
check("any(frombuffer [2,4,..])", lambda: jix.any(b2).numpy(), np.any(b2))

# 2-d, multi-cell fold path
m = np.array([[2, 1, 4], [1, 2, 1]], dtype=np.uint8).view(np.bool_)
check("all(m, dim=0)", lambda: jix.all(m, dim=0).numpy(), np.all(m, axis=0))
check("sum(m, dim=0)", lambda: jix.sum(m, dim=0).numpy(), np.sum(m, axis=0))

# element-wise consumers: comparison, bitwise and, where
x = np.array([2, 2, 0, 0, 2, 1] * 16, np.uint8).view(np.bool_)
y = np.array([1, 0, 2, 1, 2, 2] * 16, np.uint8).view(np.bool_)
check("equal(x, y)", lambda: jix.equal(x, y).numpy(), np.equal(x, y))
check("x & y", lambda: (jix.asarray(x) & jix.asarray(y)).numpy(), x & y)
xs = np.arange(96, dtype=np.int32)
check("where(x, xs, -1)", lambda: jix.where(x, xs, -1).numpy(), np.where(x, xs, -1))

# compact() stores the invalid bytes; every later read of the compressed array is affected too
check("all(compact(b))", lambda: jix.all(jix.compact(b)).numpy(), True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("all checks passed")

"""F037: float add/sub/mul/square and sum/prod/mean never report `invalid` / `over`, while divide
reports the same conditions (and under the default modes, raises for `invalid`).

Asserts the behavior consistent with jix.divide and numpy's errstate: every case below must report
under errstate(all='raise'). Exits 1 while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F037.py
"""

import sys
import warnings

import numpy as np

import jix


def run(f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = f()
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}({e})", True
    if w:
        return f"{r!r} warnings={[str(x.message) for x in w]}", True
    return repr(r), False


def c(values, dtype=np.float64):
    return jix.compact(np.array(values, dtype=dtype))


inf = c([np.inf])
failures = []

print("== default modes (invalid='raise') ==")
for label, f, nf in [
    ("inf / inf", lambda: (inf / inf).numpy(), lambda: np.array([np.inf]) / np.array([np.inf])),
    ("inf - inf", lambda: (inf - c([np.inf])).numpy(), lambda: np.array([np.inf]) - np.array([np.inf])),
    ("inf + -inf", lambda: (inf + c([-np.inf])).numpy(), lambda: np.array([np.inf]) + np.array([-np.inf])),
    ("0 * inf", lambda: (c([0.0]) * inf).numpy(), lambda: np.array([0.0]) * np.array([np.inf])),
    ("sum([inf, -inf])", lambda: jix.sum(c([np.inf, -np.inf])).numpy(), lambda: np.sum(np.array([np.inf, -np.inf]))),
    ("mean([inf, -inf])", lambda: jix.mean(c([np.inf, -np.inf])).numpy(), lambda: np.mean(np.array([np.inf, -np.inf]))),
]:
    out, reported = run(f)
    nout, _ = run(nf)
    print(f"  jix {label:22s} -> {out}")
    print(f"  np  {label:22s} -> {nout}")
    if not reported:
        failures.append(f"default modes: {label}")

print("== jix.errstate(all='raise') / np.errstate(all='raise') ==")
cases = [
    ("f64 1e308 + 1e308", lambda: (c([1e308]) + c([1e308])).numpy(), lambda: np.array([1e308]) + np.array([1e308])),
    ("f64 -1e308 - 1e308", lambda: (c([-1e308]) - c([1e308])).numpy(), lambda: np.array([-1e308]) - np.array([1e308])),
    ("f64 1e200 * 1e200", lambda: (c([1e200]) * c([1e200])).numpy(), lambda: np.array([1e200]) * np.array([1e200])),
    ("f64 square(1e200)", lambda: jix.square(c([1e200])).numpy(), lambda: np.square(np.array([1e200]))),
    ("f32 3e38 + 3e38", lambda: (c([3e38], np.float32) + c([3e38], np.float32)).numpy(),
     lambda: np.array([3e38], np.float32) + np.array([3e38], np.float32)),
    ("f16 60000 + 60000", lambda: (c([60000], np.float16) + c([60000], np.float16)).numpy(),
     lambda: np.array([60000], np.float16) + np.array([60000], np.float16)),
    ("f64 sum([1e308, 1e308])", lambda: jix.sum(c([1e308, 1e308])).numpy(), lambda: np.sum(np.array([1e308, 1e308]))),
    ("f64 prod([1e200, 1e200])", lambda: jix.prod(c([1e200, 1e200])).numpy() if hasattr(jix, "prod") else jix.product(c([1e200, 1e200])).numpy(),
     lambda: np.prod(np.array([1e200, 1e200]))),
    ("f16 sum(1000 x 100.0)", lambda: jix.sum(c(np.full(1000, 100.0), np.float16)).numpy(),
     lambda: np.sum(np.full(1000, 100.0, np.float16))),
    ("f64 1e308 / 1e-308 (divide, reference)", lambda: (c([1e308]) / c([1e-308])).numpy(),
     lambda: np.array([1e308]) / np.array([1e-308])),
]
for label, f, nf in cases:
    with jix.errstate(all="raise"):
        out, reported = run(f)
    with np.errstate(all="raise"):
        nout, _ = run(nf)
    print(f"  jix {label:40s} -> {out}")
    print(f"  np  {label:40s} -> {nout}")
    if not reported:
        failures.append(f"all=raise: {label}")

print()
print("NOT REPORTED:" if failures else "all reported", *failures, sep="\n  ")
sys.exit(1 if failures else 0)

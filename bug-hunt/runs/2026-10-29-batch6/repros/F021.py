"""F021: asarray / cast / compact with dtype= convert numpy scalars (np.generic) with numpy's own
cast instead of jix's Cast, so one value gives different results depending on whether it arrives
as a numpy scalar or as a 0-d / 1-d numpy array.

as_array.rs:65-72 sends every value that is not a jix.Array or an ndarray through
numpy.asarray(value, dtype=dtype) (the K96 fix, meant for untyped Python ints / lists). np.generic
is not an ndarray, so numpy scalars take that path too.

The repro asserts the CORRECT behavior: the numpy-scalar result equals the 0-d ndarray result
(which goes through cast_impl -> jix Cast), and no RuntimeWarning / ComplexWarning is emitted.
It fails while the bug exists.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F021.py
"""

import warnings

import numpy as np

import jix

FNS = {
    "asarray": lambda v, d: jix.asarray(v, dtype=d),
    "cast": lambda v, d: jix.cast(v, d),
    "compact": lambda v, d: jix.compact(v, dtype=d),
}

# (numpy scalar, target dtype). jix Cast float -> int saturates and sends NaN to 0
# (pinned by jix/src/ops/cast.rs cast_int_float_concrete).
CASES = [
    (np.float64(3e9), np.int32),
    (np.float64("nan"), np.int32),
    (np.float64(-1.5), np.uint8),
    (np.float32(300.7), np.uint8),
    (np.float64(1e10), np.uint32),
    (np.float64("inf"), np.int64),
    (np.float64(-3e9), np.int32),
]


def run(fn, value, dtype):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn(value, dtype).numpy()
            res = ("ok", out.dtype.str, out.tolist())
        except Exception as e:  # noqa: BLE001
            res = ("raise", type(e).__name__, None)
    return res, [f"{x.category.__name__}: {x.message}" for x in w]


failures = []
for name, fn in FNS.items():
    for value, dtype in CASES:
        got, got_w = run(fn, value, dtype)
        ref0, ref0_w = run(fn, np.array(value), dtype)  # 0-d ndarray, same value and dtype
        ref1, _ = run(fn, np.array([value]), dtype)  # 1-d ndarray
        ok = got == ref0 and not got_w and ref1[2] == [ref0[2]]
        tag = "ok " if ok else "BUG"
        print(
            f"[{tag}] {name}({type(value).__name__}({value!r}), {np.dtype(dtype).name}): "
            f"np scalar -> {got[2]} {got_w} | 0-d -> {ref0[2]} {ref0_w} | 1-d -> {ref1[2]}"
        )
        if not ok:
            failures.append((name, repr(value), np.dtype(dtype).name, got[2], ref0[2]))

# complex numpy scalar -> float: the docstring promises TypeError for complex -> float casts,
# which is what a 0-d complex ndarray gives; the numpy scalar is silently truncated instead.
for name, fn in FNS.items():
    got, got_w = run(fn, np.complex128(1 + 2j), np.float64)
    ref0, _ = run(fn, np.array(1 + 2j), np.float64)
    ok = got[0] == ref0[0] == "raise"
    tag = "ok " if ok else "BUG"
    print(f"[{tag}] {name}(np.complex128(1+2j), float64): np scalar -> {got} {got_w} | 0-d -> {ref0}")
    if not ok:
        failures.append((name, "np.complex128(1+2j)", "float64", got, ref0))

assert not failures, f"{len(failures)} numpy-scalar inputs disagree with the ndarray path: {failures}"

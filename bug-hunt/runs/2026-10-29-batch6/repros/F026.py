"""F026: a Python bool passed as a dim / dims argument is silently read as dim 0 / 1.

pyo3's int extraction (used for `dim: Option<ItemOrTuple<i32>>`, `Option<i32>`, `i32`) accepts
Python bool because bool subclasses int, so True -> dim 1 and False -> dim 0. jix's own index
parser rejects a Python bool since K33 (a[True] -> TypeError), and np.True_ as a dim is already
rejected (no __index__), so only the Python-bool spelling slips through.

numpy's C-level axis parser (PyArray_PyIntAsInt) rejects a bool axis with TypeError: every
reduction, argmax/argmin, concatenate, squeeze, transpose. numpy's Python-level helpers that go
through operator.index (flip, roll, expand_dims, stack, swapaxes) accept it, so those jix
counterparts are listed as informational only.

Repro convention: asserts the CORRECT behavior (TypeError where numpy and jix's own index parser
raise), so it exits non-zero while the bug exists. Run: .venv/bin/python <this file>
"""

import numpy as np

import jix

x = np.arange(6.0).reshape(2, 3)
a = jix.asarray(x)
m = jix.asarray(x > 0)

# (label, jix call, numpy counterpart) - numpy raises TypeError for every numpy call here.
must_raise = [
    ("jix.sum(a, True)", lambda: jix.sum(a, True), lambda: np.sum(x, True)),
    ("jix.sum(a, dim=False)", lambda: jix.sum(a, dim=False), lambda: np.sum(x, axis=False)),
    ("jix.max(a, dim=(True, False))", lambda: jix.max(a, dim=(True, False)), lambda: np.max(x, axis=(True, False))),
    ("a.mean(False)", lambda: a.mean(False), lambda: x.mean(False)),
    ("jix.var(a, True)", lambda: jix.var(a, True), lambda: np.var(x, True)),
    ("jix.all(m, True)", lambda: jix.all(m, True), lambda: np.all(x > 0, True)),
    ("jix.argmax(a, dim=True)", lambda: jix.argmax(a, dim=True), lambda: np.argmax(x, axis=True)),
    ("a.argmin(False)", lambda: a.argmin(False), lambda: x.argmin(False)),
    ("jix.concatenate([a, a], True)", lambda: jix.concatenate([a, a], True), lambda: np.concatenate([x, x], True)),
    (
        "jix.squeeze(jix.insert_dim(a, 0), False)",
        lambda: jix.squeeze(jix.insert_dim(a, 0), False),
        lambda: np.squeeze(x[None], False),
    ),
    (
        "jix.permute_dims(a, (True, False))",
        lambda: jix.permute_dims(a, (True, False)),
        lambda: np.transpose(x, (True, False)),
    ),
]
# numpy accepts a bool here (operator.index); listed to show the jix dim parsers all behave alike.
informational = [
    ("jix.insert_dim(a, True)", lambda: jix.insert_dim(a, True)),
    ("jix.stack([a, a], True)", lambda: jix.stack([a, a], True)),
    ("jix.flip(a, True)", lambda: jix.flip(a, True)),
    ("jix.roll(a, 1, True)", lambda: jix.roll(a, 1, True)),
    ("jix.swapdims(a, True, False)", lambda: jix.swapdims(a, True, False)),
]
# Same value spelled as a numpy bool is already rejected by jix.
np_bool = [
    ("jix.sum(a, np.True_)", lambda: jix.sum(a, np.True_)),
    ("jix.argmax(a, np.True_)", lambda: jix.argmax(a, np.True_)),
]


def run(call):
    try:
        r = call()
        out = np.asarray(r.numpy() if hasattr(r, "numpy") else r)
        return "ACCEPTED", f"shape {out.shape} {out.ravel()[:6]!r}"
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, str(e)


print("numpy", np.__version__)
print("a[True]:", *run(lambda: a[True]))
for label, call in np_bool:
    print(f"{label}:", *run(call))
print()
accepted = []
for label, call, npcall in must_raise:
    status, detail = run(call)
    np_status, np_detail = run(npcall)
    print(f"{label}: jix {status}: {detail}")
    print(f"{' ' * len(label)}  numpy {np_status}: {np_detail}")
    if status == "ACCEPTED":
        accepted.append(label)
print()
for label, call in informational:
    print(f"(info) {label}:", *run(call))

assert not accepted, f"{len(accepted)} calls accepted a bool dim where numpy raises TypeError: {accepted}"

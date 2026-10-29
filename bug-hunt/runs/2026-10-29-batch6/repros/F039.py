"""F039: a Python read that raises drops the warnings recorded in the same read.

With over='warn' and invalid='raise', a read that hits both an overflow and an invalid
operation should emit the overflow RuntimeWarning and then raise FloatingPointError (numpy:
warning first, then the raise). jix records the warning in the ReadContext
(jix/src/error.rs check_op_status pushes it before returning the Err), but numpy(), compact()
and write_array propagate the Err with `?` before `take_warnings()`, so the warning stays in the
cached context and the next read's configure_context_error_modes drops it.

Asserts the CORRECT behavior: fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F039.py
"""

import io
import warnings

import numpy as np

import jix

A = np.array([1e308, 0.0])
B = np.array([0.5, 0.0])
MODES = dict(over="warn", invalid="raise", divide="ignore")


def run(read):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raised = None
        try:
            read()
        except FloatingPointError as e:
            raised = str(e)
    return raised, [str(x.message) for x in w]


def numpy_ref():
    with np.errstate(**MODES):
        np.divide(A, B)


def jix_numpy():
    with jix.errstate(**MODES):
        (jix.compact(A) / jix.compact(B)).numpy()


def jix_compact():
    with jix.errstate(**MODES):
        jix.compact(jix.compact(A) / jix.compact(B))


def jix_write_array():
    with jix.errstate(**MODES):
        jix.write_array(jix.compact(A) / jix.compact(B), io.BytesIO())


def jix_numpy_multiblock():
    # overflow in the first block, invalid in the last: the warning comes from an earlier block
    a = np.full(64, 1.0)
    b = np.full(64, 1.0)
    a[0], b[0] = 1e308, 0.5
    a[-1], b[-1] = 0.0, 0.0
    x = jix.compact(a, params={"block_shape": [8]}) / jix.compact(b, params={"block_shape": [8]})
    with jix.errstate(**MODES):
        x.numpy()


def jix_numpy_nonnan_followup():
    # the dropped warning does not resurface on the next (clean) read either
    x = jix.compact(A) / jix.compact(B)
    with jix.errstate(**MODES):
        try:
            x.numpy()
        except FloatingPointError:
            pass
    return run(lambda: jix.compact(np.array([1.0])).numpy())


ref_raised, ref_warns = run(numpy_ref)
print("numpy        raised:", ref_raised, "| warnings:", ref_warns)
assert ref_raised is not None and any("overflow" in m for m in ref_warns)

failures = []
for name, fn in [
    ("numpy()", jix_numpy),
    ("compact()", jix_compact),
    ("write_array", jix_write_array),
    ("numpy() multi-block", jix_numpy_multiblock),
]:
    raised, warns = run(fn)
    print(f"jix {name:20s} raised: {raised} | warnings: {warns}")
    if raised is None:
        failures.append(f"{name}: did not raise")
    if not any("overflow" in m for m in warns):
        failures.append(f"{name}: overflow warning not emitted alongside the FloatingPointError")

print("follow-up clean read:", jix_numpy_nonnan_followup())

assert not failures, "\n".join(failures)
print("OK")

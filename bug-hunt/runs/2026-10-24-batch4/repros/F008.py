"""F008 repro: jix.sum / jix.max / jix.all over a valid lazy broadcast whose reduced element count
is > 2**64 - 5 raise PanicException ('attempt to add with overflow', jix/src/ops/reduction.rs:177)
in the debug extension instead of returning a lazy 0-d array. Asserts the correct behavior, so it
fails (exit 1) while the bug exists."""

import sys

import numpy as np

import jix

failed = False
for name, dt, fn in [
    ("sum", np.uint8, jix.sum),
    ("max", np.uint8, jix.max),
    ("all", np.bool_, jix.all),
]:
    a = jix.broadcast(jix.asarray(np.ones((1, 1), dtype=dt)), (2**63 - 1, 2))
    try:
        s = fn(a)
        print(f"{name}: ok, shape {s.shape}")
        assert s.shape == ()
    except BaseException as e:  # PanicException derives from BaseException
        print(f"{name}: FAIL {type(e).__module__}.{type(e).__name__}: {e}")
        failed = True

sys.exit(1 if failed else 0)

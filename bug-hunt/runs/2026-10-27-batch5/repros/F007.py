# F007: Broadcast evaluates its inner row for an EMPTY read (a broadcast dim requested 0..0 is
# read as inner 0..1). From Python it surfaces through concatenate, which reads a zero-length
# member in the middle of the list with an empty range on the concat dim.
# Asserts the CORRECT behavior; exits 1 while the bug exists.
import numpy as np

import jix

a = jix.asarray(np.array([[6, 7, 8]], dtype=np.int32))
b = jix.asarray(np.array([[2, 0, 4]], dtype=np.int32))
q = a // b  # lazy; element [0, 1] divides by zero
o = jix.asarray(np.ones((2, 3), dtype=np.int32))
z0 = jix.asarray(np.zeros((0, 3), dtype=np.int32))

cases = {
    # explicit broadcast to a zero-length dim
    "concatenate([o, broadcast(q, (0, 3)), o])": lambda: jix.concatenate([o, jix.broadcast(q, (0, 3)), o]),
    # implicit broadcasting of a binary op against an empty operand: (1,3) + (0,3) -> (0,3)
    "concatenate([o, q + zeros((0, 3)), o])": lambda: jix.concatenate([o, q + z0, o]),
}

# Controls: the empty member alone, and the same members with the empty one last, are fine.
print("broadcast(q, (0, 3)).numpy().shape =", jix.broadcast(q, (0, 3)).numpy().shape)
print(
    "concatenate([o, o, broadcast(q, (0, 3))]).numpy().shape =",
    jix.concatenate([o, o, jix.broadcast(q, (0, 3))]).numpy().shape,
)

failed = False
for name, f in cases.items():
    try:
        r = f().numpy()
        ok = r.shape == (4, 3) and bool((r == 1).all())
        print(f"{name}: OK shape {r.shape}, all ones: {ok}")
        failed |= not ok
    except Exception as e:  # noqa: BLE001
        print(f"{name}: raised {type(e).__name__}: {e}")
        failed = True
raise SystemExit(1 if failed else 0)

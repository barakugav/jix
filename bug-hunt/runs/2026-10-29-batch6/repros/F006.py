# F006: dtype_from_numpy (jix-py/src/dtype.rs:101, self-call at :203) recurses once per nesting
# level of a numpy struct dtype with no depth limit. A deeply nested struct dtype that numpy
# itself builds and copies without trouble overflows the native stack and the Python process
# dies with SIGSEGV (no exception).
#
# THIS REPRO CRASHES THE INTERPRETER: every conversion runs in a subprocess and the parent
# reports its exit status. It asserts the CORRECT behavior - the child exits 0, having either
# wrapped the array or raised a catchable Python exception - so it fails while the bug exists.
import subprocess
import sys

CHILD = r"""
import sys, threading
import numpy as np
import jix

entry, depth, stack_kib = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = np.dtype("u1")
for _ in range(depth):
    d = np.dtype([("a", d)])
x = np.zeros(2, dtype=d)

def work():
    y = x.copy()  # numpy handles the dtype
    assert y.dtype == d
    print("numpy ok", flush=True)
    try:
        if entry == "asarray":
            jix.asarray(x)
        elif entry == "asarray_dtype":
            jix.asarray(np.zeros(2, dtype="u1"), dtype=d)
        elif entry == "where":
            jix.where(np.ones(2, dtype=bool), x, x)
        else:
            raise SystemExit(f"unknown entry {entry}")
        print("jix ok", flush=True)
    except Exception as e:  # a catchable error is acceptable behavior
        print(f"jix raised {type(e).__name__}: {str(e)[:80]}", flush=True)

if stack_kib:
    threading.stack_size(stack_kib * 1024)
    t = threading.Thread(target=work)
    t.start()
    t.join()
else:
    work()
"""

CASES = [
    # (entry point, nesting depth, thread stack KiB; 0 = main thread with the 8 MiB rlimit)
    ("asarray", 1000, 0),
    ("asarray", 1400, 0),
    ("asarray", 100, 1024),
    ("asarray", 200, 1024),
    ("asarray_dtype", 200, 1024),
    ("where", 200, 1024),
]

failures = []
for entry, depth, kib in CASES:
    r = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, entry, str(depth), str(kib)],
        capture_output=True,
        text=True,
        cwd="/",
    )
    out = r.stdout.strip().replace("\n", " | ")
    where = "main thread" if kib == 0 else f"{kib} KiB thread"
    print(f"{entry:14s} depth {depth:5d} on {where:14s}: exit {r.returncode:4d}  {out}")
    if r.returncode != 0:
        failures.append((entry, depth, kib, r.returncode))

assert not failures, (
    f"jix crashed the interpreter (expected the array to be wrapped or a catchable error): "
    f"{failures}"
)
print("OK")

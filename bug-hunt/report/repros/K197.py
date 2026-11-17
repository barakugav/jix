"""K197: a tiny fused read over numpy memory always releases the GIL in PyArray.numpy().

jix-py/src/array.rs:631-639 keeps a read attached when the spec says ZeroCopy or Fusable and
the read is cheap, but every jix-py array is an ArrayAny, whose spec demotes Fusable to
SinglePass (jix/src/storage/any.rs:105-108). So `(jix.asarray(x) + 1).numpy()` on 16 elements
detaches.

Observed through GIL hand-off latency: a second thread spins in Python, and the switch interval
is raised to 50 ms. A read that stays attached finishes in microseconds; a read that detaches
hands the GIL to the spinner and waits about one switch interval to get it back.

Asserts the intended (#106) behavior: the tiny fused read is as cheap as the zero-copy read, so it
fails while the bug exists.
"""

import sys
import threading
import time

import numpy as np

import jix

INTERVAL = 0.05

x = np.arange(16, dtype=np.float32)
plain = jix.asarray(x)  # ZeroCopy: stays attached
fused = jix.asarray(x) + 1  # Fusable in core, SinglePass once type-erased
compact = jix.compact(x)  # SinglePass (decompression): detaching is intended

stop = False


def spin():
    n = 0
    while not stop:
        n += 1


def min_time(f, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


sys.setswitchinterval(INTERVAL)
t = threading.Thread(target=spin)
t.start()
try:
    t_plain = min_time(lambda: plain.numpy())
    t_fused = min_time(lambda: fused.numpy())
    t_compact = min_time(lambda: compact.numpy())
finally:
    stop = True
    t.join()

np.testing.assert_array_equal(fused.numpy(), x + 1)
print(f"switch interval {INTERVAL * 1e3:.0f} ms, min over 5 reads of 16 x f32 with a spinning thread:")
print(f"  plain   (ZeroCopy)              {t_plain * 1e3:8.3f} ms")
print(f"  x + 1   (Fusable -> SinglePass) {t_fused * 1e3:8.3f} ms")
print(f"  compact (SinglePass, control)   {t_compact * 1e3:8.3f} ms")
assert t_plain < INTERVAL / 5, "control: the zero-copy read should stay attached"
assert t_fused < INTERVAL / 5, (
    f"tiny fused read waited {t_fused * 1e3:.1f} ms for the GIL: numpy() detached although the read "
    "is a fused op over numpy memory costing far below 4096"
)
print("OK")

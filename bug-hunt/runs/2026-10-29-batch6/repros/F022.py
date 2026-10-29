"""F022: Python exceptions raised by a file-like's read()/write()/flush() lose their type
(KeyboardInterrupt / SystemExit / OSError(errno) -> RuntimeError or OSError), write() is called
again after it raised, and a missing file raises FileNotFoundError or RuntimeError depending on len=.

The script asserts the CORRECT behavior: it exits non-zero while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F022.py
"""

import io
import os
import sys
import tempfile

import numpy as np

import jix

failures = []


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + ": " + detail)
    if not ok:
        failures.append(name)


class RaisingWriter:
    """Delegates to a BytesIO; write() raises `exc` on call number `on_call`."""

    def __init__(self, exc, on_call=2, flush_exc=None):
        self.bio = io.BytesIO()
        self.exc = exc
        self.on_call = on_call
        self.flush_exc = flush_exc
        self.calls = 0
        self.calls_after_raise = 0
        self.raised = False

    def write(self, data):
        self.calls += 1
        if self.raised:
            self.calls_after_raise += 1
        elif self.exc is not None and self.calls == self.on_call:
            self.raised = True
            raise self.exc
        return self.bio.write(data)

    def flush(self):
        if self.flush_exc is not None:
            raise self.flush_exc

    def seek(self, *args):
        return self.bio.seek(*args)

    def tell(self):
        return self.bio.tell()


class RaisingReader:
    def __init__(self, data, exc, on_call=2):
        self.bio = io.BytesIO(data)
        self.exc = exc
        self.on_call = on_call
        self.calls = 0

    def read(self, n):
        self.calls += 1
        if self.calls == self.on_call:
            raise self.exc
        return self.bio.read(n)

    def seek(self, *args):
        return self.bio.seek(*args)

    def tell(self):
        return self.bio.tell()


def run(fn):
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 - we want KeyboardInterrupt / SystemExit too
        return e
    return None


src = jix.compact(np.random.default_rng(0).random(1 << 16))

# 1-3: write() raises on its 2nd call
for exc, want in [
    (KeyboardInterrupt(), KeyboardInterrupt),
    (SystemExit(2), SystemExit),
    (OSError(28, "No space left on device"), OSError),
    (ValueError("bad"), ValueError),
]:
    w = RaisingWriter(exc)
    got = run(lambda: jix.write_array(src, w))
    check(
        f"write() raises {type(exc).__name__}",
        type(got) is want and (not isinstance(exc, OSError) or getattr(got, "errno", None) == 28),
        f"write_array raised {type(got).__name__}: {got!r}",
    )
    check(
        f"write() not called again after raising {type(exc).__name__}",
        w.calls_after_raise == 0,
        f"write() called {w.calls_after_raise} more time(s) after it raised",
    )

# 4: flush() raises (the final writer.flush() goes through a different conversion -> OSError)
w = RaisingWriter(None, flush_exc=ValueError("flush failed"))
got = run(lambda: jix.write_array(src, w))
check("flush() raises ValueError", type(got) is ValueError, f"write_array raised {type(got).__name__}: {got!r}")

# 5: read() raises on its 2nd call
data = io.BytesIO()
jix.write_array(src, data)
data = data.getvalue()
for exc, want in [(KeyboardInterrupt(), KeyboardInterrupt), (OSError(5, "Input/output error"), OSError)]:
    r = RaisingReader(data, exc)
    got = run(lambda: jix.read_array(r))
    check(
        f"read() raises {type(exc).__name__}",
        type(got) is want,
        f"read_array raised {type(got).__name__}: {got!r}",
    )

# 6: a missing file raises the same exception type whatever len= / mmap= is
missing = os.path.join(tempfile.mkdtemp(), "missing.jix")
for kw in ({}, {"len": 100}, {"mmap": True}, {"mmap": True, "len": 100}):
    got = run(lambda: jix.read_array(missing, **kw))
    check(
        f"read_array(missing, {kw})",
        isinstance(got, FileNotFoundError),
        f"raised {type(got).__name__}: {got}",
    )

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("all checks passed")

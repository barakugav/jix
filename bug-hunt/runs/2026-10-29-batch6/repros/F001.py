"""F001: PyWriter::write (jix-py/src/archive.rs:419-439) hands write() a memoryview over Rust-owned
memory. `release()` only invalidates that one memoryview object, and its error is ignored, so a
view DERIVED from it inside write() (a slice b[:], memoryview(b), b.cast('B'), np.frombuffer(b))
stays live after write_array returns and points into memory jix reuses or frees.

The repro asserts the CORRECT behavior: every view the writer kept either raises ValueError on
access (released) or still holds exactly the bytes it was given; reading it never crashes.
It FAILS while the bug exists:
  - check 1 (in process): kept views read back bytes that differ from what was written
    (the BufWriter's internal buffer was reused), i.e. they alias reused Rust memory;
  - check 2 (child process, may SEGFAULT): after the source compact array is freed, reading a
    kept view of a large block reads freed heap memory; the child dies with SIGSEGV or prints
    garbage.

Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F001.py
"""

import gc
import subprocess
import sys

import numpy as np

import jix


class KeepingWriter:
    """Seekable writer that keeps a zero-copy view of every chunk (plus a reference copy)."""

    def __init__(self, keep):
        self.keep = keep
        self.views, self.copies = [], []
        self.pos = self.size = 0

    def write(self, b):
        self.views.append(self.keep(b))
        self.copies.append(bytes(b))
        self.pos += len(b)
        self.size = max(self.size, self.pos)
        return len(b)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos


KEEP = {
    "plain": lambda b: b,  # the documented-safe case: the original view is released
    "slice": lambda b: b[:],
    "frombuffer": lambda b: np.frombuffer(b, dtype=np.uint8),
}


def check_views(w):
    live = stale = 0
    for v, c in zip(w.views, w.copies):
        try:
            got = bytes(v)
        except ValueError:
            continue  # released: correct
        live += 1
        stale += got != c
    return live, stale


def child(kind):
    a = jix.compact(np.random.default_rng(0).random(1 << 18))  # 2 MiB, incompressible
    w = KeepingWriter(KEEP[kind])
    jix.write_array(a, w)
    i = max(range(len(w.copies)), key=lambda i: len(w.copies[i]))
    del a  # frees the compact block data the large chunk was a view of
    gc.collect()
    junk = [bytearray(b"\xaa" * (1 << 20)) for _ in range(16)]  # noqa: F841
    ok = bytes(w.views[i]) == w.copies[i]
    print(f"largest kept view ({len(w.copies[i])} B) still equals what was written: {ok}")
    sys.exit(0 if ok else 3)


def main():
    failures = []
    for kind in KEEP:
        # A lazy source: every chunk the writer sees is a view of the BufWriter's 8 KiB buffer or of
        # the encoder's output buffer, both reused between writes.
        a = jix.compact(np.random.default_rng(1).random(1 << 16))
        w = KeepingWriter(KEEP[kind])
        jix.write_array(a * 1.0, w)
        live, stale = check_views(w)
        print(f"{kind:10s}: {len(w.views)} writes, {live} views still live after write_array, "
              f"{stale} of them now hold different bytes than were written")
        if stale:
            failures.append(f"{kind}: {stale} stale views")
    for kind in ("slice", "frombuffer"):
        r = subprocess.run([sys.executable, __file__, kind], capture_output=True, text=True)
        print(f"child {kind}: exit status {r.returncode}, stdout={r.stdout.strip()!r}")
        if r.returncode != 0:
            failures.append(f"child {kind}: exit status {r.returncode}")
    assert not failures, "views outlived the Rust buffer passed to write(): " + "; ".join(failures)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()

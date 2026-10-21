# F005 repro: jix.read_array(path, len=..., mmap=True) bounds the archive sections by the caller's
# `len`, not by the mapping. With a file shorter than `len` (truncated download / interrupted copy,
# or a stale `len` from an external index), the open succeeds and reading a block whose compressed
# bytes lie past the end of the mapping makes zstd read outside it -> SIGSEGV.
#
# Run: .venv/bin/python bug-hunt/runs/2026-10-21-pilot/repros/F005.py
# Each case runs in a child process so the crash is observable as an exit status.
import os
import subprocess
import sys
import tempfile

import numpy as np


def child(path, full, mode):
    import jix

    try:
        if mode == "default-len":
            b = jix.read_array(path, mmap=True)  # len defaults to file size - offset
        else:
            b = jix.read_array(path, offset=0, len=full, mmap=(mode != "nommap"))
        print(f"  opened, shape {b.shape}", flush=True)
        x = b[-1:].numpy()
        print(f"  read returned {x}", flush=True)
    except Exception as e:
        print(f"  error: {type(e).__name__}: {e}", flush=True)


def main():
    import jix

    d = tempfile.mkdtemp()
    p = os.path.join(d, "a.jix")
    jix.write_array(jix.compact(np.random.default_rng(0).random(2_000_000)), p)
    full = os.path.getsize(p)
    with open(p, "r+b") as f:
        f.truncate(64 * 1024)
    print(f"archive length {full}, file truncated to {os.path.getsize(p)}")
    for mode in ["nommap", "default-len", "mmap"]:
        print(f"[{mode}]", flush=True)
        r = subprocess.run([sys.executable, __file__, "child", p, str(full), mode])
        print(f"  child exit status {r.returncode}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()

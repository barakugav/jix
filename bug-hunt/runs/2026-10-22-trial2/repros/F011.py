"""Repro for bug-hunt finding F011 (run 2026-10-22-trial2), Python side.

An ~8 KB .jix file that passes strict validation declares one block of 2^28 - 1 elements of a
128-byte struct dtype (34_359_738_240 bytes). Reading a single element asks for a full block of
scratch; when that allocation fails the extension calls handle_alloc_error and the WHOLE
INTERPRETER ABORTS (SIGABRT) instead of raising.

The crashing step runs in a subprocess; this script asserts the CORRECT behavior (the child exits
normally, having seen an exception either at load or at read), so it fails while the bug exists.
Whether the allocation fails depends on the machine: here 15 GB RAM, no swap,
vm.overcommit_memory=0, so the kernel refuses 34 GB.
"""

import io
import os
import subprocess
import sys
import tempfile

import numpy as np

import jix

CHILD = r"""
import sys
import jix
path = sys.argv[1]
try:
    a = jix.read_array(path, validation="strict")
except Exception as e:
    print("load rejected:", type(e).__name__, e, flush=True)
    sys.exit(0)
print("loaded under strict: shape", a.shape, "dtype", a.dtype, flush=True)
try:
    v = a[1:2].numpy()
    print("read returned a value?!", v.shape, flush=True)
    sys.exit(2)
except Exception as e:
    print("read raised:", type(e).__name__, e, flush=True)
    sys.exit(0)
"""


def build(path: str) -> None:
    n = 1 << 21  # uleb128: 80 80 80 01
    dt = np.dtype([("x", "u1", (128,))], align=True)
    a = jix.compact(np.zeros(n, dtype=dt), params={"block_shape": [n], "filters": []})
    bio = io.BytesIO()
    jix.write_array(a, bio)
    buf = bytearray(bio.getvalue())
    frame = buf.find(b"\x28\xb5\x2f\xfd")
    assert frame > 0
    patched = 0
    i = 0
    while True:
        j = buf.find(b"\x80\x80\x80\x01", i, frame)
        if j < 0:
            break
        buf[j : j + 4] = b"\xff\xff\xff\x7f"  # 2^28 - 1
        patched += 1
        i = j + 4
    assert patched == 4, patched
    with open(path, "wb") as f:
        f.write(buf)
    print(f"wrote {len(buf)} bytes to {path}")


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "huge_block.jix")
        build(path)
        r = subprocess.run([sys.executable, "-c", CHILD, path], capture_output=True, text=True, check=False)
        print("child stdout:", r.stdout.strip())
        print("child stderr:", r.stderr.strip()[:300])
        print("child exit status:", r.returncode)
        assert r.returncode == 0, f"interpreter died with status {r.returncode} instead of raising"


if __name__ == "__main__":
    main()

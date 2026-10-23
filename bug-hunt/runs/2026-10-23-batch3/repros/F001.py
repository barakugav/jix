# F001 repro (bug-hunt run 2026-10-23-batch3).
#
# A crafted .jix whose struct dtype has itemsize 1 but fields a: u8[65535] @0, b: u8[2] @65535.
# The u16 layout sum in Dtype::is_aligned_struct wraps (65535 + 2 -> 1 == itemsize).
#
# Correct behavior (asserted): jix.read_array raises a regular Exception (RuntimeError for
# InvalidArchive). The child step runs in a subprocess because it can CRASH THE PROCESS:
#   - debug extension (the installed .venv build): PanicException "attempt to add with overflow"
#   - release wheel: the archive loads and reading field 'b' reads 64 KiB past the decoded
#     buffer (SIGSEGV seen in the Rust release repro).
import io
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

import jix


def varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v == 0:
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def get_varint(buf, pos):
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, pos
        shift += 7


def vfield(f, v):
    return varint(f << 3) + varint(v)


def bfield(f, b):
    return varint((f << 3) | 2) + varint(len(b)) + b


def u8_dtype(shape):
    d = bfield(1, vfield(1, 5) + vfield(2, 1))  # SCALAR_KIND_U8, little endian
    if shape:
        d += bfield(3, b"".join(varint(s) for s in shape))
    n = 1
    for s in shape:
        n *= s
    return d + vfield(4, n) + vfield(5, 1)


def struct_dtype(fields, itemsize, align):
    st = b"".join(bfield(1, bfield(1, n.encode()) + vfield(2, o) + bfield(3, d)) for n, o, d in fields)
    return bfield(2, st) + vfield(4, itemsize) + vfield(5, align)


def replace_dtype(arc, new_dtype):
    pos = 4
    for _ in range(2):
        n, pos = get_varint(arc, pos)
        pos += n
    hdr_len_pos = pos
    n, pos = get_varint(arc, pos)
    hdr = arc[pos : pos + n]
    after = pos + n
    new_hdr = bfield(1, new_dtype)
    p = 0
    while p < len(hdr):
        start = p
        key, p = get_varint(hdr, p)
        if key & 7 == 0:
            _, p = get_varint(hdr, p)
        else:
            ln, p = get_varint(hdr, p)
            p += ln
        if key >> 3 != 1:
            new_hdr += hdr[start:p]
    out = arc[:hdr_len_pos] + varint(len(new_hdr)) + new_hdr
    delta = len(out) - after
    toc = bytearray(arc[after : after + 32])
    for s in range(2):
        (o,) = struct.unpack_from("<q", toc, s * 16)
        struct.pack_into("<q", toc, s * 16, o + delta)
    return out + bytes(toc) + arc[after + 32 :]


CHILD = r"""
import sys, jix
path = sys.argv[1]
try:
    a = jix.read_array(path)
except Exception as e:  # correct: InvalidArchive -> RuntimeError
    print("REJECTED", type(e).__name__, e)
    sys.exit(0)
except BaseException as e:  # pyo3 PanicException derives from BaseException
    print("PANIC", type(e).__name__, e)
    sys.exit(3)
print("ACCEPTED", a.dtype, flush=True)
print("field b:", jix.dtype_sub_field(a, "b").numpy(), flush=True)
sys.exit(4)
"""


def main():
    buf = io.BytesIO()
    jix.write_array(jix.asarray(np.array([1, 2, 3, 4], dtype=np.uint8)), buf)
    base = buf.getvalue()

    with tempfile.TemporaryDirectory() as td:
        # Control: a legit {a: u8 @0} struct archive reads back.
        ok_path = os.path.join(td, "ok.jix")
        with open(ok_path, "wb") as f:
            f.write(replace_dtype(base, struct_dtype([("a", 0, u8_dtype([]))], 1, 1)))
        ok = jix.read_array(ok_path)
        assert list(jix.dtype_sub_field(ok, "a").numpy()) == [1, 2, 3, 4]
        print("control ok")

        bad_path = os.path.join(td, "crafted.jix")
        with open(bad_path, "wb") as f:
            f.write(
                replace_dtype(base, struct_dtype([("a", 0, u8_dtype([65535])), ("b", 65535, u8_dtype([2]))], 1, 1))
            )
        r = subprocess.run([sys.executable, "-I", "-c", CHILD, bad_path], capture_output=True, text=True)
        print("child exit status:", r.returncode)
        print("child stdout:", r.stdout.strip())
        if r.stderr.strip():
            print("child stderr (tail):", r.stderr.strip().splitlines()[-1])
        assert r.returncode == 0 and r.stdout.startswith("REJECTED"), (
            "crafted archive with struct field past itemsize was not rejected cleanly"
        )
    print("PASS")


if __name__ == "__main__":
    main()

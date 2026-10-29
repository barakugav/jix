# F040: an array read from a .jix file does not carry the file's filters in its spec, so
# compact() and write_array of a lazy view over it re-encode with the default byte-shuffle
# instead of the file's filters. The same array built in memory keeps its filters.
#
# Asserts the CORRECT behavior (codec settings inherited from the source); fails while the bug
# exists. Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F040.py
import io
import os
import tempfile

import numpy as np

import jix


def varint(b, pos):
    v, shift = 0, 0
    while True:
        byte = b[pos]
        pos += 1
        v |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return v, pos
        shift += 7


def filters_of(buf: bytes):
    """Filters listed in the BlockTableHeader (field 5 of the 3rd length-delimited message)."""
    assert buf[:4] == b"jix1"
    pos = 4
    msgs = []
    for _ in range(3):
        n, pos = varint(buf, pos)
        msgs.append(buf[pos : pos + n])
        pos += n
    hdr, p, out = msgs[2], 0, []
    while p < len(hdr):
        tag, p = varint(hdr, p)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            _, p = varint(hdr, p)
        elif wire == 2:
            n, p = varint(hdr, p)
            body = hdr[p : p + n]
            p += n
            if field == 5:
                out.append({1: "byte-shuffle", 2: "bit-shuffle"}[body[0] >> 3])
        else:
            raise AssertionError(f"wire type {wire}")
    return out


def write(a):
    b = io.BytesIO()
    jix.write_array(a, b)
    return b.getvalue()


failures = []
x = np.arange(4096, dtype=np.uint16).reshape(64, 64)
with tempfile.TemporaryDirectory() as d:
    for filters in (["bit-shuffle"], []):
        c = jix.compact(x, params={"filters": filters})
        f = write(c)
        path = os.path.join(d, f"a{len(filters)}.jix")
        with open(path, "wb") as fh:
            fh.write(f)
        cases = {
            "file": f,
            "c.compact()": write(c.compact()),
            "c + 1": write(c + 1),
        }
        for name, r in (
            ("read_array(BytesIO)", jix.read_array(io.BytesIO(f))),
            ("read_array(path, mmap=True)", jix.read_array(path, mmap=True)),
        ):
            cases[f"{name}: copy"] = write(r)
            cases[f"{name}: r.compact()"] = write(r.compact())
            cases[f"{name}: r + 1"] = write(r + 1)
            assert np.array_equal((r + 1).numpy(), x + 1)  # data stays correct
        print(f"source filters {filters}:")
        for name, buf in cases.items():
            got = filters_of(buf)
            ok = got == filters
            print(f"  {name:40s} {got}{'' if ok else '   <-- expected ' + str(filters)}")
            if not ok:
                failures.append((filters, name, got))

assert not failures, f"re-encodes of file-read arrays dropped the file's filters: {failures}"

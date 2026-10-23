# Repro for F005 (run 2026-10-23-batch3).
#
# A .jix file whose TOC section has offset + size > u64::MAX makes jix.read_array panic
# (pyo3_runtime.PanicException, a BaseException that `except Exception` does not catch) instead of
# raising a normal error. check_section_bounds (jix/src/archive/common.rs) adds offset + size
# unchecked: the debug wheel panics on the add; a release wheel wraps, passes the check, and the
# heap reader then panics with "capacity overflow" allocating `size` bytes.
#
# Each case asserts the CORRECT behavior: read_array raises an ordinary Exception (RuntimeError /
# ValueError) for the crafted file. The script exits 1 while the bug exists.
import io
import os
import struct
import sys
import tempfile

import numpy as np

import jix


def valid_archive_bytes():
    buf = io.BytesIO()
    jix.write_array(jix.compact(np.arange(1, 9, dtype=np.uint8)), buf)
    return bytearray(buf.getvalue())


def toc_position(b):
    # "jix1" | uleb128-len FileMetadata | uleb128-len ArrayHeader | uleb128-len BlockTableHeader | TOC
    assert bytes(b[:4]) == b"jix1"
    pos = 4
    for _ in range(3):
        n, shift = 0, 0
        while True:
            c = b[pos]
            pos += 1
            n |= (c & 0x7F) << shift
            shift += 7
            if c < 0x80:
                break
        pos += n
    off, size = struct.unpack_from("<qQ", b, pos + 16)
    assert off + size == len(b), "TOC not where expected"
    return pos


def crafted(offset, size):
    b = valid_archive_bytes()
    toc = toc_position(b)
    struct.pack_into("<qQ", b, toc + 16, offset, size)  # row 1 = block_data section
    return bytes(b)


CASES = {
    "offset=i64::MAX size=2^63+2": crafted(2**63 - 1, 2**63 + 2),
    "offset=1 size=u64::MAX": crafted(1, 2**64 - 1),
}

failures = []
with tempfile.TemporaryDirectory() as d:
    for name, data in CASES.items():
        path = os.path.join(d, "crafted.jix")
        with open(path, "wb") as f:
            f.write(data)
        for label, call in [
            ("read_array(path)", lambda: jix.read_array(path)),
            ("read_array(path, mmap=True)", lambda: jix.read_array(path, mmap=True)),
            ("read_array(BytesIO, len=n)", lambda: jix.read_array(io.BytesIO(data), len=len(data))),
            # len omitted: bounds vs len are skipped by design, but an overflowing offset + size
            # must still be an error, not a panic.
            ("read_array(BytesIO)", lambda: jix.read_array(io.BytesIO(data))),
        ]:
            try:
                call()
                print(f"[{name}] {label}: loaded without error  -> FAIL")
                failures.append((name, label))
            except Exception as e:
                print(f"[{name}] {label}: {type(e).__name__}: {e}  -> ok")
            except BaseException as e:
                print(f"[{name}] {label}: NOT an Exception: {type(e).__name__}: {e}  -> FAIL")
                failures.append((name, label))
        os.remove(path)

print(f"{len(failures)} failing case(s)")
sys.exit(1 if failures else 0)

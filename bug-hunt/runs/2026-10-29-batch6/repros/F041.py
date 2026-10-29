"""F041: a Minimal-validation load of an archive with one out-of-bounds block location.

BlockTable::new only debug_asserts the block locations, so the dev (debug) extension raises
PanicException at LOAD, while a release wheel loads it and serves the intact blocks. This script
asserts the build-independent behavior (load succeeds, intact blocks read back), so it fails on
the debug extension while the bug exists.
"""

import io
import struct

import numpy as np

import jix

data = np.arange(64, dtype=np.uint32)
a = jix.compact(data, params={"block_shape": [16]})
buf = io.BytesIO()
jix.write_array(a, buf)
b = bytearray(buf.getvalue())

# TOC: [Section; 2] = {offset: i64, size: u64} x 2; first = blocks_loc (4 * 12 bytes) right after it.
toc = next(p for p in range(len(b) - 32) if struct.unpack_from("<qQ", b, p) == (p + 32, 48))
loc3 = toc + 32 + 3 * 12
struct.pack_into("<I", b, loc3 + 8, 0xFFFFFFFF)

try:
    jix.read_array(io.BytesIO(bytes(b)), len=len(b), validation="strict")
    raise AssertionError("strict should reject the archive")
except AssertionError:
    raise
except Exception as e:  # noqa: BLE001
    print("strict:", type(e).__name__, e)

try:
    m = jix.read_array(io.BytesIO(bytes(b)), len=len(b), validation="minimal")
except BaseException as e:  # noqa: BLE001 - PanicException derives from BaseException
    print("minimal load:", type(e).__name__, str(e).splitlines()[0])
    raise SystemExit(1)
got = np.asarray(m[0:48].numpy())
assert (got == data[:48]).all(), got
print("minimal load OK, intact blocks read back")

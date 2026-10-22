# Repro for F004 (run 2026-10-22-trial2): a .jix block whose zstd frame decodes to fewer bytes than
# the block (here 0 bytes, padded to the original length with a skippable frame) makes
# Array.numpy(out=dst) raise - but only after the filter decode has copied the never-written,
# uninitialized pooled scratch into dst. Asserts the correct behavior (dst untouched), so it fails
# while the bug exists.
import io
import struct

import numpy as np

import jix

rng = np.random.default_rng(1)
data = rng.integers(0, 2**32, size=1024, dtype=np.uint32)  # incompressible
buf = io.BytesIO()
jix.write_array(jix.compact(data, params={"block_shape": [1024]}), buf)
b = bytearray(buf.getvalue())
pos = b.find(b"\x28\xb5\x2f\xfd")
assert pos >= 0 and b.find(b"\x28\xb5\x2f\xfd", pos + 4) < 0
flen = len(b) - pos  # single block: its frame runs to the end of the archive
# zstd frame of 0 bytes (single segment, FCS = 0, one empty raw last block) + skippable pad frame
repl = bytearray(b"\x28\xb5\x2f\xfd\x20\x00\x01\x00\x00")
pad = flen - len(repl) - 8
repl += struct.pack("<II", 0x184D2A50, pad) + bytes(pad)
b[pos:] = repl

a = jix.read_array(io.BytesIO(bytes(b)))  # default (strict) validation accepts it
dst = np.zeros(1024, dtype=np.uint32)
try:
    a.numpy(out=dst)
    print("no exception (unexpected)")
except Exception as e:
    print("raised:", type(e).__name__, e)
nz = int(np.count_nonzero(dst))
print("nonzero elements in dst after the exception:", nz, "of", dst.size)
print("first nonzero bytes:", dst.view(np.uint8)[dst.view(np.uint8) != 0][:16])
assert nz == 0, "dst was overwritten with bytes the decoder never produced, although the read failed"

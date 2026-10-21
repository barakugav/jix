"""F001: a block whose zstd payload decodes short is accepted (only a debug_assert checks it).

Debug build of the extension (.venv): PanicException at src/storage/block.rs:193.
Release wheels: returns stale / uninitialized bytes as array data.
"""

import io

import jix
import numpy as np

MAGIC = b"\x28\xb5\x2f\xfd"


def frame_len(b: bytes, pos: int) -> int:
    """Length of the zstd frame starting at pos (minimal parser, no dictionary support needed)."""
    fhd = b[pos + 4]
    fcs_flag, single, checksum, did_flag = fhd >> 6, (fhd >> 5) & 1, (fhd >> 2) & 1, fhd & 3
    hdr = 1 + (0 if single else 1) + [0, 1, 2, 4][did_flag] + [1 if single else 0, 2, 4, 8][fcs_flag]
    i = pos + 4 + hdr
    while True:
        bh = int.from_bytes(b[i : i + 3], "little")
        last, btype, bsize = bh & 1, (bh >> 1) & 3, bh >> 3
        i += 3 + (1 if btype == 1 else bsize)
        if last:
            break
    return i + (4 if checksum else 0) - pos


def shorten_first_frame(b: bytearray, payload: bytes) -> None:
    pos = b.index(MAGIC)
    n = frame_len(b, pos)
    # magic, FHD (single segment, 1-byte FCS), FCS, block header (last, raw, size), payload
    repl = MAGIC + bytes([0x20, len(payload)]) + (1 | (len(payload) << 3)).to_bytes(3, "little") + payload
    pad = n - len(repl) - 8
    assert pad >= 0
    repl += (0x184D2A50).to_bytes(4, "little") + pad.to_bytes(4, "little") + bytes(pad)
    b[pos : pos + n] = repl


rng = np.random.default_rng(0)
a = jix.compact(rng.integers(0, 256, 64, dtype=np.uint8), params={"block_shape": [64], "filters": []})
buf = io.BytesIO()
jix.write_array(a, buf)
raw = bytearray(buf.getvalue())
shorten_first_frame(raw, b"\xee" * 8)

r = jix.read_array(io.BytesIO(bytes(raw)))  # default validation = strict; accepted
print("read_array accepted the corrupt archive, shape", r.shape)
try:
    out = r.numpy()
    print("numpy() returned (release behavior):", list(out))
except BaseException as e:  # pyo3_runtime.PanicException derives from BaseException
    print(f"{type(e).__name__}: {e}")

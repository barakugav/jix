"""F016 repro: a .jix header dim > i64::MAX overflows next_multiple_of in read_from.

Asserts the CORRECT behavior (read_array raises RuntimeError for the malformed header), so it
fails while the bug exists: the debug extension raises pyo3_runtime.PanicException ("attempt to
add with overflow"), which derives from BaseException and is not caught below.
"""

import io
import struct

import numpy as np

import jix


def read_varint(b, pos):
    v, shift = 0, 0
    while True:
        byte = b[pos]
        pos += 1
        v |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return v, pos
        shift += 7


def varint(v):
    out = bytearray()
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def array_header(shape, block_shape):
    msg = b""
    for field, vals in ((1, shape), (2, block_shape)):
        body = b"".join(varint(v) for v in vals)
        msg += varint((field << 3) | 2) + varint(len(body)) + body
    return msg


def with_header(arr, shape, block_shape):
    buf = io.BytesIO()
    jix.write_array(arr, buf)
    b = buf.getvalue()
    pos, msgs = 4, []
    for _ in range(3):
        n, pos = read_varint(b, pos)
        msgs.append(b[pos : pos + n])
        pos += n
    msgs[1] = array_header(shape, block_shape)
    out = bytearray(b[:4])
    for m in msgs:
        out += varint(len(m)) + m
    delta = len(out) - pos
    for i in range(2):
        off, size = struct.unpack_from("<qQ", b, pos + 16 * i)
        out += struct.pack("<qQ", off + delta, size)
    out += b[pos + 32 :]
    return bytes(out)


def main():
    a = jix.compact(np.array([7, 7], dtype=np.uint8))
    # Sanity: rewriting with the original header still reads.
    ok = jix.read_array(io.BytesIO(with_header(a, [2], [2])))
    assert ok.shape == (2,), ok.shape

    crafted = with_header(a, [2**64 - 1], [2])
    try:
        jix.read_array(io.BytesIO(crafted))
    except RuntimeError as e:
        print("OK: clean error:", e)
    else:
        raise AssertionError("malformed header accepted")


if __name__ == "__main__":
    main()

"""K188: ArchiveReader::new trusts the reader's tell() on every read (stream_len restores the
position with an absolute seek to tell()), so a file-like whose tell() always returns 0 - the
reader #107's test_reader_with_bad_tell_after_packed_archives was written for - silently gets
the FIRST archive back on the second of two back-to-back reads.

Asserts the CORRECT behavior: the second read returns b (or raises). Fails while the bug exists.
The NoSeekEnd case is printed for information only (not asserted).
"""

import io

import numpy as np

import jix

a = jix.compact(np.arange(10, dtype=np.int32))
b = jix.compact(np.arange(100, 110, dtype=np.int32))
buf = io.BytesIO()
jix.write_array(a, buf)
off = buf.tell()
jix.write_array(b, buf)
data = buf.getvalue()
print(f"archive a: [0, {off}), archive b: [{off}, {len(data)})")


class TellAlwaysZero(io.BytesIO):
    """BytesIO whose tell() always reports 0 (seek() still returns the true position)."""

    def tell(self):
        return 0


class NoSeekEnd(io.BytesIO):
    def seek(self, pos, whence=0):
        if whence == 2:
            raise io.UnsupportedOperation("SEEK_END not supported")
        return super().seek(pos, whence)


# Control: a well-behaved reader.
r = io.BytesIO(data)
x, y = jix.read_array(r).numpy(), jix.read_array(r).numpy()
print("control BytesIO:", x.tolist(), y.tolist())
assert y.tolist() == list(range(100, 110))

# Informational: len= given, seek(0, 2) unsupported.
try:
    print("NoSeekEnd len=off:", jix.read_array(NoSeekEnd(data), len=off).numpy().tolist())
except Exception as e:  # noqa: BLE001
    print(f"NoSeekEnd len=off: raised {type(e).__name__}: {e}")

# The bug.
r = TellAlwaysZero(data)
first = jix.read_array(r).numpy()
print("TellAlwaysZero first read:", first.tolist(), "real position after:", io.BytesIO.tell(r))
try:
    second = jix.read_array(r).numpy()
except Exception as e:  # noqa: BLE001 - an error would be acceptable
    print(f"TellAlwaysZero second read: raised {type(e).__name__}: {e} (acceptable)")
else:
    print("TellAlwaysZero second read:", second.tolist(), "real position after:", io.BytesIO.tell(r))
    assert second.tolist() == list(range(100, 110)), (
        f"second read silently returned {second.tolist()} (archive a again) instead of b"
    )
print("OK")

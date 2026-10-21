"""F007: mmap=True rejects a valid archive that mmap=False reads, when the archive sits at a file
offset with a different residue mod 8 than the stream position it was written at."""

import io
import os
import tempfile

import numpy as np

import jix

src = np.array([10.0, 20.0, 30.0, 40.0])
buf = io.BytesIO()
jix.write_array(jix.asarray(src), buf)  # archive written at stream position 0
blob = buf.getvalue()

with tempfile.TemporaryDirectory() as d:
    for k in range(8):
        p = os.path.join(d, f"embedded_{k}.jix")
        with open(p, "wb") as f:
            f.write(b"H" * k)  # e.g. a small user header, or a previous archive
            f.write(blob)
        heap = jix.read_array(p, offset=k, len=len(blob)).numpy()
        assert np.array_equal(heap, src)
        try:
            m = jix.read_array(p, offset=k, len=len(blob), mmap=True).numpy()
            res = f"ok {m.tolist()}"
        except Exception as e:  # noqa: BLE001
            res = f"{type(e).__name__}: {e}"
        print(f"offset {k}: mmap=False ok {heap.tolist()} | mmap=True {res}")

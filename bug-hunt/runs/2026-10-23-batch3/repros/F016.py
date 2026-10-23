# F016 repro: jix.write_array's docstring lists gzip.GzipFile as a supported seekable writer, but
# the archive writer seeks BACKWARDS in finalize (jix/src/archive/block.rs:214, :231) to patch the
# block-locations section and the TOC placeholder. GzipFile in write mode only supports forward
# seeks, so every write_array to a GzipFile raises.
#
# Asserts the CORRECT (documented) behavior: the write succeeds and the archive round-trips.
# Fails while the bug exists.
import gzip
import io

import numpy as np

import jix


def main():
    src = np.arange(10, dtype=np.float32)
    a = jix.compact(src)
    raw = io.BytesIO()
    err = None
    try:
        with gzip.GzipFile(fileobj=raw, mode="wb") as g:
            jix.write_array(a, g)
    except Exception as e:  # noqa: BLE001
        err = e
    print("write_array to GzipFile:", "ok" if err is None else f"{type(err).__name__}: {err}")

    # Also a lazy view (different write path: compress-on-the-fly) and an empty array.
    for name, arr in [("lazy view", jix.compact(src) + 1.0), ("empty", jix.compact(np.zeros(0, np.float32)))]:
        r = io.BytesIO()
        try:
            with gzip.GzipFile(fileobj=r, mode="wb") as g:
                jix.write_array(arr, g)
            print(f"  {name}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: {type(e).__name__}: {e}")

    # Control: the same array to a plain BytesIO works.
    ctrl = io.BytesIO()
    jix.write_array(a, ctrl)
    ctrl.seek(0)
    print("control BytesIO roundtrip ok:", np.array_equal(jix.read_array(ctrl).numpy(), src))

    assert err is None, f"write_array to gzip.GzipFile (documented as supported) failed: {err!r}"
    back = jix.read_array(io.BytesIO(gzip.decompress(raw.getvalue())))
    np.testing.assert_array_equal(back.numpy(), src)
    print("PASS")


if __name__ == "__main__":
    main()

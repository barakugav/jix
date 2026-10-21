"""F008 repro: on i686 Linux numpy aligns f8/i8/u8/c16 to 4, jix hard-codes 8.

Run with .venv/bin/python bug-hunt/runs/2026-10-21-pilot/repros/F008.py

On i686 Linux (manylinux / musllinux x86 wheel + a numpy 2 built for i686, e.g. Debian i386 or
Alpine x86 python3-numpy) every case marked [bug] raises
  ValueError: numpy dtype itemsize or alignment mismatch: (8, 4) != (8, 8)
or, for the aligned struct, 'Unsupported struct dtype: ... not in a valid packed or aligned layout'.
On x86_64 / aarch64 / armv7 / win32 numpy reports alignment 8 and every case succeeds (control).
"""

import platform
import struct
import sys

import numpy as np

import jix

print(f"python {sys.version.split()[0]} machine={platform.machine()} pointer={struct.calcsize('P') * 8}-bit")
print(f"numpy {np.__version__}")
for code in ["f8", "i8", "u8", "c16", "f4"]:
    print(f"  np.dtype({code!r}).alignment = {np.dtype(code).alignment}")
sd = np.dtype([("a", "f8"), ("b", "i4")], align=True)
print(f"  aligned struct f8,i4: itemsize={sd.itemsize} alignment={sd.alignment} offsets={[sd.fields[n][1] for n in sd.names]}")
print("jix dtype alignment for f8/i8/u8/c16 is ScalarKind::alignment() = 8 on every target")


def case(name, fn):
    try:
        r = fn()
        print(f"[ok ] {name}: {r!r}"[:200])
    except Exception as e:  # noqa: BLE001
        print(f"[bug] {name}: {type(e).__name__}: {e}")


case("asarray(np.zeros(3)).dtype", lambda: jix.asarray(np.zeros(3)).dtype)
case("asarray(np.zeros(3)).numpy()", lambda: jix.asarray(np.zeros(3)).numpy())
case("compact(np.zeros(3)).numpy()", lambda: jix.compact(np.zeros(3)).numpy())
case("compact([1.0, 2.0], dtype='float64').dtype", lambda: jix.compact([1.0, 2.0], dtype="float64").dtype)
case("(asarray(np.arange(4)) + 1).numpy()  # int64", lambda: (jix.asarray(np.arange(4)) + 1).numpy())
case("asarray(np.zeros(4, 'u8'))[1:]", lambda: jix.asarray(np.zeros(4, "u8"))[1:])
case("asarray(np.zeros(3, 'c16')).numpy()", lambda: jix.asarray(np.zeros(3, "c16")).numpy())
case("reshape(asarray(np.zeros(6)), (2, 3)).numpy()", lambda: jix.reshape(jix.asarray(np.zeros(6)), (2, 3)).numpy())
case("asarray(np.zeros(3, f32)).numpy()  # control: 4-byte dtype", lambda: jix.asarray(np.zeros(3, np.float32)).numpy())
case("asarray(np.zeros(3, aligned struct f8,i4))", lambda: jix.asarray(np.zeros(3, dtype=sd)).shape)

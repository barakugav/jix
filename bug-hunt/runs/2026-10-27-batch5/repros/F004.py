# F004 repro (run 2026-10-27-batch5): a widening cast of a valid lazy array whose WIDENED byte
# size no longer fits in u64 raises pyo3_runtime.PanicException (a BaseException, not caught by
# `except Exception`) instead of RuntimeError. jix-py's cast_impl (jix-py/src/ops/mod.rs:116)
# calls the panicking convenience `Array::cast` (jix/src/ops/cast.rs:485 = new_array(..).unwrap())
# instead of the fallible `Cast::new_array(..)`.
#
# Asserts the CORRECT behavior (RuntimeError, like the fallible paths in the control cases), so
# it FAILS while the bug exists. Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F004.py
import numpy as np
import jix

N = 2**62  # u8 byte size 2**62 fits u64 / i64; at itemsize >= 4 it does not
big_u8 = jix.broadcast(jix.asarray(np.zeros(1, np.uint8)), (N,))
big_i8 = jix.broadcast(jix.asarray(np.ones(1, np.int8)), (N,))
print("valid lazy array:", big_u8.shape, big_u8.dtype)

bug_cases = {
    "jix.cast(big_u8, np.float64)": lambda: jix.cast(big_u8, np.float64),
    "jix.asarray(big_u8, dtype=np.float32)": lambda: jix.asarray(big_u8, dtype=np.float32),
    "big_u8 + 1.5": lambda: big_u8 + 1.5,
    "big_i8 * np.int64(3)": lambda: big_i8 * np.int64(3),
    "big_u8 == np.float32(1)": lambda: big_u8 == np.float32(1),
    "jix.where(True, big_u8, 1.5)": lambda: jix.where(True, big_u8, 1.5),
    "jix.concatenate([big_u8, np.zeros(1, np.int32)])": lambda: jix.concatenate(
        [big_u8, np.zeros(1, np.int32)]
    ),
    # promotion runs before the shape check, so this panics before the shape mismatch is seen
    "jix.stack([big_u8, 1.5])": lambda: jix.stack([big_u8, 1.5]),
    "jix.compact(big_u8, dtype=np.float64)": lambda: jix.compact(big_u8, dtype=np.float64),
}
# Same overflow, but detected by a core constructor that jix-py calls fallibly -> RuntimeError.
control_cases = {
    "jix.add(1.5, big_u8)": lambda: jix.add(1.5, big_u8),
    "big_u8 + jix.asarray(np.zeros((1, 1), np.float32)) (broadcast first)": lambda: big_u8
    + jix.asarray(np.zeros((1, 1), np.float32)),
}

# Not affected (same-width cast u8 -> i8 cannot overflow): negation of unsigned works.
print("-big_u8 ->", (-big_u8).dtype, "(no error, as expected)")

failed = []
for group, cases in (("bug", bug_cases), ("control", control_cases)):
    for name, f in cases.items():
        try:
            f()
            print(f"[{group}] {name}: no error")
            failed.append(name)
        except RuntimeError as e:
            print(f"[{group}] {name}: RuntimeError (ok): {str(e)[:90]}")
        except BaseException as e:  # pyo3_runtime.PanicException derives from BaseException
            print(f"[{group}] {name}: {type(e).__module__}.{type(e).__name__} "
                  f"(isinstance Exception: {isinstance(e, Exception)}): {str(e)[:120]}")
            failed.append(name)

assert not failed, f"{len(failed)} case(s) did not raise RuntimeError: {failed}"
print("all cases raise RuntimeError")

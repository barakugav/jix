# F015 repro (bug-hunt run 2026-10-22-trial2): f64 -> f16 conversion double-rounds via f32.
# Asserts the CORRECT behavior (numpy's directly rounded float16), so it fails while the bug exists.
# Run: .venv/bin/python bug-hunt/runs/2026-10-22-trial2/repros/F015.py
import numpy as np

import jix

x = 1 + 2**-11 + 2**-40  # just above the f16 midpoint between 1.0 and 1 + 2**-10
want = float(np.float16(x))
assert want == 1 + 2**-10, want

a = np.zeros(2, np.float16)
results = {
    "add(f16 arr, py float)": jix.add(a, x).numpy(),
    "maximum(f16 arr, py float)": jix.maximum(a, x).numpy(),
    "where(c, py float, f16 arr)": jix.where(np.array([True, True]), x, a).numpy(),
    "cast(f64 arr, f16)": jix.cast(np.array([x, x]), np.float16).numpy(),
    "asarray(f64 arr, dtype=f16)": jix.asarray(np.array([x, x]), dtype=np.float16).numpy(),
}
failed = []
for name, got in results.items():
    ok = got.dtype == np.float16 and got.tolist() == [want, want]
    print(f"{name:30s} -> {got.tolist()} {got.dtype}  numpy: {[want, want]}  {'OK' if ok else 'WRONG'}")
    if not ok:
        failed.append(name)

rng = np.random.default_rng(0)
v = rng.standard_normal(1_000_000) * 100
got = jix.cast(jix.compact(v), np.float16).numpy()
exp = v.astype(np.float16)
bad = np.flatnonzero(got.view(np.uint16) != exp.view(np.uint16))
print(f"random f64 -> f16 cast: {bad.size} mismatches of {v.size};",
      [(float(v[i]), float(got[i]), float(exp[i])) for i in bad[:3]])
if bad.size:
    failed.append("random sweep")

assert not failed, f"wrong f16 rounding in: {failed}"

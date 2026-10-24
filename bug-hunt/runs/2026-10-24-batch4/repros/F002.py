"""F002 repro: jix.var / jix.std of values whose deviations overflow f64.

Asserts the correct result (+inf, as numpy gives), so it fails while the bug exists.
In the debug dev build jix.std raises pyo3_runtime.PanicException
('assertion failed: status.is_ok()'); in a release build it returns nan.
"""

import warnings

import numpy as np

import jix

warnings.simplefilter("ignore", RuntimeWarning)  # numpy warns about the overflow

cases = {
    "f64 [1e308,-1e308]": (np.array([1e308, -1e308]), None),
    "f64 [1e308,-1e308,0]": (np.array([1e308, -1e308, 0.0]), None),
    "c128 [1e308j,-1e308j]": (np.array([1e308j, -1e308j]), None),
    "f64 [[1.7e308,-1.7e308]]*3 dim=1": (np.array([[1.7e308, -1.7e308]] * 3), 1),
}

failures = []
for name, (x, dim) in cases.items():
    np_var = np.var(x, axis=dim)
    got = jix.var(jix.asarray(x), dim=dim).numpy()
    print(f"var  {name}: jix={got} numpy={np_var}")
    if not np.array_equal(got, np_var):
        failures.append(f"var {name}: jix={got} numpy={np_var}")
    np_std = np.std(x, axis=dim)
    try:
        got = jix.std(jix.asarray(x), dim=dim).numpy()
    except BaseException as e:  # PanicException derives from BaseException
        print(f"std  {name}: jix raised {type(e).__name__}: {e}")
        failures.append(f"std {name}: raised {type(e).__name__}: {e}")
        continue
    print(f"std  {name}: jix={got} numpy={np_std}")
    if not np.array_equal(got, np_std):
        failures.append(f"std {name}: jix={got} numpy={np_std}")

assert not failures, "\n".join(failures)
print("OK")

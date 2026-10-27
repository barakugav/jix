# F022: f16 tan(+-177.5) overflows to +-inf when the f32 result is rounded to f16, but jix
# reports no Overflow (no error under errstate(over="raise"), no warning by default).
# numpy reports "overflow encountered in tan"; jix's own f16 exp reports its rounding overflow.
# Asserts the CORRECT behavior, so it fails while the bug exists.
import warnings

import jix
import numpy as np

a = np.array([177.5, -177.5], dtype=np.float16)

# numpy reference
with np.errstate(over="raise"):
    try:
        np.tan(a)
        np_raised = None
    except FloatingPointError as e:
        np_raised = str(e)
print("numpy float16 tan, over=raise:", np_raised)

# jix control: f16 exp(12) overflows only by rounding to f16, and is reported
with jix.errstate(over="raise"):
    try:
        jix.exp(np.array([12.0], dtype=np.float16)).numpy()
        exp_raised = None
    except Exception as e:
        exp_raised = f"{type(e).__name__}: {e}"
print("jix f16 exp([12.0]), over=raise:", exp_raised)

# jix tan under over=raise
with jix.errstate(over="raise"):
    try:
        r = jix.tan(a).numpy()
        tan_raised = None
        print("jix f16 tan([177.5, -177.5]), over=raise: returned", r, "(no error)")
    except Exception as e:
        tan_raised = f"{type(e).__name__}: {e}"
        print("jix f16 tan, over=raise: raised", tan_raised)

# jix tan with the default errstate (over='warn')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = jix.tan(a).numpy()
    tan_warnings = [str(x.message) for x in w]
print("jix f16 tan default errstate:", r, "warnings:", tan_warnings)

assert np_raised is not None
assert exp_raised is not None
assert tan_raised is not None, "jix f16 tan overflowed to inf with over='raise' but raised nothing"
assert any("overflow" in m for m in tan_warnings), "no overflow warning in the default errstate"

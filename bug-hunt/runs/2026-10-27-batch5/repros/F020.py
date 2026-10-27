# F020: an unbounded (f32, never saturated) read-cost hint reaches +inf on a lazy pipeline, and a
# zero-length concatenate member turns it into NaN; the next multi-input op then panics.
#
# The dev extension is a debug build: it panics at the debug_assert in Array::from_storage
# (array.rs:1662) on the add that first makes the cost inf (step ~124). A release wheel carries
# the inf, the first concatenate makes the cost NaN (0 * inf), and the second concatenate (or
# `c + o`) panics in max_cost_layout_order (params.rs:912, partial_cmp().unwrap()).
#
# Asserts the CORRECT behavior, so it fails while the bug exists.
import numpy as np

import jix

x = jix.compact(np.zeros(0, dtype=np.float32))
for i in range(130):
    x = x + x  # cost c -> 2c + 1, exceeds f32::MAX (~2^128) after ~125 steps
assert x.shape == (0,)

o = jix.compact(np.array([5.0], dtype=np.float32))
c = jix.concatenate([x, o])  # zero-length member: weight 0 * inf cost = NaN
c2 = jix.concatenate([c, o])  # release: panics in max_cost_layout_order
np.testing.assert_array_equal(c2.numpy(), np.array([5.0, 5.0], dtype=np.float32))
np.testing.assert_array_equal((c + o).numpy(), np.array([10.0], dtype=np.float32))
print("OK")

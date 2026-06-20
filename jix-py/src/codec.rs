use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A context that holds reusable decompressor state across multiple array reads.
///
/// Allocating temporary buffers and initializing a codec decoder on every block read can be
/// expensive, especially for small blocks. `ReadContext` holds a long-lived decompressor
/// instance and reusable scratch buffers. Passing the same `ReadContext` to successive reads
/// amortizes these costs.
///
/// For most workloads you do not need to create one explicitly - functions that read array
/// data (such as [`Array.numpy()`][jix.Array.numpy] and [`jix.compact()`][jix.compact]) create a
/// context internally when none is provided. Pass an explicit `ReadContext` when you are doing many
/// successive reads and want to avoid the repeated initialization overhead.
///
/// A single `ReadContext` instance may be reused across multiple calls and across multiple
/// arrays.
///
/// Note:
///     `ReadContext` is intended to be used from a single thread. Although passing one to
///     concurrent calls is not a hard error (accesses are serialized internally), doing so
///     defeats the purpose - threads would block on each other and lose the benefit of reuse.
///     Create one `ReadContext` per thread for concurrent workloads instead.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact(np.arange(1000, dtype=np.int32).reshape(100, 10))
///
///     # Reuse one context across many reads to amortize decompressor setup
///     ctx = jix.ReadContext()
///     for i in range(100):
///         row = a.numpy(i, context=ctx)
///     ```
#[gen_stub_pyclass]
#[pyclass(module = "jix", frozen)]
pub struct ReadContext(Mutex<jix_core::ReadContext>);
impl ReadContext {
    pub(crate) fn from_core(ctx: jix_core::ReadContext) -> Self {
        Self(Mutex::new(ctx))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ReadContext {
    /// Creates a new `ReadContext` with the default configuration.
    ///
    /// Prefer [`Array.read_ctx()`][jix.Array.read_ctx], which derives the parameters from the array.
    ///
    /// Returns:
    ///     A new [`jix.ReadContext`][jix.ReadContext] with the default configuration.
    #[new]
    pub fn new() -> PyResult<Self> {
        let ctx = jix_core::ReadContext::default();
        Ok(Self(Mutex::new(ctx)))
    }
}
impl ReadContext {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, jix_core::ReadContext> {
        self.0.lock().unwrap()
    }
}

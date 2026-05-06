use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

use pyo3::prelude::*;

use crate::storage::DynStorage;
use crate::util::{IntoPyResult, OrKwargs};
use crate::{Array, ArrayParams, ReadContext};

/// Load a compressed array from a `.zix` file.
///
/// `path` is the path to the `.zix` file.
///
/// `offset` and `len` allow reading a single array from a byte range within a larger file —
/// useful when multiple arrays have been packed into the same file with back-to-back
/// writes. `offset` defaults to `0`; `len` defaults to the remaining file size from `offset`.
///
/// `mmap` maps the file into virtual address space instead of copying its bytes onto the heap.
/// The OS pages blocks in on demand, so startup is fast and only the blocks you actually read
/// are loaded into physical memory. Defaults to `False`. **Caution:** if the underlying file
/// is modified while the returned array (or any view derived from it) is live, the behavior is
/// undefined.
///
/// `params` controls how the array is read and decoded. Accepts a :class:`zix.ArrayParams`
/// instance or a plain `dict`. See :class:`zix.ArrayParams` for details.
///
/// # Examples
/// ```python,ignore
/// import zix
///
/// # Read the whole file
/// a = zix.read_array("data.zix")
/// assert a.shape == (4, 4)
///
/// # Read two arrays packed back-to-back in one file
/// with open("packed.zix", "wb") as f:
///     zix.write_array(a, f)
///     offset = f.tell()
///     zix.write_array(b, f)
///     total = f.tell()
/// b2 = zix.read_array("packed.zix", offset=offset, len=total - offset)
///
/// # Memory-mapped read (zero-copy, fast startup)
/// c = zix.read_array("large.zix", mmap=True)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (path, *, params=None, offset=0, len=None, mmap=false))]
pub fn read_array(
    py: Python,
    path: PathBuf,
    params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
    offset: u64,
    len: Option<u64>,
    mmap: bool,
) -> PyResult<Array> {
    let params = ArrayParams::resolve(py, params)?;

    py.detach(|| {
        let len = match len {
            Some(len) => len,
            None => {
                let file_len = path.metadata()?.len();
                file_len.checked_sub(offset).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("offset is out of bounds")
                })?
            }
        };

        Ok(if !mmap {
            let array = zix_core::Array::read_from_file_section(&path, offset, len, params)
                .into_py_result()?;
            Array::from_core_storage(array.into_storage())
        } else {
            let array = unsafe {
                zix_core::Array::read_from_file_mmap(&path, offset, len, params).into_py_result()?
            };
            Array::from_core_storage(array.into_storage())
        })
    })
}

/// Write a compressed array to a file or a file-like object.
///
/// Works for any array type: a compact array (the result of :func:`zix.compact` or
/// :func:`zix.read_array`) streams its already-compressed blocks directly to the destination
/// without decompressing — `params` is ignored in this case and no re-compression takes place.
/// A lazy view (slice, arithmetic op chain, etc.) compresses on the fly, so the full
/// data (compressed or decompressed) is never held in memory.
///
/// `path_or_writer` is either a file path or any seekable binary file-like object with
/// `.write()`, `.seek()`, and `.tell()` methods (e.g. `io.BytesIO`, an open file handle,
/// `gzip.GzipFile`).
///
/// `append` controls how a file path is opened. When `False` (default), the file is created
/// anew and must not already exist. When `True`, the array is written to the end of an existing
/// file (or a new file is created if it does not yet exist). Passing `append=True` alongside a
/// file-like object is ignored — the writer is used as-is at its current position.
/// This argument is only relevant when `path_or_writer` is a file path.
///
/// `params` controls the block layout and codec used to encode the output. Accepts a
/// :class:`zix.ArrayParams` instance or a plain `dict` (e.g. `{"block_shape": [64, 64]}`). Any
/// field not set is inherited from the source array's storage. Ignored when the source array is
/// already compact. See :class:`zix.ArrayParams` for details.
///
/// `context` is an optional :class:`zix.ReadContext` to reuse when reading the source array.
/// When omitted, a context is created internally. See :class:`zix.ReadContext` for details.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import io
///
/// a = zix.compact([[1.0, 2.0], [3.0, 4.0]])
///
/// # Write to a new file (file must not exist)
/// zix.write_array(a, "output.zix")
///
/// # Pack two arrays back-to-back into the same file
/// b = zix.compact([10, 20, 30])
/// zix.write_array(a, "packed.zix")
/// zix.write_array(b, "packed.zix", append=True)
///
/// # Write to an in-memory buffer
/// buf = io.BytesIO()
/// zix.write_array(a, buf)
///
/// # Streaming pipeline: read via mmap, apply a lazy op, write without materializing
/// src = zix.read_array("large.zix", mmap=True)
/// ctx = zix.ReadContext()
/// view = src.exp() + 1.0
/// with open("modified.zix", "wb") as f:
///     zix.write_array(view, f, context=ctx)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, path_or_writer, *, append=false, params=None, context=None))]
pub fn write_array(
    array: &Bound<'_, Array>,
    path_or_writer: &Bound<'_, PyAny>,
    append: bool,
    params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
    context: Option<&Bound<'_, ReadContext>>,
) -> PyResult<()> {
    let py = array.py();
    let path_or_writer = PathOrWriter::from_pyany(path_or_writer)?;
    let array = &array.get().arr;
    let params = ArrayParams::resolve(py, params)?;
    let context = context.map(|ctx| ctx.get());

    match path_or_writer {
        PathOrWriter::Path(path) => {
            // detach the GIL
            py.detach(|| {
                let file = if !append {
                    std::fs::File::create_new(path)?
                } else {
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .append(true)
                        .open(path)?;
                    file.seek(std::io::SeekFrom::End(0))?;
                    file
                };
                let writer = BufWriter::new(file);
                write_array_impl(array, writer, params, context)
            })
        }
        PathOrWriter::PyWriter(writer) => {
            // cant detach the GIL here since the writer is a Python object
            let writer = BufWriter::new(writer);
            write_array_impl(array, writer, params, context)
        }
    }
}
fn write_array_impl<W>(
    array: &zix_core::Array<DynStorage>,
    mut writer: W,
    params: zix_core::ArrayParams,
    context: Option<&ReadContext>,
) -> PyResult<()>
where
    W: Write + Seek,
{
    let context_guard;
    let context = match context {
        Some(ctx) => {
            context_guard = ctx.lock();
            &*context_guard
        }
        None => &array.read_ctx(),
    };

    array
        .write_to_with(&mut writer, params, context)
        .into_py_result()?;

    writer.flush()?;
    Ok(())
}

enum PathOrWriter<'a> {
    Path(PathBuf),
    PyWriter(PyWriter<'a>),
}
impl<'a> PathOrWriter<'a> {
    fn from_pyany(pyany: &'a Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(path) = pyany.extract::<PathBuf>() {
            Ok(Self::Path(path))
        } else if pyany.hasattr("write")? {
            Ok(Self::PyWriter(PyWriter::new(pyany.clone())?))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a file path or a file-like object with a 'write' method",
            ))
        }
    }
}

/// Wraps any Python file-like object that has `.write()`, `.seek()`, `.tell()` methods, and optionally `.flush()`.
#[derive(Debug)]
struct PyWriter<'py> {
    inner: Bound<'py, PyAny>,
    write_fn: Bound<'py, PyAny>,
    seek_fn: Bound<'py, PyAny>,
    tell_fn: Bound<'py, PyAny>,
    flush_fn: Option<Bound<'py, PyAny>>,
}

impl<'py> PyWriter<'py> {
    fn new(inner: Bound<'py, PyAny>) -> PyResult<Self> {
        let write_fn = inner.getattr("write").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("Object must have a 'write' method")
        })?;
        let seek_fn = inner.getattr("seek").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("Object must have a 'seek' method")
        })?;
        let tell_fn = inner.getattr("tell").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("Object must have a 'tell' method")
        })?;
        let flush_fn = inner.getattr("flush").ok();
        Ok(Self {
            inner,
            write_fn,
            seek_fn,
            tell_fn,
            flush_fn,
        })
    }
}

impl Write for PyWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let py = self.inner.py();

        // SAFETY: We create a read-only memoryview pointing directly into
        // `buf`.  This is sound as long as the memoryview does not outlive
        // this call — i.e. Python's .write() must not stash a reference to
        // the memoryview (or a slice of it) beyond returning.
        //
        // All standard library IO implementations (FileIO, BufferedWriter,
        // BytesIO, GzipFile, etc.) consume the buffer immediately.  This is
        // the same contract CPython's own C-level IO code relies on.
        let py_buf: Bound<'_, PyAny> = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                pyo3::ffi::PyMemoryView_FromMemory(
                    buf.as_ptr().cast_mut().cast::<std::ffi::c_char>(), // cast away const — we mark readonly below
                    buf.len() as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyBUF_READ, // read-only access
                ),
            )
            .map_err(py_to_io_err)?
        };

        let result = self.write_fn.call1((&py_buf,)).map_err(py_to_io_err)?;

        // Explicitly release the memoryview before we return, ensuring
        // Python can't hold a reference into `buf` after this point.
        // (The refcount drop on `mv` would do this anyway, but
        // calling .release() makes the invalidation immediate and
        // deterministic — any stashed reference becomes a dead
        // memoryview that raises ValueError on access.)
        let _ = py_buf.call_method0("release");

        if result.is_none() {
            return Ok(buf.len());
        }
        result.extract::<usize>().map_err(py_to_io_err)
    }

    fn flush(&mut self) -> io::Result<()> {
        // flush() is optional on the Python side.
        if let Some(flush_fn) = &self.flush_fn {
            flush_fn.call0().map_err(py_to_io_err)?;
        }
        Ok(())
    }
}

impl Seek for PyWriter<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // Python's seek(offset, whence) where whence:
        //   0 = SEEK_SET, 1 = SEEK_CUR, 2 = SEEK_END
        let (offset, whence): (i64, i32) = match pos {
            SeekFrom::Start(n) => (n as i64, 0),
            SeekFrom::Current(n) => (n, 1),
            SeekFrom::End(n) => (n, 2),
        };

        // seek() may return the new absolute position, or None.
        let result = self.seek_fn.call1((offset, whence)).map_err(py_to_io_err)?;

        if !result.is_none() {
            result.extract::<u64>().map_err(py_to_io_err)
        } else {
            // Fall back to tell() to get the position.
            self.stream_position()
        }
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        // Python's tell() returns the current position.
        self.tell_fn
            .call0()
            .map_err(py_to_io_err)?
            .extract::<u64>()
            .map_err(py_to_io_err)
    }
}

/// Convert a PyErr into std::io::Error.
fn py_to_io_err(e: PyErr) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::io::{Seek, SeekFrom, Write};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Run a closure with the GIL held, returning its result.
    fn with_py<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        // pyo3::prepare_freethreaded_python();
        Python::attach(f)
    }

    /// Create a `BytesIO` instance.
    fn make_bytesio<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        py.import("io")
            .unwrap()
            .getattr("BytesIO")
            .unwrap()
            .call0()
            .unwrap()
    }

    /// Read all bytes from a BytesIO (seeks to start first).
    fn read_bytesio(bio: &Bound<'_, PyAny>) -> Vec<u8> {
        bio.call_method1("seek", (0,)).unwrap();
        let data = bio.call_method0("read").unwrap();
        data.extract::<Vec<u8>>().unwrap()
    }

    /// Return the current position of a BytesIO.
    fn tell_bytesio(bio: &Bound<'_, PyAny>) -> u64 {
        bio.call_method0("tell").unwrap().extract::<u64>().unwrap()
    }

    // =======================================================================
    // Construction
    // =======================================================================

    #[test]
    fn test_new_with_bytesio() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let pw = PyWriter::new(bio);
            assert!(pw.is_ok());
            assert!(pw.unwrap().flush_fn.is_some());
        });
    }

    #[test]
    fn test_new_missing_write_method() {
        with_py(|py| {
            // Plain object with no write/seek/tell.
            let obj = PyDict::new(py);
            let err = PyWriter::new(obj.into_any()).unwrap_err();
            assert!(err.to_string().contains("write"));
        });
    }

    #[test]
    fn test_new_missing_seek_method() {
        with_py(|py| {
            // Object that has write but no seek.
            let code = c"
class WriteOnly:
    def write(self, b): pass
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("WriteOnly").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let err = PyWriter::new(obj).unwrap_err();
            assert!(err.to_string().contains("seek"));
        });
    }

    #[test]
    fn test_new_missing_tell_method() {
        with_py(|py| {
            let code = c"
class WriteSeekOnly:
    def write(self, b): pass
    def seek(self, offset, whence=0): pass
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("WriteSeekOnly").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let err = PyWriter::new(obj).unwrap_err();
            assert!(err.to_string().contains("tell"));
        });
    }

    #[test]
    fn test_new_flush_is_optional() {
        with_py(|py| {
            let code = c"
class NoFlush:
    def write(self, b): return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("NoFlush").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let pw = PyWriter::new(obj).unwrap();
            assert!(pw.flush_fn.is_none());
            // flush() should be a no-op, not an error.
            let mut pw = pw;
            assert!(pw.flush().is_ok());
        });
    }

    // =======================================================================
    // Write — basic
    // =======================================================================

    #[test]
    fn test_write_empty() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            let n = pw.write(b"").unwrap();
            assert_eq!(n, 0);
            assert_eq!(read_bytesio(&bio), b"");
        });
    }

    #[test]
    fn test_write_small() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            let n = pw.write(b"hello").unwrap();
            assert_eq!(n, 5);
            assert_eq!(read_bytesio(&bio), b"hello");
        });
    }

    #[test]
    fn test_write_multiple() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"hello ").unwrap();
            pw.write_all(b"world").unwrap();
            assert_eq!(read_bytesio(&bio), b"hello world");
        });
    }

    #[test]
    fn test_write_binary_data() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            let data: Vec<u8> = (0..=255).collect();
            pw.write_all(&data).unwrap();
            assert_eq!(read_bytesio(&bio), data);
        });
    }

    #[test]
    fn test_write_large_buffer() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            let data = vec![0xABu8; 1024 * 1024]; // 1 MiB
            pw.write_all(&data).unwrap();
            assert_eq!(read_bytesio(&bio).len(), 1024 * 1024);
        });
    }

    // =======================================================================
    // Write — return value handling
    // =======================================================================

    #[test]
    fn test_write_returns_none_treated_as_full_write() {
        with_py(|py| {
            let code = c"
class NoneWriter:
    def __init__(self):
        self.data = bytearray()
    def write(self, b):
        self.data.extend(b)
        return None  # some writers return None
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("NoneWriter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj.clone()).unwrap();
            let n = pw.write(b"test").unwrap();
            assert_eq!(n, 4);
            let stored: Vec<u8> = obj.getattr("data").unwrap().extract().unwrap();
            assert_eq!(stored, b"test");
        });
    }

    #[test]
    fn test_write_partial_return() {
        with_py(|py| {
            let code = c"
class PartialWriter:
    def __init__(self):
        self.data = bytearray()
    def write(self, b):
        # Only accept first 3 bytes each call
        chunk = bytes(b[:3])
        self.data.extend(chunk)
        return len(chunk)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("PartialWriter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj.clone()).unwrap();

            // Single write call — should report partial.
            let n = pw.write(b"abcdef").unwrap();
            assert_eq!(n, 3);

            // write_all should retry until everything is written.
            pw.write_all(b"abcdef").unwrap();
            let stored: Vec<u8> = obj.getattr("data").unwrap().extract().unwrap();
            assert_eq!(stored, b"abcabcdef");
        });
    }

    // =======================================================================
    // Write — memoryview safety
    // =======================================================================

    #[test]
    fn test_memoryview_is_readonly() {
        with_py(|py| {
            let code = c"
class CheckReadonly:
    def __init__(self):
        self.readonly = None
    def write(self, b):
        self.readonly = b.readonly
        return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("CheckReadonly").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj.clone()).unwrap();
            pw.write(b"x").unwrap();
            let readonly: bool = obj.getattr("readonly").unwrap().extract().unwrap();
            assert!(readonly, "memoryview should be read-only");
        });
    }

    #[test]
    fn test_memoryview_is_released_after_write() {
        with_py(|py| {
            // If the Python side stashes the memoryview, accessing it after
            // write() returns should raise ValueError.
            let code = c"
class StashWriter:
    def __init__(self):
        self.stashed = None
    def write(self, b):
        self.stashed = b  # hold a reference
        return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("StashWriter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj.clone()).unwrap();
            pw.write(b"dangerous").unwrap();

            // The stashed memoryview should be released — accessing it should
            // raise.
            let stashed = obj.getattr("stashed").unwrap();
            let result = stashed.call_method0("tobytes");
            assert!(
                result.is_err(),
                "released memoryview should raise on access"
            );
        });
    }

    #[test]
    fn test_memoryview_content_is_correct() {
        with_py(|py| {
            let code = c"
class CaptureWriter:
    def __init__(self):
        self.captured = None
    def write(self, b):
        self.captured = bytes(b)  # copy before release
        return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("CaptureWriter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj.clone()).unwrap();
            pw.write(b"\x00\x01\x02\xff\xfe\xfd").unwrap();
            let captured: Vec<u8> = obj.getattr("captured").unwrap().extract().unwrap();
            assert_eq!(captured, b"\x00\x01\x02\xff\xfe\xfd");
        });
    }

    // =======================================================================
    // Seek
    // =======================================================================

    #[test]
    fn test_seek_start() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"abcdefghij").unwrap();
            let pos = pw.seek(SeekFrom::Start(3)).unwrap();
            assert_eq!(pos, 3);
            assert_eq!(tell_bytesio(&bio), 3);
        });
    }

    #[test]
    fn test_seek_current_forward() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"abcdefghij").unwrap();
            pw.seek(SeekFrom::Start(2)).unwrap();
            let pos = pw.seek(SeekFrom::Current(3)).unwrap();
            assert_eq!(pos, 5);
        });
    }

    #[test]
    fn test_seek_current_backward() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"abcdefghij").unwrap();
            let pos = pw.seek(SeekFrom::Current(-4)).unwrap();
            assert_eq!(pos, 6);
        });
    }

    #[test]
    fn test_seek_end() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"abcdefghij").unwrap();
            let pos = pw.seek(SeekFrom::End(-2)).unwrap();
            assert_eq!(pos, 8);
        });
    }

    #[test]
    fn test_seek_then_overwrite() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"aaaaaaaaaa").unwrap();
            pw.seek(SeekFrom::Start(3)).unwrap();
            pw.write_all(b"BBB").unwrap();
            assert_eq!(read_bytesio(&bio), b"aaaBBBaaaa");
        });
    }

    // =======================================================================
    // stream_position / tell
    // =======================================================================

    #[test]
    fn test_stream_position_at_start() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio).unwrap();
            assert_eq!(pw.stream_position().unwrap(), 0);
        });
    }

    #[test]
    fn test_stream_position_after_write() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio).unwrap();
            pw.write_all(b"hello").unwrap();
            assert_eq!(pw.stream_position().unwrap(), 5);
        });
    }

    #[test]
    fn test_stream_position_after_seek() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio).unwrap();
            pw.write_all(b"0123456789").unwrap();
            pw.seek(SeekFrom::Start(7)).unwrap();
            assert_eq!(pw.stream_position().unwrap(), 7);
        });
    }

    // =======================================================================
    // Seek — fallback to tell() when seek() returns None
    // =======================================================================

    #[test]
    fn test_seek_returns_none_falls_back_to_tell() {
        with_py(|py| {
            let code = c"
class SeekReturnsNone:
    def __init__(self):
        self.pos = 0
    def write(self, b):
        n = len(b)
        self.pos += n
        return n
    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        return None  # does not return new position
    def tell(self):
        return self.pos
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("SeekReturnsNone").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj).unwrap();
            pw.write_all(b"abcdef").unwrap();
            let pos = pw.seek(SeekFrom::Start(2)).unwrap();
            assert_eq!(pos, 2);
        });
    }

    // =======================================================================
    // Flush
    // =======================================================================

    #[test]
    fn test_flush_is_called() {
        with_py(|py| {
            let code = c"
class FlushCounter:
    def __init__(self):
        self.flush_count = 0
    def write(self, b): return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
    def flush(self):
        self.flush_count += 1
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("FlushCounter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj.clone()).unwrap();
            pw.flush().unwrap();
            pw.flush().unwrap();
            let count: i32 = obj.getattr("flush_count").unwrap().extract().unwrap();
            assert_eq!(count, 2);
        });
    }

    // =======================================================================
    // Error propagation
    // =======================================================================

    #[test]
    fn test_write_error_propagates() {
        with_py(|py| {
            let code = c"
class BrokenWriter:
    def write(self, b):
        raise IOError('disk full')
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("BrokenWriter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj).unwrap();
            let err = pw.write(b"x").unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::Other);
            assert!(err.to_string().contains("disk full"));
        });
    }

    #[test]
    fn test_seek_error_propagates() {
        with_py(|py| {
            let code = c"
class BrokenSeeker:
    def write(self, b): return len(b)
    def seek(self, offset, whence=0):
        raise IOError('unseekable stream')
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("BrokenSeeker").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj).unwrap();
            let err = pw.seek(SeekFrom::Start(0)).unwrap_err();
            assert!(err.to_string().contains("unseekable"));
        });
    }

    #[test]
    fn test_flush_error_propagates() {
        with_py(|py| {
            let code = c"
class BrokenFlusher:
    def write(self, b): return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
    def flush(self):
        raise IOError('flush failed')
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("BrokenFlusher").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj).unwrap();
            let err = pw.flush().unwrap_err();
            assert!(err.to_string().contains("flush failed"));
        });
    }

    #[test]
    fn test_tell_error_propagates() {
        with_py(|py| {
            let code = c"
class BrokenTell:
    def write(self, b): return len(b)
    def seek(self, offset, whence=0): return None  # force fallback to tell
    def tell(self):
        raise IOError('tell broken')
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("BrokenTell").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj).unwrap();
            let err = pw.seek(SeekFrom::Start(0)).unwrap_err();
            assert!(err.to_string().contains("tell broken"));
        });
    }

    // =======================================================================
    // Integration: BufWriter<PyWriter> (the intended usage pattern)
    // =======================================================================

    #[test]
    fn test_bufwriter_wrapping() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let pw = PyWriter::new(bio.clone()).unwrap();
            let mut bw = std::io::BufWriter::with_capacity(16, pw);

            // Write less than buffer capacity — should not hit Python yet.
            bw.write_all(b"hello").unwrap();
            assert_eq!(tell_bytesio(&bio), 0); // still buffered

            // Flush forces it through.
            bw.flush().unwrap();
            assert_eq!(read_bytesio(&bio), b"hello");
        });
    }

    #[test]
    fn test_bufwriter_auto_flush_on_capacity() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let pw = PyWriter::new(bio.clone()).unwrap();
            let mut bw = std::io::BufWriter::with_capacity(8, pw);

            // Write more than buffer capacity — should auto-flush.
            bw.write_all(b"0123456789ABCDEF").unwrap();
            // At least some data should have been flushed to Python.
            assert!(tell_bytesio(&bio) > 0);
            bw.flush().unwrap();
            assert_eq!(read_bytesio(&bio), b"0123456789ABCDEF");
        });
    }

    #[test]
    fn test_bufwriter_many_small_writes() {
        with_py(|py| {
            let code = c"
class CountingWriter:
    def __init__(self):
        self.data = bytearray()
        self.write_count = 0
    def write(self, b):
        self.write_count += 1
        self.data.extend(b)
        return len(b)
    def seek(self, offset, whence=0): return 0
    def tell(self): return len(self.data)
    def flush(self): pass
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("CountingWriter").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let pw = PyWriter::new(obj.clone()).unwrap();
            let mut bw = std::io::BufWriter::with_capacity(1024, pw);

            // 500 tiny writes should be batched.
            for i in 0..500u16 {
                bw.write_all(&i.to_le_bytes()).unwrap();
            }
            bw.flush().unwrap();

            let write_count: i32 = obj.getattr("write_count").unwrap().extract().unwrap();
            let data: Vec<u8> = obj.getattr("data").unwrap().extract().unwrap();
            assert_eq!(data.len(), 1000); // 500 * 2 bytes
                                          // BufWriter should have batched — far fewer than 500 Python calls.
            assert!(
                write_count < 10,
                "expected batching, got {write_count} Python write calls for 500 writes"
            );
        });
    }

    // =======================================================================
    // Integration: with real io.BufferedWriter / gzip
    // =======================================================================

    #[test]
    fn test_with_gzip_file() {
        with_py(|py| {
            let code = c"
import io, gzip
buf = io.BytesIO()
gz = gzip.GzipFile(fileobj=buf, mode='wb')
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let gz = locals.get_item("gz").unwrap().unwrap();
            // let buf = locals.get_item("buf").unwrap().unwrap();

            // GzipFile has write/seek/tell but seek is limited.
            // We can at least write through it.
            let mut pw = PyWriter::new(gz.clone()).unwrap();
            pw.write_all(b"compressed payload").unwrap();
            pw.flush().unwrap();
            gz.call_method0("close").unwrap();

            // Decompress and verify.
            let verify = c"
import io, gzip
buf.seek(0)
result = gzip.decompress(buf.read())
";
            py.run(verify, None, Some(&locals)).unwrap();
            let result: Vec<u8> = locals
                .get_item("result")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(result, b"compressed payload");
        });
    }

    #[test]
    fn test_with_bytesio_seek_overwrite_pattern() {
        // Common pattern: write a header placeholder, write body,
        // seek back and patch the header.
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();

            // Write placeholder header (4 bytes) + body.
            pw.write_all(&[0u8; 4]).unwrap();
            pw.write_all(b"BODY_CONTENT").unwrap();
            let body_len = 12u32;

            // Seek back to start and patch the header.
            pw.seek(SeekFrom::Start(0)).unwrap();
            pw.write_all(&body_len.to_le_bytes()).unwrap();

            let data = read_bytesio(&bio);
            assert_eq!(&data[..4], &12u32.to_le_bytes());
            assert_eq!(&data[4..], b"BODY_CONTENT");
        });
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn test_write_returns_wrong_type_is_error() {
        with_py(|py| {
            let code = c"
class BadReturn:
    def write(self, b): return 'not an int'
    def seek(self, offset, whence=0): return 0
    def tell(self): return 0
";
            let locals = PyDict::new(py);
            py.run(code, None, Some(&locals)).unwrap();
            let cls = locals.get_item("BadReturn").unwrap().unwrap();
            let obj = cls.call0().unwrap();
            let mut pw = PyWriter::new(obj).unwrap();
            let err = pw.write(b"x").unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::Other);
        });
    }

    #[test]
    fn test_seek_to_start_0_is_rewind() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            pw.write_all(b"data").unwrap();
            pw.rewind().unwrap(); // calls seek(SeekFrom::Start(0))
            assert_eq!(pw.stream_position().unwrap(), 0);
        });
    }

    #[test]
    fn test_write_all_zeros() {
        with_py(|py| {
            let bio = make_bytesio(py);
            let mut pw = PyWriter::new(bio.clone()).unwrap();
            let data = vec![0u8; 4096];
            pw.write_all(&data).unwrap();
            let result = read_bytesio(&bio);
            assert_eq!(result.len(), 4096);
            assert!(result.iter().all(|&b| b == 0));
        });
    }
}

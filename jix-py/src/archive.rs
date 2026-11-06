use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict};
use pyo3::{type_hint_identifier, type_hint_subscript};

use crate::array::resolve_array_params;
use crate::ops::asarray;
use crate::util::IntoPyResult;
use crate::Array;

/// Load a compressed array from a `.jix` file or a file-like object.
///
/// When reading an archive, jix performs sanity checks to ensure the file is well-formed before
/// trusting its contents. These checks fall into two cost categories:
///
/// - **O(1) checks** are *always* performed regardless of the validation mode. They examine
///   only a constant amount of metadata - the file magic header, the archive type tag,
///   declared shape and block-shape consistency, and similar fixed-size fields. The cost
///   is negligible, so there is never a reason to skip them.
///
/// - **O(data_size) checks** scan structures whose size grows with the array (most notably
///   the per-block offset table, which has one entry per block). For very large arrays this
///   can become non-trivial - on the order of memory-touch cost over the whole offset table.
///   These checks are only performed in `strict` mode.
///
/// Args:
///     path_or_reader: A file path or any seekable binary file-like object with `.read()`,
///         `.seek()`, and `.tell()` methods (e.g. `io.BytesIO`, an open file handle).
///     offset: Byte offset within the file to start reading from. Only supported when
///         `path_or_reader` is a file path. Defaults to `0`.
///     len: Number of bytes to read. Defaults to the remaining file size from `offset`.
///         When `path_or_reader` is a file-like object, still used for bounds checking if provided.
///     mmap: If `True`, maps the file into virtual address space instead of copying bytes onto
///         the heap; blocks are paged in on demand. Defaults to `False`. **Caution:** modifying
///         the file while the returned array is live has undefined behavior. Not supported for
///         file-like objects.
///     params: Block layout and codec configuration. See [`jix.compact()`][jix.compact] for
///         details. These never change the loaded data: the block shape comes from the archive
///         (`block_shape` is ignored) and the blocks are decoded with the codec they were written
///         with. They mostly affect arrays later created from the result (e.g. via `compact()` or
///         `write_array()`), which inherit them, so their codec settings come from `params`.
///     validation: Controls how strictly the archive is validated. Can be one of:
///
///         - `None` (default): Currently the same as the `strict` mode, may change in the future.
///         - `minimal`: Perform only the constant-time (O(1)) consistency checks.
///             Suitable for archives produced by a trusted source where the O(data_size) scans
///             would add measurable overhead with no expected benefit.
///         - `strict`: Perform all checks, including those that scan structures whose size grows
///             with the array (O(data_size)). This is the right choice unless you have a specific
///             reason to skip the extra scans. Use it whenever the archive's origin is untrusted
///             or unverified.
///
/// Returns:
///     A [`jix.Array`][jix.Array] loaded from the file.
///
/// Examples:
///     ```python
///     import jix
///     import io
///
///     # Read the whole file
///     a = jix.read_array("data.jix")
///     assert a.shape == (4, 4)
///
///     # Read two arrays packed back-to-back in one file
///     with open("packed.jix", "wb") as f:
///         jix.write_array(a, f)
///         offset = f.tell()
///         jix.write_array(a, f)
///         total = f.tell()
///     b2 = jix.read_array("packed.jix", offset=offset, len=total - offset)
///
///     # Memory-mapped read (zero-copy, fast startup)
///     c = jix.read_array("large.jix", mmap=True)
///
///     # Read from an in-memory buffer
///     buf = io.BytesIO()
///     jix.write_array(a, buf)
///     buf.seek(0)
///     d = jix.read_array(buf)
///     ```
#[pyfunction]
#[pyo3(signature = (
    path_or_reader,
    *,
    params=None,
    offset=None,
    len=None,
    mmap=false,
    validation=None,
))]
pub fn read_array(
    py: Python,
    path_or_reader: &Bound<'_, PyAny>,
    params: Option<Bound<'_, PyDict>>,
    offset: Option<u64>,
    len: Option<u64>,
    mmap: bool,
    validation: Option<ArchiveValidation>,
) -> PyResult<Array> {
    let params = resolve_array_params(py, params)?;
    let path_or_reader = PathOrReader::from_pyany(path_or_reader)?;
    let validation = validation.map(|v| v.0).unwrap_or_default();

    match path_or_reader {
        PathOrReader::Path(path) => py.detach(|| {
            let offset = offset.unwrap_or(0);
            let len = match len {
                Some(len) => len,
                None => {
                    let file_len = path.metadata()?.len();
                    file_len
                        .checked_sub(offset)
                        .ok_or_else(|| PyValueError::new_err("offset is out of bounds"))?
                }
            };

            Ok(if !mmap {
                let array =
                    jix_core::Array::read_from_file_with(&path, offset, len, params, validation)
                        .into_py_result()?;
                Array::from_core(array.into_any())
            } else {
                let array = unsafe {
                    jix_core::Array::read_from_file_mmap(&path, offset, len, params, validation)
                        .into_py_result()?
                };
                Array::from_core(array.into_any())
            })
        }),
        PathOrReader::PyReader(reader) => {
            if offset.is_some() {
                return Err(PyValueError::new_err(
                    "offset is not supported for file-like readers",
                ));
            }
            if mmap {
                return Err(PyValueError::new_err(
                    "mmap=True is not supported for file-like readers",
                ));
            }
            // Can't detach the GIL since the reader is a Python object.
            let mut reader = BufReader::new(reader);
            let array = jix_core::Array::read_from(&mut reader, len, params, validation)
                .into_py_result()?;

            // Leave the reader at the current position after reading the array, BufReader may have read ahead
            let pos = reader.stream_position()?;
            reader.into_inner().seek(SeekFrom::Start(pos))?;

            Ok(Array::from_core(array.into_any()))
        }
    }
}

/// The `validation` mode of [`crate::archive::read_array`].
pub struct ArchiveValidation(pub jix_core::ArchiveValidation);

impl<'a, 'py> FromPyObject<'a, 'py> for ArchiveValidation {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("typing", "Literal"),
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("minimal")
        },
        PyStaticExpr::Constant {
            value: PyStaticConstant::Str("strict")
        }
    );

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match obj.extract::<&str>()? {
            "minimal" => Ok(Self(jix_core::ArchiveValidation::Minimal)),
            "strict" => Ok(Self(jix_core::ArchiveValidation::Strict)),
            other => Err(PyValueError::new_err(format!(
                "invalid validation value: '{other}', expected 'minimal' or 'strict'"
            ))),
        }
    }
}

/// Write a compressed array to a file or a file-like object.
///
/// Works for any array type: a compact array (the result of [`jix.compact()`][jix.compact] or
/// [`jix.read_array()`][jix.read_array]) streams its already-compressed blocks directly to the destination
/// without decompressing - `params` is ignored in this case and no re-compression takes place.
/// A lazy view (slice, arithmetic op chain, etc.) compresses on the fly, so the full
/// data (compressed or decompressed) is never held in memory.
///
/// Args:
///     array: The array to write.
///     path_or_writer: A file path or any seekable binary file-like object with `.write()`,
///         `.seek()`, and `.tell()` methods (e.g. `io.BytesIO`, an open file handle). The writer
///         must support seeking backwards. For the same reason a file object opened in append
///         mode (`"ab"`) raises; use `append=True` with a path instead. As with `io.RawIOBase`,
///         `.write(b)` must only access `b` during the call, and must not keep `b` or any view of
///         it (a slice, `np.frombuffer(b)`, ...) after it returns.
///     append: When `False` (default), the file is created anew and must not already exist.
///         When `True`, the array is appended to an existing file (or a new one is created).
///         Ignored when `path_or_writer` is a file-like object.
///     params: Controls the block layout and codec for encoding. Unset fields are
///         inherited from the source array. Ignored when the source is already compact.
///         See [`jix.compact()`][jix.compact] for details.
///
/// Examples:
///     ```python
///     import jix
///     import io
///
///     a = jix.compact([[1.0, 2.0], [3.0, 4.0]])
///
///     # Write to a new file (file must not exist)
///     jix.write_array(a, "output.jix")
///
///     # Pack two arrays back-to-back into the same file
///     b = jix.compact([10, 20, 30])
///     jix.write_array(a, "packed.jix")
///     jix.write_array(b, "packed.jix", append=True)
///
///     # Write to an in-memory buffer
///     buf = io.BytesIO()
///     jix.write_array(a, buf)
///
///     # Streaming pipeline: read via mmap, apply a lazy op, write without materializing
///     src = jix.read_array("large.jix", mmap=True)
///     view = src.exp() + 1.0
///     with open("modified.jix", "wb") as f:
///         jix.write_array(view, f)
///     ```
#[pyfunction]
#[pyo3(signature = (array, path_or_writer, *, append=false, params=None))]
pub fn write_array(
    array: &Bound<'_, PyAny>,
    path_or_writer: &Bound<'_, PyAny>,
    append: bool,
    params: Option<Bound<'_, PyDict>>,
) -> PyResult<()> {
    let array = asarray(array, None, params.clone())?;
    let py = array.py();
    let array = array.get();
    let path_or_writer = PathOrWriter::from_pyany(path_or_writer)?;
    let params = resolve_array_params(py, params)?;

    match path_or_writer {
        PathOrWriter::Path(path) => {
            py.detach(|| {
                let (file, original_file_size) = if !append {
                    (std::fs::File::create_new(&path)?, None)
                } else {
                    let file_exists = path.exists();
                    // O_APPEND would force every write to EOF, breaking the
                    // seek-then-write used to patch the block TOC placeholder.
                    // Instead, open for writing and seek to the end manually.
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .open(&path)?;
                    let original_size = file.seek(std::io::SeekFrom::End(0))?;
                    let original_size = (file_exists || original_size > 0).then_some(original_size);
                    (file, original_size)
                };
                let written = write_array_impl(array, BufWriter::new(&file), params);
                if written.is_err() {
                    // Don't leave a partial archive behind
                    let revert_result = match original_file_size {
                        Some(original_size) => file.set_len(original_size),
                        None => {
                            drop(file);
                            std::fs::remove_file(&path)
                        }
                    };
                    if let Err(err) = revert_result {
                        eprintln!("Error cleaning up partial archive: {err}");
                    }
                }
                written
            })
        }
        PathOrWriter::PyWriter(writer) => {
            // cant detach the GIL here since the writer is a Python object
            let mut writer = BufWriter::new(writer);
            let written = write_array_impl(array, &mut writer, params);
            // Drop the buffer of a failed write instead of writing it again on drop
            let _ = writer.into_parts();
            written
        }
    }
}
fn write_array_impl(
    array: &Array,
    mut writer: impl Write + Seek,
    params: jix_core::ArrayParams,
) -> PyResult<()> {
    let context = array.read_ctx()?;
    array
        .arr
        .write_to(&mut writer, params, context.as_ref())
        .into_py_result()?;
    Ok(writer.flush()?)
}

enum PathOrReader<'a> {
    Path(PathBuf),
    PyReader(PyReader<'a>),
}
impl<'a> PathOrReader<'a> {
    fn from_pyany(pyany: &'a Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(path) = pyany.extract::<PathBuf>() {
            Ok(Self::Path(path))
        } else if pyany.hasattr("read")? {
            Ok(Self::PyReader(PyReader::new(pyany.clone())?))
        } else {
            Err(PyTypeError::new_err(
                "Expected a file path or a file-like object with a 'read' method",
            ))
        }
    }
}

/// Wraps any Python file-like object that has `.read()`, `.seek()`, and `.tell()` methods.
struct PyReader<'py> {
    read_fn: Bound<'py, PyAny>,
    seek_fn: Bound<'py, PyAny>,
    tell_fn: Bound<'py, PyAny>,
}

impl<'py> PyReader<'py> {
    fn new(inner: Bound<'py, PyAny>) -> PyResult<Self> {
        let read_fn = inner
            .getattr("read")
            .map_err(|_| PyTypeError::new_err("Object must have a 'read' method"))?;
        let seek_fn = inner
            .getattr("seek")
            .map_err(|_| PyTypeError::new_err("Object must have a 'seek' method"))?;
        let tell_fn = inner
            .getattr("tell")
            .map_err(|_| PyTypeError::new_err("Object must have a 'tell' method"))?;
        Ok(Self {
            read_fn,
            seek_fn,
            tell_fn,
        })
    }
}

impl Read for PyReader<'_> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let result = self.read_fn.call1((buf.len(),)).map_err(py_to_io_err)?;
        if result.is_none() {
            return Ok(0); // EOF signalled by None
        }

        let bytes_vec;
        let bytes = if let Ok(bytes) = result.cast::<PyBytes>() {
            bytes.as_bytes()
        } else {
            bytes_vec = result.extract::<Vec<u8>>().map_err(py_to_io_err)?;
            bytes_vec.as_slice()
        };

        let n = bytes.len();
        if n > buf.len() {
            return Err(io::Error::other(format!(
                "read() returned {n} bytes for a request of {}",
                buf.len()
            )));
        }
        buf[..n].copy_from_slice(bytes);
        Ok(n)
    }
}

impl Seek for PyReader<'_> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let (offset, whence): (i64, i32) = match pos {
            SeekFrom::Start(n) => (n as i64, 0),
            SeekFrom::Current(n) => (n, 1),
            SeekFrom::End(n) => (n, 2),
        };
        let result = self.seek_fn.call1((offset, whence)).map_err(py_to_io_err)?;
        if !result.is_none() {
            result.extract::<u64>().map_err(py_to_io_err)
        } else {
            self.stream_position()
        }
    }

    #[inline]
    fn stream_position(&mut self) -> io::Result<u64> {
        self.tell_fn
            .call0()
            .map_err(py_to_io_err)?
            .extract::<u64>()
            .map_err(py_to_io_err)
    }
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
            Err(PyTypeError::new_err(
                "Expected a file path or a file-like object with a 'write' method",
            ))
        }
    }
}

/// Wraps any Python file-like object that has `.write()`, `.seek()`, `.tell()` methods, and optionally `.flush()`.
struct PyWriter<'py> {
    write_fn: Bound<'py, PyAny>,
    seek_fn: Bound<'py, PyAny>,
    tell_fn: Bound<'py, PyAny>,
    flush_fn: Option<Bound<'py, PyAny>>,
}

impl<'py> PyWriter<'py> {
    fn new(inner: Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            write_fn: inner
                .getattr("write")
                .map_err(|_| PyTypeError::new_err("Object must have a 'write' method"))?,
            seek_fn: inner
                .getattr("seek")
                .map_err(|_| PyTypeError::new_err("Object must have a 'seek' method"))?,
            tell_fn: inner
                .getattr("tell")
                .map_err(|_| PyTypeError::new_err("Object must have a 'tell' method"))?,
            flush_fn: inner.getattr("flush").ok(),
        })
    }
}

impl Write for PyWriter<'_> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let py = self.write_fn.py();

        // SAFETY: A read-only memoryview pointing directly into `buf`. Python's write() must not
        // keep the memoryview or any view derived from it after returning, the same contract
        // CPython's BufferedWriter puts on raw.write() (documented in write_array). A view that
        // holds a buffer export makes release() fail, which is reported as an error below.
        let py_buf: Bound<'_, PyAny> = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                pyo3::ffi::PyMemoryView_FromMemory(
                    buf.as_ptr().cast_mut().cast::<std::ffi::c_char>(), // cast away const - we mark readonly below
                    buf.len() as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyBUF_READ, // read-only access
                ),
            )
            .map_err(py_to_io_err)?
        };

        let result = self.write_fn.call1((&py_buf,));

        static MEMORY_VIEW_RELEASE_FN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        let memory_view_release_fn = MEMORY_VIEW_RELEASE_FN.get_or_init(py, || {
            py.get_type::<pyo3::types::PyMemoryView>()
                .getattr("release")
                .expect("Failed to get memoryview release function")
                .into()
        });
        memory_view_release_fn
            .bind(py)
            .call1((&py_buf,))
            .map_err(|_| {
                py_to_io_err(PyRuntimeError::new_err(
                    "write() kept a reference to its buffer after returning",
                ))
            })?;

        let result = result.map_err(py_to_io_err)?;
        if result.is_none() {
            return Ok(buf.len());
        }
        let n = result.extract::<usize>().map_err(py_to_io_err)?;
        if n > buf.len() {
            return Err(io::Error::other(format!(
                "write() reported {n} bytes written for a buffer of {}",
                buf.len()
            )));
        }
        Ok(n)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        // flush() is optional on the Python side.
        if let Some(flush_fn) = &self.flush_fn {
            flush_fn.call0().map_err(py_to_io_err)?;
        }
        Ok(())
    }
}

impl Seek for PyWriter<'_> {
    #[inline]
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

    #[inline]
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
    io::Error::other(e)
}

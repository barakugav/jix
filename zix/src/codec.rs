use std::cell::UnsafeCell;
use std::io;

use crate::dtype::Alignment;
use crate::util::AlignedBytes;

pub(crate) struct Encoder {
    #[cfg(not(miri))]
    inner: zstd::bulk::Compressor<'static>,
}
impl Encoder {
    pub(crate) fn new(level: i32) -> io::Result<Self> {
        Ok(Self {
            #[cfg(not(miri))]
            inner: zstd::bulk::Compressor::new(level)?,
        })
    }

    pub(crate) fn encode(&mut self, data: &[u8], dst: &mut [u8]) -> io::Result<usize> {
        cfg_if::cfg_if! { if #[cfg(not(miri))] {
            self.inner.compress_to_buffer(data, dst)
        } else {
            dst.copy_from_slice(data);
            Ok(data.len())
        } }
    }

    pub(crate) fn encode_bound(&self, src_size: usize) -> usize {
        cfg_if::cfg_if! { if #[cfg(not(miri))] {
            zstd::zstd_safe::compress_bound(src_size)
        } else {
            src_size
        } }
    }
}
pub struct ReadContext {
    #[cfg(not(miri))]
    inner: UnsafeCell<zstd::bulk::Decompressor<'static>>,
    tmp_buffers: TmpBufferPool,
}
impl ReadContext {
    pub(crate) fn new() -> io::Result<Self> {
        Ok(Self {
            #[cfg(not(miri))]
            inner: UnsafeCell::new(zstd::bulk::Decompressor::new()?),
            tmp_buffers: TmpBufferPool::new(),
        })
    }

    pub(crate) fn decode(&self, cdata: &[u8], dst: &mut [u8]) -> io::Result<usize> {
        cfg_if::cfg_if! { if #[cfg(not(miri))] {
            let inner = unsafe { &mut *self.inner.get() };
            inner.decompress_to_buffer(cdata, dst)
        } else {
            dst.copy_from_slice(cdata);
            Ok(cdata.len())
        } }
    }

    pub(crate) fn tmp_buf(&self, size: usize, alignment: Alignment) -> TmpBuf<'_> {
        self.tmp_buffers.get(size, alignment)
    }
}

struct TmpBufferPool {
    align1: UnsafeCell<Vec<AlignedBytes>>,
    align2: UnsafeCell<Vec<AlignedBytes>>,
    align4: UnsafeCell<Vec<AlignedBytes>>,
    align8: UnsafeCell<Vec<AlignedBytes>>,
    align_other: UnsafeCell<Vec<(Alignment, Vec<AlignedBytes>)>>,
}
impl TmpBufferPool {
    fn new() -> Self {
        Self {
            align1: UnsafeCell::new(Vec::new()),
            align2: UnsafeCell::new(Vec::new()),
            align4: UnsafeCell::new(Vec::new()),
            align8: UnsafeCell::new(Vec::new()),
            align_other: UnsafeCell::new(Vec::new()),
        }
    }

    fn get(&self, size: usize, alignment: Alignment) -> TmpBuf<'_> {
        let pool = self.get_pool(alignment);
        let pool = unsafe { &mut *pool };
        let tmp_buf = pool
            .pop()
            .unwrap_or_else(|| AlignedBytes::with_capacity(alignment as usize, size));
        let mut buf = TmpBuf {
            buf: tmp_buf,
            buffers: self,
        };
        buf.set_len(size);
        buf
    }

    fn return_buf(&self, mut buf: AlignedBytes) {
        buf.clear();
        let pool = self.get_pool(buf.alignment() as Alignment);
        let pool = unsafe { &mut *pool };
        pool.push(buf);
    }

    fn get_pool(&self, alignment: Alignment) -> *mut Vec<AlignedBytes> {
        match alignment {
            1 => self.align1.get(),
            2 => self.align2.get(),
            4 => self.align4.get(),
            8 => self.align8.get(),
            _ => {
                let align_other = unsafe { &mut *self.align_other.get() };
                let pool = align_other
                    .iter_mut()
                    .find(|(align, _)| *align == alignment)
                    .map(|(_, pool)| pool);
                match pool {
                    Some(pool) => pool,
                    None => {
                        align_other.push((alignment, Vec::new()));
                        &mut align_other.last_mut().unwrap().1
                    }
                }
            }
        }
    }
}
pub(crate) struct TmpBuf<'a> {
    buf: AlignedBytes,
    buffers: &'a TmpBufferPool,
}
impl TmpBuf<'_> {
    pub(crate) fn set_len(&mut self, new_len: usize) {
        self.buf.clear();
        self.buf.reserve(new_len);
        unsafe { self.buf.set_len(new_len) };
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
}
// impl AsRef<[u8]> for TmpBuf<'_> {
//     fn as_ref(&self) -> &[u8] {
//         self.buf.as_ref()
//     }
// }
// impl AsMut<[u8]> for TmpBuf<'_> {
//     fn as_mut(&mut self) -> &mut [u8] {
//         self.buf.as_mut()
//     }
// }
impl Drop for TmpBuf<'_> {
    fn drop(&mut self) {
        // take self.buf
        let mut buf = AlignedBytes::new(self.buf.alignment());
        std::mem::swap(&mut self.buf, &mut buf);

        self.buffers.return_buf(buf);
    }
}

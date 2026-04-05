use std::fs::File;
use std::io::{self, BufReader, Read, Seek, Write};

use zerocopy::{FromBytes, IntoBytes};

use crate::archive::{ArchiveReader, ArchiveWriter, Section};
use crate::codec::{Encoder, ReadContext};
use crate::dtype::Dtype;
use crate::schema::{self, ArchiveType};
use crate::util::{cast_slice, cast_slice_mut};

const _: () = const {
    assert!(
        cfg!(target_endian = "little"),
        "Only little-endian is supported"
    );
};

pub(crate) type BlockSize = u32;

/// Storage of 1D array items, organized in blocks.
///
/// The number of items must be divisible by the block length, there is no support for partial blocks.
/// At all times the storage holds the invariants:
/// - `block_size > 0`
/// - `nitems % block_size == 0`
pub(crate) struct BlockTable<A> {
    pub(crate) dtype: Dtype,
    pub(crate) nitems: usize,

    pub(crate) nblocks: usize,
    /// The number of items in each block. All blocks are full (nitems is divisible by block_size).
    /// Note the units are items, not bytes.
    pub(crate) block_size: BlockSize,

    allocation: A,
}
impl<A> BlockTable<A> {
    pub(crate) fn new(allocation: A, dtype: Dtype, nitems: usize, block_size: BlockSize) -> Self
    where
        A: BlockTableAllocation,
    {
        assert!(block_size > 0);
        assert!(nitems % block_size as usize == 0);
        let nblocks = nitems / block_size as usize;
        let cdata = allocation.cdata();
        let block_offsets = allocation.block_offsets();
        if nblocks == 0 {
            assert_eq!(block_offsets.len(), 0);
        } else {
            assert_eq!(block_offsets.len(), nblocks + 1);
            debug_assert!(block_offsets.windows(2).all(|w| w[0] < w[1]));
            debug_assert!(*block_offsets.last().unwrap() <= cdata.len() as u64);
        }
        Self {
            allocation,
            dtype,
            nitems,
            nblocks,
            block_size,
        }
    }

    pub(crate) unsafe fn swap_allocation<A2>(self, map: impl FnOnce(A) -> A2) -> BlockTable<A2> {
        BlockTable {
            allocation: map(self.allocation),
            dtype: self.dtype,
            nitems: self.nitems,
            nblocks: self.nblocks,
            block_size: self.block_size,
        }
    }

    /// Get the dtype of items in this storage.
    pub(crate) fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    /// Get the total number of items in this storage.
    pub(crate) fn nitems(&self) -> usize {
        self.nitems
    }

    /// Get the length of a block in this storage.
    ///
    /// Note that the units are in items, not bytes.
    pub(crate) fn block_len(&self) -> BlockSize {
        self.block_size
    }

    /// Read a block of items into the provided buffer.
    ///
    /// # Arguments
    ///
    /// - `block_idx`: The index of the block to read, in the range `0..(nitems / block_len)`.
    /// - `buf`: The buffer to read the block into. Must be of size `block_len * dtype.itemsize()`.
    /// - `context`: a read context containing global configuration and reuseable buffers.
    pub(crate) fn read_block(
        &self,
        block_idx: usize,
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()>
    where
        A: BlockTableAllocation,
    {
        let b_size_bytes = self.block_len() as usize * self.dtype.itemsize() as usize;
        if buf.len() < b_size_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer too small",
            ));
        }

        let block_offsets = self.allocation.block_offsets();
        let begin = block_offsets[block_idx] as usize;
        let end = block_offsets[block_idx + 1] as usize;
        let b_cdata = &self.allocation.cdata()[begin..end];

        let nbytes = context.decode(b_cdata, buf)?;
        debug_assert_eq!(nbytes, b_size_bytes);
        Ok(())
    }

    pub(crate) fn write_to<W>(&self, writer: W) -> io::Result<()>
    where
        W: Write + Seek,
        A: BlockTableAllocation,
    {
        let mut writer = ArchiveWriter::new(writer, schema::ArchiveType::BlockTable)?;
        self.write_content(&mut writer)
    }

    pub(crate) fn write_content<W>(&self, writer: &mut ArchiveWriter<W>) -> io::Result<()>
    where
        W: Write + Seek,
        A: BlockTableAllocation,
    {
        // Write header
        let header = schema::BlockTableHeader {
            dtype: Some(self.dtype.to_proto()),
            nitems: self.nitems as u64,
            block_size: self.block_size as u64,
            table_of_contents: vec![
                schema::block_table_header::TableOfContents::BlockOffsets as i32,
                schema::block_table_header::TableOfContents::Cdata as i32,
            ],
        };
        writer.write_message(&header)?;

        // Write table of contents (placeholder for now, will be overwritten later)
        let mut toc = [Section::default(); 2];
        let toc_offset = writer.stream_position()?;
        writer.write_all(toc.as_bytes())?;

        // Write body data sections
        let cdata = writer.write_section(&self.allocation.cdata(), align_of::<u8>())?;
        let block_offsets = writer.write_section(
            unsafe { cast_slice::<u64, u8>(self.allocation.block_offsets()) },
            align_of::<u64>(),
        )?;

        // Go back and write table of contents
        toc = [block_offsets, cdata];
        let current_pos = writer.stream_position()?;
        writer.seek(io::SeekFrom::Start(toc_offset))?;
        writer.write_all(toc.as_bytes())?;
        writer.seek(io::SeekFrom::Start(current_pos))?;

        Ok(())
    }
}
impl BlockTable<Owned> {
    pub(crate) fn read_from<R>(reader: R, len: u64) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        let mut reader = ArchiveReader::new(reader, len)?;
        let f_meta = reader.read_file_meta()?;
        if f_meta.archive_type != schema::ArchiveType::BlockTable as i32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unexpected zix file type: expected {:?}, actual {:?}",
                    schema::ArchiveType::BlockTable,
                    ArchiveType::try_from(f_meta.archive_type)
                ),
            ));
        }
        Self::read_content(&mut reader, Owned::read_from)
    }
}

impl<A> BlockTable<A> {
    pub(crate) fn read_content<R>(
        reader: &mut ArchiveReader<R>,
        read_sections: impl FnOnce(&mut ArchiveReader<R>, Section, Section) -> io::Result<A>,
    ) -> io::Result<Self>
    where
        R: Read + Seek,
        A: BlockTableAllocation,
    {
        let header = reader.read_message::<schema::BlockTableHeader>()?;

        if header.table_of_contents.len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected 2 sections in table of contents",
            ));
        }
        let toc = <[Section; 2]>::read_from_io(reader.inner_mut())?;
        let mut cdata_section = None;
        let mut block_offsets_section = None;
        for (toc_idx, toc_entry) in header.table_of_contents().enumerate() {
            match toc_entry {
                schema::block_table_header::TableOfContents::Unspecified => {} // fail later
                schema::block_table_header::TableOfContents::Cdata => {
                    cdata_section = Some(toc[toc_idx])
                }
                schema::block_table_header::TableOfContents::BlockOffsets => {
                    block_offsets_section = Some(toc[toc_idx])
                }
            }
        }
        let (Some(cdata_section), Some(block_offsets_section)) =
            (cdata_section, block_offsets_section)
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing sections in table of contents",
            ));
        };

        // Read body data sections
        let allocation = read_sections(reader, cdata_section, block_offsets_section)?;

        Ok(Self::new(
            allocation,
            Dtype::from_proto(header.dtype.as_ref().unwrap())?,
            header.nitems as usize,
            header.block_size as BlockSize,
        ))
    }
}

impl BlockTable<Owned> {
    pub fn build_from_data(
        data: &[u8],
        dtype: Dtype,
        block_size: BlockSize,
        encoder: &mut Encoder,
    ) -> io::Result<Self> {
        let itemsize = dtype.itemsize();
        assert!(itemsize > 0);
        assert!(block_size > 0);
        assert!(data.len() % itemsize as usize == 0);
        let nitems = data.len() / itemsize as usize;
        assert!(nitems % block_size as usize == 0);
        let nblocks = nitems / block_size as usize;

        let b_size_bytes = block_size as usize * itemsize as usize;
        let mut cdata = Vec::<u8>::new();
        let mut block_offsets =
            Vec::<u64>::with_capacity(if nblocks == 0 { 0 } else { nblocks + 1 });
        if nblocks > 0 {
            block_offsets.push(0);
        }
        let max_blk_cdata_len = encoder.encode_bound(b_size_bytes);
        for b_data in data.chunks(b_size_bytes) {
            let cdata_len = cdata.len();

            cdata.reserve(max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + max_blk_cdata_len) };
            let blk_buf = &mut cdata[cdata_len..];

            let blk_cdata_len = encoder.encode(b_data, blk_buf)?;
            debug_assert!(blk_cdata_len <= max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + blk_cdata_len) };

            block_offsets.push(cdata.len() as u64);
        }

        Ok(Self::new(
            Owned {
                cdata,
                block_offsets,
            },
            dtype,
            nitems,
            block_size,
        ))
    }
}

#[doc(hidden)]
pub trait BlockTableAllocation {
    fn cdata(&self) -> &[u8];
    fn block_offsets(&self) -> &[u64];
}
#[doc(hidden)]
pub struct Owned {
    pub(crate) cdata: Vec<u8>,
    pub(crate) block_offsets: Vec<u64>,
}
impl Owned {
    pub(crate) fn read_from<R>(
        reader: &mut ArchiveReader<R>,
        cdata_section: Section,
        block_offsets_section: Section,
    ) -> io::Result<Owned>
    where
        R: Read + Seek,
    {
        let cdata = reader.read_section(&cdata_section)?;

        let block_offsets = {
            let block_offsets_section = &block_offsets_section;
            let block_offsets_len =
                block_offsets_section.size as usize / std::mem::size_of::<u64>();
            let mut block_offsets = Vec::<u64>::with_capacity(block_offsets_len);
            unsafe { block_offsets.set_len(block_offsets_len) };
            reader.read_section_into(block_offsets_section, unsafe {
                cast_slice_mut::<u64, u8>(block_offsets.as_mut_slice())
            })?;
            block_offsets
        };

        Ok(Owned {
            cdata,
            block_offsets,
        })
    }
}
#[doc(hidden)]
pub struct Borrowed<'a> {
    pub(crate) cdata: &'a [u8],
    pub(crate) block_offsets: &'a [u64],
}
#[doc(hidden)]
pub struct Mmap {
    pub(crate) cdata: &'static [u8],
    pub(crate) block_offsets: &'static [u64],
    pub(crate) mmap: memmap2::Mmap,
}
impl Mmap {
    pub(crate) fn new(mmap: memmap2::Mmap, cdata: Section, block_offsets: Section) -> Self {
        let cdata = {
            let offset = cdata.offset as usize;
            let size = cdata.size as usize;
            let slice = &mmap[offset..offset + size];
            // SAFETY: We require that the mmap outlives the returned slice, and that the caller does not mutate the slice.
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(slice) }
        };
        let block_offsets = {
            let offset = block_offsets.offset as usize;
            let size = block_offsets.size as usize;
            let buf = &mmap[offset..offset + size];
            let slice = unsafe { cast_slice::<u8, u64>(buf) };
            unsafe { std::mem::transmute::<&[u64], &'static [u64]>(slice) }
        };
        Self {
            cdata,
            block_offsets,
            mmap,
        }
    }

    pub(crate) fn read_from<R>(
        reader: &mut ArchiveReader<R>,
        mut cdata_section: Section,
        mut block_offsets_section: Section,
        mmap: memmap2::Mmap,
    ) -> io::Result<Mmap>
    where
        R: Read + Seek,
    {
        let base_offset = reader.base_offset() as i64;
        cdata_section.offset += base_offset;
        block_offsets_section.offset += base_offset;
        Ok(Mmap::new(mmap, cdata_section, block_offsets_section))
    }
}
impl BlockTableAllocation for Owned {
    fn cdata(&self) -> &[u8] {
        &self.cdata
    }
    fn block_offsets(&self) -> &[u64] {
        &self.block_offsets
    }
}
impl<'a> BlockTableAllocation for Borrowed<'a> {
    fn cdata(&self) -> &[u8] {
        self.cdata
    }
    fn block_offsets(&self) -> &[u64] {
        self.block_offsets
    }
}
impl BlockTableAllocation for Mmap {
    fn cdata(&self) -> &[u8] {
        self.cdata
    }
    fn block_offsets(&self) -> &[u64] {
        self.block_offsets
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor};

    use super::{BlockSize, BlockTable};
    use crate::block::{BlockTableAllocation, Owned};
    use crate::codec::{Encoder, ReadContext};
    use crate::dtype::Dtyped;
    use crate::util::cast_slice;

    fn make_encoder() -> Encoder {
        Encoder::new(3).unwrap()
    }
    fn make_decoder() -> ReadContext {
        ReadContext::new().unwrap()
    }

    fn decode_block<A>(table: &BlockTable<A>, idx: usize, context: &mut ReadContext) -> Vec<u8>
    where
        A: BlockTableAllocation,
    {
        let block_bytes = table.block_len() as usize * table.dtype().itemsize() as usize;
        let mut buf = vec![0u8; block_bytes];
        table.read_block(idx, &mut buf, context).unwrap();
        buf
    }

    fn build_from_items<T>(
        items: &[T],
        block_size: BlockSize,
        encoder: &mut Encoder,
    ) -> io::Result<BlockTable<Owned>>
    where
        T: Dtyped,
    {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::dtype(),
            block_size,
            encoder,
        )
    }

    #[test]
    fn build_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 8, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 1);
        assert_eq!(table.nitems, 8);
        assert_eq!(decode_block(&table, 0, &mut make_decoder()), items);
    }

    #[test]
    fn build_multiple_blocks_exact_divisor() {
        // 12 items, block_size=4 → 3 full blocks
        let items: Vec<u8> = (0u8..12).collect();
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 4, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.nitems, 12);
        let mut decoder = make_decoder();
        assert_eq!(decode_block(&table, 0, &mut decoder), items[0..4]);
        assert_eq!(decode_block(&table, 1, &mut decoder), items[4..8]);
        assert_eq!(decode_block(&table, 2, &mut decoder), items[8..12]);
    }

    #[test]
    fn build_multiple_blocks_non_divisible_panics() {
        // 10 items, block_size=4 → not divisible, should panic
        let items: Vec<u8> = (0u8..10).collect();
        let mut encoder = make_encoder();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_from_items(&items, 4, &mut encoder).unwrap();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn build_with_itemsize_greater_than_one() {
        // 4 u32 values, block_size=2
        let items: Vec<u32> = vec![10, 20, 30, 40];
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 2, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 2);
        assert_eq!(table.nitems, 4);
        let mut decoder = make_decoder();
        assert_eq!(decode_block(&table, 0, &mut decoder), unsafe {
            cast_slice::<u32, u8>(&items[0..2])
        });
        assert_eq!(decode_block(&table, 1, &mut decoder), unsafe {
            cast_slice::<u32, u8>(&items[2..4])
        });
    }

    // -----------------------------------------------------------------------
    // write_to / read_from round-trip
    // -----------------------------------------------------------------------

    fn round_trip<T: Dtyped>(items: &[T], block_size: BlockSize) -> BlockTable<Owned> {
        let mut encoder = make_encoder();
        let table = build_from_items(items, block_size, &mut encoder).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        table.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let len = bytes.len() as u64;
        BlockTable::read_from(Cursor::new(bytes), len).unwrap()
    }

    #[test]
    fn round_trip_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let table2 = round_trip(&items, 8);
        assert_eq!(table2.nblocks, 1);
        assert_eq!(table2.nitems, 8);
        assert_eq!(table2.block_size, 8);
        assert_eq!(table2.dtype, u8::dtype());
        assert_eq!(decode_block(&table2, 0, &mut make_decoder()), items);
    }

    #[test]
    fn round_trip_multiple_blocks() {
        let items: Vec<u8> = (0u8..12).collect();
        let table = round_trip(&items, 4);
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.nitems, 12);
        let mut decoder = make_decoder();
        let recovered: Vec<u8> = (0..table.nblocks)
            .flat_map(|i| decode_block(&table, i, &mut decoder))
            .collect();
        assert_eq!(recovered, items);
    }

    #[test]
    fn round_trip_preserves_block_offsets_ordering() {
        let items: Vec<u8> = (0u8..12).collect();
        let table2 = round_trip(&items, 3);
        let offs = table2.allocation.block_offsets();
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
    }

    // -----------------------------------------------------------------------
    // write_to_file / read_from_file round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_file() {
        let items: Vec<u32> = (0u32..18).collect();
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 3, &mut encoder).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path();
        table
            .write_to(&mut std::fs::File::create(path).unwrap())
            .unwrap();

        let file = std::fs::File::open(path).unwrap();
        let reader_len = file.metadata().unwrap().len();
        let table2 = BlockTable::read_from(file, reader_len).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(table2.nblocks, 6);
        assert_eq!(table2.nitems, 18);
        let mut decoder = make_decoder();
        let recovered: Vec<u8> = (0..table2.nblocks)
            .flat_map(|i| decode_block(&table2, i, &mut decoder))
            .collect();
        assert_eq!(recovered, unsafe { cast_slice::<u32, u8>(&items) });
    }

    fn make_storage<T: Dtyped>(items: &[T], block_len: BlockSize) -> BlockTable<Owned> {
        let mut encoder = Encoder::new(3).unwrap();
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            T::dtype(),
            block_len,
            &mut encoder,
        )
        .unwrap()
    }

    fn read_block_items<T: Dtyped, A: BlockTableAllocation>(
        storage: &BlockTable<A>,
        idx: usize,
    ) -> Vec<T> {
        let mut context = ReadContext::new().unwrap();
        let block_bytes = storage.block_len() as usize * storage.dtype().itemsize() as usize;
        let mut buf = vec![0u8; block_bytes];
        storage.read_block(idx, &mut buf, &mut context).unwrap();
        unsafe { cast_slice::<u8, T>(&buf) }.to_vec()
    }

    #[test]
    fn single_block_u8_round_trips() {
        let items: Vec<u8> = (0..8).collect();
        let s = make_storage(&items, 8);
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 8);
        assert_eq!(s.dtype(), &u8::dtype());
        assert_eq!(read_block_items::<u8, _>(&s, 0), items);
    }

    #[test]
    fn two_blocks_i32_round_trips() {
        let items: Vec<i32> = (0..8).collect();
        let s = make_storage(&items, 4);
        assert_eq!(s.nitems(), 8);
        assert_eq!(s.block_len(), 4);
        assert_eq!(read_block_items::<i32, _>(&s, 0), items[..4]);
        assert_eq!(read_block_items::<i32, _>(&s, 1), items[4..]);
    }

    #[test]
    fn multiple_blocks_f32_round_trips() {
        let items: Vec<f32> = (0..12).map(|x| x as f32 * 0.5).collect();
        let s = make_storage(&items, 4);
        assert_eq!(s.nitems(), 12);
        assert_eq!(s.block_len(), 4);
        for b in 0..3 {
            assert_eq!(read_block_items::<f32, _>(&s, b), items[b * 4..(b + 1) * 4]);
        }
    }

    #[test]
    fn buffer_too_small_returns_error() {
        let items: Vec<u8> = (0..4).collect();
        let s = make_storage(&items, 4);
        let mut buf = vec![0u8; 3]; // one byte short
        let mut context = ReadContext::new().unwrap();
        assert!(s.read_block(0, &mut buf, &mut context).is_err());
    }
}

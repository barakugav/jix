use std::borrow::Cow;
use std::io::{self, Read, Seek, Write};
use std::path::Path;

use prost::Message;

use crate::dtype::{Dtyped, Itemsize};
use crate::schema::{self, ArchiveType};
use crate::storage::codec::Encoder;
use crate::storage::common::{ArchiveReader, ArchiveWriter};
use crate::util::{cast_slice, cast_slice_mut};

const _: () = const {
    assert!(
        cfg!(target_endian = "little"),
        "Only little-endian is supported"
    );
};

pub(crate) type BlockSize = u32;

pub(crate) struct Block<'a> {
    pub(crate) cdata: &'a [u8],
    pub(crate) itemsize: Itemsize,
    pub(crate) nitems: BlockSize,
}

pub(crate) struct BlockTable<'a> {
    cdata: Cow<'a, [u8]>,
    block_offsets: Cow<'a, [u64]>, // (nblocks-1,)

    itemsize: Itemsize,
    nitems: usize,
    nblocks: usize,

    /// The number of items in each block, except possibly the last block which may have fewer items.
    /// Note the units of `block_size` are in items, not bytes.
    block_size: BlockSize,
}
impl<'a> BlockTable<'a> {
    pub(crate) fn new(
        cdata: Cow<'a, [u8]>,
        block_offsets: Cow<'a, [u64]>,
        itemsize: Itemsize,
        nitems: usize,
        block_size: BlockSize,
    ) -> Self {
        let nblocks = block_offsets.len() + 1;
        debug_assert!(block_offsets.windows(2).all(|w| w[0] < w[1]));
        debug_assert!(
            block_offsets.is_empty() || *block_offsets.last().unwrap() < cdata.len() as u64
        );
        Self {
            cdata,
            block_offsets,
            itemsize,
            nitems,
            nblocks,
            block_size,
        }
    }

    pub fn get_block(&self, idx: usize) -> Block {
        let offs = &self.block_offsets;
        let first_blk = idx == 0;
        let is_last = idx == offs.len();
        let begin = if first_blk { 0 } else { offs[idx - 1] as usize };
        let end = if is_last {
            self.cdata.len()
        } else {
            offs[idx] as usize
        };
        Block {
            cdata: &self.cdata[begin..end],
            itemsize: self.itemsize,
            nitems: if is_last {
                (self.nitems - idx * self.block_size as usize) as BlockSize
            } else {
                self.block_size
            },
        }
    }

    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let mut writer = std::fs::File::create(path)?;
        self.write_to(&mut writer)
    }

    pub fn write_to<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write + Seek,
    {
        let mut writer = ArchiveWriter::new(writer)?;

        // Write body data sections
        let cdata = writer.write_section(&self.cdata, align_of::<u8>())?;
        let block_offsets = writer.write_section(
            unsafe { cast_slice::<u64, u8>(self.block_offsets.as_ref()) },
            align_of::<u64>(),
        )?;

        // Write table and footer
        let table = schema::BlockTable {
            itemsize: self.itemsize as u32,
            nitems: self.nitems as u64,
            block_size: self.block_size as u64,
            cdata: Some(cdata),
            block_offsets: Some(block_offsets),
        };
        writer.write_main_section_and_footer(&table, schema::ArchiveType::BlockTable)?;

        Ok(())
    }

    pub fn read_from_file(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader_len = file.metadata()?.len();
        Self::read_from(file, reader_len)
    }

    pub fn read_from<R>(reader: R, len: u64) -> io::Result<Self>
    where
        R: Read + Seek,
    {
        let mut reader = ArchiveReader::new(reader, len)?;

        // Read footer and table
        let (f_meta, table_bytes) = reader.read_file_meta_and_main_section()?;
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
        let table = schema::BlockTable::decode_length_delimited(table_bytes)?;

        // Read body data sections
        let cdata = reader.read_section(table.cdata.as_ref().unwrap())?;
        let block_offsets = {
            let block_offsets_section = table.block_offsets.as_ref().unwrap();
            let block_offsets_len =
                block_offsets_section.size as usize / std::mem::size_of::<u64>();
            let mut block_offsets = Vec::<u64>::with_capacity(block_offsets_len);
            unsafe { block_offsets.set_len(block_offsets_len) };
            reader.read_section_into(block_offsets_section, unsafe {
                cast_slice_mut::<u64, u8>(block_offsets.as_mut_slice())
            })?;
            block_offsets
        };

        Ok(Self::new(
            Cow::Owned(cdata),
            Cow::Owned(block_offsets),
            table.itemsize as Itemsize,
            table.nitems as usize,
            table.block_size as BlockSize,
        ))
    }
}

impl BlockTable<'static> {
    pub fn build_from_data(
        data: &[u8],
        itemsize: Itemsize,
        block_size: BlockSize,
        encoder: &mut Encoder,
    ) -> io::Result<Self> {
        assert!(itemsize > 0);
        assert!(data.len() % itemsize as usize == 0);
        let nitems = data.len() / itemsize as usize;
        let nblocks = nitems.div_ceil(block_size as usize);
        let block_size_bytes = block_size as usize * itemsize as usize;
        let mut cdata = Vec::<u8>::new();
        let mut block_offsets = Vec::<u64>::with_capacity(nblocks.saturating_sub(1));
        let max_full_blk_cdata_len = encoder.encode_bound(block_size_bytes);
        for b_data in data.chunks(block_size_bytes) {
            let max_blk_cdata_len = if b_data.len() == block_size_bytes {
                max_full_blk_cdata_len
            } else {
                encoder.encode_bound(b_data.len())
            };

            let cdata_len = cdata.len();
            if cdata_len > 0 {
                block_offsets.push(cdata_len as u64);
            }

            cdata.reserve(max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + max_blk_cdata_len) };
            let blk_buf = &mut cdata[cdata_len..];

            let blk_cdata_len = encoder.encode(b_data, blk_buf)?;
            debug_assert!(blk_cdata_len <= max_blk_cdata_len);
            unsafe { cdata.set_len(cdata_len + blk_cdata_len) };
        }
        Ok(Self::new(
            Cow::Owned(cdata),
            Cow::Owned(block_offsets),
            itemsize,
            nitems,
            block_size,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor};

    use super::{BlockSize, BlockTable};
    use crate::dtype::{Dtyped, Itemsize};
    use crate::storage::codec::{Decoder, Encoder};
    use crate::util::cast_slice;

    fn make_encoder() -> Encoder {
        Encoder::new(3).unwrap()
    }
    fn make_decoder() -> Decoder {
        Decoder::new().unwrap()
    }

    fn decode_block(table: &BlockTable, idx: usize, decoder: &mut Decoder) -> Vec<u8> {
        let blk = table.get_block(idx);
        let out_len = blk.itemsize as usize * blk.nitems as usize;
        let mut out = vec![0u8; out_len];
        decoder.decode(&blk, &mut out).unwrap();
        out
    }

    fn build_from_items<T>(
        items: &[T],
        block_size: BlockSize,
        encoder: &mut Encoder,
    ) -> io::Result<BlockTable<'static>>
    where
        T: Dtyped,
    {
        BlockTable::build_from_data(
            unsafe { cast_slice::<T, u8>(items) },
            size_of::<T>() as Itemsize,
            block_size,
            encoder,
        )
    }

    #[test]
    fn build_single_block() {
        let items: Vec<u8> = (0u8..8).collect();
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 16, &mut encoder).unwrap();
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
    fn build_multiple_blocks_partial_last() {
        // 10 items, block_size=4 → blocks of 4, 4, 2
        let items: Vec<u8> = (0u8..10).collect();
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 4, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.get_block(2).nitems, 2);
        assert_eq!(decode_block(&table, 2, &mut make_decoder()), items[8..10]);
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

    fn round_trip<T: Dtyped>(items: &[T], block_size: BlockSize) -> BlockTable<'static> {
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
        let table2 = round_trip(&items, 16);
        assert_eq!(table2.nblocks, 1);
        assert_eq!(table2.nitems, 8);
        assert_eq!(table2.block_size, 16);
        assert_eq!(table2.itemsize, 1);
        assert_eq!(decode_block(&table2, 0, &mut make_decoder()), items);
    }

    #[test]
    fn round_trip_multiple_blocks() {
        let items: Vec<u8> = (0u8..10).collect();
        let table = round_trip(&items, 4);
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.nitems, 10);
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
        let offs = table2.block_offsets.as_ref();
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
    }

    // -----------------------------------------------------------------------
    // write_to_file / read_from_file round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_file() {
        let items: Vec<u32> = (0u32..17).collect();
        let mut encoder = make_encoder();
        let table = build_from_items(&items, 3, &mut encoder).unwrap();

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path();
        table.write_to_file(&path).unwrap();
        let table2 = BlockTable::read_from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(table2.nblocks, 6);
        assert_eq!(table2.nitems, 17);
        let mut decoder = make_decoder();
        let recovered: Vec<u8> = (0..table2.nblocks)
            .flat_map(|i| decode_block(&table2, i, &mut decoder))
            .collect();
        assert_eq!(recovered, unsafe { cast_slice::<u32, u8>(&items) });
    }
}

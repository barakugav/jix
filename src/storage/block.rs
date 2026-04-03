use std::borrow::Cow;
use std::io::{self, Read, Seek, Write};
use std::path::Path;

use crate::dtype::Itemsize;
use crate::schema;
use crate::storage::codec::Encoder;
use crate::storage::common::{Reader, Writer};

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
        let mut writer = Writer::new(writer)?;

        // Write header
        let footer_spec = writer.new_footer_spec();
        writer.write_header(Some(&footer_spec))?;

        // Write body data sections
        let cdata = writer.write_section(&self.cdata, align_of::<u8>())?;
        let block_offsets = writer.write_section(
            cast_slice_u64_u8(self.block_offsets.as_ref()),
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
        writer.write_main_section_and_footer(&table, &footer_spec)?;

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
        // Read header
        let mut reader = Reader::new(reader, len)?;
        let header = reader.read_header()?;

        // Read footer and table
        let (_footer, table) =
            reader.read_footer_and_main_section::<schema::BlockTable>(&header.footer_spec)?;

        // Read body data sections
        let cdata = reader.read_section(table.cdata.as_ref().unwrap())?;
        let block_offsets = {
            let block_offsets_section = table.block_offsets.as_ref().unwrap();
            let block_offsets_len =
                block_offsets_section.size as usize / std::mem::size_of::<u64>();
            let mut block_offsets = Vec::<u64>::with_capacity(block_offsets_len);
            unsafe { block_offsets.set_len(block_offsets_len) };
            reader.read_section_into(
                block_offsets_section,
                cast_slice_mut_u64_u8(block_offsets.as_mut_slice()),
            )?;
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

fn cast_slice_u64_u8(slice: &[u64]) -> &[u8] {
    let (ptr, len) = (slice.as_ptr(), slice.len());
    unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len * size_of::<u64>()) }
}
fn cast_slice_mut_u64_u8(slice: &mut [u64]) -> &mut [u8] {
    let (ptr, len) = (slice.as_mut_ptr(), slice.len());
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), len * size_of::<u64>()) }
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
        let max_blk_cdata_len = encoder.encode_bound(block_size_bytes);
        for b_data in data.chunks(block_size_bytes) {
            let max_blk_cdata_len = if b_data.len() == block_size_bytes {
                max_blk_cdata_len
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
    use std::io::Cursor;

    use super::{BlockSize, BlockTable};
    use crate::dtype::Itemsize;
    use crate::storage::codec::{Decoder, Encoder};

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

    #[test]
    fn build_single_block() {
        let data: Vec<u8> = (0u8..8).collect();
        let mut encoder = make_encoder();
        let table = BlockTable::build_from_data(&data, 1, 16, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 1);
        assert_eq!(table.nitems, 8);
        let decoded = decode_block(&table, 0, &mut make_decoder());
        assert_eq!(decoded, data);
    }

    #[test]
    fn build_multiple_blocks_exact_divisor() {
        // 12 items, block_size=4 → 3 full blocks
        let data: Vec<u8> = (0u8..12).collect();
        let mut encoder = make_encoder();
        let table = BlockTable::build_from_data(&data, 1, 4, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.nitems, 12);
        let mut decoder = make_decoder();
        assert_eq!(decode_block(&table, 0, &mut decoder), &data[0..4]);
        assert_eq!(decode_block(&table, 1, &mut decoder), &data[4..8]);
        assert_eq!(decode_block(&table, 2, &mut decoder), &data[8..12]);
    }

    #[test]
    fn build_multiple_blocks_partial_last() {
        // 10 items, block_size=4 → blocks of 4, 4, 2
        let data: Vec<u8> = (0u8..10).collect();
        let mut encoder = make_encoder();
        let table = BlockTable::build_from_data(&data, 1, 4, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.get_block(2).nitems, 2);
        let mut decoder = make_decoder();
        assert_eq!(decode_block(&table, 2, &mut decoder), &data[8..10]);
    }

    #[test]
    fn build_with_itemsize_greater_than_one() {
        // 4 u32 values → 16 bytes, itemsize=4, block_size=2
        let values: Vec<u32> = vec![10, 20, 30, 40];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut encoder = make_encoder();
        let table = BlockTable::build_from_data(&data, 4, 2, &mut encoder).unwrap();
        assert_eq!(table.nblocks, 2);
        assert_eq!(table.nitems, 4);
        let mut decoder = make_decoder();
        let blk0 = decode_block(&table, 0, &mut decoder);
        let blk1 = decode_block(&table, 1, &mut decoder);
        assert_eq!(&blk0, &data[0..8]);
        assert_eq!(&blk1, &data[8..16]);
    }

    // -----------------------------------------------------------------------
    // write_to / read_from round-trip
    // -----------------------------------------------------------------------

    fn round_trip(data: &[u8], itemsize: Itemsize, block_size: BlockSize) -> BlockTable<'static> {
        let mut encoder = make_encoder();
        let table = BlockTable::build_from_data(data, itemsize, block_size, &mut encoder).unwrap();
        let mut buf = Cursor::new(Vec::<u8>::new());
        table.write_to(&mut buf).unwrap();
        let bytes = buf.into_inner();
        let len = bytes.len() as u64;
        BlockTable::read_from(Cursor::new(bytes), len).unwrap()
    }

    #[test]
    fn round_trip_single_block() {
        let data: Vec<u8> = (0u8..8).collect();
        let table2 = round_trip(&data, 1, 16);
        assert_eq!(table2.nblocks, 1);
        assert_eq!(table2.nitems, 8);
        assert_eq!(table2.block_size, 16);
        assert_eq!(table2.itemsize, 1);
        let decoded = decode_block(&table2, 0, &mut make_decoder());
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_multiple_blocks() {
        let data: Vec<u8> = (0u8..10).collect();
        let table = round_trip(&data, 1, 4);
        assert_eq!(table.nblocks, 3);
        assert_eq!(table.nitems, 10);
        let mut decoder = make_decoder();
        let recovered: Vec<u8> = (0..table.nblocks)
            .flat_map(|i| decode_block(&table, i, &mut decoder))
            .collect();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_preserves_block_offsets_ordering() {
        let data: Vec<u8> = (0u8..12).collect();
        let table2 = round_trip(&data, 1, 3);
        // block_offsets should be strictly increasing
        let offs = table2.block_offsets.as_ref();
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
    }

    // -----------------------------------------------------------------------
    // write_to_file / read_from_file round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_file() {
        let data: Vec<u8> = (0u32..17).flat_map(|v| v.to_le_bytes()).collect();
        let mut encoder = make_encoder();
        let table = BlockTable::build_from_data(&data, 4, 3, &mut encoder).unwrap();

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
        assert_eq!(recovered, data);
    }
}

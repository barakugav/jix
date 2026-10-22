// Repro for bug-hunt finding F005 (run 2026-10-22-trial2). 32-bit targets only (x86 / armv7 wheels).
//
// Nothing bounds a block's byte size `block_size * itemsize` to `usize`: the header checks and
// `ArrayParams::tune` bound the block element count by u32 and the itemsize by u16, and the u64
// `check_shape_overflow` passes. Every consumer then computes the product unchecked in usize
// (`BlockTable::read_block` b_size_bytes, `read_data_slow` full_buf_len, `compact_into_builder`
// block_size_bytes / chunk_bytes, `inner_loop_untyped_broadcast` total). On i686 they all wrap to
// the same small value, so the buffer-size and decoded-size checks agree with each other, while
// element offsets inside the block (`inner_offset * block_stride`) are not wrapped and index past
// the wrapped scratch buffer.
//
// Run natively as 32-bit (release shows the real memory-safety behavior, debug panics with
// "attempt to multiply with overflow"):
//   bug-hunt/tools/cargo-i686 test --release --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_22_trial2__f005 -- --nocapture --test-threads=1
//
// Every non-ignored test asserts the CORRECT behavior (an error), so it fails while the bug exists.
// The `#[ignore]`d test `far_element_read_after_compact` CRASHES THE PROCESS (SIGSEGV) on i686
// release; run it on its own with `--ignored far_element`.
use std::io::Cursor;

use jix::storage::Compact;
use jix::{ArchiveValidation, Array, ArrayParams, Error, Filter, ReadContext, Ty};
use ndarray::{Array1, Ix1};

const ITEM: usize = 4096;
type Item = [u8; ITEM];
/// `N * ITEM = 2^32 + 4096`: wraps to 4096 in a 32-bit usize.
const N: u64 = (1 << 20) + 1;

/// A lazy array of `N` x [7u8; 4096] (4 GiB + 4 KiB logical, 4 KiB real), compacted into ONE
/// block of `N` elements with the safe public API.
fn compact_one_huge_block() -> (Result<Array<Compact<Ty<Item>>>, Error>, ReadContext) {
    let one = Array::compact_ndarray(&Array1::from_elem(1, [7u8; ITEM])).unwrap();
    let big = one.broadcast(&[N]);
    let ctx = big.read_ctx();
    let mut params = ArrayParams::new();
    params.block_shape(&[N as u32]);
    (big.compact_with(params, &ctx), ctx)
}

#[test]
fn compact_with_block_over_address_space_is_rejected() {
    if usize::BITS == 64 {
        return; // 64-bit would really compact a 4 GiB block; the bug needs a 32-bit usize
    }
    let (res, ctx) = compact_one_huge_block();
    if let Ok(c) = &res {
        // What the silently truncated array returns. Element 1 is read from the 4 KiB right after
        // the 4 KiB block scratch, element 2 from 4..8 KiB past it.
        for idx in [0u64, 1, 2] {
            match c.to_ndarray_sub::<Ix1>(&[idx..idx + 1], &ctx) {
                Err(e) => eprintln!("[{idx}] err: {e}"),
                Ok(v) => {
                    let n7 = v[0].iter().filter(|&&b| b == 7).count();
                    eprintln!("[{idx}] ok: {n7} of {ITEM} bytes are 7");
                }
            }
        }
    }
    // Correct on 32-bit: an error (a block of 4 GiB + 4 KiB does not fit the address space).
    assert!(
        res.is_err(),
        "compact_with returned Ok for a block of {} bytes on a {}-bit target",
        N * ITEM as u64,
        usize::BITS
    );
}

#[test]
#[ignore = "SIGSEGV on i686 release: reads 4 MB past a 4 KiB heap buffer"]
fn far_element_read_after_compact() {
    if usize::BITS == 64 {
        return;
    }
    let (res, ctx) = compact_one_huge_block();
    if let Ok(c) = res {
        eprintln!("compact_with returned Ok; reading element 1000");
        let r = c.to_ndarray_sub::<Ix1>(&[1000..1001], &ctx);
        // Correct: an error (or never getting here because compact_with failed).
        assert!(r.is_err(), "element 1000 of a truncated block read Ok");
    }
}

/// Byte position and length of the first zstd frame in `bytes`.
fn find_frame(bytes: &[u8]) -> (usize, usize) {
    let magic = [0x28, 0xB5, 0x2F, 0xFD];
    let pos = (0..bytes.len() - 4).find(|&i| bytes[i..i + 4] == magic).unwrap();
    let len = zstd::zstd_safe::find_frame_compressed_size(&bytes[pos..]).unwrap();
    (pos, len)
}

/// A Strict-valid archive of shape [N] x [u8; 4096], block_shape [N], no filters, whose single
/// block is a zstd frame that decodes to 4096 x 0xCD (= (N * 4096) mod 2^32 bytes).
///
/// Built from a valid archive of shape [16385] (varint 81 80 01) by patching the four header
/// varints (shape, block_shape, nitems, block_size) to N = 1048577 (varint 81 80 40, same length).
fn crafted_archive() -> Vec<u8> {
    let n = 16385usize;
    let data = Array1::from_elem(n, [1u8; ITEM]);
    let mut params = ArrayParams::new();
    params.block_shape(&[n as u32]);
    params.filters(&[] as &[Filter]).unwrap();
    let a = Array::compact_ndarray_with(&data, params).unwrap();
    let mut cur = Cursor::new(Vec::new());
    a.write_to(&mut cur).unwrap();
    let mut b = cur.into_inner();

    let (pos, len) = find_frame(&b);
    let mut patched = 0;
    for i in 0..pos - 2 {
        if b[i..i + 3] == [0x81, 0x80, 0x01] {
            b[i + 2] = 0x40;
            patched += 1;
        }
    }
    assert_eq!(patched, 4, "expected shape, block_shape, nitems, block_size varints");
    // single-segment frame, 2-byte content size (4096 - 256), one RLE last block of 4096 x 0xCD,
    // then a skippable frame padding it back to the original frame length.
    let mut repl = vec![0x28, 0xB5, 0x2F, 0xFD, 0x60];
    repl.extend_from_slice(&((ITEM - 256) as u16).to_le_bytes());
    repl.extend_from_slice(&(1u32 | (1 << 1) | ((ITEM as u32) << 3)).to_le_bytes()[..3]);
    repl.push(0xCD);
    let pad = len - repl.len() - 8;
    repl.extend_from_slice(&0x184D2A50u32.to_le_bytes());
    repl.extend_from_slice(&(pad as u32).to_le_bytes());
    repl.resize(len, 0);
    b[pos..pos + len].copy_from_slice(&repl);
    b
}

#[test]
fn crafted_archive_block_over_address_space_is_rejected() {
    let res = Array::read_from_reader(
        Cursor::new(crafted_archive()),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    );
    let a = match res {
        Err(e) => {
            eprintln!("rejected at load: {e}");
            return; // correct
        }
        Ok(a) => a.into_typed::<Item>().unwrap(),
    };
    eprintln!("loaded (Strict): shape {:?}", a.shape());
    let ctx = a.read_ctx();
    let mut read_ok = Vec::new();
    for idx in [0u64, 1, 2] {
        match a.to_ndarray_sub::<Ix1>(&[idx..idx + 1], &ctx) {
            Err(e) => eprintln!("[{idx}] err: {e}"),
            Ok(v) => {
                let ncd = v[0].iter().filter(|&&b| b == 0xCD).count();
                eprintln!("[{idx}] ok: {ncd} of {ITEM} bytes are 0xCD (the only bytes decoded)");
                read_ok.push(idx);
            }
        }
    }
    // Correct: the block decodes to 4096 bytes, not 4 GiB + 4 KiB, so every read must fail.
    assert!(read_ok.is_empty(), "elements {read_ok:?} were read Ok from a short block");
}

// Repro for bug-hunt finding F004 (run 2026-10-22-trial2).
//
// `Decoder::decode` (jix/src/codec/mod.rs) runs the whole filter chain over the destination before
// anyone compares the number of bytes zstd produced with the block size; the K1 fix only checks the
// length afterwards, in `BlockTable::read_block` (jix/src/storage/block.rs). With a short zstd
// frame (a valid frame of 8 or 0 bytes, padded to the original length with a skippable frame), the
// filter reads the never-written tail of the pooled decompression scratch - stale bytes of another
// array on a warm `ReadContext`, uninitialized heap memory on a fresh one - and writes it into the
// caller's `out` buffer, and only then is the `Err` returned.
//
// Every test asserts the CORRECT behavior (the failed read leaves nothing in `out` that the decoder
// did not produce), so each fails while the bug exists.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams, Filter, ReadContext};
use ndarray::Array1;

/// Replace the (only) zstd frame of `bytes` in place by a frame that decodes to `short`, followed
/// by a skippable frame that pads it back to the original length. All archive offsets stay valid.
fn shorten_only_frame(bytes: &mut [u8], short: &[u8]) {
    let magic = [0x28, 0xB5, 0x2F, 0xFD];
    let pos = (0..bytes.len() - 4).rev().find(|&i| bytes[i..i + 4] == magic).unwrap();
    let len = zstd::zstd_safe::find_frame_compressed_size(&bytes[pos..]).unwrap();
    assert_eq!(pos + len, bytes.len(), "the block's frame should end the archive");
    let n = short.len();
    // single-segment frame, 1-byte content size, one raw (last) block of n bytes
    let mut repl = vec![0x28, 0xB5, 0x2F, 0xFD, 0x20, n as u8];
    repl.extend_from_slice(&(1u32 | ((n as u32) << 3)).to_le_bytes()[..3]);
    repl.extend_from_slice(short);
    let pad = len - repl.len() - 8;
    repl.extend_from_slice(&0x184D2A50u32.to_le_bytes()); // skippable frame magic
    repl.extend_from_slice(&(pad as u32).to_le_bytes());
    repl.resize(len, 0);
    bytes[pos..pos + len].copy_from_slice(&repl);
}

fn archive<T: jix::dtype::Dtyped + Copy>(data: Vec<T>, filters: &[Filter]) -> Vec<u8> {
    let mut params = ArrayParams::new();
    params.block_shape(&[data.len() as u32]);
    params.filters(filters).unwrap();
    let a = Array::compact_ndarray_with(&Array1::from(data), params).unwrap();
    let mut cur = Cursor::new(Vec::new());
    a.write_to(&mut cur).unwrap();
    cur.into_inner()
}

fn load(bytes: Vec<u8>) -> Array<impl jix::ArrayStorage> {
    Array::read_from_reader(Cursor::new(bytes), None, ArrayParams::default(), ArchiveValidation::Strict)
        .unwrap()
}

/// u8, one ByteShuffle filter (a memcpy for itemsize 1): the tail of the failed read is another
/// array's data, read earlier on the same context.
#[test]
fn f004_stale_bytes_of_another_array_reach_out_on_error() {
    let ctx = ReadContext::default();
    let other = load(archive(vec![0x5Au8; 64], &[Filter::ByteShuffle]));
    let mut obuf = vec![0u8; 64];
    other.to_ndarray_slice(&[0..64], &mut obuf, &ctx).unwrap();

    let mut bytes = archive((0..64u8).collect(), &[Filter::ByteShuffle]);
    shorten_only_frame(&mut bytes, &[0xEE; 8]);
    let victim = load(bytes);

    let mut out = vec![0u8; 64];
    let res = victim.to_ndarray_slice(&[0..64], &mut out, &ctx);
    eprintln!("res = {:?}", res.as_ref().map_err(|e| e.to_string()));
    eprintln!("out after Err = {out:?}");
    assert!(res.is_err());
    // Correct: `out` holds only its own bytes (0) or bytes the decoder produced (0xEE).
    assert!(
        out.iter().all(|&b| b == 0 || b == 0xEE),
        "{} bytes of a different array (0x5A) were written into the caller's buffer",
        out.iter().filter(|&&b| b == 0x5A).count()
    );
}

/// u32 through BitShuffle and through a 3-filter chain: the bytes left in `out` by a failed read
/// depend on which array was read earlier on the context. They must not.
#[test]
fn f004_failed_read_output_depends_on_unrelated_pool_contents() {
    let chains: [&[Filter]; 2] =
        [&[Filter::BitShuffle], &[Filter::ByteShuffle, Filter::BitShuffle, Filter::ByteShuffle]];
    for filters in chains {
        // incompressible payload, so the frame is long enough to hold the replacement
        let noise = (0..64u32).map(|i| i.wrapping_mul(2654435761).rotate_left(13)).collect();
        let mut bytes = archive(noise, filters);
        shorten_only_frame(&mut bytes, &[0xEE; 8]);
        let victim = load(bytes);

        let mut outs = Vec::new();
        for secret in [0x5A5A5A5Au32, 0xA5A5A5A5] {
            let ctx = ReadContext::default();
            let other = load(archive(vec![secret; 64], filters));
            let mut obuf = vec![0u8; 256];
            other.to_ndarray_slice(&[0..64], &mut obuf, &ctx).unwrap();

            let mut out = vec![0u8; 256];
            let res = victim.to_ndarray_slice(&[0..64], &mut out, &ctx);
            assert!(res.is_err());
            eprintln!("{filters:?} after reading [{secret:#x}; 64]: out[..48] = {:?}", &out[..48]);
            outs.push(out);
        }
        assert!(
            outs[0] == outs[1],
            "{filters:?}: the caller's buffer after a failed read depends on another array's data"
        );
    }
}

/// Fresh context: the decompression scratch is a fresh, uninitialized pool allocation, and its
/// bytes (none of which zstd wrote: the frame decodes to 0 bytes) are copied into `out`.
/// Valgrind reports the uninitialized origin at BufferPool::get <- Decoder::decode.
#[test]
fn f004_uninit_scratch_reaches_out_on_error() {
    let mut bytes = archive(vec![0x11u8; 4096], &[Filter::ByteShuffle]);
    shorten_only_frame(&mut bytes, &[]);
    let victim = load(bytes);
    // Leave garbage in the freed heap so a reused chunk is visibly non-zero.
    drop(std::hint::black_box(vec![0x77u8; 8192]));
    let ctx = ReadContext::default();
    let mut out = vec![0u8; 4096];
    let res = victim.to_ndarray_slice(&[0..4096], &mut out, &ctx);
    eprintln!("res = {:?}", res.as_ref().map_err(|e| e.to_string()));
    assert!(res.is_err());
    let nonzero = out.iter().filter(|&&b| b != 0).count();
    eprintln!("nonzero bytes in out after Err: {nonzero} of 4096; first 16: {:?}", &out[..16]);
    assert_eq!(nonzero, 0, "out was overwritten with bytes the decoder never produced");
}

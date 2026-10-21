// Verifier repro for F001: a block whose zstd payload decodes to fewer bytes than the block size
// is accepted (only `debug_assert_eq!` at storage/block.rs:193 checks the decoded length).
// Debug build: every test panics at that debug_assert. Release build: every read returns Ok and
// the never-written tail of the destination is returned as array data.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams, Filter};
use ndarray::{Array1, Ix1};

fn lcg_bytes(n: usize, mut s: u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// (offset, len) of every zstd frame in `bytes`.
fn find_frames(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i..i + 4] == [0x28, 0xB5, 0x2F, 0xFD]
            && let Ok(n) = zstd::zstd_safe::find_frame_compressed_size(&bytes[i..])
        {
            out.push((i, n));
            i += n;
            continue;
        }
        i += 1;
    }
    out
}

/// Replace frame `k` in place by a hand-made zstd frame holding `short.len()` raw bytes, followed
/// by a skippable frame padding it to the original length. Every offset / length in the archive
/// stays valid; the block simply decodes to `short.len()` bytes.
fn shorten_frame(bytes: &mut [u8], k: usize, short: &[u8]) {
    let (pos, len) = find_frames(bytes)[k];
    let n = short.len();
    // magic, FHD (single segment, 1-byte FCS), FCS, block header (last, raw, size n), payload
    let mut repl = vec![0x28, 0xB5, 0x2F, 0xFD, 0x20, n as u8];
    let bh = 1u32 | ((n as u32) << 3);
    repl.extend_from_slice(&bh.to_le_bytes()[..3]);
    repl.extend_from_slice(short);
    let pad = len - repl.len() - 8;
    repl.extend_from_slice(&0x184D2A50u32.to_le_bytes());
    repl.extend_from_slice(&(pad as u32).to_le_bytes());
    repl.resize(len, 0);
    bytes[pos..pos + len].copy_from_slice(&repl);
}

fn archive(data: &[u8], block: u32, filters: &[Filter]) -> Vec<u8> {
    let mut params = ArrayParams::new();
    params.block_shape(&[block]);
    params.filters(filters).unwrap();
    let a = Array::compact_ndarray_with(&Array1::from(data.to_vec()), params).unwrap();
    let mut cur = Cursor::new(Vec::new());
    a.write_to(&mut cur).unwrap();
    cur.into_inner()
}

fn load(bytes: Vec<u8>) -> Array<impl jix::storage::ArrayStorageTyped<Item = u8>> {
    Array::read_from_reader(
        Cursor::new(bytes),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .unwrap()
    .into_typed::<u8>()
    .unwrap()
}

/// read_data_slow path: the pooled tmp buffer still holds a previously read array's block, and the
/// short decode of the corrupt block leaves it there -> another array's bytes come back as data.
#[test]
fn f001_stale_pool_bytes_from_other_array() {
    let secret = load(archive(&[0x5Au8; 64], 64, &[]));
    let mut bytes = archive(&lcg_bytes(128, 1), 64, &[]);
    shorten_frame(&mut bytes, 1, &[0xEE; 8]);
    let victim = load(bytes);

    let ctx = victim.read_ctx();
    // Unaligned single-block read: decodes into the pooled tmp buffer, which now holds 0x5A x64.
    let s = secret.to_ndarray_sub::<Ix1>(&[1..64], &ctx).unwrap();
    assert!(s.iter().all(|&b| b == 0x5A));
    // Same ReadContext, corrupt block 1 of another array.
    let res = victim.to_ndarray_sub::<Ix1>(&[64..127], &ctx);
    match res {
        Ok(got) => panic!("corrupt block accepted: got {:?}", got.to_vec()),
        Err(e) => eprintln!("ok, error: {e}"),
    }
}

/// Fast path (aligned single block, contiguous out): decodes straight into the
/// `ndarray::Array::uninit` buffer, then `assume_init` (array.rs:800) on 56 unwritten bytes.
#[test]
fn f001_uninit_ndarray_tail() {
    let mut bytes = archive(&lcg_bytes(64, 2), 64, &[]);
    shorten_frame(&mut bytes, 0, &[0xEE; 8]);
    let a = load(bytes);
    // Leave a recognizable freed chunk of the same size class for malloc to hand back.
    drop(std::hint::black_box(vec![0x77u8; 64]));
    match a.to_ndarray::<Ix1>() {
        Ok(got) => panic!("corrupt block accepted: got {:?}", got.to_vec()),
        Err(e) => eprintln!("ok, error: {e}"),
    }
}

/// Filter path: decompression target is the pooled tmp from Decoder::decode (codec/mod.rs:276);
/// the byte-shuffle decode then reads its stale tail.
#[test]
fn f001_byteshuffle_stale_pool_tail() {
    let secret = load(archive(&[0x5Au8; 64], 64, &[Filter::ByteShuffle]));
    let mut bytes = archive(&lcg_bytes(64, 3), 64, &[Filter::ByteShuffle]);
    shorten_frame(&mut bytes, 0, &[0xEE; 8]);
    let victim = load(bytes);
    let ctx = victim.read_ctx();
    let s = secret.to_ndarray_sub::<Ix1>(&[0..64], &ctx).unwrap();
    assert!(s.iter().all(|&b| b == 0x5A));
    match victim.to_ndarray_sub::<Ix1>(&[0..64], &ctx) {
        Ok(got) => panic!("corrupt block accepted: got {:?}", got.to_vec()),
        Err(e) => eprintln!("ok, error: {e}"),
    }
}

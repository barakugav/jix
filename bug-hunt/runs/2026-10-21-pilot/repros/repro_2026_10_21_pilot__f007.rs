//! F007 repro: the mmap reader rejects valid archives whose block-locations section is not
//! 8-aligned in absolute file position, while the heap reader accepts the same bytes.
use std::io::{Cursor, Write};
use std::path::Path;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::Ix1;

fn archive_bytes(v: &[f64]) -> Vec<u8> {
    let a = Array::compact_ndarray(&ndarray::Array1::from(v.to_vec())).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    buf.into_inner()
}

fn read_both(
    path: &Path,
    off: u64,
    len: u64,
) -> (Result<Vec<f64>, String>, Result<Vec<f64>, String>) {
    let heap = Array::read_from_file_section(
        path,
        off,
        len,
        ArrayParams::default(),
        ArchiveValidation::default(),
    )
    .map(|x| {
        x.into_typed::<f64>()
            .unwrap()
            .to_ndarray::<Ix1>()
            .unwrap()
            .to_vec()
    })
    .map_err(|e| e.to_string());
    // Safety: the file is not modified while the array is alive.
    let mmap = unsafe {
        Array::read_from_file_mmap(
            path,
            off,
            len,
            ArrayParams::default(),
            ArchiveValidation::default(),
        )
    }
    .map(|x| {
        x.into_typed::<f64>()
            .unwrap()
            .to_ndarray::<Ix1>()
            .unwrap()
            .to_vec()
    })
    .map_err(|e| e.to_string());
    (heap, mmap)
}

fn rd_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Locate the block-table TOC (2 x {i64 offset, u64 size}) in an archive written at base 0.
/// Returns (toc_pos, blocks_loc_offset).
fn find_toc(b: &[u8]) -> Option<(usize, usize)> {
    for t in 0..b.len().saturating_sub(32) {
        let o0 = rd_u64(b, t) as usize;
        let s0 = rd_u64(b, t + 8) as usize;
        let o1 = rd_u64(b, t + 16) as usize;
        let s1 = rd_u64(b, t + 24) as usize;
        if o0 % 8 == 0
            && o0 >= t + 32
            && o0 < t + 40
            && b[t + 32..o0].iter().all(|&x| x == 0)
            && s0 > 0
            && s0 % 24 == 0
            && o1 == o0 + s0
            && o1 + s1 == b.len()
        {
            return Some((t, o0));
        }
    }
    None
}

#[test]
fn f007_mmap_rejects_relocated_archive() {
    let dir = tempfile::tempdir().unwrap();
    let vals = [10.0f64, 20.0, 30.0, 40.0];
    let b = archive_bytes(&vals);

    // (1) The same archive bytes placed after k bytes (embedded / concatenated / copied).
    for k in 0..8usize {
        let path = dir.path().join(format!("prefix_{k}.jix"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&vec![0xAAu8; k]).unwrap();
        f.write_all(&b).unwrap();
        drop(f);
        let (heap, mmap) = read_both(&path, k as u64, b.len() as u64);
        println!("prefix {k}: heap={heap:?} mmap={mmap:?}");
        assert_eq!(heap.as_deref(), Ok(&vals[..]));
        if k % 8 == 0 {
            assert_eq!(mmap.as_deref(), Ok(&vals[..]));
        } else {
            assert_eq!(
                mmap,
                Err("data section offset is not properly aligned".into())
            );
        }
    }

    // (2) `cat a.jix b.jix`, read b by (offset, len).
    let a = archive_bytes(&[1.0, 2.0, 3.0]);
    let path = dir.path().join("cat.jix");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&a).unwrap();
    f.write_all(&b).unwrap();
    drop(f);
    let (heap, mmap) = read_both(&path, a.len() as u64, b.len() as u64);
    println!("cat a b (len(a)={}): heap={heap:?} mmap={mmap:?}", a.len());
    assert_eq!(heap.as_deref(), Ok(&vals[..]));
    assert!(mmap.is_err());
}

/// (3) Simulate a file written by the i686-unknown-linux-{gnu,musl} wheel, where
/// `align_of::<BlockLocation2>() == align_of::<u64>() == 4`. The writer then pads the TOC end to a
/// multiple of 4, not 8. All other layout (Section = 16 B, BlockLocation2 = 24 B) is identical, so
/// the i686 bytes are the x86_64 bytes with 4 fewer padding bytes and both TOC offsets minus 4.
#[test]
fn f007_mmap_rejects_i686_written_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut done = false;
    for n in (1..200usize).chain([1000, 5000, 20000, 100000, 1_000_000]) {
        let vals: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let b = archive_bytes(&vals);
        let (t, o0) = find_toc(&b).expect("toc not found");
        let pad64 = o0 - (t + 32);
        if pad64 < 4 {
            // i686 writer would produce the same bytes as x86_64 for this array.
            continue;
        }
        // Build the i686 byte image: drop 4 padding bytes, shift both section offsets by -4.
        let mut b32 = Vec::with_capacity(b.len() - 4);
        b32.extend_from_slice(&b[..t]);
        let o0n = rd_u64(&b, t) - 4;
        let o1n = rd_u64(&b, t + 16) - 4;
        b32.extend_from_slice(&o0n.to_le_bytes());
        b32.extend_from_slice(&b[t + 8..t + 16]);
        b32.extend_from_slice(&o1n.to_le_bytes());
        b32.extend_from_slice(&b[t + 24..t + 32]);
        b32.extend_from_slice(&b[t + 32..t + 32 + pad64 - 4]);
        b32.extend_from_slice(&b[o0..]);
        assert_eq!(o0n % 4, 0);
        assert_eq!(o0n % 8, 4);

        let path = dir.path().join(format!("i686_{n}.jix"));
        std::fs::write(&path, &b32).unwrap();
        let (heap, mmap) = read_both(&path, 0, b32.len() as u64);
        println!(
            "n={n}: x86_64 padding={pad64}, i686 padding={}, blocks_loc at {o0n} (mod 8 = {})",
            pad64 - 4,
            o0n % 8
        );
        println!(
            "  heap={:?}\n  mmap={mmap:?}",
            heap.as_ref().map(|v| (v.len(), v[n - 1]))
        );
        assert_eq!(heap.as_deref(), Ok(&vals[..]));
        assert_eq!(
            mmap,
            Err("data section offset is not properly aligned".into())
        );
        done = true;
        break;
    }
    assert!(done);
}

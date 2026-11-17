//! K190 (run 2026-11-16-scan-106-107): `ArrayParams::level` / `filters` / `codec` each do
//! `encoder_params.get_or_insert_with(EncoderParams::default)`, and `inherit_unset_from_storage`
//! inherits the encoder params only as a whole, so setting ONE codec field resets the source's
//! other codec fields to their defaults. `compact_with` documents "Fields not set in `params` are
//! inherited from the source storage" (and the Python `compact` / `write_array` docs say the same
//! per key), so an unset `filters` must keep the source's filters when only `level` is set, and an
//! unset `level` must keep the source's level when only `filters` is set.
//!
//! Each test compares the archive bytes against an explicitly spelled-out reference. The tests
//! assert the documented behavior, so they fail while the bug exists.
//!
//! Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//!          --test repro_k190 -- --nocapture
use jix::{Array, ArrayParams, Codec, Filter, ArrayStorage};
use std::io::Cursor;

fn archive<S: ArrayStorage>(a: &Array<S>) -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf, ArrayParams::new(), &a.read_ctx())
        .unwrap();
    buf.into_inner()
}

fn data() -> ndarray::Array1<i32> {
    ndarray::Array1::from_iter(0..(1i32 << 16))
}

#[test]
fn level_only_keeps_source_filters() {
    let x = data();
    let src = Array::compact_ndarray_with(
        &x,
        ArrayParams::new().filters(&[]).unwrap().level(3).unwrap(),
    )
    .unwrap();
    let ctx = src.read_ctx();
    let want = archive(&src.compact_with(ArrayParams::new(), &ctx).unwrap());
    // The source's level is already 3, so only `filters` is unset here.
    let got = archive(
        &src.compact_with(ArrayParams::new().level(3).unwrap(), &ctx)
            .unwrap(),
    );
    println!("level only: got {} bytes, expected {} bytes", got.len(), want.len());
    assert!(got == want, "filters=[] of the source was not inherited");
}

#[test]
fn codec_only_keeps_source_filters() {
    let x = data();
    let src = Array::compact_ndarray_with(&x, ArrayParams::new().filters(&[]).unwrap()).unwrap();
    let ctx = src.read_ctx();
    let want = archive(&src.compact_with(ArrayParams::new(), &ctx).unwrap());
    let got = archive(
        &src.compact_with(ArrayParams::new().codec(Codec::Zstd), &ctx)
            .unwrap(),
    );
    println!("codec only: got {} bytes, expected {} bytes", got.len(), want.len());
    assert!(got == want, "filters=[] of the source was not inherited");
}

#[test]
fn filters_only_keeps_source_level() {
    // Low-entropy random walk, where level 19 beats level 3.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut acc = 0i32;
    let y = ndarray::Array1::from_iter((0..(1 << 16)).map(|_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        acc = acc.wrapping_add((s % 4) as i32);
        acc
    }));
    let src = Array::compact_ndarray_with(&y, ArrayParams::new().level(19).unwrap()).unwrap();
    let ctx = src.read_ctx();
    let want = archive(&src.compact_with(ArrayParams::new(), &ctx).unwrap());
    // The source's filters are already [ByteShuffle], so only `level` is unset here.
    let got = archive(
        &src.compact_with(
            ArrayParams::new().filters(&[Filter::ByteShuffle]).unwrap(),
            &ctx,
        )
        .unwrap(),
    );
    println!("filters only: got {} bytes, expected {} bytes", got.len(), want.len());
    assert!(got == want, "level 19 of the source was not inherited");
}

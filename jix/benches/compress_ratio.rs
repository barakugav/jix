mod common;

use jix::{Array, ArrayParams};

use crate::common::{create_data, Profile};

fn main() {
    let args = [
        (
            // array_shape
            [600, 32],
            // block_shapes
            [[4, 4], [1, 32], [32, 32]].as_slice(),
        ),
        (
            // array_shape
            [11_000, 460],
            // block_shapes
            &[[4, 4], [1, 32], [512, 32], [1000, 230]],
        ),
    ];
    let levels = [3, 9];

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("array.jix");
    println!(
        "{:<12} {:>14} {:>10} {:>6} {:>14} {:>14} {:>8}",
        "profile", "array_shape", "block", "level", "raw_bytes", "stored_bytes", "ratio"
    );
    for (shape, block_shapes) in args {
        for profile in Profile::ALL {
            let data = create_data::<i32>(profile, &shape, 0x1567c2dbd6e16813);
            let raw_bytes = data.len() * std::mem::size_of::<i32>();
            for block_shape in block_shapes {
                for level in levels {
                    let mut params = ArrayParams::new();
                    params.block_shape(block_shape);
                    params.level(level).unwrap();
                    let array = Array::compact_ndarray_with(&data, params).unwrap();

                    if path.exists() {
                        std::fs::remove_file(&path).unwrap();
                    }
                    array.write_to_file(&path).unwrap();
                    let stored_bytes = std::fs::metadata(&path).unwrap().len();

                    let ratio = raw_bytes as f64 / stored_bytes as f64;
                    println!(
                        "{:<12} {:>14} {:>10} {:>6} {:>14} {:>14} {:>8.2}",
                        profile.name(),
                        format!("{shape:?}"),
                        format!("{block_shape:?}"),
                        level,
                        raw_bytes,
                        stored_bytes,
                        ratio,
                    );
                }
            }
        }
    }
}

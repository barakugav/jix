use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Generate protobuf types
    let proto_dir = crate_dir.join("proto");
    let proto_gen_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("proto_gen");
    fs::create_dir_all(&proto_gen_dir).unwrap();
    prost_build::Config::new()
        .out_dir(proto_gen_dir)
        .include_file("_includes.rs")
        .compile_protos(&["zix/v1/block.proto"], &[&proto_dir])
        .expect("Failed to compile protos");
}

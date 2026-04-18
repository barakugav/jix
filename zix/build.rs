use std::path::{Path, PathBuf};
use std::{env, fs, io};

fn main() {
    // TODO: publish the generated protobuf code, generate in a standalone script

    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Generate protobuf types
    let proto_dir = crate_dir.join("proto");
    let proto_files = find_files_recursively(&proto_dir, "proto").unwrap();
    let proto_gen_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("proto_gen");
    fs::create_dir_all(&proto_gen_dir).unwrap();
    prost_build::Config::new()
        .out_dir(proto_gen_dir)
        .include_file("_includes.rs")
        .compile_protos(
            &proto_files
                .iter()
                .map(|p| p.strip_prefix(&proto_dir).unwrap())
                .collect::<Vec<_>>(),
            &[&proto_dir],
        )
        .expect("Failed to compile protos");
}

fn find_files_recursively(dir: &Path, extension: &str) -> io::Result<Vec<PathBuf>> {
    assert!(dir.is_dir(), "Expected a directory: {}", dir.display());
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(find_files_recursively(&path, extension)?);
        } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(extension) {
            files.push(path);
        }
    }
    Ok(files)
}

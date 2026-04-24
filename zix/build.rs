use std::path::{Path, PathBuf};
use std::{fs, io};

fn main() {
    // TODO: add CI to check committed schema matches the generated one
    #[cfg(feature = "build-schema")]
    build_schema();
}

#[cfg(feature = "build-schema")]
fn build_schema() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Generate protobuf types
    let proto_dir = crate_dir.join("proto");
    let proto_files = find_files_recursively(&proto_dir, "proto").unwrap();
    let proto_gen_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("proto_gen");
    fs::create_dir_all(&proto_gen_dir).unwrap();
    prost_build::Config::new()
        .out_dir(&proto_gen_dir)
        .include_file("_includes.rs")
        .compile_protos(
            &proto_files
                .iter()
                .map(|p| p.strip_prefix(&proto_dir).unwrap())
                .collect::<Vec<_>>(),
            &[&proto_dir],
        )
        .expect("Failed to compile protos");

    println!("cargo:rerun-if-env-changed=ZIX_SCHEMA_GEN_UPDATE");
    if std::env::var("ZIX_SCHEMA_GEN_UPDATE").as_deref() == Ok("1") {
        let published_proto_gen = crate_dir
            .join("src")
            .join("archive")
            .join("schema")
            .join("proto_gen");
        if published_proto_gen.exists() {
            fs::remove_dir_all(&published_proto_gen).unwrap();
        }
        for gen_file in find_files_recursively(&proto_gen_dir, "rs").unwrap() {
            let dst = published_proto_gen.join(gen_file.strip_prefix(&proto_gen_dir).unwrap());
            let dst_parent = dst.parent().unwrap();
            if !dst_parent.exists() {
                fs::create_dir_all(dst_parent).unwrap();
            }
            fs::copy(gen_file, dst).unwrap();
        }
    };
}

#[allow(unused)]
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

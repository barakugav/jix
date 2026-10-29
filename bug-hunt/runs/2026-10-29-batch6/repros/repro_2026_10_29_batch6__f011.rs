// Repro for F011 (bug-hunt run 2026-10-29-batch6).
//
// `impl Dtyped for [T; N]` asserts `N < Itemsize::MAX` (jix/src/dtype.rs:1209-1212), so
// `<[u8; 65535] as Dtyped>::DTYPE` is a const-eval error, although the runtime dtype `u8` with
// shape [65535] (itemsize 65535 == Itemsize::MAX) is valid and accepted by `set_shape`.
//
// The defect is a COMPILE error, so a test that names `[u8; 65535]` directly would break the
// build of this target. Instead this test compiles a tiny program against the already built
// `libjix` rlib with rustc at run time and asserts that it compiles (and runs) - so it fails while
// the bug exists and passes once the bound is fixed to `N <= Itemsize::MAX`.
//
// A control program with N = 65534 is compiled first to select a usable rlib and to show that the
// harness itself works.

use std::path::{Path, PathBuf};
use std::process::Command;

fn program(n: u32) -> String {
    format!(
        r#"
use jix::dtype::{{Dtype, Dtyped, ScalarKind}};
fn main() {{
    let mut d = Dtype::new_scalar(ScalarKind::U8);
    d.set_shape(&[{n}]).unwrap();
    assert_eq!(d.itemsize() as u32, {n});
    let t = <[u8; {n}] as Dtyped>::DTYPE;
    assert_eq!(d, t);
    println!("ok: [u8; {n}]::DTYPE == runtime u8 shape [{n}], itemsize {{}}", t.itemsize());
}}
"#
    )
}

/// Compiles and runs `src` against `rlib`. Returns (compiled_and_ran_ok, combined output).
fn compile_and_run(deps: &Path, rlib: &Path, src: &str, tmp: &Path, name: &str) -> (bool, String) {
    let src_path = tmp.join(format!("{name}.rs"));
    let bin_path = tmp.join(name);
    std::fs::write(&src_path, src).unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-type=bin")
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("--extern")
        .arg(format!("jix={}", rlib.display()))
        .arg("-o")
        .arg(&bin_path)
        .arg(&src_path)
        .output()
        .expect("failed to spawn rustc");
    let mut log = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        return (false, log);
    }
    let run = Command::new(&bin_path).output().expect("failed to run program");
    log.push_str(&String::from_utf8_lossy(&run.stdout));
    log.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), log)
}

#[test]
fn u8_array_of_65535_is_dtyped() {
    let exe = std::env::current_exe().unwrap();
    let deps: PathBuf = exe.parent().unwrap().to_path_buf();
    let mut rlibs: Vec<PathBuf> = std::fs::read_dir(&deps)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy();
            n.starts_with("libjix-") && n.ends_with(".rlib")
        })
        .collect();
    rlibs.sort_by_key(|p| std::cmp::Reverse(p.metadata().unwrap().modified().unwrap()));
    let tmp = tempfile::tempdir().unwrap();

    // Control: N = 65534 must compile and run against the chosen rlib.
    let mut chosen = None;
    for rlib in &rlibs {
        let (ok, log) = compile_and_run(&deps, rlib, &program(65534), tmp.path(), "control");
        eprintln!("control with {}:\n{log}", rlib.display());
        if ok {
            chosen = Some(rlib.clone());
            break;
        }
    }
    let rlib = chosen.expect("harness problem: control program (N = 65534) did not build");

    // The defect: N = 65535 (itemsize == Itemsize::MAX) must also be Dtyped.
    let (ok, log) = compile_and_run(&deps, &rlib, &program(65535), tmp.path(), "bug");
    eprintln!("N = 65535:\n{log}");
    assert!(
        ok,
        "<[u8; 65535] as Dtyped>::DTYPE should equal the runtime dtype u8 shape [65535] \
         (itemsize 65535 == Itemsize::MAX), but the program failed:\n{log}"
    );
}

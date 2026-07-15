fn main() {
    let check_cfg = true; // msrv 1.89

    println!("cargo::rerun-if-env-changed=JIX_DENY_WARNINGS");
    let deny_warnings = std::env::var("JIX_DENY_WARNINGS").as_deref() == Ok("1");
    if check_cfg {
        println!("cargo:rustc-check-cfg=cfg(deny_warnings)");
    }
    if deny_warnings {
        println!("cargo:rustc-cfg=deny_warnings");
    }

    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap()
    );
}

fn main() {
    println!("cargo::rerun-if-env-changed=JIX_DENY_WARNINGS");
    let deny_warnings = std::env::var("JIX_DENY_WARNINGS").as_deref() == Ok("1");
    println!("cargo:rustc-check-cfg=cfg(deny_warnings)");
    if deny_warnings {
        println!("cargo:rustc-cfg=deny_warnings");
    }
}

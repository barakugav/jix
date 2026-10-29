// F010: #[derive(Dtyped)] on a struct with a raw-identifier field (`r#type`).
//
// Correct behavior: the derive compiles, and the recorded field name is the unprefixed
// identifier ("type"), so the derived dtype equals a struct dtype built at runtime (e.g. from
// an archive or a numpy record dtype) with a field named "type".
//
// While the bug exists this file FAILS TO COMPILE: the proc macro panics with
// `"FIELD_0_R#TYPE_DTYPE"` is not a valid identifier (jix-macros/src/lib.rs:147-148).
use jix::dtype::{Dtype, Dtyped};

#[derive(Copy, Clone, jix::dtype::Dtyped)]
#[repr(C)]
struct S {
    r#type: u32,
    x: u32,
}

#[test]
fn raw_ident_field_name_is_unprefixed() {
    let names: Vec<String> = S::DTYPE
        .fields()
        .unwrap()
        .iter()
        .map(|f| f.0.to_string())
        .collect();
    assert_eq!(names, vec!["type".to_string(), "x".to_string()]);
}

#[test]
fn raw_ident_dtype_equals_runtime_dtype() {
    let runtime = Dtype::from_fields(vec![
        ("type".to_string(), 0, u32::DTYPE),
        ("x".to_string(), 4, u32::DTYPE),
    ])
    .unwrap();
    assert_eq!(S::DTYPE, runtime);
}

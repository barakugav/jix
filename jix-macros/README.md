# jix-macros

Procedural macros for the [`jix`](https://crates.io/crates/jix) crate. Currently provides `#[derive(Dtyped)]`,
which implements the `jix::dtype::Dtyped` trait for `#[repr(C)]` structs so they can be
used as element types in a `jix::Array`. Structs with padding bytes are rejected; add the padding
as an explicit field (e.g. `_pad: [u8; 3]`) instead. See the crate documentation for details.

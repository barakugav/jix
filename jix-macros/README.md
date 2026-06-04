# jix-macros

Procedural macros for the [`jix`](../jix/) crate. Currently provides `#[derive(Dtyped)]`,
which implements the `jix::dtype::Dtyped` trait for `#[repr(C)]` structs so they can be
used as element types in a `jix::Array`. See the crate documentation for details.

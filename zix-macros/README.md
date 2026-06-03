# zix-macros

Procedural macros for the [`zix`](../zix/) crate. Currently provides `#[derive(Dtyped)]`,
which implements the `zix::dtype::Dtyped` trait for `#[repr(C)]` structs so they can be
used as element types in a `zix::Array`. See the crate documentation for details.

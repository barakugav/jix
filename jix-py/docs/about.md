# About

## Links

- Source code: [github.com/barakugav/jix](https://github.com/barakugav/jix)
- Rust crate documentation: [docs.rs/jix](https://docs.rs/jix) - the lower-level typed
  Rust API used by these Python bindings.


## Inspirations and credits

This project would not exist without the work of several upstream authors. In particular:

- **[C-Blosc2](https://github.com/Blosc/c-blosc2)** by Francesc Alted and the Blosc
  Development Team is the conceptual ancestor of jix's block-and-codec design. Jix can
  reasonably be described as a port of those ideas into Rust.
- The bit-shuffle filter is the work of **Kiyoshi Masui**
  ([kiyo-masui/bitshuffle](https://github.com/kiyo-masui/bitshuffle)).
- Portions of the aligned allocation code are derived from the
  [`aligned-vec`](https://github.com/sarah-quinones/aligned-vec) crate by Sarah Quinones.

Full attribution and license text is in
[`NOTICE`](https://github.com/barakugav/jix/blob/main/NOTICE) at the repository root.


## License

Apache-2.0. See [`LICENSE`](https://github.com/barakugav/jix/blob/main/LICENSE).


## Author

Barak Ugav - [barakugav@gmail.com](mailto:barakugav@gmail.com),
[github.com/barakugav](https://github.com/barakugav).

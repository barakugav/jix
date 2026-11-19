mod array;
mod block;
mod common;
mod schema;
mod version;

/// How thoroughly an archive's consistency is verified at load time.
///
/// - **O(1) checks** are always performed. They examine a constant amount of metadata: the file
///   magic, the archive type tag, shape and block-shape consistency, and similar fixed-size
///   fields.
/// - **O(data_size) checks** scan structures that grow with the array, most notably the
///   per-block offset table. They are only performed in [`ArchiveValidation::Strict`] mode.
///
/// For example, [`ArchiveValidation::Strict`] currently verifies that every block's
/// `(offset, len)` location lies within the block-data section, an O(nblocks) scan. Future
/// versions may add, remove or move checks between the two categories, so do not rely on any
/// specific check: treat this enum as a coarse "be paranoid" vs "trust the source" knob.
///
/// The default is [`ArchiveValidation::Strict`]. Use [`ArchiveValidation::Minimal`] when the
/// archive comes from a trusted source and the extra scan cost is measurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ArchiveValidation {
    /// Perform only the O(1) checks. Suitable for archives from a trusted source.
    Minimal,
    /// Perform all checks, including the O(data_size) scans. The default; use it whenever the
    /// archive's origin is untrusted or unverified.
    #[default]
    Strict,
}

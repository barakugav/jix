mod array;
mod block;
mod common;
mod schema;
mod version;

/// Controls how aggressively an archive's internal consistency is verified at load time.
///
/// When reading an archive, jix performs sanity checks to ensure the file is well-formed before
/// trusting its contents. These checks fall into two cost categories:
///
/// - **O(1) checks** are *always* performed regardless of the validation mode. They examine
///   only a constant amount of metadata - the file magic header, the archive type tag,
///   declared shape and block-shape consistency, and similar fixed-size fields. The cost
///   is negligible, so there is never a reason to skip them.
///
/// - **O(data_size) checks** scan structures whose size grows with the array (most notably
///   the per-block offset table, which has one entry per block). For very large arrays this
///   can become non-trivial - on the order of memory-touch cost over the whole offset table.
///   These checks are only performed in [`ArchiveValidation::Strict`] mode.
///
/// As a concrete example, [`ArchiveValidation::Strict`] currently verifies that the block
/// offset table is monotonically non-decreasing and that its final entry lies within the
/// block-data section - an O(nblocks) scan. This is illustrative only: future versions of
/// the library may add new checks, remove existing ones, or move checks between the two
/// categories. Do not rely on any specific check being present or absent in either mode -
/// treat this enum as a coarse "be paranoid" vs "trust the source" knob, not as a contract
/// over a particular set of validations.
///
/// The default is [`ArchiveValidation::Strict`]. Use [`ArchiveValidation::Minimal`] when the
/// archive comes from a trusted source (e.g. produced by your own pipeline) and the extra
/// scan cost is measurable in your workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ArchiveValidation {
    /// Perform only the constant-time (O(1)) consistency checks.
    ///
    /// Suitable for archives produced by a trusted source where the O(data_size) scans would
    /// add measurable overhead with no expected benefit.
    Minimal,
    /// Perform all checks, including those that scan structures whose size grows with the
    /// array (O(data_size)).
    ///
    /// This is the default and the right choice unless you have a specific reason to skip the
    /// extra scans. Use it whenever the archive's origin is untrusted or unverified.
    #[default]
    Strict,
}

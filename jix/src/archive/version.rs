pub const VERSION: Version = Version::new(
    const_atoi(env!("CARGO_PKG_VERSION_MAJOR").as_bytes()),
    const_atoi(env!("CARGO_PKG_VERSION_MINOR").as_bytes()),
    const_atoi(env!("CARGO_PKG_VERSION_PATCH").as_bytes()),
    parse_pre_release(env!("CARGO_PKG_VERSION_PRE").as_bytes()),
    0, // TODO
);

/// Library version encoded as a 64-bit unsigned integer.
///
/// Bit field layout:
///   major          :  8 [ 0.. 8]
///   minor          :  8 [ 8..16]
///   patch          : 10 [16..26]
///   pre-release    :  6 [26..32]
///   build-metadata : 32 [32..64]
pub const VERSION_U64: u64 = VERSION.encode();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Version {
    pub(crate) major: u32,
    pub(crate) minor: u32,
    pub(crate) patch: u32,
    pub(crate) pre_release: u32,
    pub(crate) build_metadata: u64,
}

impl Version {
    const MAJOR_MAX: u32 = (1 << 8) - 1;
    const MINOR_MAX: u32 = (1 << 8) - 1;
    const PATCH_MAX: u32 = (1 << 10) - 1;
    const PRE_RELEASE_MAX: u32 = (1 << 6) - 1;
    const BUILD_METADATA_MAX: u64 = (1u64 << 32) - 1;

    pub const fn new(
        major: u32,
        minor: u32,
        patch: u32,
        pre_release: u32,
        build_metadata: u64,
    ) -> Self {
        assert!(major <= Self::MAJOR_MAX, "major must fit in 8 bits");
        assert!(minor <= Self::MINOR_MAX, "minor must fit in 8 bits");
        assert!(patch <= Self::PATCH_MAX, "patch must fit in 10 bits");
        assert!(
            pre_release <= Self::PRE_RELEASE_MAX,
            "pre_release must fit in 6 bits"
        );
        assert!(
            build_metadata <= Self::BUILD_METADATA_MAX,
            "build_metadata must fit in 32 bits"
        );
        Self {
            major,
            minor,
            patch,
            pre_release,
            build_metadata,
        }
    }

    pub const fn encode(&self) -> u64 {
        (self.major as u64)
            | ((self.minor as u64) << 8)
            | ((self.patch as u64) << 16)
            | ((self.pre_release as u64) << 26)
            | ((self.build_metadata) << 32)
    }

    #[allow(dead_code)]
    pub const fn decode(v: u64) -> Self {
        Self {
            major: (v & 0xFF) as u32,
            minor: ((v >> 8) & 0xFF) as u32,
            patch: ((v >> 16) & 0x3FF) as u32,
            pre_release: ((v >> 26) & 0x3F) as u32,
            build_metadata: v >> 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Compile-time parsing from Cargo env vars
// ---------------------------------------------------------------------------

const fn const_atoi(s: &[u8]) -> u32 {
    assert!(!s.is_empty(), "empty version component");
    let mut r = 0u32;
    let mut i = 0;
    while i < s.len() {
        assert!(
            s[i] >= b'0' && s[i] <= b'9',
            "non-digit in version component"
        );
        r = r * 10 + (s[i] - b'0') as u32;
        i += 1;
    }
    r
}

/// Parse pre-release tag into a 6-bit number.
///
/// - `""` -> 0 (stable)
/// - `"3"` -> 3
/// - `"alpha.1"` -> 1 (trailing numeric segment)
/// - `"rc"` -> 1 (non-numeric, just marks pre-release)
const fn parse_pre_release(s: &[u8]) -> u32 {
    if s.is_empty() {
        return 0;
    }
    let mut i = s.len();
    while i > 0 && s[i - 1] >= b'0' && s[i - 1] <= b'9' {
        i -= 1;
    }
    if i == s.len() {
        return 1;
    }
    let mut r = 0u32;
    while i < s.len() {
        r = r * 10 + (s[i] - b'0') as u32;
        i += 1;
    }
    assert!(
        r <= Version::PRE_RELEASE_MAX,
        "pre-release must fit in 6 bits"
    );
    r
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn version_strategy() -> impl Strategy<Value = Version> {
        (
            0..=Version::MAJOR_MAX,
            0..=Version::MINOR_MAX,
            0..=Version::PATCH_MAX,
            0..=Version::PRE_RELEASE_MAX,
            0..=Version::BUILD_METADATA_MAX,
        )
            .prop_map(|(major, minor, patch, pre_release, build_metadata)| {
                Version::new(major, minor, patch, pre_release, build_metadata)
            })
    }

    proptest! {
        #[test]
        fn roundtrip(v in version_strategy()) {
            prop_assert_eq!(Version::decode(v.encode()), v);
        }

        #[test]
        fn fields_are_isolated(a in version_strategy(), b in version_strategy()) {
            // Changing one field must not affect any other field after decode.
            let mutated = Version::new(a.major, b.minor, a.patch, a.pre_release, a.build_metadata);
            let decoded = Version::decode(mutated.encode());
            prop_assert_eq!(decoded.major, a.major);
            prop_assert_eq!(decoded.minor, b.minor);
            prop_assert_eq!(decoded.patch, a.patch);
            prop_assert_eq!(decoded.pre_release, a.pre_release);
            prop_assert_eq!(decoded.build_metadata, a.build_metadata);
        }
    }

    #[test]
    fn bit_layout_spot_check() {
        let v = Version::new(0xAB, 0xCD, 0x3FF, 0x3F, 0xDEAD_BEEF);
        let bits = v.encode();
        assert_eq!(bits & 0xFF, 0xAB);
        assert_eq!((bits >> 8) & 0xFF, 0xCD);
        assert_eq!((bits >> 16) & 0x3FF, 0x3FF);
        assert_eq!((bits >> 26) & 0x3F, 0x3F);
        assert_eq!(bits >> 32, 0xDEAD_BEEF);
    }

    #[test]
    fn cargo_version_matches_env() {
        let v = VERSION;
        assert_eq!(v.major.to_string(), env!("CARGO_PKG_VERSION_MAJOR"));
        assert_eq!(v.minor.to_string(), env!("CARGO_PKG_VERSION_MINOR"));
        assert_eq!(v.patch.to_string(), env!("CARGO_PKG_VERSION_PATCH"));
    }

    #[test]
    fn const_atoi_parses_digits() {
        assert_eq!(const_atoi(b"0"), 0);
        assert_eq!(const_atoi(b"1204"), 1204);
    }

    #[test]
    #[should_panic(expected = "empty version component")]
    fn const_atoi_rejects_empty() {
        const_atoi(b"");
    }

    #[test]
    #[should_panic(expected = "non-digit in version component")]
    fn const_atoi_rejects_non_digit() {
        const_atoi(b"1a");
    }

    #[test]
    fn parse_pre_release_reads_the_trailing_number() {
        assert_eq!(parse_pre_release(b""), 0);
        assert_eq!(parse_pre_release(b"3"), 3);
        assert_eq!(parse_pre_release(b"alpha.1"), 1);
        assert_eq!(parse_pre_release(b"rc"), 1);
    }

    #[test]
    #[should_panic(expected = "pre-release must fit in 6 bits")]
    fn parse_pre_release_rejects_overflow() {
        parse_pre_release(b"alpha.64");
    }
}

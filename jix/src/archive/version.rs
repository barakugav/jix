use crate::archive::schema;

const MAJOR: u32 = const_atoi(env!("CARGO_PKG_VERSION_MAJOR").as_bytes());
const MINOR: u32 = const_atoi(env!("CARGO_PKG_VERSION_MINOR").as_bytes());
const PATCH: u32 = const_atoi(env!("CARGO_PKG_VERSION_PATCH").as_bytes());

pub(crate) fn lib_version() -> schema::Semver {
    schema::Semver {
        major: MAJOR,
        minor: MINOR,
        patch: PATCH,
        pre_release: env!("CARGO_PKG_VERSION_PRE").to_owned(),
        build_metadata: String::new(),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lib_version_matches_cargo() {
        let v = lib_version();
        let mut s = format!("{}.{}.{}", v.major, v.minor, v.patch);
        if !v.pre_release.is_empty() {
            s += &format!("-{}", v.pre_release);
        }
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
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
}

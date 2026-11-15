use core::fmt;

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

impl fmt::Display for schema::Semver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre_release.is_empty() {
            write!(f, "-{}", self.pre_release)?;
        }
        if !self.build_metadata.is_empty() {
            write!(f, "+{}", self.build_metadata)?;
        }
        Ok(())
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
        assert_eq!(lib_version().to_string(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn semver_display() {
        let v = schema::Semver {
            major: 1,
            minor: 2,
            patch: 3,
            pre_release: "rc.1".to_owned(),
            build_metadata: "1a2b3c4d".to_owned(),
        };
        assert_eq!(v.to_string(), "1.2.3-rc.1+1a2b3c4d");
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

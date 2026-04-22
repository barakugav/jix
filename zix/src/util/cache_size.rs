//! CPU cache size detection.
//!
//! Returns the data L1 and unified L2/L3 cache sizes visible to this thread.
//! Detection runs once on first call and the result is cached for the rest
//! of the process lifetime via a `OnceLock`. When detection fails — either
//! because the OS is unsupported or because a particular level can't be
//! read — conservative defaults are used (32 KiB / 256 KiB / 1 MiB, matching
//! blosc2).
//!
//! Supported platforms:
//! - Linux: reads `/sys/devices/system/cpu/cpu0/cache/index*/` (all arches).
//! - macOS: calls `sysctlbyname`, preferring the Apple Silicon performance
//!   cluster keys (`hw.perflevel0.l*cachesize`) and falling back to the
//!   legacy Intel Mac keys (`hw.l*cachesize`).
//! - Anything else: the defaults, unmodified.
//!
//! # Cargo.toml
//! ```toml
//! [target.'cfg(target_os = "macos")'.dependencies]
//! libc = "0.2"
//! ```
//!
//! # Caveats
//! - On big.LITTLE / heterogeneous ARM systems, `cpu0` may be a small core
//!   with smaller caches than the big cores. The sysfs layout exposes per-cpu
//!   caches, but we only probe `cpu0` to stay simple.
//! - The reported L3 is the raw size the OS reports, *not* divided by the
//!   number of threads or CCXes that share it. Callers doing per-thread
//!   cache blocking on L3 may want to divide by `shared_cpu_list`'s cardinality
//!   (Linux) or the number of performance cores (macOS).

use std::sync::OnceLock;

/// Sizes of the caches visible to one thread, in bytes.
///
/// The values are monotonic: `l1_data <= l2 <= l3`. If a level is absent
/// from the hardware (e.g. no L3 on Apple Silicon M1/M2/M3) or detection
/// fails for that level, its field equals the next level down rather than
/// an arbitrary default. This way callers doing cache-blocked kernels can
/// safely use `l3` as "the largest working set that fits in on-chip cache"
/// without special-casing missing levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSizes {
    /// L1 data cache.
    pub l1_data: usize,
    /// L2 cache (unified on all mainstream CPUs).
    pub l2: usize,
    /// L3 cache, or L2 size if there is no L3.
    pub l3: usize,
}

impl CacheSizes {
    /// Conservative defaults used when detection fails, per-level. These
    /// match the defaults in blosc2's `get_cpu_info`.
    pub const DEFAULT: Self = Self {
        l1_data: 32 * 1024,
        l2: 256 * 1024,
        l3: 1024 * 1024,
    };
}

/// Returns the cache sizes for the current CPU, detecting them on first call.
pub fn cache_sizes() -> &'static CacheSizes {
    static CACHE: OnceLock<CacheSizes> = OnceLock::new();
    CACHE.get_or_init(detect)
}

fn detect() -> CacheSizes {
    let mut sizes = CacheSizes::DEFAULT;

    #[cfg(target_os = "linux")]
    linux::fill(&mut sizes);

    #[cfg(target_os = "macos")]
    macos::fill(&mut sizes);

    // Enforce l1_data <= l2 <= l3. Handles (a) Apple Silicon, which has no
    // L3, so the L3 default of 1 MiB would otherwise be smaller than a
    // correctly-detected L2 of 12-16 MiB, and (b) partial-detection cases
    // where some levels succeed and others fall back to defaults.
    sizes.l2 = sizes.l2.max(sizes.l1_data);
    sizes.l3 = sizes.l3.max(sizes.l2);

    sizes
}

// -------------------------------------------------------------------------
// Linux: sysfs
// -------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod linux {
    use super::CacheSizes;
    use std::fs;
    use std::path::Path;

    /// Fill in whatever levels we find from sysfs; leave others at their
    /// default. The canonical layout is:
    ///
    /// ```text
    /// /sys/devices/system/cpu/cpu0/cache/index{N}/
    ///     level           -> "1", "2", "3"
    ///     type            -> "Data", "Instruction", "Unified"
    ///     size            -> "32K", "8192K", "16M"
    /// ```
    ///
    /// We don't hardcode the index-to-level mapping: we read `level` and
    /// `type` and dispatch. This is more robust than blosc2's approach on
    /// systems where the kernel enumerates caches in a different order.
    pub(super) fn fill(out: &mut CacheSizes) {
        let Ok(entries) = fs::read_dir("/sys/devices/system/cpu/cpu0/cache") else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !name.starts_with("index") {
                continue;
            }
            let Some(size) = read_trimmed(&path.join("size")).and_then(|s| parse_size(&s)) else {
                continue;
            };
            let Some(level) = read_trimmed(&path.join("level")).and_then(|s| s.parse::<u8>().ok())
            else {
                continue;
            };
            let ty = read_trimmed(&path.join("type")).unwrap_or_default();

            // L1: only the data (or unified, on architectures without a split)
            // cache is interesting. L2/L3 are virtually always unified.
            match (level, ty.as_str()) {
                (1, "Data") | (1, "Unified") => out.l1_data = size,
                (2, _) => out.l2 = size,
                (3, _) => out.l3 = size,
                _ => {}
            }
        }
    }

    fn read_trimmed(path: &Path) -> Option<String> {
        fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
    }

    /// Parse sysfs cache size strings: `"32K"`, `"8192K"`, `"16M"`, `"1G"`,
    /// or a bare decimal (taken as bytes). Binary multipliers, as the kernel
    /// uses throughout sysfs.
    fn parse_size(s: &str) -> Option<usize> {
        let s = s.trim();
        let (digits, mult) = match s.as_bytes().last()? {
            b'K' | b'k' => (&s[..s.len() - 1], 1usize << 10),
            b'M' | b'm' => (&s[..s.len() - 1], 1usize << 20),
            b'G' | b'g' => (&s[..s.len() - 1], 1usize << 30),
            b'0'..=b'9' => (s, 1),
            _ => return None,
        };
        digits.parse::<usize>().ok()?.checked_mul(mult)
    }

    #[cfg(test)]
    mod tests {
        use super::parse_size;

        #[test]
        fn parses_common_sysfs_sizes() {
            assert_eq!(parse_size("32K"), Some(32 * 1024));
            assert_eq!(parse_size("8192K"), Some(8192 * 1024));
            assert_eq!(parse_size("16M"), Some(16 * 1024 * 1024));
            assert_eq!(parse_size("1G"), Some(1 << 30));
            assert_eq!(parse_size("4096"), Some(4096));
            assert_eq!(parse_size("  32K  "), Some(32 * 1024));
            assert_eq!(parse_size(""), None);
            assert_eq!(parse_size("abc"), None);
        }
    }
}

// -------------------------------------------------------------------------
// macOS: sysctlbyname
// -------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod macos {
    use super::CacheSizes;
    use std::ffi::c_void;
    use std::ffi::CStr;
    use std::mem;
    use std::ptr;

    pub(super) fn fill(out: &mut CacheSizes) {
        // Prefer the performance cluster on Apple Silicon; fall back to the
        // legacy unprefixed keys for Intel Macs (where `perflevel0` doesn't
        // exist and the query fails).
        if let Some(v) = query(c"hw.perflevel0.l1dcachesize").or_else(|| query(c"hw.l1dcachesize"))
        {
            out.l1_data = v;
        }
        if let Some(v) = query(c"hw.perflevel0.l2cachesize").or_else(|| query(c"hw.l2cachesize")) {
            out.l2 = v;
        }
        if let Some(v) = query(c"hw.perflevel0.l3cachesize").or_else(|| query(c"hw.l3cachesize")) {
            out.l3 = v;
        }
    }

    /// Query a `sysctlbyname` key that returns an integer cache size.
    ///
    /// On macOS these keys return `int64_t`. We read into a `u64` initialized
    /// to zero, so if sysctl writes only 4 bytes (older variants) or 8 bytes,
    /// the low half holds the value and the high half stays zero — either way
    /// we get the right number for any plausible cache size.
    fn query(name: &CStr) -> Option<usize> {
        let mut value: u64 = 0;
        let mut len: libc::size_t = mem::size_of::<u64>();

        // SAFETY: `name` is a nul-terminated C string.
        // `value` and `len` point to valid, initialized memory whose sizes
        // match what we pass to sysctl. `newp` is NULL with `newlen == 0`,
        // so no write path is taken.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr() as *const libc::c_char,
                &mut value as *mut u64 as *mut c_void,
                &mut len,
                ptr::null_mut(),
                0,
            )
        };

        if rc == 0 && value != 0 {
            Some(value as usize)
        } else {
            None
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_plausible_sizes() {
        let s = cache_sizes();
        // Any real or default cache must be at least the defaults and hierarchical.
        assert!(s.l1_data >= 1024, "l1_data = {}", s.l1_data);
        assert!(s.l2 >= s.l1_data, "l2 {} < l1_data {}", s.l2, s.l1_data);
        assert!(s.l3 >= s.l2, "l3 {} < l2 {}", s.l3, s.l2);
    }

    #[test]
    fn is_cached() {
        // Same pointer on repeat calls (OnceLock behavior).
        let a = cache_sizes() as *const _;
        let b = cache_sizes() as *const _;
        assert_eq!(a, b);
    }
}

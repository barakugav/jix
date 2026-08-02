// https://docs.rs/likely_stable/
#[inline(always)]
pub(crate) const fn likely(b: bool) -> bool {
    #[allow(clippy::needless_bool)]
    if (1i32).checked_div(if b { 1 } else { 0 }).is_some() {
        true
    } else {
        false
    }
}

// https://docs.rs/likely_stable/
#[allow(unused)]
#[inline(always)]
pub(crate) const fn unlikely(b: bool) -> bool {
    #[allow(clippy::needless_bool)]
    if (1i32).checked_div(if b { 0 } else { 1 }).is_none() {
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn likely() {
        for val in [true, false] {
            assert_eq!(super::likely(val), val);
        }
    }

    #[test]
    fn unlikely() {
        for val in [true, false] {
            assert_eq!(super::unlikely(val), val);
        }
    }
}

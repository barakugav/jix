pub(crate) fn test_rng(file: &str, test_name: &str) -> fastrand::Rng {
    let seed = [file, test_name]
        .join(" ")
        .as_bytes()
        .iter()
        .fold(1337, |acc: u64, b| {
            acc.wrapping_mul(31).wrapping_add(*b as u64)
        });
    fastrand::Rng::with_seed(seed)
}

pub(crate) fn test_lengths(rand: &mut fastrand::Rng) -> Vec<usize> {
    let mut lengths = Vec::new();
    lengths.extend(0..=10); // always test small lengths
    for _ in 0..20 {
        lengths.push(rand.usize(..1000));
    }
    lengths
}

use crate::dtype::Dtyped;
use crate::util::AlignedBytes;

/// Trait for types that can be randomly sampled for use in tests.
pub(crate) trait Sampleable: Dtyped {
    fn sample(rng: &mut fastrand::Rng) -> Self;
}

impl Sampleable for u8 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.u8(..)
    }
}
impl Sampleable for u16 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.u16(..)
    }
}
impl Sampleable for u32 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.u32(..)
    }
}
impl Sampleable for u64 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.u64(..)
    }
}
impl Sampleable for i8 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.i8(..)
    }
}
impl Sampleable for i16 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.i16(..)
    }
}
impl Sampleable for i32 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.i32(..)
    }
}
impl Sampleable for i64 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.i64(..)
    }
}
impl Sampleable for f32 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.f32()
    }
}
impl Sampleable for f64 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.f64()
    }
}
impl Sampleable for bool {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        rng.bool()
    }
}

#[cfg(feature = "half")]
impl Sampleable for crate::dtype::f16 {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        crate::dtype::f16::from_f32(rng.f32())
    }
}

impl Sampleable for crate::dtype::Complex<f32> {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        crate::dtype::Complex {
            re: rng.f32(),
            im: rng.f32(),
        }
    }
}

impl Sampleable for crate::dtype::Complex<f64> {
    fn sample(rng: &mut fastrand::Rng) -> Self {
        crate::dtype::Complex {
            re: rng.f64(),
            im: rng.f64(),
        }
    }
}

pub(crate) fn gen_data<T: Sampleable>(len: usize, rand: &mut fastrand::Rng) -> Vec<T> {
    (0..len).map(|_| T::sample(rand)).collect()
}

pub(crate) fn gen_data_bytes<T: Sampleable>(len: usize, rand: &mut fastrand::Rng) -> AlignedBytes {
    let items = gen_data(len, rand);
    let bytes = unsafe { crate::util::cast_slice::<T, u8>(&items) };
    AlignedBytes::from_slice(T::DTYPE.alignment() as usize, bytes)
}

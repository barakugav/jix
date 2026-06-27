#![allow(dead_code)]

use fastrand::Rng;
use jix::storage::Compact;
use jix::{Array, ArrayParams, DimDyn, Ty};

pub fn create_data<T: Scalar>(profile: Profile, shape: &[u64], seed: u64) -> ndarray::ArrayD<T> {
    let dims = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let n = dims.iter().product::<usize>();
    let mut rng = Rng::with_seed(seed);
    let flat = T::random(&profile, &mut rng).take(n).collect();
    ndarray::ArrayD::from_shape_vec(dims, flat).unwrap()
}

pub fn create_ndarray(shape: &[u64], rng: &mut Rng) -> ndarray::ArrayD<i32> {
    let dims = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
    ndarray::ArrayD::from_shape_simple_fn(dims.as_slice(), || rng.i32(..))
}

pub fn create_compact<'a>(
    shape: &[u64],
    block_shape: impl Into<Option<&'a [u32]>>,
    read_size: impl Into<Option<(u64, u64)>>,
    rng: &mut Rng,
) -> Array<Compact<Ty<i32>, DimDyn>> {
    let data = create_ndarray(shape, rng);
    let mut params = ArrayParams::new();
    if let Some(block_shape) = block_shape.into() {
        params.block_shape(block_shape);
    }
    if let Some(read_size) = read_size.into() {
        params.read_size(read_size);
    }
    Array::compact_ndarray_with(&data, params).unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Uniform random over all values; incompressible.
    Random,
    /// Structured field (sine + gradient); moderately compressible.
    Smooth,
    /// Few distinct small values (~8 unique values); highly compressible.
    LowEntropy,
}

impl Profile {
    pub const ALL: [Profile; 3] = [Profile::Random, Profile::Smooth, Profile::LowEntropy];

    pub fn name(self) -> &'static str {
        match self {
            Profile::Random => "random",
            Profile::Smooth => "smooth",
            Profile::LowEntropy => "low_entropy",
        }
    }
}

pub trait Scalar {
    fn random<'a>(profile: &Profile, rng: &'a mut Rng) -> impl Iterator<Item = Self> + 'a
    where
        Self: Sized;
}
impl Scalar for i32 {
    fn random<'a>(profile: &Profile, rng: &'a mut Rng) -> impl Iterator<Item = Self> + 'a {
        let iter: Box<dyn Iterator<Item = Self>> = match profile {
            Profile::Random => Box::new(std::iter::repeat_with(|| rng.i32(..))),
            Profile::Smooth => Box::new(f64::random(profile, rng).map(move |x| x.round() as i32)),
            Profile::LowEntropy => Box::new(std::iter::repeat_with(|| rng.i32(0..8))),
        };
        iter
    }
}
impl Scalar for f32 {
    fn random<'a>(profile: &Profile, rng: &'a mut Rng) -> impl Iterator<Item = Self> + 'a {
        f64::random(profile, rng).map(|x| x as f32)
    }
}
impl Scalar for f64 {
    fn random<'a>(profile: &Profile, rng: &'a mut Rng) -> impl Iterator<Item = Self> + 'a {
        let iter: Box<dyn Iterator<Item = Self>> = match profile {
            Profile::Random => Box::new(std::iter::repeat_with(|| rng.f64())),
            Profile::Smooth => {
                let phase = rng.f64() * 2.0 * std::f64::consts::PI;
                Box::new((0u64..).map(move |i| {
                    let t = 8.0 * std::f64::consts::PI * (i as f64) / 10000.0;
                    let grad = 0.25 * (i as f64) / 10000.0;
                    ((t + phase).sin() + grad) * 1000.0
                }))
            }
            Profile::LowEntropy => Box::new(std::iter::repeat_with(|| rng.i32(0..8) as f64)),
        };
        iter
    }
}

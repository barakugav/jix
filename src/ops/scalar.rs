use crate::array::{Array, ArrayStorage, BlocksLayout};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, f16};
use crate::util::{DimArray, cast_slice, cast_slice_mut};

pub(crate) trait ScalarOp2 {
    const SUPPORT_F16: bool = false;
    const SUPPORT_COMPLEX: bool = false;
    const SUPPORT_BOOL: bool = false;
    fn apply_i8(a: i8, b: i8) -> i8;
    fn apply_i16(a: i16, b: i16) -> i16;
    fn apply_i32(a: i32, b: i32) -> i32;
    fn apply_i64(a: i64, b: i64) -> i64;
    fn apply_u8(a: u8, b: u8) -> u8;
    fn apply_u16(a: u16, b: u16) -> u16;
    fn apply_u32(a: u32, b: u32) -> u32;
    fn apply_u64(a: u64, b: u64) -> u64;
    fn apply_f16(a: f16, b: f16) -> f16;
    fn apply_f32(a: f32, b: f32) -> f32;
    fn apply_f64(a: f64, b: f64) -> f64;
    fn apply_complex_f32(a: Complex<f32>, b: Complex<f32>) -> Complex<f32>;
    fn apply_complex_f64(a: Complex<f64>, b: Complex<f64>) -> Complex<f64>;
    fn apply_bool(a: bool, b: bool) -> bool;

    type S1;
    type S2;
    fn base(&self) -> &ScalarOp2Base<Self::S1, Self::S2>;
}

pub(crate) struct ScalarOp2Base<S1, S2> {
    pub(crate) a: Array<S1>,
    pub(crate) b: Array<S2>,

    pub(crate) dtype: Dtype,
    pub(crate) shape: DimArray<usize>,
    pub(crate) blocks_layout: BlocksLayout,
}
impl<S1, S2> ScalarOp2Base<S1, S2> {
    fn read_data<Op>(
        &self,
        index: &[std::ops::Range<usize>],
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> std::io::Result<()>
    where
        Op: ScalarOp2<S1 = S1, S2 = S2>,
        Op::S1: ArrayStorage,
        Op::S2: ArrayStorage,
    {
        let mut buf2 = vec![0u8; buf.len()];
        self.a.storage.read_data(index, buf, context)?;
        self.b.storage.read_data(index, &mut buf2, context)?;

        macro_rules! impl_add {
            ($ty:ty, $apply:ident) => {
                let buf1 = unsafe { cast_slice_mut::<u8, $ty>(buf) };
                let buf2 = unsafe { cast_slice::<u8, $ty>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a = Op::$apply(*a, *b);
                }
            };
        }

        Ok(match self.dtype.try_to_scalar() {
            Some(DtypeScalarKind::I8) => {
                impl_add!(i8, apply_i8);
            }
            Some(DtypeScalarKind::I16) => {
                impl_add!(i16, apply_i16);
            }
            Some(DtypeScalarKind::I32) => {
                impl_add!(i32, apply_i32);
            }
            Some(DtypeScalarKind::I64) => {
                impl_add!(i64, apply_i64);
            }
            Some(DtypeScalarKind::U8) => {
                impl_add!(u8, apply_u8);
            }
            Some(DtypeScalarKind::U16) => {
                impl_add!(u16, apply_u16);
            }
            Some(DtypeScalarKind::U32) => {
                impl_add!(u32, apply_u32);
            }
            Some(DtypeScalarKind::U64) => {
                impl_add!(u64, apply_u64);
            }
            Some(DtypeScalarKind::F16) => {
                cfg_if::cfg_if! {  if #[cfg(feature = "half")] {
                    impl_add!(f16, apply_f16);
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "f16 support requires the `half` feature",
                    ));
                } }
            }
            Some(DtypeScalarKind::F32) => {
                impl_add!(f32, apply_f32);
            }
            Some(DtypeScalarKind::F64) => {
                impl_add!(f64, apply_f64);
            }
            Some(DtypeScalarKind::ComplexF32) => {
                cfg_if::cfg_if! {  if #[cfg(feature = "num-complex")] {
                    impl_add!(Complex<f32>, apply_complex_f32);
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "complex f32 support requires the `num-complex` feature",
                    ));
                } }
            }
            Some(DtypeScalarKind::ComplexF64) => {
                cfg_if::cfg_if! {  if #[cfg(feature = "num-complex")] {
                    impl_add!(Complex<f64>, apply_complex_f64);
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "complex f64 support requires the `num-complex` feature",
                    ));
                } }
            }
            Some(DtypeScalarKind::Bool) | None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "unsupported dtype for addition",
                ));
            }
        })
    }
}

impl<Op> ArrayStorage for Op
where
    Op: ScalarOp2,
    Op::S1: ArrayStorage,
    Op::S2: ArrayStorage,
{
    fn dtype(&self) -> &Dtype {
        &self.base().dtype
    }

    fn shape(&self) -> &[usize] {
        &self.base().shape
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.base().blocks_layout
    }

    fn read_data(
        &self,
        index: &[std::ops::Range<usize>],
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> std::io::Result<()> {
        self.base().read_data::<Self>(index, buf, context)
    }
}

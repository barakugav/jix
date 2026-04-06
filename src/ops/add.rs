use crate::array::{Array, ArrayStorage, BlocksLayout};
use crate::dtype::{Dtype, DtypeScalarKind};
use crate::util::{DimArray, cast_slice, cast_slice_mut};

pub struct Add<S1, S2> {
    a: Array<S1>,
    b: Array<S2>,

    dtype: Dtype,
    shape: DimArray<usize>,
    blocks_layout: BlocksLayout,
}
impl<S1, S2> ArrayStorage for Add<S1, S2>
where
    S1: ArrayStorage,
    S2: ArrayStorage,
{
    fn dtype(&self) -> &crate::dtype::Dtype {
        &self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }

    fn read_block(
        &self,
        block_idx: usize,
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> std::io::Result<()> {
        let mut buf2 = vec![0u8; buf.len()];
        self.a.storage.read_block(block_idx, buf, context)?;
        self.b.storage.read_block(block_idx, &mut buf2, context)?;
        Ok(match self.a.dtype().try_to_scalar() {
            Some(DtypeScalarKind::I8) => {
                let buf1 = unsafe { cast_slice_mut::<u8, i8>(buf) };
                let buf2 = unsafe { cast_slice::<u8, i8>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::I16) => {
                let buf1 = unsafe { cast_slice_mut::<u8, i16>(buf) };
                let buf2 = unsafe { cast_slice::<u8, i16>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::I32) => {
                let buf1 = unsafe { cast_slice_mut::<u8, i32>(buf) };
                let buf2 = unsafe { cast_slice::<u8, i32>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::I64) => {
                let buf1 = unsafe { cast_slice_mut::<u8, i64>(buf) };
                let buf2 = unsafe { cast_slice::<u8, i64>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::U8) => {
                let buf1 = unsafe { cast_slice_mut::<u8, u8>(buf) };
                let buf2 = unsafe { cast_slice::<u8, u8>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::U16) => {
                let buf1 = unsafe { cast_slice_mut::<u8, u16>(buf) };
                let buf2 = unsafe { cast_slice::<u8, u16>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::U32) => {
                let buf1 = unsafe { cast_slice_mut::<u8, u32>(buf) };
                let buf2 = unsafe { cast_slice::<u8, u32>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::U64) => {
                let buf1 = unsafe { cast_slice_mut::<u8, u64>(buf) };
                let buf2 = unsafe { cast_slice::<u8, u64>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::F16) => {
                cfg_if::cfg_if! {  if #[cfg(feature = "half")] {
                    let buf1 = unsafe { cast_slice_mut::<u8, half::f16>(buf) };
                    let buf2 = unsafe { cast_slice::<u8, half::f16>(&buf2) };
                    for (a, b) in buf1.iter_mut().zip(buf2) {
                        *a += *b;
                    }
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "f16 support requires the `half` feature",
                    ));
                } }
            }
            Some(DtypeScalarKind::F32) => {
                let buf1 = unsafe { cast_slice_mut::<u8, f32>(buf) };
                let buf2 = unsafe { cast_slice::<u8, f32>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::F64) => {
                let buf1 = unsafe { cast_slice_mut::<u8, f64>(buf) };
                let buf2 = unsafe { cast_slice::<u8, f64>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a += *b;
                }
            }
            Some(DtypeScalarKind::ComplexF32) => {
                cfg_if::cfg_if! {  if #[cfg(feature = "num-complex")] {
                    let buf1 = unsafe { cast_slice_mut::<u8, num_complex::Complex<f32>>(buf) };
                    let buf2 = unsafe { cast_slice::<u8, num_complex::Complex<f32>>(&buf2) };
                    for (a, b) in buf1.iter_mut().zip(buf2) {
                        *a += *b;
                    }
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "complex f32 support requires the `num-complex` feature",
                    ));
                } }
            }
            Some(DtypeScalarKind::ComplexF64) => {
                cfg_if::cfg_if! {  if #[cfg(feature = "num-complex")] {
                    let buf1 = unsafe { cast_slice_mut::<u8, num_complex::Complex<f64>>(buf) };
                    let buf2 = unsafe { cast_slice::<u8, num_complex::Complex<f64>>(&buf2) };
                    for (a, b) in buf1.iter_mut().zip(buf2) {
                        *a += *b;
                    }
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
impl<'a, 'b, S1, S2> core::ops::Add<&'b Array<S2>> for &'a Array<S1>
where
    S1: ArrayStorage,
    S2: ArrayStorage,
{
    type Output = Array<Add<&'a S1, &'b S2>>;
    fn add(self, b: &'b Array<S2>) -> Array<Add<&'a S1, &'b S2>> {
        // TODO: check shapes, dtype
        Array {
            storage: Add {
                a: Array {
                    storage: &self.storage,
                },
                b: Array {
                    storage: &b.storage,
                },

                dtype: self.dtype().clone(),
                shape: self.shape().try_into().unwrap(),
                blocks_layout: self.blocks_layout().clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;

    #[test]
    fn add_1d_i32() {
        let a = ndarray::array![1i32, 2, 3, 4].into_dyn();
        let b = ndarray::array![10i32, 20, 30, 40].into_dyn();
        let za = Array::from_ndarray(&a, &[4]).unwrap();
        let zb = Array::from_ndarray(&b, &[4]).unwrap();
        let got = (&za + &zb).data().to_ndarray::<i32>().unwrap();
        assert_eq!(got, &a + &b);
    }

    #[test]
    fn add_1d_i32_multi_block() {
        let a = ndarray::array![1i32, 2, 3, 4, 5, 6].into_dyn();
        let b = ndarray::array![10i32, 20, 30, 40, 50, 60].into_dyn();
        let za = Array::from_ndarray(&a, &[2]).unwrap();
        let zb = Array::from_ndarray(&b, &[2]).unwrap();
        let got = (&za + &zb).data().to_ndarray::<i32>().unwrap();
        assert_eq!(got, &a + &b);
    }

    #[test]
    fn add_1d_f64() {
        let a = ndarray::array![1.0f64, 2.5, 3.0].into_dyn();
        let b = ndarray::array![0.5f64, 1.5, 2.0].into_dyn();
        let za = Array::from_ndarray(&a, &[3]).unwrap();
        let zb = Array::from_ndarray(&b, &[3]).unwrap();
        let got = (&za + &zb).data().to_ndarray::<f64>().unwrap();
        assert_eq!(got, &a + &b);
    }

    #[test]
    fn add_2d_f32() {
        let a = ndarray::array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let b = ndarray::array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn();
        let za = Array::from_ndarray(&a, &[2, 3]).unwrap();
        let zb = Array::from_ndarray(&b, &[2, 3]).unwrap();
        let got = (&za + &zb).data().to_ndarray::<f32>().unwrap();
        assert_eq!(got, &a + &b);
    }

    #[test]
    fn add_2d_i32_multi_block() {
        #[rustfmt::skip]
        let a = ndarray::array![
            [1i32,  2,  3,  4],
            [5,     6,  7,  8],
            [9,    10, 11, 12],
            [13,   14, 15, 16],
        ].into_dyn();
        #[rustfmt::skip]
        let b = ndarray::array![
            [100i32, 200, 300, 400],
            [500,    600, 700, 800],
            [900,   1000, 1100, 1200],
            [1300,  1400, 1500, 1600],
        ].into_dyn();
        let za = Array::from_ndarray(&a, &[2, 2]).unwrap();
        let zb = Array::from_ndarray(&b, &[2, 2]).unwrap();
        let got = (&za + &zb).data().to_ndarray::<i32>().unwrap();
        assert_eq!(got, &a + &b);
    }

    #[test]
    fn add_three_arrays_1d_i32() {
        let a = ndarray::array![1i32, 2, 3, 4].into_dyn();
        let b = ndarray::array![10i32, 20, 30, 40].into_dyn();
        let c = ndarray::array![100i32, 200, 300, 400].into_dyn();
        let za = Array::from_ndarray(&a, &[4]).unwrap();
        let zb = Array::from_ndarray(&b, &[4]).unwrap();
        let zc = Array::from_ndarray(&c, &[4]).unwrap();
        let zab = &za + &zb;
        let got = (&zab + &zc).data().to_ndarray::<i32>().unwrap();
        assert_eq!(got, &(&a + &b) + &c);
    }
}

// use crate::array::Array;
// use crate::dtype::{Complex, Dtype, DtypeScalarKind, Dtyped, f16};
// use crate::error::Error;
// use crate::ops::apply::{Op1, apply_op1};

// pub struct MulScalar<S> {
//     storage: S,
//     scalar: f64,
// }
// impl<S: ArrayStorage> ArrayStorage for MulScalar<S> {
//     fn dtype(&self) -> &Dtype {
//         self.storage.dtype()
//     }
//     fn shape(&self) -> &[usize] {
//         self.storage.shape()
//     }
//     fn chunks_layout(&self) -> &ChunksLayoutInfo {
//         self.storage.chunks_layout()
//     }
//     fn get_chunk_data(
//         &self,
//         chunk_idx: &[usize],
//         inner_offset: &[usize],
//         buf: &mut ChunkDataBuf,
//     ) -> Result<(), Error> {
//         self.storage.get_chunk_data(chunk_idx, inner_offset, buf)?;
//         match self.dtype() {
//             d if d == &f64::dtype() => {
//                 struct MulOp {
//                     scalar: f64,
//                 }
//                 impl Op1 for MulOp {
//                     fn i8(&mut self, _ptr: *mut i8, _idx: &[usize]) {
//                         unreachable!()
//                     }
//                     fn i16(&mut self, _ptr: *mut i16, _idx: &[usize]) {
//                         unreachable!()
//                     }
//                     fn i32(&mut self, _ptr: *mut i32, _idx: &[usize]) {
//                         unreachable!()
//                     }

//                     fn i64(&mut self, ptr: *mut i64, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn u8(&mut self, ptr: *mut u8, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn u16(&mut self, ptr: *mut u16, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn u32(&mut self, ptr: *mut u32, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn u64(&mut self, ptr: *mut u64, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn f16(&mut self, ptr: *mut f16, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn f32(&mut self, ptr: *mut f32, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn f64(&mut self, ptr: *mut f64, idx: &[usize]) {
//                         unsafe { *ptr *= self.scalar };
//                     }

//                     fn complex_f32(&mut self, ptr: *mut Complex<f32>, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn complex_f64(&mut self, ptr: *mut Complex<f64>, idx: &[usize]) {
//                         todo!()
//                     }

//                     fn bool(&mut self, ptr: *mut bool, idx: &[usize]) {
//                         todo!()
//                     }
//                 }
//                 unsafe {
//                     apply_op1(
//                         buf.buffer.cast(),
//                         buf.shape,
//                         buf.strides,
//                         self.dtype(),
//                         &mut MulOp {
//                             scalar: self.scalar,
//                         },
//                     )
//                 };
//             }
//             _ => unimplemented!("MulScalar is only implemented for f64"),
//         }
//         Ok(())
//     }
// }
// impl<S> core::ops::Mul<f64> for Array<S>
// where
//     S: ArrayStorage,
// {
//     type Output = Array<MulScalar<S>>;
//     fn mul(self, rhs: f64) -> Self::Output {
//         Array::new(MulScalar {
//             storage: self.storage,
//             scalar: rhs,
//         })
//     }
// }

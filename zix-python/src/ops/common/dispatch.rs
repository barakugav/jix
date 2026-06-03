use std::borrow::Cow;

use pyo3::prelude::*;
use zix_core::dtype::{Dtype, DtypeScalarKind, Dtyped};
use zix_core::ops::ToType;
use zix_core::storage::ArrayStorageAny;
use zix_core::{Array as ZixArray, ArrayAny, Ty};

use crate::ops::astype_impl;
use crate::ops::common::{CastKind, Operand, Precision, Scalar};
use crate::util::{IntoPyResult, ItemOrSequence, IterExt};

type TypedOperand<T> = ZixArray<ToType<ArrayStorageAny, Ty<T>>>;

pub(crate) struct OpDescriptor<const IN_N: usize, ExtraArgs> {
    name: &'static str,
    fns: Vec<OpFnDescriptor<IN_N, ExtraArgs>>,
}
pub(crate) struct OpFnDescriptor<const IN_N: usize, ExtraArgs> {
    f: Box<dyn Fn([ArrayAny; IN_N], ExtraArgs) -> PyResult<ArrayAny> + Send + Sync>,
    input_desc: [OpFnInputDescriptor; IN_N],
}
pub(crate) struct OpFnInputDescriptor {
    dtype: DtypeScalarKind,
    allowed_cast: CastKind,
}

impl<const IN_N: usize, ExtraArgs> OpDescriptor<IN_N, ExtraArgs> {
    pub(crate) fn new(name: &'static str, fns: Vec<OpFnDescriptor<IN_N, ExtraArgs>>) -> Self {
        Self { name, fns }
    }
}

// impl<const IN_N: usize> OpFnDescriptor<IN_N, ()> {
//     fn new(
//         input_desc: [OpFnInputDescriptor; IN_N],
//         f: impl Fn([ArrayAny; IN_N]) -> PyResult<ArrayAny> + Send + Sync + 'static,
//     ) -> Self {
//         Self::new_args(input_desc, move |inputs, _| f(inputs))
//     }
// }
impl<const IN_N: usize, ExtraArgs> OpFnDescriptor<IN_N, ExtraArgs> {
    fn new_args(
        input_desc: [OpFnInputDescriptor; IN_N],
        f: impl Fn([ArrayAny; IN_N], ExtraArgs) -> PyResult<ArrayAny> + Send + Sync + 'static,
    ) -> Self {
        Self {
            input_desc,
            f: Box::new(f),
        }
    }
}
impl<ExtraArgs> OpFnDescriptor<1, ExtraArgs> {
    pub(crate) fn new1_args<T>(
        allowed_cast: CastKind,
        f: impl Fn(TypedOperand<T>, ExtraArgs) -> PyResult<ArrayAny> + Send + Sync + 'static,
    ) -> Self
    where
        T: Dtyped,
    {
        let input_desc = [OpFnInputDescriptor {
            dtype: T::DTYPE.try_to_scalar().unwrap(),
            allowed_cast,
        }];

        Self::new_args(input_desc, move |inputs, extra_args| {
            let [input] = inputs;
            let input = input.to_typed::<T>().unwrap();
            f(input, extra_args)
        })
    }
}
impl OpFnDescriptor<1, ()> {
    pub(crate) fn new1<T>(
        allowed_cast: CastKind,
        f: impl Fn(TypedOperand<T>) -> PyResult<ArrayAny> + Send + Sync + 'static,
    ) -> Self
    where
        T: Dtyped,
    {
        Self::new1_args(allowed_cast, move |input, ()| f(input))
    }
}
impl OpFnDescriptor<2, ()> {
    pub(crate) fn new2<T1, T2>(
        allowed_cast: impl Into<ItemOrSequence<CastKind>>,
        f: impl Fn(TypedOperand<T1>, TypedOperand<T2>) -> PyResult<ArrayAny> + Send + Sync + 'static,
    ) -> Self
    where
        T1: Dtyped,
        T2: Dtyped,
    {
        let f = move |inputs: [ArrayAny; 2]| {
            let [input1, input2] = inputs;
            let input1 = input1.to_typed::<T1>().unwrap();
            let input2 = input2.to_typed::<T2>().unwrap();
            f(input1, input2)
        };

        let allowed_cast: [CastKind; 2] = match allowed_cast.into() {
            ItemOrSequence::Item(cast) => [cast; 2],
            ItemOrSequence::Sequence(items) => items.try_into().unwrap(),
        };
        let [allowed_cast1, allowed_cast2] = allowed_cast;
        let input_desc = [
            OpFnInputDescriptor {
                dtype: T1::DTYPE.try_to_scalar().unwrap(),
                allowed_cast: allowed_cast1,
            },
            OpFnInputDescriptor {
                dtype: T2::DTYPE.try_to_scalar().unwrap(),
                allowed_cast: allowed_cast2,
            },
        ];

        Self::new_args(input_desc, move |inputs, _| f(inputs))
    }
}

impl<const IN_N: usize, ExtraArgs> OpDescriptor<IN_N, ExtraArgs> {
    #[inline(never)]
    pub(crate) fn dispatch_args(
        &self,
        inputs: [Operand; IN_N],
        extra_args: ExtraArgs,
    ) -> PyResult<ArrayAny> {
        let in_dtypes = inputs.each_ref().map(|input| input.rank_precision());
        let in_dtypes = in_dtypes
            .map(|x| x.ok_or(()))
            .into_iter()
            .try_collect_array::<_, _, IN_N>()
            .transpose()
            .unwrap()
            .ok();

        if let Some(in_dtypes) = in_dtypes {
            for op_fn in &self.fns {
                let dtypes_supported = in_dtypes.iter().zip(op_fn.input_desc.iter()).all(
                    |(input_dtype, input_desc)| {
                        input_desc
                            .allowed_cast
                            .is_cast_allowed(*input_dtype, input_desc.dtype)
                    },
                );
                if !dtypes_supported {
                    continue;
                }

                let inputs: [ArrayAny; IN_N] = inputs
                    .into_iter()
                    .zip(op_fn.input_desc.iter())
                    .map(|(input, input_desc)| {
                        let input = input.into_array()?;
                        astype_impl(input, &Dtype::of_scalar(input_desc.dtype))
                    })
                    .try_collect_array()?
                    .unwrap();

                return (op_fn.f)(inputs, extra_args);
            }
        }

        Err(zix_core::Error::new(
            zix_core::ErrorKind::UnsupportedDtype,
            format!(
                "Op {} does not support operands with dtypes {:?}",
                self.name,
                inputs.each_ref().map(|dtype| match dtype {
                    Operand::Array(array) => Cow::Owned(format!("{}", array.dtype())),
                    Operand::Scalar {
                        value,
                        shape: _,
                        precision,
                    } => Cow::Borrowed(match (value, precision) {
                        (Scalar::Bool(_), _) => "bool",
                        (Scalar::UInt(_), None) => "uint",
                        (Scalar::UInt(_), Some(Precision::P1)) => "u8",
                        (Scalar::UInt(_), Some(Precision::P2)) => "u16",
                        (Scalar::UInt(_), Some(Precision::P4)) => "u32",
                        (Scalar::UInt(_), Some(Precision::P8)) => "u64",
                        (Scalar::Int(_), None) => "int",
                        (Scalar::Int(_), Some(Precision::P1)) => "i8",
                        (Scalar::Int(_), Some(Precision::P2)) => "i16",
                        (Scalar::Int(_), Some(Precision::P4)) => "i32",
                        (Scalar::Int(_), Some(Precision::P8)) => "i64",
                        (Scalar::Float(_), None) => "float",
                        (Scalar::Float(_), Some(Precision::P2)) => "f16",
                        (Scalar::Float(_), Some(Precision::P4)) => "f32",
                        (Scalar::Float(_), Some(Precision::P8)) => "f64",
                        (Scalar::Float(_), Some(_)) => "float<?>",
                        (Scalar::Complex(_), None) => "Complex",
                        (Scalar::Complex(_), Some(Precision::P4)) => "Complex<f32>",
                        (Scalar::Complex(_), Some(Precision::P8)) => "Complex<f64>",
                        (Scalar::Complex(_), Some(_)) => "Complex<?>",
                    }),
                })
            ),
        ))
        .into_py_result()
    }
}
impl<const IN_N: usize> OpDescriptor<IN_N, ()> {
    fn dispatch(&self, inputs: [Operand; IN_N]) -> PyResult<ArrayAny> {
        Self::dispatch_args(self, inputs, ())
    }
}
impl OpDescriptor<1, ()> {
    pub(crate) fn dispatch1(&self, array: Operand) -> PyResult<ArrayAny> {
        Self::dispatch(self, [array])
    }
}
impl OpDescriptor<2, ()> {
    pub(crate) fn dispatch2(&self, a: Operand, b: Operand) -> PyResult<ArrayAny> {
        Self::dispatch(self, [a, b])
    }
}

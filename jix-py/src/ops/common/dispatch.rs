use std::borrow::Cow;

use jix_core::dtype::{Dtype, Dtyped, ScalarKind};
use jix_core::ops::IntoType;
use jix_core::storage::ArrayStorageAny;
use jix_core::{Array as CoreArray, ArrayAny, Ty};
use pyo3::prelude::*;

use crate::ops::common::{CastKind, Operand, Precision, Rank, Scalar};
use crate::util::{array_try_map, ItemOrTuple, IterExt};

type TypedOperand<T> = CoreArray<IntoType<ArrayStorageAny, Ty<T>>>;

/// Runtime dtype-dispatch table for an op with `IN_N` array inputs.
///
/// Holds one [`OpFnDescriptor`] per supported (typed) implementation. At call time
/// [`dispatch_args`](Self::dispatch_args) walks `fns` in order and picks the first whose input
/// dtypes are reachable from the operands' dtypes under each descriptor's [`CastKind`], casting
/// the operands to that descriptor's dtypes before invoking it. Order therefore matters: list
/// exact/narrower dtypes before wider cast targets. Returns `UnsupportedDtype` if none match.
pub(crate) struct OpDescriptor<const IN_N: usize, ExtraArgs> {
    name: &'static str,
    fns: Vec<OpFnDescriptor<IN_N, ExtraArgs>>,
}
pub(crate) struct OpFnDescriptor<const IN_N: usize, ExtraArgs> {
    f: Box<dyn Fn([ArrayAny; IN_N], ExtraArgs) -> PyResult<ArrayAny> + Send + Sync>,
    input_desc: [OpFnInputDescriptor; IN_N],
}
pub(crate) struct OpFnInputDescriptor {
    dtype: ScalarKind,
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
            let input = input.into_typed::<T>().unwrap();
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
        allowed_cast: impl Into<ItemOrTuple<CastKind>>,
        f: impl Fn(TypedOperand<T1>, TypedOperand<T2>) -> PyResult<ArrayAny> + Send + Sync + 'static,
    ) -> Self
    where
        T1: Dtyped,
        T2: Dtyped,
    {
        let f = move |inputs: [ArrayAny; 2]| {
            let [input1, input2] = inputs;
            let input1 = input1.into_typed::<T1>().unwrap();
            let input2 = input2.into_typed::<T2>().unwrap();
            f(input1, input2)
        };

        let allowed_cast: [CastKind; 2] = match allowed_cast.into() {
            ItemOrTuple::Item(cast) => [cast; 2],
            ItemOrTuple::Sequence(items) => items.try_into().unwrap(),
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
impl OpFnDescriptor<3, ()> {
    pub(crate) fn new3<T>(
        allowed_cast: CastKind,
        f: impl Fn(TypedOperand<T>, TypedOperand<T>, TypedOperand<T>) -> PyResult<ArrayAny>
            + Send
            + Sync
            + 'static,
    ) -> Self
    where
        T: Dtyped,
    {
        let input_desc = [(); 3].map(|()| OpFnInputDescriptor {
            dtype: T::DTYPE.try_to_scalar().unwrap(),
            allowed_cast,
        });
        Self::new_args(input_desc, move |inputs, ()| {
            let [a, b, c] = inputs.map(|input| input.into_typed::<T>().unwrap());
            f(a, b, c)
        })
    }
}
impl<ExtraArgs> OpFnDescriptor<2, ExtraArgs> {
    /// Like [`new2`] but passes `extra_args` through to the closure.
    /// Both inputs must have the same dtype `T`.
    pub(crate) fn new2_args<T>(
        allowed_cast: CastKind,
        f: impl Fn(TypedOperand<T>, TypedOperand<T>, ExtraArgs) -> PyResult<ArrayAny>
            + Send
            + Sync
            + 'static,
    ) -> Self
    where
        T: Dtyped,
    {
        let input_desc = [
            OpFnInputDescriptor {
                dtype: T::DTYPE.try_to_scalar().unwrap(),
                allowed_cast,
            },
            OpFnInputDescriptor {
                dtype: T::DTYPE.try_to_scalar().unwrap(),
                allowed_cast,
            },
        ];
        Self::new_args(input_desc, move |inputs, extra_args| {
            let [input1, input2] = inputs;
            let input1 = input1.into_typed::<T>().unwrap();
            let input2 = input2.into_typed::<T>().unwrap();
            f(input1, input2, extra_args)
        })
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

        if let Some(mut in_dtypes) = in_dtypes {
            cast_python_scalars_to_default_precision(&mut in_dtypes);
            let has_float16_input = has_float16_input(&in_dtypes);

            for op_fn in &self.fns {
                if !has_float16_input {
                    // if none of the inputs is f16, never dispatch a function that requires f16
                    let has_f16_desc = op_fn
                        .input_desc
                        .iter()
                        .any(|input_desc| input_desc.dtype == ScalarKind::F16);
                    if has_f16_desc {
                        continue;
                    }
                }

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
                    .map(|(input, input_desc)| input.cast(input_desc.dtype)?.into_array())
                    .try_collect_array()?
                    .unwrap();

                return (op_fn.f)(inputs, extra_args);
            }
        }

        Err(unsupported_dtype_error(self.name, &inputs))
    }
}

pub(crate) fn operands_dtype_promotion<const N: usize>(
    name: &str,
    operands: [Operand; N],
) -> PyResult<[Operand; N]> {
    match promotion_target(name, &operands)? {
        None => Ok(operands),
        Some(dtype) => array_try_map(operands, |_i, operand| operand.cast(dtype)),
    }
}

pub(crate) fn operands_dtype_promotion_vec(
    name: &str,
    operands: Vec<Operand>,
) -> PyResult<Vec<Operand>> {
    match promotion_target(name, &operands)? {
        None => Ok(operands),
        Some(dtype) => operands
            .into_iter()
            .map(|operand| operand.cast(dtype))
            .collect(),
    }
}

/// The dtype to cast `operands` to, or `None` if they are arrays that already share a dtype.
fn promotion_target(name: &str, operands: &[Operand]) -> PyResult<Option<ScalarKind>> {
    fn array_dtype(operand: &Operand) -> Option<&Dtype> {
        match operand {
            Operand::PyArray(array) => Some(array.get().arr.dtype()),
            Operand::Array(array) => Some(array.dtype()),
            Operand::Scalar { .. } => None,
        }
    }
    let first_dtype = operands.first().and_then(array_dtype);
    if first_dtype.is_some() && operands.iter().all(|op| array_dtype(op) == first_dtype) {
        return Ok(None);
    }
    match promotion_target_impl(operands) {
        Some(dtype) => Ok(Some(dtype)),
        None => Err(unsupported_dtype_error(name, operands)),
    }
}

pub(crate) fn promotion_target_impl(operands: &[Operand]) -> Option<ScalarKind> {
    let mut in_dtypes = operands
        .iter()
        .map(Operand::rank_precision)
        .collect::<Option<Vec<_>>>()?;
    cast_python_scalars_to_default_precision(&mut in_dtypes);
    let has_float16_input = has_float16_input(&in_dtypes);

    let promotion_target_impls = [
        ScalarKind::Bool,
        ScalarKind::U8,
        ScalarKind::I8,
        ScalarKind::U16,
        ScalarKind::I16,
        ScalarKind::U32,
        ScalarKind::I32,
        ScalarKind::U64,
        ScalarKind::I64,
        ScalarKind::F16,
        ScalarKind::F32,
        ScalarKind::F64,
        ScalarKind::ComplexF32,
        ScalarKind::ComplexF64,
    ];
    for &dtype in &promotion_target_impls {
        if dtype == ScalarKind::F16 && !has_float16_input {
            continue;
        }
        if in_dtypes
            .iter()
            .all(|&in_dtype| CastKind::Safe.is_cast_allowed(in_dtype, dtype))
        {
            return Some(dtype);
        }
    }
    None
}

/// If all operands have no precision (all are python scalars), set precision to 8 (i64, f64).
fn cast_python_scalars_to_default_precision(in_dtypes: &mut [(Rank, Option<Precision>)]) {
    let all_without_precision = in_dtypes
        .iter()
        .all(|(_rank, precision)| precision.is_none());
    if all_without_precision {
        for (rank, precision) in in_dtypes {
            *precision = match rank {
                Rank::Bool => Some(Precision::P1),
                Rank::UInt | Rank::Int | Rank::Float | Rank::Complex => Some(Precision::P8),
            };
        }
    }
}

/// Whether one of the inputs is f16. If none is, a function that requires f16 is never dispatched.
fn has_float16_input(in_dtypes: &[(Rank, Option<Precision>)]) -> bool {
    in_dtypes.iter().any(|(rank, precision)| {
        matches!(rank, Rank::Float | Rank::Complex) && matches!(precision, Some(Precision::P2))
    })
}

#[cold]
#[inline(never)]
fn unsupported_dtype_error(name: &str, inputs: &[Operand]) -> PyErr {
    let ops_types = inputs
        .iter()
        .map(|dtype| match dtype {
            Operand::PyArray(array) => Cow::Owned(format!("{}", array.get().arr.dtype())),
            Operand::Array(array) => Cow::Owned(format!("{}", array.dtype())),
            Operand::Scalar {
                value,
                shape: _,
                precision,
                params: _,
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
        .collect::<Vec<_>>();
    let ops_types_str = ops_types.join(", ");
    pyo3::exceptions::PyTypeError::new_err(format!(
        "Op '{name}' does not support operands with dtypes [{ops_types_str}]"
    ))
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
impl OpDescriptor<3, ()> {
    pub(crate) fn dispatch3(&self, a: Operand, b: Operand, c: Operand) -> PyResult<ArrayAny> {
        Self::dispatch(self, [a, b, c])
    }
}

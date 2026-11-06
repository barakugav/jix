use jix_core::dtype::{Dtype, Itemsize, ScalarKind};
use pyo3::prelude::*;

use crate::ops::common::{scalar_kind_to_rank_precision, unsupported_dtype_error, Operand};
use crate::util::array_try_map;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Rank {
    Bool = 0,
    UInt = 1,
    Int = 2,
    Float = 3,
    Complex = 4,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Precision {
    P1 = 0,
    P2 = 1,
    P4 = 2,
    P8 = 3,
}

impl Precision {
    #[inline(always)]
    pub(crate) fn from_itemsize(itemsize: Itemsize) -> Self {
        match itemsize {
            1 => Self::P1,
            2 => Self::P2,
            4 => Self::P4,
            8 => Self::P8,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn higher(self) -> Option<Self> {
        match self {
            Self::P1 => Some(Self::P2),
            Self::P2 => Some(Self::P4),
            Self::P4 => Some(Self::P8),
            Self::P8 => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CastKind {
    /// No casting
    None,
    /// f16->f32->f64
    /// i8->i16->i32->i64
    /// u8->u16->u32->u64
    /// u8->i16
    /// u16->i32
    /// u32->i64
    /// u64->f64
    Safe,
    /// Allow any cast, even if it loses information
    Unsafe,
}

impl CastKind {
    pub(crate) fn is_cast_allowed(&self, src: (Rank, Option<Precision>), dst: ScalarKind) -> bool {
        let (mut src_rank, src_precision) = src;
        let (dst_rank, dst_precision) = scalar_kind_to_rank_precision(dst);
        let dst_precision = dst_precision.unwrap();

        let is_python_scalar = src_precision.is_none();
        if is_python_scalar {
            // a python int is not signed and not unsigned, it can fit both
            if src_rank == Rank::Int && dst_rank == Rank::UInt {
                src_rank = Rank::UInt;
            }
        }

        match self {
            CastKind::None => {
                src_rank == dst_rank && src_precision.is_none_or(|p| p == dst_precision)
            }
            CastKind::Safe => {
                if src_rank > dst_rank {
                    return false;
                }

                match (src_rank, dst_rank) {
                    // bool can be cast to anything
                    (Rank::Bool, _) => true,

                    // equal rank
                    (Rank::UInt, Rank::UInt)
                    | (Rank::Int, Rank::Int)
                    | (Rank::Float, Rank::Float)
                    | (Rank::Complex, Rank::Complex) |
                    // float to complex
                    (Rank::Float, Rank::Complex) => {
                        // Same precision is OK
                        src_precision.is_none_or(|p| p <= dst_precision)
                    }

                    // uint to int
                    (Rank::UInt, Rank::Int) => {
                        // Higher precision is required
                        #[allow(clippy::unnecessary_unwrap)]
                        if src_precision.is_none() {
                            true
                        } else if let Some(src_precision) = src_precision.unwrap().higher() {
                            src_precision <= dst_precision
                        } else {
                            false
                        }
                    }
                    // u/int to float/complex
                    (Rank::UInt | Rank::Int, Rank::Float)
                    | (Rank::UInt | Rank::Int, Rank::Complex) => {
                        // Higher precision is required
                        #[allow(clippy::unnecessary_unwrap)]
                        if src_precision.is_none() {
                            true
                        } else if dst_precision == Precision::P8 {
                            // Allow any precision to be cast to 64-bit float/complex
                            true
                        } else if let Some(src_precision) = src_precision.unwrap().higher() {
                            src_precision <= dst_precision
                        } else {
                            false
                        }
                    }

                    (_, Rank::Bool)
                    | (Rank::Int, Rank::UInt)
                    | (Rank::Float, Rank::UInt | Rank::Int)
                    | (Rank::Complex, Rank::UInt | Rank::Int | Rank::Float) => {
                        unreachable!() // src_rank <= dst_rank
                    }
                }
            }
            CastKind::Unsafe => match (src_rank, dst_rank) {
                (Rank::Bool | Rank::UInt | Rank::Int | Rank::Float, _) => true,
                (Rank::Complex, Rank::Bool | Rank::Complex) => true,
                (Rank::Complex, Rank::UInt | Rank::Int | Rank::Float) => false,
            },
        }
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
        ScalarKind::Bool8,
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
pub(crate) fn cast_python_scalars_to_default_precision(
    in_dtypes: &mut [(Rank, Option<Precision>)],
) {
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
pub(crate) fn has_float16_input(in_dtypes: &[(Rank, Option<Precision>)]) -> bool {
    in_dtypes.iter().any(|(rank, precision)| {
        matches!(rank, Rank::Float | Rank::Complex) && matches!(precision, Some(Precision::P2))
    })
}

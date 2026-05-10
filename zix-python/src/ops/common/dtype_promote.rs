use zix_core::dtype::DtypeScalarKind;

use crate::ops::common::Precision;
use crate::ops::{Operand, Scalar};

pub(crate) fn promote(ops: &[&Operand]) -> Option<DtypeScalarKind> {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Rank {
        Bool = 0,
        UInt = 1,
        Int = 2,
        Float = 3,
        Complex = 4,
    }

    fn to_rank_precision(value: &Operand) -> Option<(Rank, Option<Precision>)> {
        fn to_rank_precision_impl(kind: DtypeScalarKind) -> (Rank, Option<Precision>) {
            let (rank, precision) = match kind {
                _ if kind.is_bool() => return (Rank::Bool, None),
                _ if kind.is_unsigned_integer() => (Rank::UInt, kind.itemsize()),
                _ if kind.is_integer() => (Rank::Int, kind.itemsize()),
                _ if kind.is_float() => (Rank::Float, kind.itemsize()),
                _ if kind.is_complex() => (Rank::Complex, kind.itemsize() / 2),
                _ => unreachable!(),
            };
            (rank, Some(Precision::from_itemsize(precision)))
        }

        match value {
            Operand::Zix(arr) => arr
                .get()
                .arr
                .dtype()
                .try_to_scalar()
                .map(to_rank_precision_impl),
            Operand::Numpy(arr) => arr.dtype().try_to_scalar().map(to_rank_precision_impl),
            Operand::Scalar {
                value,
                precision,
                shape: _,
            } => {
                let kind = match value {
                    Scalar::Bool(_) => Rank::Bool,
                    Scalar::UInt(_) => Rank::UInt,
                    Scalar::Int(_) => Rank::Int,
                    Scalar::Float(_) => Rank::Float,
                    Scalar::Complex(_) => Rank::Complex,
                };
                Some((kind, *precision))
            }
        }
    }

    if ops.is_empty() {
        return None;
    }
    let mut ranks_precisions = ops
        .iter()
        .map(|op| to_rank_precision(op))
        .collect::<Option<Vec<_>>>()?
        .into_iter();

    let (mut result_rank, mut result_precision) = ranks_precisions.next().unwrap().clone();
    for (mut b_rank, mut b_precision) in ranks_precisions {
        let (mut a_rank, mut a_precision) = (result_rank, result_precision);

        if a_rank < b_rank {
            std::mem::swap(&mut a_rank, &mut b_rank);
            std::mem::swap(&mut a_precision, &mut b_precision);
        }
        debug_assert!(a_rank >= b_rank);

        (result_rank, result_precision) = match (a_rank, b_rank) {
            (Rank::Bool, Rank::Bool)
            | (Rank::UInt, Rank::UInt)
            | (Rank::Int, Rank::Int)
            | (Rank::Float, Rank::Float)
            | (Rank::Complex, Rank::Complex) => (a_rank, std::cmp::max(a_precision, b_precision)),

            (Rank::Bool, _) => (b_rank, b_precision),
            (_, Rank::Bool) => unreachable!(), // a_rank >= b_rank

            (Rank::Int, Rank::UInt) => {
                if let Some(b_precision) = b_precision {
                    // increase unsigned precision to prevent overflow, e.g. u8 + i8 -> i16 instead of i8
                    if let Some(b_promoted_prec) = b_precision.higher() {
                        (Rank::Int, std::cmp::max(a_precision, Some(b_promoted_prec)))
                    } else {
                        // if unsigned precision is already max (u64), promote to f64
                        (Rank::Float, Some(Precision::P8))
                    }
                } else {
                    (Rank::Int, a_precision)
                }
            }
            (Rank::UInt, Rank::Int) => unreachable!(), // a_rank >= b_rank
            (Rank::Float, Rank::UInt | Rank::Int) => {
                let mut precision = a_precision;
                if let Some(b_precision) = b_precision {
                    // when promoting int/uint to float, increase number of bytes to preserve precision
                    // f32 + i/u8 -> f32
                    // f32 + i/u16 -> f32
                    // f32 + i/u32 -> f64
                    // f32 + i/u64 -> f64
                    let b_promoted_prec = b_precision.higher().unwrap_or(Precision::P8);
                    precision = std::cmp::max(a_precision, Some(b_promoted_prec));
                }
                (Rank::Float, precision)
            }
            (Rank::UInt | Rank::Int, Rank::Float) => unreachable!(), // a_rank >= b_rank

            (Rank::Complex, Rank::UInt | Rank::Int) => {
                let mut precision = a_precision;
                if let Some(b_precision) = b_precision {
                    // when promoting int/uint to complex, increase number of bytes to preserve precision
                    // c<f32> + i/u8 -> c<f32>
                    // c<f32> + i/u16 -> c<f32>
                    // c<f32> + i/u32 -> c<f64>
                    // c<f32> + i/u64 -> c<f64>
                    let b_promoted_prec = b_precision.higher().unwrap_or(Precision::P8);
                    precision = std::cmp::max(a_precision, Some(b_promoted_prec));
                }
                (Rank::Complex, precision)
            }
            (Rank::Complex, Rank::Float) => {
                (Rank::Complex, std::cmp::max(a_precision, b_precision))
            }
            (Rank::UInt | Rank::Int | Rank::Float, Rank::Complex) => {
                unreachable!() // a_rank >= b_rank
            }
        };
    }

    let (rank, precision) = (result_rank, result_precision);
    Some(match rank {
        Rank::Bool => match precision {
            None => DtypeScalarKind::Bool,
            Some(_) => unreachable!(),
        },
        Rank::UInt => match precision {
            Some(Precision::P1) => DtypeScalarKind::U8,
            Some(Precision::P2) => DtypeScalarKind::U16,
            Some(Precision::P4) => DtypeScalarKind::U32,
            Some(Precision::P8) => DtypeScalarKind::U64,
            None => DtypeScalarKind::U64,
        },
        Rank::Int => match precision {
            Some(Precision::P1) => DtypeScalarKind::I8,
            Some(Precision::P2) => DtypeScalarKind::I16,
            Some(Precision::P4) => DtypeScalarKind::I32,
            Some(Precision::P8) => DtypeScalarKind::I64,
            None => DtypeScalarKind::I64,
        },
        Rank::Float => match precision {
            Some(Precision::P1) => unreachable!(),
            Some(Precision::P2) => DtypeScalarKind::F16,
            Some(Precision::P4) => DtypeScalarKind::F32,
            Some(Precision::P8) => DtypeScalarKind::F64,
            None => DtypeScalarKind::F64,
        },
        Rank::Complex => match precision {
            Some(Precision::P1) | Some(Precision::P2) => unreachable!(),
            Some(Precision::P4) => DtypeScalarKind::ComplexF32,
            Some(Precision::P8) => DtypeScalarKind::ComplexF64,
            None => DtypeScalarKind::ComplexF64,
        },
    })
}

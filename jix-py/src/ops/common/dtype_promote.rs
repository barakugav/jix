use jix_core::dtype::{Itemsize, ScalarKind};

use crate::ops::common::scalar_kind_to_rank_precision;

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
        let (src_rank, src_precision) = src;
        let (dst_rank, dst_precision) = scalar_kind_to_rank_precision(dst);
        let dst_precision = dst_precision.unwrap();

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

use pyo3::prelude::*;
use jix_core::ErrorKind;
use jix_core::{ops::Broadcast, NDIM_MAX};

use crate::ops::common::Operand;
use crate::util::{DimArray, IntoPyResult};

pub(crate) fn broadcast_operands<const N: usize>(operands: [Operand; N]) -> PyResult<[Operand; N]> {
    let operands = operands.into_iter().collect::<Vec<_>>();
    let operands = broadcast_operands_dyn(operands)?;
    let operands: [Operand; N] = operands.try_into().map_err(|_| unreachable!()).unwrap();
    Ok(operands)
}

// TODO pub broadcast_arrays
pub(crate) fn broadcast_operands_dyn(operands: Vec<Operand>) -> PyResult<Vec<Operand>> {
    if operands.len() < 2 {
        return Ok(operands);
    }

    let shape = {
        let mut shapes = operands.iter().map(|operand| match operand {
            Operand::Array(arr) => arr.shape(),
            Operand::Scalar { shape, .. } => shape.as_slice(),
        });

        let first_shape: DimArray<_> = shapes.next().unwrap().try_into().unwrap();
        shapes
            .try_fold(first_shape, |shape1, shape2| {
                broadcast_shapes(&shape1, shape2)
            })
            .into_py_result()?
    };

    operands
        .into_iter()
        .map(|operand| broadcast_operand(operand, &shape))
        .collect::<Result<Vec<_>, _>>()
        .into_py_result()
}

fn broadcast_operand(operand: Operand, shape: &[u64]) -> Result<Operand, jix_core::Error> {
    assert!(shape.len() <= NDIM_MAX);

    let array = match operand {
        Operand::Array(arr) => arr,
        Operand::Scalar {
            value,
            precision,
            shape: _,
        } => {
            return Ok(Operand::Scalar {
                value,
                precision,
                shape: shape.try_into().unwrap(),
            });
        }
    };

    let array = if let Some(missing_dims) = shape.len().checked_sub(array.ndim()) {
        Either::Left(array.insert_axis(&vec![0; missing_dims]))
    } else {
        Either::Right(array)
    };

    let array = match array {
        Either::Left(arr) => Broadcast::new_array(arr, shape)?.into_any(),
        Either::Right(arr) => Broadcast::new_array(arr, shape)?.into_any(),
    };

    Ok(Operand::Array(array))
}

pub(crate) enum Either<L, R> {
    Left(L),
    Right(R),
}

fn broadcast_shapes(shape1: &[u64], shape2: &[u64]) -> Result<DimArray<u64>, jix_core::Error> {
    let ndim = shape1.len().max(shape2.len());
    let mut result = DimArray::new();

    for i in 0..ndim {
        let d1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let d2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if d1 == d2 {
            result.push(d1);
        } else if d1 == 1 {
            result.push(d2);
        } else if d2 == 1 {
            result.push(d1);
        } else {
            return Err(jix_core::Error::new(
                ErrorKind::InvalidShapeOperation,
                "operands could not be broadcast together",
            ));
        }
    }

    result.reverse();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(shape1: &[u64], shape2: &[u64], expected: &[u64]) {
        let result = broadcast_shapes(shape1, shape2).unwrap();
        assert_eq!(
            &result, expected,
            "broadcast({shape1:?}, {shape2:?}) = {result:?}, expected {expected:?}"
        );
    }

    fn err(shape1: &[u64], shape2: &[u64]) {
        assert!(
            broadcast_shapes(shape1, shape2).is_err(),
            "broadcast({shape1:?}, {shape2:?}) should fail but got {:?}",
            broadcast_shapes(shape1, shape2).unwrap(),
        );
    }

    // ── Identical shapes ──────────────────────────────────────────────

    #[test]
    fn identical_scalar() {
        ok(&[], &[], &[]);
    }

    #[test]
    fn identical_1d() {
        ok(&[5], &[5], &[5]);
    }

    #[test]
    fn identical_2d() {
        ok(&[3, 4], &[3, 4], &[3, 4]);
    }

    #[test]
    fn identical_3d() {
        ok(&[2, 3, 4], &[2, 3, 4], &[2, 3, 4]);
    }

    // ── Scalar (0-d) broadcasting ─────────────────────────────────────

    #[test]
    fn scalar_with_1d() {
        ok(&[], &[5], &[5]);
        ok(&[5], &[], &[5]);
    }

    #[test]
    fn scalar_with_2d() {
        ok(&[], &[3, 4], &[3, 4]);
        ok(&[3, 4], &[], &[3, 4]);
    }

    #[test]
    fn scalar_with_3d() {
        ok(&[], &[2, 3, 4], &[2, 3, 4]);
    }

    // ── Ones expand ───────────────────────────────────────────────────

    #[test]
    fn one_expands_trailing() {
        ok(&[3, 1], &[3, 4], &[3, 4]);
        ok(&[3, 4], &[3, 1], &[3, 4]);
    }

    #[test]
    fn one_expands_leading() {
        ok(&[1, 4], &[3, 4], &[3, 4]);
        ok(&[3, 4], &[1, 4], &[3, 4]);
    }

    #[test]
    fn one_expands_both_dims() {
        ok(&[1, 1], &[3, 4], &[3, 4]);
    }

    #[test]
    fn both_have_ones_different_dims() {
        ok(&[1, 4], &[3, 1], &[3, 4]);
        ok(&[3, 1], &[1, 4], &[3, 4]);
    }

    #[test]
    fn one_by_one() {
        ok(&[1], &[1], &[1]);
    }

    #[test]
    fn one_by_n() {
        ok(&[1], &[7], &[7]);
        ok(&[7], &[1], &[7]);
    }

    // ── Prepending dimensions (different ndim) ────────────────────────

    #[test]
    fn prepend_1d_to_2d() {
        ok(&[4], &[3, 4], &[3, 4]);
        ok(&[3, 4], &[4], &[3, 4]);
    }

    #[test]
    fn prepend_1d_to_3d() {
        ok(&[4], &[2, 3, 4], &[2, 3, 4]);
    }

    #[test]
    fn prepend_2d_to_3d() {
        ok(&[3, 4], &[2, 3, 4], &[2, 3, 4]);
        ok(&[2, 3, 4], &[3, 4], &[2, 3, 4]);
    }

    #[test]
    fn prepend_1d_to_4d() {
        ok(&[5], &[2, 3, 4, 5], &[2, 3, 4, 5]);
    }

    #[test]
    fn prepend_with_ones() {
        // (1,) vs (2,3) → implicit (1,1) vs (2,3) → (2,3)
        ok(&[1], &[2, 3], &[2, 3]);
    }

    // ── Combined prepend + expand ─────────────────────────────────────

    #[test]
    fn prepend_and_expand() {
        // (1, 4) vs (2, 3, 1) → (1, 1, 4) vs (2, 3, 1) → (2, 3, 4)
        ok(&[1, 4], &[2, 3, 1], &[2, 3, 4]);
    }

    #[test]
    fn prepend_and_expand_reversed() {
        ok(&[2, 3, 1], &[1, 4], &[2, 3, 4]);
    }

    #[test]
    fn complex_broadcast_3d() {
        // (1, 5, 1) vs (3, 1, 4) → (3, 5, 4)
        ok(&[1, 5, 1], &[3, 1, 4], &[3, 5, 4]);
    }

    #[test]
    fn complex_broadcast_4d() {
        // (8, 1, 6, 1) vs (7, 1, 5) → (8, 7, 6, 5)
        // This is the classic NumPy docs example
        ok(&[8, 1, 6, 1], &[7, 1, 5], &[8, 7, 6, 5]);
    }

    // ── All-ones shapes ───────────────────────────────────────────────

    #[test]
    fn all_ones_same_ndim() {
        ok(&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]);
    }

    #[test]
    fn all_ones_different_ndim() {
        ok(&[1], &[1, 1, 1], &[1, 1, 1]);
    }

    #[test]
    fn all_ones_vs_real_shape() {
        ok(&[1, 1, 1], &[2, 3, 4], &[2, 3, 4]);
    }

    // ── Large dimensions ──────────────────────────────────────────────

    #[test]
    fn large_dims() {
        ok(&[1, 1000000], &[1000000, 1], &[1000000, 1000000]);
    }

    #[test]
    fn u64_max_dim() {
        let big = u64::MAX;
        ok(&[big], &[1], &[big]);
        ok(&[1], &[big], &[big]);
        ok(&[big], &[big], &[big]);
    }

    // ── High-rank tensors ─────────────────────────────────────────────

    #[test]
    fn broadcast_6d() {
        ok(
            &[1, 2, 1, 4, 1, 6],
            &[7, 1, 3, 1, 5, 1],
            &[7, 2, 3, 4, 5, 6],
        );
    }

    #[test]
    fn broadcast_rank_mismatch_big() {
        ok(&[3], &[5, 4, 3, 2, 1, 3], &[5, 4, 3, 2, 1, 3]);
    }

    // ── Zero-size dimensions ──────────────────────────────────────────
    // NumPy allows 0 in shapes: (0,) broadcasts like any other dim

    #[test]
    fn zero_with_zero() {
        ok(&[0], &[0], &[0]);
    }

    #[test]
    fn zero_with_one() {
        // np.broadcast_shapes((0,), (1,)) → (0,)
        ok(&[0], &[1], &[0]);
        ok(&[1], &[0], &[0]);
    }

    #[test]
    fn zero_with_nonzero_fails() {
        // np.broadcast_shapes((0,), (3,)) → error
        err(&[0], &[3]);
        err(&[3], &[0]);
    }

    #[test]
    fn zero_in_higher_dim() {
        ok(&[3, 0], &[3, 1], &[3, 0]);
        ok(&[1, 0], &[3, 1], &[3, 0]);
    }

    #[test]
    fn zero_broadcast_prepend() {
        ok(&[0], &[3, 0], &[3, 0]);
    }

    // ── Commutativity ─────────────────────────────────────────────────

    #[test]
    fn commutative_success() {
        let pairs: Vec<(&[u64], &[u64])> = vec![
            (&[3, 1], &[1, 4]),
            (&[8, 1, 6, 1], &[7, 1, 5]),
            (&[5], &[2, 3, 4, 5]),
            (&[], &[3, 4]),
            (&[1], &[256, 256, 3]),
        ];
        for (a, b) in pairs {
            let ab = broadcast_shapes(a, b).unwrap();
            let ba = broadcast_shapes(b, a).unwrap();
            assert_eq!(ab, ba, "commutativity failed for {a:?} vs {b:?}");
        }
    }

    #[test]
    fn commutative_failure() {
        let pairs: Vec<(&[u64], &[u64])> =
            vec![(&[3], &[4]), (&[2, 3], &[4, 5]), (&[2, 1], &[8, 4, 3])];
        for (a, b) in pairs {
            assert!(broadcast_shapes(a, b).is_err());
            assert!(broadcast_shapes(b, a).is_err());
        }
    }

    // ── Incompatible shapes (errors) ──────────────────────────────────

    #[test]
    fn err_simple_mismatch() {
        err(&[3], &[4]);
    }

    #[test]
    fn err_trailing_mismatch() {
        err(&[2, 3], &[2, 4]);
    }

    #[test]
    fn err_leading_mismatch() {
        err(&[3, 4], &[5, 4]);
    }

    #[test]
    fn err_inner_mismatch() {
        err(&[2, 3, 4], &[2, 5, 4]);
    }

    #[test]
    fn err_no_ones_to_save_it() {
        err(&[2, 3], &[4, 5]);
    }

    #[test]
    fn err_one_dim_ok_another_not() {
        // trailing matches, but inner doesn't
        err(&[2, 1, 4], &[3, 3, 4]);
    }

    #[test]
    fn err_prepend_still_mismatches() {
        err(&[2, 1], &[8, 4, 3]);
    }

    #[test]
    fn err_multiple_bad_dims() {
        err(&[2, 3, 4], &[5, 6, 7]);
    }

    // ── NumPy docs examples ───────────────────────────────────────────
    // Taken straight from numpy.broadcast_shapes / broadcasting docs

    #[test]
    fn numpy_docs_example_1() {
        // np.broadcast_shapes((5,4), (1,)) → (5,4)
        ok(&[5, 4], &[1], &[5, 4]);
    }

    #[test]
    fn numpy_docs_example_2() {
        // np.broadcast_shapes((5,4), (4,)) → (5,4)
        ok(&[5, 4], &[4], &[5, 4]);
    }

    #[test]
    fn numpy_docs_example_3() {
        // np.broadcast_shapes((15,3,5), (15,1,5)) → (15,3,5)
        ok(&[15, 3, 5], &[15, 1, 5], &[15, 3, 5]);
    }

    #[test]
    fn numpy_docs_example_4() {
        // np.broadcast_shapes((15,3,5), (3,5)) → (15,3,5)
        ok(&[15, 3, 5], &[3, 5], &[15, 3, 5]);
    }

    #[test]
    fn numpy_docs_example_5() {
        // np.broadcast_shapes((15,3,5), (3,1)) → (15,3,5)
        ok(&[15, 3, 5], &[3, 1], &[15, 3, 5]);
    }

    #[test]
    fn numpy_docs_example_arange() {
        // The classic (4,1) + (1,3) → (4,3)
        ok(&[4, 1], &[1, 3], &[4, 3]);
    }

    #[test]
    fn numpy_docs_image_example() {
        // (256,256,3) + (3,) → (256,256,3)
        ok(&[256, 256, 3], &[3], &[256, 256, 3]);
    }
}

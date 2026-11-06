use jix_core::error::ErrorKind;
use jix_core::NDIM_MAX;
use pyo3::prelude::*;

use crate::ops::broadcast_array_impl;
use crate::ops::common::Operand;
use crate::util::{array_try_map, DimArray, IntoPyResult};

pub(crate) fn broadcast_operands<const N: usize>(
    mut operands: [Operand; N],
) -> PyResult<[Operand; N]> {
    if let Some(shape) = broadcast_operands_shape(&operands)? {
        operands = array_try_map(operands, |_i, operand| broadcast_operand(operand, &shape))
            .into_py_result()?;
    };
    Ok(operands)
}

pub(crate) fn broadcast_operands_shape(operands: &[Operand]) -> PyResult<Option<DimArray<u64>>> {
    if operands.len() < 2 {
        return Ok(None);
    }

    let mut shapes = operands.iter().map(|operand| match operand {
        Operand::PyArray(arr) => arr.get().arr.shape(),
        Operand::Array(arr) => arr.shape(),
        Operand::Scalar { shape, .. } => shape.as_slice(),
    });

    let first_shape: DimArray<_> = shapes.next().unwrap().try_into().unwrap();
    let shape = shapes
        .try_fold(first_shape, |shape1, shape2| {
            broadcast_shapes(&shape1, shape2)
        })
        .into_py_result()?;
    Ok(Some(shape))
}

fn broadcast_operand(operand: Operand, shape: &[u64]) -> Result<Operand, jix_core::Error> {
    let op_shape = match &operand {
        Operand::PyArray(py) => py.get().arr.shape(),
        Operand::Array(array) => array.shape(),
        Operand::Scalar { shape, .. } => shape.as_slice(),
    };
    if shape == op_shape {
        return Ok(operand);
    }
    assert!(shape.len() <= NDIM_MAX);

    let array = match operand {
        Operand::PyArray(arr) => arr.get().arr.clone(),
        Operand::Array(arr) => arr,
        Operand::Scalar {
            value,
            precision,
            shape: _,
            params,
        } => {
            return Ok(Operand::Scalar {
                value,
                precision,
                shape: shape.try_into().unwrap(),
                params,
            });
        }
    };

    let array = broadcast_array_impl(array, shape)?;

    Ok(Operand::Array(array))
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

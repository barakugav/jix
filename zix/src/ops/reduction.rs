use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
#[allow(unused_imports)]
use crate::dtype::{f16, Complex};
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::util::iter::NdIter;
use crate::util::{default_strides, dim_arr, DimArray};
use crate::Array;

pub(crate) trait ReductionOpKernel {
    fn reduce<'a>(
        &self,
        slice_iter: impl Iterator<Item = (impl Iterator<Item = &'a [u8]> + Clone, &'a mut [u8])>,
        input_dtype: &Dtype,
    ) -> Result<()>;

    fn output_dtype(&self, input_dtype: &Dtype) -> Result<Dtype>;
    fn supports_empty(&self) -> bool;
}

pub(crate) struct ReductionOp<Op, S> {
    op: Op,

    array: Array<S>,
    is_reduced: DimArray<bool>,
    keepdims: bool,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<Op, S> ReductionOp<Op, S> {
    pub(crate) fn new(op: Op, array: Array<S>, axes: &[usize], keepdims: bool) -> Result<Self>
    where
        Op: ReductionOpKernel,
        S: ArrayStorage,
    {
        let output_dtype = op.output_dtype(array.dtype())?;

        let input_ndim = array.shape().len();
        let mut is_reduced = dim_arr(input_ndim, |_| false);
        for &ax in axes {
            ensure!(
                ax < input_ndim,
                InvalidArgument,
                "axis {ax} out of bounds for array of ndim {input_ndim}"
            );

            ensure!(!is_reduced[ax], InvalidArgument, "duplicate axis {ax}");
            is_reduced[ax] = true;
        }

        if !op.supports_empty()
            && array
                .shape()
                .iter()
                .zip(&is_reduced)
                .any(|(&s, &reduced)| reduced && s == 0)
        {
            bail!(
                InvalidArgument,
                "reduction on empty dimension not supported"
            );
        }

        let shape = array
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                if is_reduced[i] {
                    keepdims.then_some(1)
                } else {
                    Some(s)
                }
            })
            .collect::<DimArray<_>>();

        let mut b_layout = array.blocks_layout().clone();
        b_layout.block_shape_hint = (0..input_ndim)
            .filter_map(|d| {
                if is_reduced[d] {
                    keepdims.then_some(1)
                } else {
                    Some(b_layout.block_shape_hint[d])
                }
            })
            .collect();
        b_layout.block_shape_tag = (0..input_ndim)
            .filter_map(|d| {
                if is_reduced[d] {
                    keepdims.then_some(crate::storage::BlockShapeTag::Any)
                } else {
                    Some(b_layout.block_shape_tag[d])
                }
            })
            .collect();
        b_layout.preferred_read_shape = (0..input_ndim)
            .filter_map(|d| {
                if is_reduced[d] {
                    keepdims.then_some(1)
                } else {
                    Some(b_layout.preferred_read_shape[d])
                }
            })
            .collect();

        Ok(Self {
            op,
            dtype: output_dtype,
            shape,
            blocks_layout: b_layout,
            array,
            is_reduced,
            keepdims,
        })
    }
}
impl<Op, S> ArrayStorage for ReductionOp<Op, S>
where
    Op: ReductionOpKernel,
    S: ArrayStorage,
{
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, &self.dtype, buf)?;

        let orig_shape = self.array.shape();
        let orig_ndim = orig_shape.len();

        // Build inner_index: reduced dims span the full original range,
        // non-reduced dims forward the requested output range.
        //
        // With keepdims=false the output has fewer dims than the input, so we
        // use `out_dim` to step through `index`.  With keepdims=true the output
        // has the same number of dims (reduced ones are size-1), so `index[d]`
        // maps directly to input dim `d`.
        let mut out_dim = 0usize;
        let inner_index = (0..orig_ndim)
            .map(|in_d| {
                if self.is_reduced[in_d] {
                    if self.keepdims {
                        out_dim += 1; // skip the size-1 keepdim slot
                    }
                    // TODO: we could read it in chunks
                    0..orig_shape[in_d]
                } else {
                    let r = index[out_dim].clone();
                    out_dim += 1;
                    r
                }
            })
            .collect::<DimArray<_>>();

        let src_dtype = self.array.dtype();
        let dst_dtype = self.dtype();

        let inner_read_shape = inner_index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<DimArray<_>>();
        let n_inner: usize = inner_read_shape.iter().product();

        let out_shape = index
            .iter()
            .map(|r| (r.end - r.start) as usize)
            .collect::<DimArray<_>>();

        // Read the full inner block into a temp buffer.
        let tmp_buf_size = n_inner * src_dtype.itemsize() as usize;
        let mut tmp_buf = context.tmp_buf(tmp_buf_size, src_dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array
            .storage
            .read_data(&inner_index, tmp_buf, context)?;

        // C-contiguous byte strides for inner and output layouts.
        let inner_strides = default_strides(&inner_read_shape, src_dtype.itemsize() as usize);
        let out_strides = default_strides(&out_shape, dst_dtype.itemsize() as usize);

        // Strides used by out_iter to advance the `base_ptr` into tmp_buf.
        // For reduced dims the outer loop visits exactly one position (size 1
        // when keepdims=true, or the dim is absent when keepdims=false), so
        // their stride contribution to base_ptr is 0.
        let tmp_buf_strides = inner_strides
            .iter()
            .zip(&self.is_reduced)
            .filter_map(|(&s, &reduced)| {
                if reduced {
                    self.keepdims.then_some(0)
                } else {
                    Some(s)
                }
            })
            .collect::<DimArray<_>>();

        let out_iter = NdIter::new(
            &out_shape,
            (
                NdIterExtStridesPtr::new(&tmp_buf_strides, tmp_buf.as_ptr()),
                NdIterExtStridesPtrMut::new(&out_strides, buf.as_mut_ptr()),
            ),
        );
        let reduction_shape = dim_arr(orig_ndim, |d| {
            if self.is_reduced[d] {
                orig_shape[d]
            } else {
                1
            }
        });

        let slice_iter = out_iter.map(|(_out_idx, (base_ptr, out_ptr))| {
            let reduction_iter = NdIter::new(
                &reduction_shape,
                NdIterExtStridesPtr::new(&inner_strides, base_ptr),
            );
            let reduction_iter = reduction_iter.map(|(_idx, in_ptr)| unsafe {
                std::slice::from_raw_parts(in_ptr, src_dtype.itemsize() as usize)
            });
            let out_entry =
                unsafe { std::slice::from_raw_parts_mut(out_ptr, dst_dtype.itemsize() as usize) };
            (reduction_iter, out_entry)
        });
        self.op.reduce(slice_iter, src_dtype)
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage._spec()
        }
    }
}

macro_rules! define_reduction_op {
    (
        $(#[$meta:meta])*
        $Name:ident,
        $NameKernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = $types:tt,
        single_axis = true
    ) => {
        $(#[$meta])*
        pub struct $Name<S>(crate::ops::reduction::ReductionOp<$NameKernel, S>);
        impl<S> $Name<S> {
            /// Creates a new view storage applying the operation by reducing the specified axis.
            ///
            /// See the struct-level documentation for details on supported dtypes, output dtype, and semantics.
            pub fn new(array: crate::Array<S>, axis: usize, keepdims: bool $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                let kernel = $NameKernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(kernel, array, &[axis], keepdims)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        define_reduction_op_kernel!(
            $NameKernel,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = $types
        );
    };

    (
        $(#[$meta:meta])*
        $Name:ident,
        $NameKernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = $types:tt
    ) => {
        $(#[$meta])*
        pub struct $Name<S>(crate::ops::reduction::ReductionOp<$NameKernel, S>);
        impl<S> $Name<S> {
            /// Creates a new view storage applying the operation by reducing the specified axes.
            ///
            /// See the struct-level documentation for details on supported dtypes, output dtype, and semantics.
            pub fn new(array: crate::Array<S>, axes: &[usize], keepdims: bool $(, $extra_arg: $extra_ty)*) -> crate::error::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                let kernel = $NameKernel { $($extra_arg),* };
                Ok(Self(crate::ops::reduction::ReductionOp::new(kernel, array, axes, keepdims)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        define_reduction_op_kernel!(
            $NameKernel,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = $types
        );
    };
}

macro_rules! define_reduction_op_kernel {
    (
        $NameKernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {
            input = [$($scalar:tt),* $(,)?],
            output = "same"
        }
    ) => {
        define_reduction_op_kernel!(
            $NameKernel,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = {$($scalar => $scalar),*}
        );
    };

    (
        $NameKernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {
            input = [$($scalar:tt),* $(,)?],
            output = $output_type:tt
        }
    ) => {
        define_reduction_op_kernel!(
            $NameKernel,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = {$($scalar => $output_type),*}
        );
    };

    (
        $NameKernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {$([$($scalar:tt),*] => $output_type:tt),* $(,)?}
    ) => {
        define_reduction_op_kernel!(
            $NameKernel,
            support_empty = $support_empty,
            |$arg_items $(, $extra_arg : $extra_ty)*| { $body },
            types = {$($($scalar => $output_type),*),*}
        );
    };

    (
        $NameKernel:ident,
        support_empty = $support_empty:expr,
        |$arg_items:ident $(, $extra_arg:ident : $extra_ty:ty)*| { $body:expr },
        types = {$($scalar:tt => $reduction_type:tt),* $(,)?}
    ) => {
        struct $NameKernel { $($extra_arg: $extra_ty,)* }
        impl crate::ops::reduction::ReductionOpKernel for $NameKernel {
            fn reduce<'a>(
                &self,
                slice_iter: impl Iterator<Item = (impl Iterator<Item = &'a [u8]> + Clone, &'a mut [u8])>,
                input_dtype: &Dtype,
            ) -> crate::error::Result<()> {
                macro_rules! apply_loop_impl {
                    ($scalar2:ty, $reduction_type2:ty) => {{
                        for (slice, out) in slice_iter {
                            let items = slice.map(|x| unsafe { x.as_ptr().cast::<$scalar2>().read() });
                            let result: $reduction_type2 = {
                                #[allow(unused)]
                                type ReductionType = $reduction_type2;
                                $(let $extra_arg = self.$extra_arg;)*
                                let $arg_items = items;
                                { $body }
                            };
                            unsafe { out.as_mut_ptr().cast::<$reduction_type2>().write(result) };
                        }
                        return Ok(())
                    }};
                }
                macro_rules! apply_loop {
                    (f16, $reduction_type2:ty) => {
                        #[cfg(feature = "half")]
                        apply_loop_impl!(f16, $reduction_type2)
                    };
                    ((Complex<f32>), $reduction_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(Complex<f32>, $reduction_type2)
                    };
                    ((Complex<f64>), $reduction_type2:ty) => {
                        #[cfg(feature = "num-complex")]
                        apply_loop_impl!(Complex<f64>, $reduction_type2)
                    };
                    ($scalar2:ty, $reduction_type2:ty) => {
                        apply_loop_impl!($scalar2, $reduction_type2)
                    };
                }
                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($scalar)) => {
                        apply_loop!($scalar, $reduction_type)
                    },)*
                    _ => {}
                }
                bail!(UnsupportedDtype, "Reduction op not supported for dtype {input_dtype:#?}");
            }

            fn output_dtype(&self, input_dtype: &crate::dtype::Dtype) -> crate::error::Result<crate::dtype::Dtype> {
                #[allow(unused_parens)]
                match input_dtype.try_to_scalar() {
                    $(Some(crate::ops::common::scalar_kind!($scalar)) => {
                        return Ok(<$reduction_type as crate::dtype::Dtyped>::DTYPE);
                    },)*
                    _ => {},

                };
                bail!(UnsupportedDtype, "Reduction op not supported for dtype {input_dtype:#?}");
            }

            fn supports_empty(&self) -> bool {
                $support_empty
            }
        }
    };
}
// pub(crate) use {define_reduction_op, define_reduction_op_kernel};

define_reduction_op!(
    /// Reduces one or more axes by taking the maximum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the input dtype.
    ///
    /// For **float** types, `NaN` values are ignored: if at least one non-`NaN` value
    /// is present, the result is the maximum of the non-`NaN` values. If all elements
    /// are `NaN`, the result is `NaN`. This deviates from the element-wise [`Maximum`](crate::ops::Maximum)
    /// op (which propagates `NaN`) but matches `numpy.max`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all axes → scalar
    /// let scalar = Array::compact_array(&nd)?
    ///     .max(&[0, 1], false).to_ndarray::<i32>()?;
    /// assert_eq!(scalar[[]], 6);
    ///
    /// // Reduce axis 0, keepdims=false → shape [3]
    /// let col_max = Array::compact_array(&nd)?
    ///     .max(&[0], false).to_ndarray::<i32>()?;
    /// assert_eq!(col_max.as_slice().unwrap(), &[4, 5, 6]);
    ///
    /// // Reduce axis 0, keepdims=true → shape [1, 3]
    /// let col_max_k = Array::compact_array(&nd)?
    ///     .max(&[0], true).to_ndarray::<i32>()?;
    /// assert_eq!(col_max_k.shape(), &[1, 3]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Max,
    MaxKernel,
    support_empty = false,
    |items| { items.reduce(|m, x| m.max(x)).unwrap() },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = "same"
    }
);
define_reduction_op!(
    /// Reduces one or more axes by taking the minimum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the input dtype.
    ///
    /// For **float** types, `NaN` values are ignored: if at least one non-`NaN` value
    /// is present, the result is the minimum of the non-`NaN` values. If all elements
    /// are `NaN`, the result is `NaN`. This deviates from the element-wise [`Minimum`](crate::ops::Minimum)
    /// op (which propagates `NaN`) but matches `numpy.min`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all axes → scalar
    /// let scalar = Array::compact_array(&nd)?
    ///     .min(&[0, 1], false).to_ndarray::<i32>()?;
    /// assert_eq!(scalar[[]], 1);
    ///
    /// // Reduce axis 0, keepdims=false → shape [3]
    /// let col_min = Array::compact_array(&nd)?
    ///     .min(&[0], false).to_ndarray::<i32>()?;
    /// assert_eq!(col_min.as_slice().unwrap(), &[1, 2, 3]);
    ///
    /// // Reduce axis 0, keepdims=true → shape [1, 3]
    /// let col_min_k = Array::compact_array(&nd)?
    ///     .min(&[0], true).to_ndarray::<i32>()?;
    /// assert_eq!(col_min_k.shape(), &[1, 3]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Min,
    MinKernel,
    support_empty = false,
    |items| { items.reduce(|m, x| m.min(x)).unwrap() },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = "same"
    }
);
define_reduction_op!(
    /// Reduces a single axis by returning the index of the maximum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype is `u64`.
    ///
    /// Unlike [`Max`], this op accepts only a single axis. If multiple elements share
    /// the maximum value, the index of the first occurrence is returned.
    /// For **float** types, `NaN` values are treated as less than any non-`NaN` value,
    /// so they are never selected unless all elements are `NaN`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of max along axis 1 (per row), keepdims=false → shape [2]
    /// let idx = Array::compact_array(&nd)?
    ///     .argmax(1, false).to_ndarray::<u64>()?;
    /// assert_eq!(idx.as_slice().unwrap(), &[1, 2]); // max of row 0 at col 1, row 1 at col 2
    ///
    /// // Index of max along axis 0 (per column), keepdims=false → shape [3]
    /// let col_idx = Array::compact_array(&nd)?
    ///     .argmax(0, false).to_ndarray::<u64>()?;
    /// assert_eq!(col_idx.as_slice().unwrap(), &[1, 0, 1]); // max of col 0 at row 1, col 1 at row 0, col 2 at row 1
    ///
    /// // keepdims=true → shape [2, 1]
    /// let idx_k = Array::compact_array(&nd)?
    ///     .argmax(1, true).to_ndarray::<u64>()?;
    /// assert_eq!(idx_k.shape(), &[2, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    ArgMax,
    ArgMaxKernel,
    support_empty = false,
    |items| {
        items
            .enumerate()
            .reduce(|(m_idx, m), (idx, x)| if x > m { (idx, x) } else { (m_idx, m) })
            .unwrap()
            .0 as u64
    },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = u64
    },
    single_axis = true
);
define_reduction_op!(
    /// Reduces a single axis by returning the index of the minimum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype is `u64`.
    ///
    /// Unlike [`Min`], this op accepts only a single axis. If multiple elements share
    /// the minimum value, the index of the first occurrence is returned.
    /// For **float** types, `NaN` values are treated as greater than any non-`NaN` value,
    /// so they are never selected unless all elements are `NaN`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of min along axis 1 (per row), keepdims=false → shape [2]
    /// let idx = Array::compact_array(&nd)?
    ///     .argmin(1, false).to_ndarray::<u64>()?;
    /// assert_eq!(idx.as_slice().unwrap(), &[0, 1]); // min of row 0 at col 0, row 1 at col 1
    ///
    /// // Index of min along axis 0 (per column), keepdims=false → shape [3]
    /// let col_idx = Array::compact_array(&nd)?
    ///     .argmin(0, false).to_ndarray::<u64>()?;
    /// assert_eq!(col_idx.as_slice().unwrap(), &[0, 1, 0]); // min of col 0 at row 0, col 1 at row 1, col 2 at row 0
    ///
    /// // keepdims=true → shape [2, 1]
    /// let idx_k = Array::compact_array(&nd)?
    ///     .argmin(1, true).to_ndarray::<u64>()?;
    /// assert_eq!(idx_k.shape(), &[2, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    ArgMin,
    ArgMinKernel,
    support_empty = false,
    |items| {
        items
            .enumerate()
            .reduce(|(m_idx, m), (idx, x)| if x < m { (idx, x) } else { (m_idx, m) })
            .unwrap()
            .0 as u64
    },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool],
        output = u64
    },
    single_axis = true
);
define_reduction_op!(
    /// Reduces one or more axes by summing all elements along those axes.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64`, `bool` | `u64` |
    /// | `f16`, `f32`, `f64` | `f64` |
    /// | `Complex<f32>`, `Complex<f64>` | `Complex<f64>` |
    ///
    /// The output dtype is always widened to avoid overflow on large reductions.
    /// An empty reduction (zero elements along the reduced axes) returns `0`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Sum all elements → i64
    /// let total = Array::compact_array(&nd)?
    ///     .sum(&[0, 1], false).to_ndarray::<i64>()?;
    /// assert_eq!(total[[]], 21);
    ///
    /// // Sum along axis 0, keepdims=false → shape [3]
    /// let col_sums = Array::compact_array(&nd)?
    ///     .sum(&[0], false).to_ndarray::<i64>()?;
    /// assert_eq!(col_sums.as_slice().unwrap(), &[5, 7, 9]);
    ///
    /// // Sum along axis 1, keepdims=true → shape [2, 1]
    /// let row_sums = Array::compact_array(&nd)?
    ///     .sum(&[1], true).to_ndarray::<i64>()?;
    /// assert_eq!(row_sums.shape(), &[2, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Sum,
    SumKernel,
    support_empty = true,
    |items| { items.fold(crate::ops::astype::cast(0), |m, x| m + crate::ops::astype::cast_as(x, &m)) },
    types = {
        [i8, i16, i32, i64] => i64,
        [u8, u16, u32, u64, bool] => u64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => (Complex::<f64>),
    }
);
define_reduction_op!(
    /// Reduces one or more axes by multiplying all elements along those axes.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64` | `u64` |
    /// | `f16`, `f32`, `f64` | `f64` |
    /// | `Complex<f32>`, `Complex<f64>` | `Complex<f64>` |
    ///
    /// Note: `bool` is not supported. The output dtype is always widened.
    /// An empty reduction (zero elements along the reduced axes) returns `1`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Product of all elements → i64
    /// let total = Array::compact_array(&nd)?
    ///     .product(&[0, 1], false).to_ndarray::<i64>()?;
    /// assert_eq!(total[[]], 720);
    ///
    /// // Product along axis 0, keepdims=false → shape [3]
    /// let col_products = Array::compact_array(&nd)?
    ///     .product(&[0], false).to_ndarray::<i64>()?;
    /// assert_eq!(col_products.as_slice().unwrap(), &[4, 10, 18]);
    ///
    /// // Product along axis 1, keepdims=true → shape [2, 1]
    /// let row_products = Array::compact_array(&nd)?
    ///     .product(&[1], true).to_ndarray::<i64>()?;
    /// assert_eq!(row_products.shape(), &[2, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Product,
    ProductKernel,
    support_empty = true,
    |items| { items.fold(crate::ops::astype::cast(1), |m, x| m * crate::ops::astype::cast_as(x, &m)) },
    types = {
        [i8, i16, i32, i64] => i64,
        [u8, u16, u32, u64] => u64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => (Complex::<f64>),
    }
);
define_reduction_op!(
    /// Reduces one or more axes by computing the arithmetic mean.
    ///
    /// Supported dtypes: all numeric types and `bool`. Output dtype is `f64` for all
    /// scalar inputs; `Complex<f64>` for `Complex<f32>` and `Complex<f64>` inputs.
    ///
    /// Reducing an empty slice (zero elements) panics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Mean of all elements → f64
    /// let total = Array::compact_array(&nd)?
    ///     .mean(&[0, 1], false).to_ndarray::<f64>()?;
    /// assert_eq!(total[[]], 3.5);
    ///
    /// // Mean along axis 0, keepdims=false → shape [3]
    /// let col_means = Array::compact_array(&nd)?
    ///     .mean(&[0], false).to_ndarray::<f64>()?;
    /// assert_eq!(col_means.as_slice().unwrap(), &[2.5, 3.5, 4.5]);
    ///
    /// // Mean along axis 0, keepdims=true → shape [1, 3]
    /// let col_means_k = Array::compact_array(&nd)?
    ///     .mean(&[0], true).to_ndarray::<f64>()?;
    /// assert_eq!(col_means_k.shape(), &[1, 3]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Mean,
    MeanKernel,
    support_empty = false,
    |items| {{
        let (size, size_high) = items.size_hint();
        assert_eq!(Some(size), size_high);
        assert!(size > 0);
        let sum = items.fold(crate::ops::astype::cast::<_, ReductionType>(0), |m, x| m + crate::ops::astype::cast_as(x, &m));
        sum / size as f64
     }},
    types = {
        [i8, i16, i32, i64] => f64,
        [u8, u16, u32, u64] => f64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => (Complex::<f64>),
        [bool] => f64,
    }
);
define_reduction_op!(
    /// Reduces one or more axes by computing the variance.
    ///
    /// Supported dtypes: all integer and float types, `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `f64`. For complex inputs the result is the real-valued variance
    /// `E[|x - mean|²]`.
    ///
    /// The `ddof` parameter (delta degrees of freedom) adjusts the divisor: the variance
    /// is computed as `sum((x - mean)²) / (n - ddof)`. Use `ddof=0` for the population
    /// variance and `ddof=1` for the sample variance (Bessel's correction). If
    /// `n - ddof <= 0`, the result is `NaN`.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population variance (ddof=0) of all elements → f64
    /// let var_all = Array::compact_array(&nd)?
    ///     .var(&[0, 1], false, 0.0).to_ndarray::<f64>()?;
    /// assert!((var_all[[]] - 2.9167).abs() < 0.001);
    ///
    /// // Sample variance (ddof=1) along axis 0, keepdims=false → shape [3]
    /// let col_vars = Array::compact_array(&nd)?
    ///     .var(&[0], false, 1.0).to_ndarray::<f64>()?;
    /// assert_eq!(col_vars.as_slice().unwrap(), &[4.5, 4.5, 4.5]);
    ///
    /// // Population variance along axis 0, keepdims=true → shape [1, 3]
    /// let col_vars_k = Array::compact_array(&nd)?
    ///     .var(&[0], true, 0.0).to_ndarray::<f64>()?;
    /// assert_eq!(col_vars_k.shape(), &[1, 3]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Variance,
    VarianceKernel,
    support_empty = false,
    |items, ddof: f64| { variance_impl(items, ddof) },
    types = {
        [i8, i16, i32, i64] => f64,
        [u8, u16, u32, u64] => f64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => f64,
    }
);
define_reduction_op!(
    /// Reduces one or more axes by computing the standard deviation.
    ///
    /// Supported dtypes: all integer and float types, `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `f64`. For complex inputs the result is the real-valued standard
    /// deviation `sqrt(E[|x - mean|²])`.
    ///
    /// Equivalent to `sqrt(variance)`. The `ddof` parameter has the same meaning as in
    /// [`Variance`]: use `ddof=0` for population std and `ddof=1` for sample std.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population std (ddof=0) of all elements → f64
    /// let std_all = Array::compact_array(&nd)?
    ///     .std(&[0, 1], false, 0.0).to_ndarray::<f64>()?;
    /// assert!((std_all[[]] - 1.7078).abs() < 0.001);
    ///
    /// // Sample std (ddof=1) along axis 0, keepdims=false → shape [3]
    /// let col_stds = Array::compact_array(&nd)?
    ///     .std(&[0], false, 1.0).to_ndarray::<f64>()?;
    /// assert!((col_stds[[0]] - 2.1213).abs() < 0.001);
    ///
    /// // Population std along axis 0, keepdims=true → shape [1, 3]
    /// let col_stds_k = Array::compact_array(&nd)?
    ///     .std(&[0], true, 0.0).to_ndarray::<f64>()?;
    /// assert_eq!(col_stds_k.shape(), &[1, 3]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    StandardDeviation,
    StandardDeviationKernel,
    support_empty = false,
    |items, ddof: f64| { variance_impl(items, ddof).sqrt() },
    types = {
        [i8, i16, i32, i64] => f64,
        [u8, u16, u32, u64] => f64,
        [f16, f32, f64] => f64,
        [(Complex<f32>), (Complex<f64>)] => f64,
    }
);
fn variance_impl<T>(items: impl Iterator<Item = T>, ddof: f64) -> f64
where
    T: VarianceImpl,
    i32: crate::ops::astype::Cast<T::MeanType>,
    T: crate::ops::astype::Cast<T::MeanType>,
    T::MeanType: core::ops::Sub<T::MeanType, Output = T::MeanType>
        + core::ops::Div<f64, Output = T::MeanType>
        + core::ops::AddAssign<T::MeanType>
        + Copy,
{
    let mut mean: T::MeanType = crate::ops::astype::cast(0);
    let mut m2 = 0.0_f64;
    let mut n = 0_u64;

    for x in items {
        let x: T::MeanType = crate::ops::astype::cast(x);
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        let delta2 = x - mean;
        m2 += T::update_m2(delta, delta2);
    }

    let denom = n as f64 - ddof;
    if denom <= 0.0 {
        f64::NAN
    } else {
        m2 / denom
    }
}
trait VarianceImpl {
    type MeanType;
    fn update_m2(delta: Self::MeanType, delta2: Self::MeanType) -> f64;
}
macro_rules! impl_num_variance_impl {
    ($ty:ty) => {
        impl VarianceImpl for $ty {
            type MeanType = f64;
            fn update_m2(delta: f64, delta2: f64) -> f64 {
                delta * delta2
            }
        }
    };
}
macro_rules! impl_complex_variance_impl {
    ($f_ty:ty) => {
        impl VarianceImpl for Complex<$f_ty> {
            type MeanType = Complex<f64>;
            fn update_m2(delta: Complex<f64>, delta2: Complex<f64>) -> f64 {
                delta.re * delta2.re + delta.im * delta2.im
            }
        }
    };
}
impl_num_variance_impl!(i8);
impl_num_variance_impl!(i16);
impl_num_variance_impl!(i32);
impl_num_variance_impl!(i64);
impl_num_variance_impl!(u8);
impl_num_variance_impl!(u16);
impl_num_variance_impl!(u32);
impl_num_variance_impl!(u64);
#[cfg(feature = "half")]
impl_num_variance_impl!(f16);
impl_num_variance_impl!(f32);
impl_num_variance_impl!(f64);
impl_complex_variance_impl!(f32);
impl_complex_variance_impl!(f64);
impl_num_variance_impl!(bool);

define_reduction_op!(
    /// Reduces one or more axes by testing whether all elements are truthy.
    ///
    /// Supported dtypes: all numeric types, `bool`, `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `bool`.
    ///
    /// Each element is cast to `bool` before reduction (zero → `false`, any non-zero
    /// → `true`; for `bool` this is the identity; for complex, non-zero means at least
    /// one component is non-zero). Returns `true` only when every element is truthy.
    /// An empty reduction returns `true`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 0, 3], [4, 5, 6]];
    ///
    /// // All elements truthy? → false (contains a zero)
    /// let all_true = Array::compact_array(&nd)?
    ///     .all(&[0, 1], false).to_ndarray::<bool>()?;
    /// assert_eq!(all_true[[]], false);
    ///
    /// // All truthy along axis 0, keepdims=false → shape [3]
    /// let col_all = Array::compact_array(&nd)?
    ///     .all(&[0], false).to_ndarray::<bool>()?;
    /// assert_eq!(col_all.as_slice().unwrap(), &[true, false, true]);
    ///
    /// // All truthy along axis 1, keepdims=true → shape [2, 1]
    /// let row_all = Array::compact_array(&nd)?
    ///     .all(&[1], true).to_ndarray::<bool>()?;
    /// assert_eq!(row_all.shape(), &[2, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    All,
    AllKernel,
    support_empty = true,
    |items| { items.fold(true, |m, x| m && crate::ops::astype::cast::<_, bool>(x)) },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
        output = bool
    }
);
define_reduction_op!(
    /// Reduces one or more axes by testing whether any element is truthy.
    ///
    /// Supported dtypes: all numeric types, `bool`, `Complex<f32>`, `Complex<f64>`.
    /// Output dtype is `bool`.
    ///
    /// Each element is cast to `bool` before reduction (zero → `false`, any non-zero
    /// → `true`; for `bool` this is the identity; for complex, non-zero means at least
    /// one component is non-zero). Returns `true` when at least one element is truthy.
    /// An empty reduction returns `false`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// use ndarray::array;
    ///
    /// let nd = array![[0i32, 0, 0], [4, 5, 6]];
    ///
    /// // Any element truthy? → true
    /// let any_true = Array::compact_array(&nd)?
    ///     .any(&[0, 1], false).to_ndarray::<bool>()?;
    /// assert_eq!(any_true[[]], true);
    ///
    /// // Any truthy along axis 0, keepdims=false → shape [3]
    /// let col_any = Array::compact_array(&nd)?
    ///     .any(&[0], false).to_ndarray::<bool>()?;
    /// assert_eq!(col_any.as_slice().unwrap(), &[true, true, true]);
    ///
    /// // Any truthy along axis 1, keepdims=true → shape [2, 1]
    /// let row_any = Array::compact_array(&nd)?
    ///     .any(&[1], true).to_ndarray::<bool>()?;
    /// assert_eq!(row_any.shape(), &[2, 1]);
    /// # Ok::<(), zix::Error>(())
    /// ```
    Any,
    AnyKernel,
    support_empty = true,
    |items| { items.fold(false, |m, x| m || crate::ops::astype::cast::<_, bool>(x)) },
    types = {
        input = [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>), bool],
        output = bool
    }
);

macro_rules! define_array_reduction_method {
    ($op:ident : $Name:ident, single_axis = true $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op(self, axis: usize, keepdims: bool $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Name<S>> {
            let op = $Name::new(self, axis, keepdims $($(, $extra_arg)*)?).unwrap();
            crate::Array::from_storage(op)
        }
    };

    ($op:ident : $Name:ident $(, extra_args = ($($extra_arg:ident : $extra_ty:ty),*))?) => {
        #[doc = concat!("Applies the [`", stringify!($Name), "`] operation, see the op struct docs for details.")]
        #[track_caller]
        pub fn $op(self, axes: &[usize], keepdims: bool $($(, $extra_arg: $extra_ty)*)?) -> crate::Array<$Name<S>> {
            let op = $Name::new(self, axes, keepdims $($(, $extra_arg)*)?).unwrap();
            crate::Array::from_storage(op)
        }
    };
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(max: Max);
    define_array_reduction_method!(min: Min);
    define_array_reduction_method!(argmax: ArgMax, single_axis = true);
    define_array_reduction_method!(argmin: ArgMin, single_axis = true);
    define_array_reduction_method!(sum: Sum);
    define_array_reduction_method!(product: Product);
    define_array_reduction_method!(mean: Mean);
    define_array_reduction_method!(var: Variance, extra_args = (ddof: f64));
    define_array_reduction_method!(std: StandardDeviation, extra_args = (ddof: f64));
    define_array_reduction_method!(all: All);
    define_array_reduction_method!(any: Any);
}

#[cfg(test)]
pub(crate) mod tests {
    use ::std::rc::Rc;

    use ndarray::{array, ArrayD};

    #[cfg(feature = "half")]
    use crate::dtype::f16;
    #[cfg(feature = "num-complex")]
    use crate::dtype::Complex;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::dtype::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::dtype::Complex<f64>;

    use crate::{array::Array, storage::Compact};

    use proptest::prelude::*;

    pub(crate) fn axis_strategy(ndim: usize) -> impl proptest::strategy::Strategy<Value = usize> {
        0..ndim
    }

    pub(crate) fn axes_strategy(ndim: usize) -> proptest::strategy::BoxedStrategy<Vec<usize>> {
        if ndim == 0 {
            return proptest::strategy::Just(vec![]).boxed();
        }
        let axis_strategy = axis_strategy(ndim).prop_map(|axis| vec![axis]);
        let multi_axes_strategy = prop::collection::vec(0..ndim, 1..=ndim).prop_map(|mut axes| {
            axes.sort_unstable();
            axes.dedup();
            axes
        });
        prop::strategy::Union::new_weighted(vec![
            (3, axis_strategy.boxed()),
            (1, multi_axes_strategy.boxed()),
        ])
        .boxed()
    }

    fn reduction_shape_strategy() -> impl proptest::strategy::Strategy<Value = Vec<usize>> {
        prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=100, 1)),
            (2, proptest::collection::vec(100..=1000, 1)),
            // 2D
            (8, proptest::collection::vec(1..=16, 2)),
            (2, proptest::collection::vec(16..=37, 2)),
            // 3D
            (5, proptest::collection::vec(1..=12, 3)),
            // 4D
            (5, proptest::collection::vec(1..=8, 4)),
            // Many dims
            (3, proptest::collection::vec(1..=4, 1..=8)),
        ])
    }

    pub(crate) fn carray_strategy_for_reduction<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<Array<Compact>>, Vec<usize>, bool)>
    {
        let shape = reduction_shape_strategy();
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axes = axes_strategy(nd.ndim());
                let keepdims = proptest::bool::ANY;
                (Just(nd), Just(za), axes, keepdims)
            })
    }

    pub(crate) fn carray_strategy_for_reduction_single_axis<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<Array<Compact>>, usize, bool)>
    {
        let shape = reduction_shape_strategy();
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axis = axis_strategy(nd.ndim());
                let keepdims = proptest::bool::ANY;
                (Just(nd), Just(za), axis, keepdims)
            })
    }

    pub(crate) fn carray_strategy_for_reduction_small<T: crate::util::ScalarStrategy>(
        elem_strategy: impl proptest::strategy::Strategy<Value = T> + Clone,
    ) -> impl proptest::strategy::Strategy<Value = (ArrayD<T>, Rc<Array<Compact>>, Vec<usize>, bool)>
    {
        let shape = prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=4, 1)),
            // 2D
            (8, proptest::collection::vec(1..=2, 2)),
        ]);
        let array = crate::util::carray_strategy_from_shape::<T>(shape, elem_strategy);
        array
            .prop_map(|(nd, za)| (nd, Rc::new(za)))
            .prop_flat_map(|(nd, za)| {
                let axes = axes_strategy(nd.ndim());
                let keepdims = proptest::bool::ANY;
                (Just(nd), Just(za), axes, keepdims)
            })
    }

    macro_rules! test_reduction_dtype {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axes, keepdims) in crate::ops::reduction::tests::carray_strategy_for_reduction::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(&axes, keepdims);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &axes, keepdims,
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };

        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident,
            small_data = true
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axes, keepdims) in crate::ops::reduction::tests::carray_strategy_for_reduction_small::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(&axes, keepdims);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &axes, keepdims,
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };

        (
            $op_method:ident,
            single_axis = true,
            |$items:ident| { $body:expr },
            $dtype:ident,
            $strategy:ident
        ) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<$op_method _ $dtype>](
                        (nd, za, axis, keepdims) in crate::ops::reduction::tests::carray_strategy_for_reduction_single_axis::<$dtype>(
                            <$dtype as crate::util::ScalarStrategy>::$strategy()
                        )
                    ) {
                        let result = (*za).as_ref().$op_method(axis, keepdims);
                        let expected = crate::ops::reduction::tests::ndarray_reduce(
                            &nd, &[axis], keepdims,
                            |arr| {
                                let $items = arr.iter().cloned();
                                $body
                            }
                        );
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };

    }

    macro_rules! test_reduction {
        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };

        (
            $op_method:ident,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*,
            small_data = true
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                |$items| { $body },
                $dtype,
                $strategy,
                small_data = true
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy,
                    small_data = true
                );
            )+)*
        };

        (
            $op_method:ident,
            single_axis = true,
            |$items:ident| { $body:expr },
            [$($dtype:ident),+ $(,)?], $strategy:ident
            $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+ $(,)?])*
        ) => {
            $(crate::ops::reduction::tests::test_reduction_dtype!(
                $op_method,
                single_axis = true,
                |$items| { $body },
                $dtype,
                $strategy
            );)+
            $($(
                #[cfg($cfg)]
                crate::ops::reduction::tests::test_reduction_dtype!(
                    $op_method,
                    single_axis = true,
                    |$items| { $body },
                    $cfg_dtype,
                    $strategy
                );
            )+)*
        };
    }

    #[allow(unused_imports)]
    pub(crate) use {test_reduction, test_reduction_dtype};

    test_reduction!(
        max,
        |items| { items.reduce(|m, x| if x > m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_reduction!(
        min,
        |items| { items.reduce(|m, x| if x < m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_reduction!(
        argmax,
        single_axis = true,
        |items| {
            items
                .enumerate()
                .reduce(|(m_i, m), (i, x)| if x > m { (i, x) } else { (m_i, m) })
                .unwrap()
                .0 as u64
        },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_reduction!(
        argmin,
        single_axis = true,
        |items| {
            items
                .enumerate()
                .reduce(|(m_i, m), (i, x)| if x < m { (i, x) } else { (m_i, m) })
                .unwrap()
                .0 as u64
        },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_reduction!(
        sum,
        |items| { items.fold(0u64, |m, x| m + x as u64) },
        [u8, u16, u32, u64, bool],
        op_safe_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0i64, |m, x| m + x as i64) },
        [i8, i16, i32, i64],
        op_safe_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0.0f64, |m, x| m + crate::ops::astype::cast::<_, f64>(x)) },
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        sum,
        |items| {
            items.fold(Complex::<f64>::default(), |m, x| {
                m + crate::ops::astype::cast::<_, Complex<f64>>(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    test_reduction!(
        product,
        |items| { items.fold(1u64, |m, x| m * x as u64) },
        [u8, u16, u32, u64],
        op_safe_strategy,
        small_data = true
    );
    test_reduction!(
        product,
        |items| { items.fold(1i64, |m, x| m * x as i64) },
        [i8, i16, i32, i64],
        op_safe_strategy,
        small_data = true
    );
    test_reduction!(
        product,
        |items| { items.fold(1.0f64, |m, x| m * crate::ops::astype::cast::<_, f64>(x)) },
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        small_data = true
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        product,
        |items| {
            items.fold(
                Complex::<f64>::default() + Complex::<f64>::new(1.0, 1.0),
                |m, x| m * crate::ops::astype::cast::<_, Complex<f64>>(x),
            )
        },
        [complex_f32, complex_f64],
        op_safe_strategy,
        small_data = true
    );
    // mean
    test_reduction!(
        mean,
        |items| {
            {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for x in items {
                    sum += crate::ops::astype::cast::<_, f64>(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        mean,
        |items| {
            {
                let mut sum: Complex<f64> = Complex::default();
                let mut count: usize = 0;
                for x in items {
                    sum += crate::ops::astype::cast::<_, Complex<f64>>(x);
                    count += 1;
                }
                sum / Complex::new(count as f64, 0.0)
            }
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    #[test]
    fn variance() {
        let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let var_all = a
            .as_ref()
            .var(&[0, 1], false, 0.0)
            .to_ndarray::<f64>()
            .unwrap();
        assert!((var_all[[]] - 2.9166).abs() < 0.001);
        let var_col = a
            .as_ref()
            .var(&[0], false, 0.0)
            .to_ndarray::<f64>()
            .unwrap();
        assert!((var_col[[0]] - 2.25).abs() < 0.001);
        let var_row = a.as_ref().var(&[1], true, 0.0).to_ndarray::<f64>().unwrap();
        assert!((var_row[[0, 0]] - 0.6666).abs() < 0.001);
    }
    #[test]
    fn std() {
        let a = Array::compact_array(&array![[7i32, 8, 9], [4, 5, 6]]).unwrap();
        let std_all = a
            .as_ref()
            .std(&[0, 1], false, 0.0)
            .to_ndarray::<f64>()
            .unwrap();
        assert!((std_all[[]] - 1.7078).abs() < 0.001);
        let std_col = a
            .as_ref()
            .std(&[0], false, 0.0)
            .to_ndarray::<f64>()
            .unwrap();
        assert!((std_col[[0]] - 1.5).abs() < 0.001);
        let std_row = a.as_ref().std(&[1], true, 0.0).to_ndarray::<f64>().unwrap();
        assert!((std_row[[0, 0]] - 0.8164).abs() < 0.001);
    }
    test_reduction!(
        all,
        |items| { items.fold(true, |m, x| m && crate::ops::astype::cast::<_, bool>(x)) },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        logical_op_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_reduction!(
        any,
        |items| { items.fold(false, |m, x| m || crate::ops::astype::cast::<_, bool>(x)) },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        logical_op_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );

    fn ndarray_reduce<'a, S, D, O>(
        array: &'a ndarray::ArrayBase<S, D>,
        axes: &[usize],
        keepdims: bool,
        f: impl Fn(&ndarray::ArrayViewD<'a, S::Elem>) -> O,
    ) -> ndarray::ArrayD<O>
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        // Output shape = original with reduction axes removed
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();

        let out_shape: Vec<usize> = array
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &s)| s)
            .collect();

        // Iterate over kept axes, each view spans the reduction axes → f collapses it to scalar
        let values: Vec<O> = ndarray_reduction_iter(array, &axes)
            .map(|(_, view)| f(&view))
            .collect();

        let mut result =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), values).unwrap();
        if keepdims {
            // Insert singleton dimensions at the reduced axes
            let mut final_shape = result.shape().to_vec();
            for &ax in axes.iter() {
                final_shape.insert(ax, 1);
            }
            result = result.into_shape_with_order(final_shape).unwrap();
        }
        result
    }

    /// Iterates over all index combinations of the **kept** axes (i.e. axes NOT in `axes`),
    /// yielding for each combination the multi-index into the kept axes and a view spanning
    /// the reduction axes.
    fn ndarray_reduction_iter<'a, S, D>(
        array: &'a ndarray::ArrayBase<S, D>,
        axes: &[usize],
    ) -> impl Iterator<Item = (Vec<usize>, ndarray::ArrayViewD<'a, S::Elem>)> + 'a
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
    {
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();

        // Kept axes = all axes not being reduced
        let ndim = array.ndim();
        let kept_axes: Vec<usize> = (0..ndim).filter(|i| !axes.contains(i)).collect();

        // Shape of the kept axes — this is what we iterate over
        let kept_shape: Vec<usize> = kept_axes.iter().map(|&ax| array.shape()[ax]).collect();
        let total: usize = kept_shape.iter().product();

        (0..total).map(move |flat_idx| {
            // Convert flat index to multi-index over the kept axes
            let mut remaining = flat_idx;
            let mut kept_indices: Vec<usize> = Vec::with_capacity(kept_axes.len());
            for &dim_size in kept_shape.iter().rev() {
                kept_indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            kept_indices.reverse();

            // Fix each kept axis to its index, remove in descending order.
            // We remove kept axes (which are the non-reduction axes), leaving
            // a view over the reduction axes.
            let mut view = array.view().into_dyn();

            // We must track axis offset: as we remove axes, remaining axis
            // indices shift down. Process kept axes in descending order.
            let mut pairs: Vec<(usize, usize)> = kept_axes
                .iter()
                .copied()
                .zip(kept_indices.iter().copied())
                .collect();
            pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));

            for (ax, idx) in &pairs {
                view = view.index_axis_move(ndarray::Axis(*ax), *idx);
            }

            (kept_indices, view)
        })
    }

    mod ndarray_reduce_tests {
        use super::{ndarray_reduce, ndarray_reduction_iter};
        use ndarray::*;

        #[cfg(test)]
        mod tests {
            use super::*;
            use ndarray::{array, Array, ArrayD, IxDyn};

            #[test]
            fn single_axis_0() {
                // Shape [2, 3], reduce axis 0 → 3 views of shape [2]
                let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0]).collect();

                assert_eq!(views.len(), 3);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[2]);
                }

                // kept axis is 1, so indices are [0], [1], [2]
                // view[0] = a[:, 0] = [1, 4]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(views[0].1, array![1, 4].into_dyn());
                // view[1] = a[:, 1] = [2, 5]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(views[1].1, array![2, 5].into_dyn());
                // view[2] = a[:, 2] = [3, 6]
                assert_eq!(views[2].0, vec![2]);
                assert_eq!(views[2].1, array![3, 6].into_dyn());
            }

            #[test]
            fn single_axis_1() {
                // Shape [2, 3], reduce axis 1 → 2 views of shape [3]
                let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[1]).collect();

                assert_eq!(views.len(), 2);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[3]);
                }

                // kept axis is 0, so indices are [0], [1]
                // view[0] = a[0, :] = [1, 2, 3]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(views[0].1, array![1, 2, 3].into_dyn());
                // view[1] = a[1, :] = [4, 5, 6]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(views[1].1, array![4, 5, 6].into_dyn());
            }

            #[test]
            fn multi_axis_3d() {
                // Shape [2, 3, 4], reduce axes [0, 2] → 3 views of shape [2, 4]
                let a = Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect()).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0, 2]).collect();

                assert_eq!(views.len(), 3);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[2, 4]);
                }

                // kept axis is 1, indices are [0], [1], [2]
                // view[0] = a[:, 0, :] = [[0,1,2,3],[12,13,14,15]]
                assert_eq!(views[0].0, vec![0]);
                assert_eq!(
                    views[0].1,
                    array![[0, 1, 2, 3], [12, 13, 14, 15]].into_dyn()
                );
                // view[1] = a[:, 1, :] = [[4,5,6,7],[16,17,18,19]]
                assert_eq!(views[1].0, vec![1]);
                assert_eq!(
                    views[1].1,
                    array![[4, 5, 6, 7], [16, 17, 18, 19]].into_dyn()
                );
                // view[2] = a[:, 2, :] = [[8,9,10,11],[20,21,22,23]]
                assert_eq!(views[2].0, vec![2]);
                assert_eq!(
                    views[2].1,
                    array![[8, 9, 10, 11], [20, 21, 22, 23]].into_dyn()
                );
            }

            #[test]
            fn reduce_all_axes() {
                // Shape [2, 3], reduce both → 1 view of shape [2, 3] (no kept axes)
                let a =
                    Array::from_shape_vec(IxDyn(&[2, 3]), vec![10, 20, 30, 40, 50, 60]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0, 1]).collect();

                assert_eq!(views.len(), 1);
                assert_eq!(views[0].0, Vec::<usize>::new());
                assert_eq!(views[0].1, array![[10, 20, 30], [40, 50, 60]].into_dyn());
            }

            #[test]
            fn no_axes_returns_scalar_views() {
                // Reduce no axes → 6 scalar views (iterate over everything)
                let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[]).collect();

                assert_eq!(views.len(), 6);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[] as &[usize]);
                }

                assert_eq!(views[0].0, vec![0, 0]);
                assert_eq!(*views[0].1.first().unwrap(), 1);
                assert_eq!(views[1].0, vec![0, 1]);
                assert_eq!(*views[1].1.first().unwrap(), 2);
                assert_eq!(views[5].0, vec![1, 2]);
                assert_eq!(*views[5].1.first().unwrap(), 6);
            }

            #[test]
            fn axes_order_independent() {
                // [0, 2] and [2, 0] should yield identical results
                let a = Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect()).unwrap();

                let v1: Vec<_> = ndarray_reduction_iter(&a, &[0, 2]).collect();
                let v2: Vec<_> = ndarray_reduction_iter(&a, &[2, 0]).collect();

                assert_eq!(v1.len(), v2.len());
                for ((idx1, view1), (idx2, view2)) in v1.iter().zip(v2.iter()) {
                    assert_eq!(idx1, idx2);
                    assert_eq!(view1, view2);
                }
            }

            #[test]
            fn dim_1_reduce_axis_0() {
                // Shape [5], reduce axis 0 → 1 view of shape [5] (no kept axes)
                let a = Array::from_shape_vec(IxDyn(&[5]), vec![10, 20, 30, 40, 50]).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[0]).collect();

                assert_eq!(views.len(), 1);
                assert_eq!(views[0].0, Vec::<usize>::new());
                assert_eq!(views[0].1, array![10, 20, 30, 40, 50].into_dyn());
            }

            #[test]
            fn reduce_middle_axis() {
                // Shape [2, 3, 4], reduce axis 1 → 2*4=8 views of shape [3]
                let a = Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect()).unwrap();
                let views: Vec<_> = ndarray_reduction_iter(&a, &[1]).collect();

                assert_eq!(views.len(), 8);
                for (_, v) in &views {
                    assert_eq!(v.shape(), &[3]);
                }

                // kept axes are [0, 2]
                // view[0]: kept=[0,0] → a[0, :, 0] = [0, 4, 8]
                assert_eq!(views[0].0, vec![0, 0]);
                assert_eq!(views[0].1, array![0, 4, 8].into_dyn());
                // view[3]: kept=[0,3] → a[0, :, 3] = [3, 7, 11]
                assert_eq!(views[3].0, vec![0, 3]);
                assert_eq!(views[3].1, array![3, 7, 11].into_dyn());
                // view[4]: kept=[1,0] → a[1, :, 0] = [12, 16, 20]
                assert_eq!(views[4].0, vec![1, 0]);
                assert_eq!(views[4].1, array![12, 16, 20].into_dyn());
                // view[7]: kept=[1,3] → a[1, :, 3] = [15, 19, 23]
                assert_eq!(views[7].0, vec![1, 3]);
                assert_eq!(views[7].1, array![15, 19, 23].into_dyn());
            }

            // --- ndarray_reduce tests ---

            #[test]
            fn reduce_sum_axis_0() {
                // np.sum(a, axis=0) for shape [2, 3]
                let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[0], false, |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[3]);
                assert_eq!(result, array![5, 7, 9].into_dyn());
            }

            #[test]
            fn reduce_sum_axis_1() {
                // np.sum(a, axis=1) for shape [2, 3]
                let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[1], false, |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[2]);
                assert_eq!(result, array![6, 15].into_dyn());
            }

            #[test]
            fn reduce_sum_multi_axis() {
                // np.sum(a, axis=(0, 2)) for shape [2, 3, 4]
                let a: ArrayD<i32> =
                    Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect()).unwrap();
                let result = ndarray_reduce(&a, &[0, 2], false, |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[3]);
                // axis 1 index 0: sum of a[:, 0, :] = sum(0..4) + sum(12..16) = 6 + 54 = 60
                // axis 1 index 1: sum of a[:, 1, :] = sum(4..8) + sum(16..20) = 22 + 70 = 92
                // axis 1 index 2: sum of a[:, 2, :] = sum(8..12) + sum(20..24) = 38 + 86 = 124
                assert_eq!(result, array![60, 92, 124].into_dyn());
            }

            #[test]
            fn reduce_all_axes_to_scalar() {
                // np.sum(a) — reduce everything
                let a: ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>, i32> =
                    Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[0, 1], false, |v| v.iter().sum::<i32>());

                assert_eq!(result.shape(), &[] as &[usize]);
                assert_eq!(*result.first().unwrap(), 21);
            }

            #[test]
            fn reduce_no_axes_identity() {
                // Reducing no axes → same shape, each element passed through f
                let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
                let result = ndarray_reduce(&a, &[], false, |v| *v.first().unwrap());

                assert_eq!(result.shape(), &[2, 3]);
                assert_eq!(result, array![[1, 2, 3], [4, 5, 6]].into_dyn());
            }

            #[test]
            fn reduce_max_axis() {
                // np.max(a, axis=0)
                let a = Array::from_shape_vec(IxDyn(&[3, 2]), vec![5, 1, 3, 8, 7, 2]).unwrap();
                let result = ndarray_reduce(&a, &[0], false, |v| *v.iter().max().unwrap());

                assert_eq!(result.shape(), &[2]);
                assert_eq!(result, array![7, 8].into_dyn());
            }
        }
    }
}

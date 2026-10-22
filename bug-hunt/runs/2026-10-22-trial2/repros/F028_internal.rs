// F028 internal repro: paste into the #[cfg(test)] mod tests of jix/src/ops/reduction.rs
// (ReductionOp / ArgMaxKernel / ArgMinKernel are pub(crate), unreachable from jix/tests).
// Run: cargo test --manifest-path jix/Cargo.toml --all-features --lib f028_ -- --nocapture
// Verified on a scratch copy of the crate at commit d1fe11d: both tests FAIL (bug present).

    #[test]
    fn f028_multi_dim_argmax_with_reduced_dims_permuted_in_memory() {
        use crate::{Array, ArrayParams, Ty};
        use ndarray::array;

        // Shape (r0, r1, k) = [2, 2, 3]; reduce r0 and r1, keep k. Byte strides put k innermost,
        // then r0, then r1 outermost: the reduced dims are in the reverse of their logical order.
        // Stream position of (r0, r1) is r0 * 2 + r1.
        fn run<T: Copy + crate::dtype::Dtyped, K>(
            strides: [usize; 3],
            vals: impl Fn(usize, usize, usize) -> T,
            kernel: K,
        ) -> ndarray::Array1<u64>
        where
            K: super::ReductionOpKernel<T, Output = u64>,
            super::ReductionOp<crate::storage::Plain<&'static (), Ty<T>>, K>:
                crate::ArrayStorage<ElementType = Ty<u64>>,
        {
            let itemsize = size_of::<T>();
            let mut backing = vec![0u64; 64]; // 8-aligned, 512 bytes
            let bytes = backing.as_mut_ptr().cast::<u8>();
            for r0 in 0..2 {
                for r1 in 0..2 {
                    for k in 0..3 {
                        let off = r0 * strides[0] + r1 * strides[1] + k * strides[2];
                        assert!(off + itemsize <= 512);
                        unsafe { bytes.add(off).cast::<T>().write_unaligned(vals(r0, r1, k)) };
                    }
                }
            }
            let arr: Array<crate::storage::Plain<&'static (), Ty<T>>> = unsafe {
                Array::plain_ndarray_ptr(
                    backing.as_ptr().cast::<u8>(),
                    &[2u64, 2, 3],
                    &strides,
                    T::DTYPE,
                    ArrayParams::new(),
                )
            }
            .unwrap();
            let op = super::ReductionOp::new(arr.into_storage(), kernel, [0usize, 1].as_slice())
                .unwrap();
            let got = Array::from_storage(op).to_ndarray::<ndarray::Ix1>().unwrap();
            drop(backing);
            got
        }

        let c_order_i32 = [24usize, 12, 4];
        let permuted_i32 = [24usize, 48, 4];
        // Tied maxima / minima at stream positions 1 = (0, 1) and 2 = (1, 0): the first is 1.
        let tie = |r0: usize, r1: usize, k: usize| -> i32 {
            let p = r0 * 2 + r1;
            if p == 1 || p == 2 { 100 + k as i32 } else { k as i32 }
        };
        let tie_min = |r0: usize, r1: usize, k: usize| -> i32 { -tie(r0, r1, k) };
        // NaN at positions 1 and 2: the first NaN, 1.
        let nan = |r0: usize, r1: usize, k: usize| -> f32 {
            let p = r0 * 2 + r1;
            if p == 1 || p == 2 { f32::NAN } else { k as f32 }
        };

        let mut failures = Vec::new();
        let mut check = |name: &str, got: ndarray::Array1<u64>| {
            eprintln!("{name}: {got}");
            if got != array![1u64, 1, 1] {
                failures.push(format!("{name}: got {got}, expected [1, 1, 1]"));
            }
        };
        check("argmax tie, C order", run(c_order_i32, tie, super::ArgMaxKernel {}));
        check("argmax tie, permuted", run(permuted_i32, tie, super::ArgMaxKernel {}));
        check("argmin tie, permuted", run(permuted_i32, tie_min, super::ArgMinKernel {}));
        check("argmax NaN, C order", run(c_order_i32, nan, super::ArgMaxKernel {}));
        check("argmax NaN, permuted", run(permuted_i32, nan, super::ArgMaxKernel {}));
        check("argmin NaN, permuted", run(permuted_i32, nan, super::ArgMinKernel {}));
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn f028_multi_dim_argmax_c_order_multi_tile() {
        use crate::Array;
        // C-order [r0, r1, k] = [4, 4, 2], 2x2x2 blocks, read window 8 items -> [2, 2, 2] tiles.
        // Maxima tied at stream positions 2 = (0, 2) and 4 = (1, 0): the first is 2.
        let nd = ndarray::Array3::<i32>::from_shape_fn((4, 4, 2), |(r0, r1, k)| {
            let p = r0 * 4 + r1;
            if p == 2 || p == 4 { 100 + k as i32 } else { k as i32 }
        });
        let mut params = crate::ArrayParams::new();
        params.block_shape(&[2, 2, 2]).read_size((32, 32));
        let za = Array::compact_ndarray_with(&nd, params).unwrap();
        let op = super::ReductionOp::new(za.into_storage(), super::ArgMaxKernel {}, [0usize, 1].as_slice())
            .unwrap();
        let got = Array::from_storage(op).to_ndarray::<ndarray::Ix1>().unwrap();
        eprintln!("argmax tie, C order multi-tile: {got}");
        assert_eq!(got, ndarray::array![2u64, 2]);
    }


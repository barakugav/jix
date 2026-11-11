use crate::ops::elementwise::op1::{define_array_op1_method, define_op1};
use crate::ops::prelude::*;

pub(crate) mod traits {
    use crate::scalar::traits_util::{define_scalar_op1_trait, impl_scalar_op1};

    define_scalar_op1_trait!(
        /// Zero-bit count operation, as [`u32::count_zeros`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Equivalent to `bit_width - count_ones`. For signed integers the full bit
        /// representation (including the sign bit) is used.
        CountZeros,
        /// Count the unset bits (`0`s).
        count_zeros,
        bulk_fn: count_zeros_bulk,
    );
    impl_scalar_op1!(
        impl CountZeros for [u32] {
            type Output = u32;
            fn count_zeros = |x| x.count_zeros(),
            fn count_zeros_bulk = |x| (!x).count_ones(),
        }
    );
    impl_scalar_op1!(
        impl CountZeros for [i32] {
            type Output = u32;
            fn count_zeros = |x| x.count_zeros(),
            fn count_zeros_bulk = |x| (!x).count_ones().bitcast::<S::u32s>(),
        }
    );
    impl_scalar_op1!(
        impl CountZeros for [i8, i16, i64, u8, u16, u64] {
            type Output = u32;
            fn count_zeros = |x| x.count_zeros(),
        }
    );

    macro_rules! int_op1 {
        (
            $(#[$trait_meta:meta])* $Trait:ident,
            $(#[$fn_meta:meta])* $f:ident,
            $f_bulk:ident, $Out:ty
        ) => {
            define_scalar_op1_trait!(
                $(#[$trait_meta])* $Trait,
                $(#[$fn_meta])* $f,
                bulk_fn: $f_bulk,
            );
            impl_scalar_op1!(
                impl $Trait for [i8, i16, i32, i64, u8, u16, u32, u64] {
                    type Output = $Out;
                    fn $f = |x| x.$f(),
                }
            );
        };
    }
    int_op1!(
        /// Set-bit count operation, as [`u32::count_ones`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Also known as the population count or Hamming weight. For signed integers the
        /// bit representation (including the sign bit) is used.
        CountOnes,
        /// Count the set bits (`1`s).
        count_ones,
        count_ones_bulk,
        u32
    );
    int_op1!(
        /// Leading-zero count operation, as [`u32::leading_zeros`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Counts zeros from the most-significant bit down to (but not including) the first
        /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
        /// `0u32`).
        LeadingZeros,
        /// Count the leading zero bits.
        leading_zeros,
        leading_zeros_bulk,
        u32
    );
    int_op1!(
        /// Trailing-zero count operation, as [`u32::trailing_zeros`].
        ///
        /// Output dtype is `u32`.
        ///
        /// Counts zeros from the least-significant bit up to (but not including) the first
        /// set bit. Returns the bit width of the type for a value of zero (e.g. `32` for
        /// `0u32`).
        TrailingZeros,
        /// Count the trailing zero bits.
        trailing_zeros,
        trailing_zeros_bulk,
        u32
    );
    int_op1!(
        /// Byte-order reversal operation, as [`u32::swap_bytes`].
        ///
        /// Converts between big-endian and little-endian representation. A no-op for the
        /// single-byte `i8` and `u8`.
        SwapBytes,
        /// Reverse the byte order.
        swap_bytes,
        swap_bytes_bulk,
        Self
    );
    int_op1!(
        /// Bit-order reversal operation, as [`u32::reverse_bits`].
        ///
        /// The most-significant bit becomes the least-significant and vice versa.
        ReverseBits,
        /// Reverse the bit order.
        reverse_bits,
        reverse_bits_bulk,
        Self
    );
}

define_op1!(
    /// Counts the number of set bits (`1`s) in each element.
    ///
    /// See [`jix::scalar::CountOnes`](crate::scalar::CountOnes) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::count_ones()`](crate::Array::count_ones).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![0b00001111u8, 0b11001100u8, 0b11111111u8])?;
    /// let result = a.count_ones().to_ndarray()?;
    /// assert_eq!(result, array![4, 4, 8]);
    ///
    /// // Zero has no set bits.
    /// let b = Array::compact_ndarray(&array![0u8, 0u8])?;
    /// let result = b.count_ones().to_ndarray()?;
    /// assert_eq!(result, array![0, 0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    CountOnes,
    CountOnesKernel,
    crate::scalar::CountOnes::count_ones,
    simd: count_ones_bulk,
);
define_op1!(
    /// Counts the number of unset bits (`0`s) in each element.
    ///
    /// See [`jix::scalar::CountZeros`](crate::scalar::CountZeros) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::count_zeros()`](crate::Array::count_zeros).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0b11110000u8, 0b00001111u8, 0b11111111u8])?;
    /// let result = a.count_zeros().to_ndarray()?;
    /// assert_eq!(result, array![4, 4, 0]);
    ///
    /// // Zero has all bits unset: count_zeros == bit width.
    /// let b = Array::compact_ndarray(&array![0u8])?;
    /// let result = b.count_zeros().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 8); // u8 has 8 bits
    /// # Ok::<(), jix::Error>(())
    /// ```
    CountZeros,
    CountZerosKernel,
    crate::scalar::CountZeros::count_zeros,
    simd: count_zeros_bulk,
);
define_op1!(
    /// Counts the number of leading zero bits in each element.
    ///
    /// See [`jix::scalar::LeadingZeros`](crate::scalar::LeadingZeros) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::leading_zeros()`](crate::Array::leading_zeros).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0x00010000u32, 0x80000000u32, 0x00000001u32])?;
    /// let result = a.leading_zeros().to_ndarray()?;
    /// assert_eq!(result, array![15, 0, 31]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = Array::compact_ndarray(&array![0u32])?;
    /// let result = b.leading_zeros().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), jix::Error>(())
    /// ```
    LeadingZeros,
    LeadingZerosKernel,
    crate::scalar::LeadingZeros::leading_zeros,
    simd: leading_zeros_bulk,
);
define_op1!(
    /// Counts the number of trailing zero bits in each element.
    ///
    /// See [`jix::scalar::TrailingZeros`](crate::scalar::TrailingZeros) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::trailing_zeros()`](crate::Array::trailing_zeros).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0x00010000u32, 0x80000000u32, 0x00000001u32])?;
    /// let result = a.trailing_zeros().to_ndarray()?;
    /// assert_eq!(result, array![16, 31, 0]);
    ///
    /// // Zero returns the bit width of the type (32 for u32).
    /// let b = Array::compact_ndarray(&array![0u32])?;
    /// let result = b.trailing_zeros().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 32);
    /// # Ok::<(), jix::Error>(())
    /// ```
    TrailingZeros,
    TrailingZerosKernel,
    crate::scalar::TrailingZeros::trailing_zeros,
    simd: trailing_zeros_bulk,
);
define_op1!(
    /// Reverses the byte order of each element.
    ///
    /// See [`jix::scalar::SwapBytes`](crate::scalar::SwapBytes) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::swap_bytes()`](crate::Array::swap_bytes).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0x00FF0000u32, 0x0000FF00u32])?;
    /// let result = a.swap_bytes().to_ndarray()?;
    /// assert_eq!(result, array![0x0000FF00, 0x00FF0000]);
    ///
    /// // Classic endian-swap example.
    /// let b = Array::compact_ndarray(&array![0x12345678u32])?;
    /// let result = b.swap_bytes().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0x78563412u32);
    /// # Ok::<(), jix::Error>(())
    /// ```
    SwapBytes,
    SwapBytesKernel,
    crate::scalar::SwapBytes::swap_bytes,
    simd: swap_bytes_bulk,
);
define_op1!(
    /// Reverses the bit order of each element.
    ///
    /// See [`jix::scalar::ReverseBits`](crate::scalar::ReverseBits) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::reverse_bits()`](crate::Array::reverse_bits).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0b00000001u8, 0b10000000u8, 0b10101010u8])?;
    /// let result = a.reverse_bits().to_ndarray()?;
    /// assert_eq!(result, array![0b10000000, 0b00000001, 0b01010101]);
    ///
    /// // Reversing bits of 0 gives 0.
    /// let b = Array::compact_ndarray(&array![0u8])?;
    /// let result = b.reverse_bits().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0u8);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ReverseBits,
    ReverseBitsKernel,
    crate::scalar::ReverseBits::reverse_bits,
    simd: reverse_bits_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(count_ones: CountOnes, crate::scalar::CountOnes);
    define_array_op1_method!(count_zeros: CountZeros, crate::scalar::CountZeros);
    define_array_op1_method!(leading_zeros: LeadingZeros, crate::scalar::LeadingZeros);
    define_array_op1_method!(trailing_zeros: TrailingZeros, crate::scalar::TrailingZeros);
    define_array_op1_method!(swap_bytes: SwapBytes, crate::scalar::SwapBytes);
    define_array_op1_method!(reverse_bits: ReverseBits, crate::scalar::ReverseBits);
}

#[cfg(test)]
mod tests {
    use crate::ops::elementwise::op1::tests::test_op1;

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics, on every SIMD level of the CPU, over edge cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::{CountOnesKernel, CountZerosKernel};
        use crate::ops::elementwise::op1::Op1Kernel;
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case1 {
                ($kernel:ident, $t:ty, $f:expr) => {
                    let xs = <$t>::simd_test_values(0);
                    let what = concat!(stringify!($kernel), " ", stringify!($t));
                    assert_same_elements(
                        $kernel.apply_bulk(xs, &mut Default::default(), simd),
                        |i| $f(xs[i]),
                        what,
                    );
                };
            }
            simd.vectorize(|| {
                case1!(CountOnesKernel, i32, i32::count_ones);
                case1!(CountOnesKernel, u32, u32::count_ones);
                case1!(CountZerosKernel, i32, i32::count_zeros);
                case1!(CountZerosKernel, u32, u32::count_zeros);
            });
        }
        for_each_simd_level!(check);
    }

    test_op1!(count_ones, |a| a.count_ones(), [u8, u32], any_strategy);

    // Edge-value inputs shared by the `*_concrete` tests below: 0, MAX (all bits set), a
    // single set bit, and an alternating 0xAA bit pattern, one array per byte width.
    const EDGE_U8: [u8; 4] = [0, u8::MAX, 1, 0xAA];
    const EDGE_U16: [u16; 4] = [0, u16::MAX, 1, 0xAAAA];
    const EDGE_U32: [u32; 4] = [0, u32::MAX, 1, 0xAAAA_AAAA];
    const EDGE_U64: [u64; 4] = [0, u64::MAX, 1, 0xAAAA_AAAA_AAAA_AAAA];

    // A single unary-op assertion for one input array: compact it, apply `$method`, and check
    // against the `ndarray`-computed reference `$body`.
    macro_rules! concrete_op1_case {
        ($method:ident, |$a:ident| $body:expr, $arr:expr) => {{
            use crate::Array;
            let nd = ndarray::arr1(&$arr);
            let za = Array::compact_ndarray(&nd).unwrap();
            let expected = nd.mapv(|$a| $body);
            crate::util::assert_array_matches(&za.view().$method(), &expected);
        }};
    }

    // A `#[test]` fn running `concrete_op1_case!` for each input array in `[...]`.
    macro_rules! concrete_op1 {
        ($name:ident, $method:ident, |$a:ident| $body:expr, [$($arr:expr),+ $(,)?]) => {
            #[test]
            fn $name() {
                $(concrete_op1_case!($method, |$a| $body, $arr);)+
            }
        };
    }

    concrete_op1!(
        count_ones_concrete,
        count_ones,
        |a| a.count_ones(),
        [EDGE_U16, EDGE_U64]
    );

    concrete_op1!(
        count_zeros_concrete,
        count_zeros,
        |a| a.count_zeros(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    concrete_op1!(
        leading_zeros_concrete,
        leading_zeros,
        |a| a.leading_zeros(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    concrete_op1!(
        trailing_zeros_concrete,
        trailing_zeros,
        |a| a.trailing_zeros(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );

    // No u8 case: single-byte types don't support swap_bytes (swapping one byte is a no-op),
    // see the doc comment on `SwapBytes` above.
    concrete_op1!(
        swap_bytes_concrete,
        swap_bytes,
        |a| a.swap_bytes(),
        [EDGE_U16, EDGE_U32, EDGE_U64]
    );

    concrete_op1!(
        reverse_bits_concrete,
        reverse_bits,
        |a| a.reverse_bits(),
        [EDGE_U8, EDGE_U16, EDGE_U32, EDGE_U64]
    );
}

use crate::dtype::Dtyped;
use crate::scalar::Complex;
use crate::storage::{ArrayStorageInfo, ArrayStorageTyped};
use crate::{Array, ArrayStorage};

/// Extracts the real part of each complex element.
///
/// Supported input dtypes: `Complex<f32>`, `Complex<f64>`. Output dtype is the
/// corresponding real component type (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`).
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::real()`](crate::Array::real).
///
/// # Examples
/// ```
/// use jix::scalar::Complex;
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![
///     Complex { re: 1.0f32, im: 2.0 },
///     Complex { re: 3.0, im: -4.0 },
/// ])?;
/// let result = a.real().to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[1.0f32, 3.0]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Real<S, T>(crate::ops::op1::Op1<S, RealKernel<T>>);
struct RealKernel<T>(std::marker::PhantomData<T>);
impl<T> crate::ops::op1::Op1Kernel<Complex<T>> for RealKernel<T> {
    type Output = T;
    #[inline(always)]
    fn apply(&self, x: Complex<T>) -> Self::Output {
        x.re
    }
}
impl<S, T> Real<S, T>
where
    S: ArrayStorageTyped<Item = Complex<T>>,
    Complex<T>: Dtyped,
    T: Dtyped,
{
    /// Constructs a [`Real`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> crate::error::Result<Self> {
        let kernel = RealKernel(std::marker::PhantomData);
        Ok(Self(crate::ops::op1::Op1::new(array, kernel)?))
    }

    /// Constructs an array with [`Real`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> crate::error::Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, T> ArrayStorage for Real<S, T>
where
    S: ArrayStorageTyped<Item = Complex<T>>,
    Complex<T>: Dtyped,
    T: Dtyped,
{
    type ElementType = crate::Ty<T>;
    type Dimension = S::Dimension;
    crate::storage::impl_array_storage_forward!('a, T2, <S, T>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Real", [&self.0.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Real<S::DimensionChange<NewD>, T>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Real(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
}

/// Extracts the imaginary part of each complex element.
///
/// Supported input dtypes: `Complex<f32>`, `Complex<f64>`. Output dtype is the
/// corresponding real component type (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`).
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::imag()`](crate::Array::imag).
///
/// # Examples
/// ```
/// use jix::scalar::Complex;
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![
///     Complex { re: 1.0f32, im: 2.0 },
///     Complex { re: 3.0, im: -4.0 },
/// ])?;
/// let result = a.imag().to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[2.0f32, -4.0]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Imaginary<S, T>(crate::ops::op1::Op1<S, ImaginaryKernel<T>>);
struct ImaginaryKernel<T>(std::marker::PhantomData<T>);
impl<T> crate::ops::op1::Op1Kernel<Complex<T>> for ImaginaryKernel<T> {
    type Output = T;
    #[inline(always)]
    fn apply(&self, x: Complex<T>) -> Self::Output {
        x.im
    }
}
impl<S, T> Imaginary<S, T>
where
    S: ArrayStorageTyped<Item = Complex<T>>,
    Complex<T>: Dtyped,
    T: Dtyped,
{
    /// Constructs a [`Imaginary`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> crate::error::Result<Self> {
        let kernel = ImaginaryKernel(std::marker::PhantomData);
        Ok(Self(crate::ops::op1::Op1::new(array, kernel)?))
    }

    /// Constructs an array with [`Imaginary`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> crate::error::Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, T> ArrayStorage for Imaginary<S, T>
where
    S: ArrayStorageTyped<Item = Complex<T>>,
    Complex<T>: Dtyped,
    T: Dtyped,
{
    type ElementType = crate::Ty<T>;
    type Dimension = S::Dimension;
    crate::storage::impl_array_storage_forward!('a, T2, <S, T>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Imaginary", [&self.0.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Imaginary<S::DimensionChange<NewD>, T>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Imaginary(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
}

impl<S, T> Array<S>
where
    S: ArrayStorageTyped<Item = Complex<T>>,
    Complex<T>: Dtyped,
    T: Dtyped,
{
    /// Extracts the real part of each complex element. See [`Real`] for details and examples.
    #[track_caller]
    pub fn real(self) -> Array<Real<S, T>> {
        Real::new_array(self).unwrap()
    }

    /// Extracts the imaginary part of each complex element. See [`Imaginary`] for details and examples.
    #[track_caller]
    pub fn imag(self) -> Array<Imaginary<S, T>> {
        Imaginary::new_array(self).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::scalar::Complex;
    use crate::Array;

    #[test]
    fn real_complex_f32() {
        let a = Array::compact_ndarray(&array![
            Complex {
                re: 1.0f32,
                im: 2.0
            },
            Complex { re: 3.0, im: -4.0 },
            Complex { re: -5.0, im: 6.0 },
        ])
        .unwrap();
        let result = a.real().to_ndarray().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0f32, 3.0, -5.0]);
    }

    #[test]
    fn real_complex_f64() {
        let a = Array::compact_ndarray(&array![
            Complex {
                re: 1.0f64,
                im: 2.0
            },
            Complex { re: 3.0, im: -4.0 },
        ])
        .unwrap();
        let result = a.real().to_ndarray().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0f64, 3.0]);
    }

    #[test]
    fn imag_complex_f32() {
        let a = Array::compact_ndarray(&array![
            Complex {
                re: 1.0f32,
                im: 2.0
            },
            Complex { re: 3.0, im: -4.0 },
            Complex { re: -5.0, im: 6.0 },
        ])
        .unwrap();
        let result = a.imag().to_ndarray().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[2.0f32, -4.0, 6.0]);
    }

    #[test]
    fn imag_complex_f64() {
        let a = Array::compact_ndarray(&array![
            Complex {
                re: 1.0f64,
                im: 2.0
            },
            Complex { re: 3.0, im: -4.0 },
        ])
        .unwrap();
        let result = a.imag().to_ndarray().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[2.0f64, -4.0]);
    }

    #[test]
    fn real_imag_preserve_2d_shape() {
        let a = Array::compact_ndarray(&array![
            [
                Complex {
                    re: 1.0f32,
                    im: 10.0
                },
                Complex { re: 2.0, im: 20.0 }
            ],
            [Complex { re: 3.0, im: 30.0 }, Complex { re: 4.0, im: 40.0 }],
        ])
        .unwrap();
        assert_eq!(a.real().shape(), &[2, 2]);
        let a = Array::compact_ndarray(&array![
            [
                Complex {
                    re: 1.0f32,
                    im: 10.0
                },
                Complex { re: 2.0, im: 20.0 }
            ],
            [Complex { re: 3.0, im: 30.0 }, Complex { re: 4.0, im: 40.0 }],
        ])
        .unwrap();
        assert_eq!(
            a.imag().to_ndarray().unwrap().as_slice().unwrap(),
            &[10.0f32, 20.0, 30.0, 40.0]
        );
    }
}

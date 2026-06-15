use std::hint::assert_unchecked;
use std::ops::{Index, IndexMut};

use crate::error::{bail, Result};

/// Maximum number of dimensions supported by the library for an array.
pub const NDIM_MAX: usize = 8;

/// A type-level representation of the number of axes in an array.
///
/// Every [`ArrayStorage`](crate::ArrayStorage) carries an associated
/// `type Dimension: Dimension` that records how many axes the array has. The compiler propagates
/// this through a chain of lazy operations, such as unary and binary operations, and shape changing
/// operations: `insert_axis`, `remove_axis`, `reshape_view`, and
/// reductions all adjust the dimension type of their output. When the argument type is known
/// statically (e.g. `usize` or `[u64; N]`), the dimension change is encoded in the return type
/// so callers can reason about shape purely in types.
///
/// # Two implementors
///
/// There are exactly two types that implement `Dimension`:
///
/// - **[`Dim<N>`]** - static dimension. The compiler knows `ndim == N` at compile time. The
///   const generic `N` is the number of axes. Implemented for `N = 0..=8`.
/// - **[`DimDyn`]** - dynamic dimension. The ndim is only known at runtime, and stored in a
///   dynamic allocated array.
///
/// # How operations propagate dimension
///
/// Element wise operations (e.g. `+`, `*`, casts) do not change the dimension, so the output
/// dimension is the same as the input dimension.
/// Shape-changing operations adjust the dimension either by using the `Smaller` / `Larger`
/// associated types of the input dimension, or by taking a argument that determines the output
/// dimension such as `reshape` (that accept [`IntoDimension`] arguments) or reduction operations
/// (that accept [`AxesArg`](crate::ops::AxesArg)).
///
/// Because `DimDyn::Smaller = DimDyn` and `DimDyn::Larger = DimDyn`, operations on a dynamic
/// array always return a dynamic array. Use [`Array::into_dim`](crate::Array::into_dim) to
/// recover static tracking once the ndim becomes known.
///
/// # Boundary cases
///
/// - `Dim<0>::Smaller = DimDyn` - reducing below zero dimensions has no static representation.
/// - `Dim<8>::Larger = DimDyn` - exceeding [`NDIM_MAX`] is caught at construction time and
///   returns an error; the type maps to `DimDyn` to keep the trait implementable.
///
/// # Example: static vs dynamic dimension tracking
///
/// ```
/// use jix::{Array, Dim};
///
/// // Passing a dynamically-dimensioned ndarray produces Array<Compact<DimDyn>>.
/// // Arrays loaded from files via Array::read_from_file also carry DimDyn.
/// let a = Array::compact_ndarray(&ndarray::ArrayD::<i32>::zeros(vec![2, 3]))?;
/// // a: Array<Compact<DimDyn>>
///
/// // Asserting "I know this is 2-D" converts to static Dim<2>.
/// // Returns Err if a.ndim() != 2.
/// let a2d = a.into_dim::<Dim<2>>()?;
///
/// // insert_axis(0): usize arg -> D::Larger = Dim<3>
/// let a3d = a2d.insert_axis(0);
///
/// // reshape_view([6u64]): [u64; 1] arg -> Dim<1> output
/// let flat = a3d.reshape_view([6u64]);
///
/// // reshape_view(&shape[..]): &[u64] arg -> DimDyn output
/// let flat_dyn = flat.reshape_view(&[6u64][..]);
/// assert_eq!(flat_dyn.shape(), &[6]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Every `Dimension` value also implements [`IntoDimension`] (with `Dimension = Self`), so a
/// `Dim<N>` or `DimDyn` can be passed directly to any function that accepts `IntoDimension`.
pub trait Dimension:
    Index<usize, Output = u64>
    + IndexMut<usize, Output = u64>
    + IntoDimension<Dimension = Self>
    + ndarray::IntoDimension<Dim: IntoDimension>
    + Clone
    + Send
    + Sync
{
    /// The number of axes, if known at compile time.
    ///
    /// - `Some(N)` for [`Dim<N>`]: the compiler can prove `ndim == N`.
    /// - `None` for [`DimDyn`]: ndim is determined at runtime.
    ///
    /// This constant is used internally to skip runtime length checks when the ndim is statically
    /// known.
    const NDIM: Option<usize>;

    /// The dimension type that results from removing one axis.
    ///
    /// For `Dim<N>` where `N >= 1`: `Dim<N-1>`.
    /// For `Dim<0>`: `DimDyn` (cannot represent -1 statically).
    /// For `DimDyn`: `DimDyn` (removing an axis keeps the dimension dynamic).
    type Smaller: Dimension;

    /// The dimension type that results from adding one axis.
    ///
    /// For `Dim<N>` where `N < 8`: `Dim<N+1>`.
    /// For `Dim<8>`: `DimDyn` (exceeds [`NDIM_MAX`], caught at construction).
    /// For `DimDyn`: `DimDyn` (adding an axis keeps the dimension dynamic).
    type Larger: Dimension;

    /// The type of the index pattern for this dimension.
    ///
    /// - For `Dim<1>`: `u64`
    /// - For `Dim<2>`: `(u64, u64)`
    /// - For `Dim<3>`: `(u64, u64, u64)`
    /// - ...
    /// - For `DimDyn`: `&[u64]`
    type Index<'a>: IntoDimension<Dimension = Self> + Clone
    where
        Self: 'a;

    /// Construct a `Dimension` by calling `f(i)` for each axis index `i` in `0..ndim`.
    ///
    /// Returns an error if `ndim` does not match the statically expected ndim (for `Dim<N>`)
    /// or exceeds [`NDIM_MAX`] (for `DimDyn`).
    fn from_fn(ndim: usize, f: impl FnMut(usize) -> u64) -> Result<Self>
    where
        Self: Sized;

    /// Construct a `Dimension` value from a shape slice.
    ///
    /// Returns an error if the slice length does not match the statically expected ndim (for
    /// `Dim<N>`) or exceeds [`NDIM_MAX`] (for `DimDyn`).
    #[inline(always)]
    fn from_slice(slice: &[u64]) -> Result<Self>
    where
        Self: Sized,
    {
        Self::from_fn(slice.len(), |i| slice[i])
    }

    /// Return the shape as a slice, with one element per dimension.
    fn as_slice(&self) -> &[u64];

    /// Return the shape as a mutable slice, with one element per dimension.
    fn as_mut_slice(&mut self) -> &mut [u64];

    /// Return the number of dimensions.
    ///
    /// Equivalent to `self.as_slice().len()`. For `Dim<N>` the compiler can constant-fold this
    /// to `N`; for `DimDyn` it is a runtime array length.
    #[inline(always)]
    fn ndim(&self) -> usize {
        self.as_slice().len()
    }

    /// Return the size of dimension `dim`.
    ///
    /// Panics if `dim >= self.ndim()`.
    #[inline(always)]
    fn size(&self, dim: usize) -> u64 {
        self.as_slice()[dim]
    }

    /// Convert the dimension into its index pattern type.
    fn to_index(&self) -> Self::Index<'_>;
}

/// A dynamically-dimensioned shape whose ndim is only known at runtime.
///
/// `DimDyn` stores the shape in a stack-allocated array with capacity [`NDIM_MAX`].
/// It is the fallback when the number of axes cannot be determined at compile
/// time: arrays loaded from files, operations that take `&[usize]` axis arguments, and
/// dimension-changing operations applied to a `DimDyn` array all produce `DimDyn`.
///
/// `DimDyn::Smaller = DimDyn` and `DimDyn::Larger = DimDyn`: the dimension type stays
/// dynamic through any chain of shape-changing operations. Use
/// [`Array::into_dim`](crate::Array::into_dim) to recover a static [`Dim<N>`] when the ndim
/// becomes known.
#[derive(Clone)]
pub struct DimDyn(DimArray<u64>);
impl Dimension for DimDyn {
    const NDIM: Option<usize> = None;
    type Smaller = Self;
    type Larger = Self;
    type Index<'a> = &'a [u64];

    #[inline(always)]
    fn from_fn(ndim: usize, f: impl FnMut(usize) -> u64) -> Result<Self> {
        if ndim > NDIM_MAX {
            bail!(
                TooManyDimensions,
                "cannot create DimDyn with ndim {ndim}: exceeds NDIM_MAX ({NDIM_MAX})"
            );
        }
        Ok(Self(dim_arr(ndim, f)))
    }

    #[inline(always)]
    fn as_slice(&self) -> &[u64] {
        let s = self.0.as_slice();
        unsafe { assert_unchecked(s.len() <= NDIM_MAX) };
        s
    }
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [u64] {
        let s = self.0.as_mut_slice();
        unsafe { assert_unchecked(s.len() <= NDIM_MAX) };
        s
    }

    #[inline(always)]
    fn to_index(&self) -> Self::Index<'_> {
        self.as_slice()
    }
}
impl Index<usize> for DimDyn {
    type Output = u64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}
impl IndexMut<usize> for DimDyn {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}
impl ndarray::IntoDimension for DimDyn {
    type Dim = ndarray::IxDyn;

    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        let dim = self.as_slice();
        let mut nd_dim = Self::Dim::zeros(dim.len());
        for (i, &size) in dim.iter().enumerate() {
            nd_dim[i] = size as usize;
        }
        nd_dim
    }
}

/// A statically-dimensioned shape with exactly `NDIM` axes, known at compile time.
///
/// `Dim<N>` is the preferred dimension type when the number of axes is determined by the code
/// structure rather than runtime data. Passing typed arguments to shape-manipulating operations
/// (e.g. `[u64; 2]` to `reshape_view`, `usize` to `insert_axis`) causes the compiler to select
/// `Dim<N>` automatically, so callers rarely need to name this type explicitly.
///
/// # Neighbors
///
/// `Dim<N>::Smaller = Dim<N-1>` (for N >= 1; `Dim<0>::Smaller = DimDyn`).
/// `Dim<N>::Larger  = Dim<N+1>` (for N <= 7; `Dim<8>::Larger  = DimDyn`).
///
/// This means that N consecutive `insert_axis(0)` calls on a `Dim<M>` array produce a
/// `Dim<M+N>` result, as long as `M + N <= 8`.
#[derive(Clone)]
pub struct Dim<const NDIM: usize>([u64; NDIM]);
impl<const NDIM: usize> Dim<NDIM> {
    /// Construct a `Dim<NDIM>` from a fixed-size array of axis sizes.
    #[inline(always)]
    pub fn from_array(arr: [u64; NDIM]) -> Self {
        Self(arr)
    }
}
macro_rules! impl_dim {
    (
        $dim:literal,
        $(nd_dim = $nd_dim:ty,)?
        $(smaller = $smaller:ty,)?
        $(larger = $larger:ty,)?
        index = { $index:ty, |$to_index_in:ident| $to_index:expr }
    ) => {
        impl Dimension for Dim<$dim> {
            const NDIM: Option<usize> = Some($dim);
            type Smaller = crate::util::or_else!($({ $smaller })? or { Dim<{ $dim - 1 }> });
            type Larger = crate::util::or_else!($({ $larger })? or { Dim<{ $dim + 1 }> });
            type Index<'a> = $index;

            #[inline(always)]
            fn from_fn(ndim: usize, mut f: impl FnMut(usize) -> u64) -> Result<Self> {
                if ndim != $dim {
                    bail!(
                        TooManyDimensions,
                        "cannot create Dim<{}> with ndim {ndim}",
                        $dim
                    );
                }
                Ok(Self(std::array::from_fn(|i| f(i))))
            }

            #[inline(always)]
            fn as_slice(&self) -> &[u64] {
                self.0.as_slice()
            }
            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [u64] {
                self.0.as_mut_slice()
            }

            #[inline(always)]
            fn to_index(&self) -> Self::Index<'_> {
                #[allow(unused_variables)]
                let $to_index_in = self.as_slice();
                $to_index
            }
        }
        impl Index<usize> for Dim<$dim> {
            type Output = u64;

            #[inline(always)]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_slice()[index]
            }
        }
        impl IndexMut<usize> for Dim<$dim> {
            #[inline(always)]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }
        impl ndarray::IntoDimension for Dim<$dim> {
            type Dim = crate::util::or_else!($({ $nd_dim })? or { ndarray::Dim<[usize; $dim]> });

            #[inline(always)]
            fn into_dimension(self) -> Self::Dim {
                let dim = self.as_slice();
                let mut nd_dim = <Self::Dim as ndarray::Dimension>::zeros(dim.len());
                for (i, &size) in dim.iter().enumerate() {
                    nd_dim[i] = size as usize;
                }
                nd_dim
            }
        }
    };
}
impl_dim!(0, smaller = DimDyn, larger = Dim<1>, index = { (), |i| () });
impl_dim!(1, index = { u64, |i| i[0] });
impl_dim!(2, index = { (u64, u64), |i| (i[0], i[1]) });
impl_dim!(3, index = { (u64, u64, u64), |i| (i[0], i[1], i[2]) });
impl_dim!(4, index = { (u64, u64, u64, u64), |i| (i[0], i[1], i[2], i[3]) });
impl_dim!(5, index = { (u64, u64, u64, u64, u64), |i| (i[0], i[1], i[2], i[3], i[4]) });
impl_dim!(6, index = { (u64, u64, u64, u64, u64, u64), |i| (i[0], i[1], i[2], i[3], i[4], i[5]) });
impl_dim!(7, nd_dim = ndarray::IxDyn, index = { (u64, u64, u64, u64, u64, u64, u64), |i| (i[0], i[1], i[2], i[3], i[4], i[5], i[6]) });
impl_dim!(8, nd_dim = ndarray::IxDyn, smaller = Dim<7>, larger = DimDyn, index = { (u64, u64, u64, u64, u64, u64, u64, u64), |i| (i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]) });

/// Conversion into a [`Dimension`] value, encoding the ndim in the type.
///
/// This trait is the mechanism by which ordinary Rust types (integers, arrays, tuples,
/// ndarray dimension types) become valid shape arguments. The associated type `Dimension`
/// carries the ndim information statically when the argument type allows it.
///
/// # Implemented for
///
/// | Type | `Dimension` | Notes |
/// |---|---|---|
/// | Any `D: Dimension` | `D` | Identity - a dimension is already a dimension. |
/// | `u64` | `Dim<1>` | Treated as a 1-D shape `[n]`. |
/// | `[u64; N]` / `&[u64; N]` | `Dim<N>` | Fixed-size array, `N = 0..=8`. |
/// | `(u64,)` .. `(u64, u64, u64, u64, u64, u64, u64, u64)` | `Dim<1>` .. `Dim<8>` | Tuples up to 8 elements. |
/// | `()` | `Dim<0>` | Zero-dimensional (scalar) shape. |
/// | `&[u64]` / `&Vec<u64>` | `DimDyn` | Slice/vec - ndim only known at runtime. |
/// | `ndarray::Dim<[usize; N]>` (N=0..=6) | `Dim<N>` | ndarray static dimension. |
/// | `ndarray::IxDyn` | `DimDyn` | ndarray dynamic dimension. |
///
/// # Return value
///
/// `into_dimension` returns `Option<Self::Dimension>`. It returns `None` only when the input
/// cannot fit (e.g. a slice longer than [`NDIM_MAX`]). For all fixed-size types (arrays, tuples,
/// `u64`) the conversion is infallible.
pub trait IntoDimension {
    /// The [`Dimension`] type produced by this conversion.
    type Dimension: Dimension;

    /// Convert `self` into a `Dimension` value.
    ///
    /// Returns an error if the input cannot be converted (e.g. a slice whose length exceeds
    /// [`NDIM_MAX`]). For statically-sized types the conversion is infallible.
    fn into_dimension(self) -> Result<Self::Dimension>;
}
impl<D> IntoDimension for D
where
    D: Dimension,
{
    type Dimension = D;

    #[inline(always)]
    fn into_dimension(self) -> Result<Self::Dimension> {
        Ok(self)
    }
}

impl IntoDimension for u64 {
    type Dimension = Dim<1>;

    #[inline(always)]
    fn into_dimension(self) -> Result<Self::Dimension> {
        Ok(Dim::<1>::from_array([self]))
    }
}
impl IntoDimension for &[u64] {
    type Dimension = DimDyn;

    #[inline(always)]
    fn into_dimension(self) -> Result<Self::Dimension> {
        DimDyn::from_slice(self)
    }
}
impl IntoDimension for &Vec<u64> {
    type Dimension = DimDyn;

    #[inline(always)]
    fn into_dimension(self) -> Result<Self::Dimension> {
        self.as_slice().into_dimension()
    }
}

macro_rules! impl_into_dimension_array {
    ($n:expr) => {
        impl IntoDimension for [u64; $n] {
            type Dimension = Dim<$n>;

            #[inline(always)]
            fn into_dimension(self) -> Result<Self::Dimension> {
                Ok(Dim::<$n>::from_array(self))
            }
        }
        impl IntoDimension for &[u64; $n] {
            type Dimension = Dim<$n>;

            #[inline(always)]
            fn into_dimension(self) -> Result<Self::Dimension> {
                Ok(Dim::<$n>::from_array(*self))
            }
        }
    };
}
impl_into_dimension_array!(0);
impl_into_dimension_array!(1);
impl_into_dimension_array!(2);
impl_into_dimension_array!(3);
impl_into_dimension_array!(4);
impl_into_dimension_array!(5);
impl_into_dimension_array!(6);
impl_into_dimension_array!(7);
impl_into_dimension_array!(8);

macro_rules! impl_into_dimension_tuple {
    ($($idx:tt),+ $(,)?) => {
        impl IntoDimension for ($(impl_into_dimension_tuple!(@replace $idx u64),)+) {
            type Dimension = Dim<{ impl_into_dimension_tuple!(@count $($idx)*) }>;

            #[inline(always)]
            fn into_dimension(self) -> Result<Self::Dimension> {
                Ok(Dim::from_array([$(self.$idx,)+]))
            }
        }
    };

    (@count ) => { 0 };
    (@count $head:tt $($tail:tt)*) => { 1 + impl_into_dimension_tuple!(@count $($tail)*) };
    (@replace $_t:tt $sub:ty) => { $sub };
}

impl IntoDimension for () {
    type Dimension = Dim<0>;

    #[inline(always)]
    fn into_dimension(self) -> Result<Self::Dimension> {
        Ok(Dim::from_array([]))
    }
}
impl_into_dimension_tuple!(0);
impl_into_dimension_tuple!(0, 1);
impl_into_dimension_tuple!(0, 1, 2);
impl_into_dimension_tuple!(0, 1, 2, 3);
impl_into_dimension_tuple!(0, 1, 2, 3, 4);
impl_into_dimension_tuple!(0, 1, 2, 3, 4, 5);
impl_into_dimension_tuple!(0, 1, 2, 3, 4, 5, 6);
impl_into_dimension_tuple!(0, 1, 2, 3, 4, 5, 6, 7);

macro_rules! impl_into_dimension_ndarray {
    ($n:expr) => {
        impl IntoDimension for ndarray::Dim<[usize; $n]> {
            type Dimension = Dim<$n>;

            #[inline(always)]
            fn into_dimension(self) -> Result<Self::Dimension> {
                let mut arr = [0u64; $n];
                #[allow(clippy::reversed_empty_ranges)]
                for i in 0..$n {
                    arr[i] = self[i] as u64;
                }
                Ok(Dim::from_array(arr))
            }
        }
    };
}

impl_into_dimension_ndarray!(0);
impl_into_dimension_ndarray!(1);
impl_into_dimension_ndarray!(2);
impl_into_dimension_ndarray!(3);
impl_into_dimension_ndarray!(4);
impl_into_dimension_ndarray!(5);
impl_into_dimension_ndarray!(6);
// impl_into_dimension_ndarray!(7);
// impl_into_dimension_ndarray!(8);
impl IntoDimension for ndarray::IxDyn {
    type Dimension = DimDyn;

    #[inline(always)]
    fn into_dimension(self) -> Result<Self::Dimension> {
        let dim = <Self as ndarray::Dimension>::as_array_view(&self);
        let dim = dim.as_slice().unwrap();
        if dim.len() > NDIM_MAX {
            bail!(
                TooManyDimensions,
                "ndarray dimension length {} exceeds NDIM_MAX ({})",
                dim.len(),
                NDIM_MAX
            );
        }
        let dim = dim_arr(dim.len(), |i| dim[i] as u64);
        DimDyn::from_slice(&dim)
    }
}

/// Stack-allocated vector for per-dimension data, with capacity [`NDIM_MAX`].
///
/// Used throughout the library to store shapes, strides, block shapes, and other per-axis
/// values without heap allocation. The capacity is always exactly [`NDIM_MAX`] (8), so no
/// array with more dimensions than supported can overflow this container.
pub(crate) type DimArray<T> = crate::util::arrayvec::ArrayVec<T, NDIM_MAX>;

/// Build a [`DimArray`] by applying `f` to each axis index `0..ndim`.
///
/// Panics if `ndim > NDIM_MAX`.
#[inline(always)]
pub(crate) fn dim_arr<T>(ndim: usize, f: impl FnMut(usize) -> T) -> DimArray<T> {
    (0..ndim).map(f).collect()
}

/// Build a [`DimArray`] by applying a fallible `f` to each axis index `0..ndim`.
///
/// Returns the first error encountered, or `Ok` with the full array on success.
/// Panics if `ndim > NDIM_MAX`.
#[inline(always)]
pub(crate) fn try_dim_arr<T, E>(
    ndim: usize,
    f: impl FnMut(usize) -> Result<T, E>,
) -> Result<DimArray<T>, E> {
    (0..ndim).map(f).collect()
}

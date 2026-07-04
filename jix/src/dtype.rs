#![allow(rustdoc::redundant_explicit_links)]
//! Element type descriptors and related primitives.
//!
//! The central type is [`Dtype`] - a runtime descriptor that captures all layout information
//! needed to interpret the raw bytes of an array element. See its documentation for a full
//! explanation, including scalar vs. struct dtypes and the inner-shape mechanism.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`Dtype`] | Runtime element-type descriptor (kind, itemsize, alignment, inner shape) |
//! | [`ScalarKind`] | Enum of every supported scalar primitive |
//! | [`Dtyped`] | Trait (and derive macro) mapping a Rust type to its `Dtype` at compile time |
//! | [`Alignment`] | Newtype for alignment values; guarantees power-of-two and non-zero |
//! | [`Itemsize`] | Alias for `u16`, used for per-element byte sizes and field offsets |
//!
//! # Numeric type coverage
//!
//! Scalar dtypes span integers, floats, complex numbers, and booleans:
//!
//! - **Integers**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
//! - **Floats**: `f16`, `f32`, `f64`
//! - **Complex**: `Complex<f32>`, `Complex<f64>`
//! - **Boolean**: `bool`

use std::borrow::Cow;
use std::collections::HashSet;
use std::hint::assert_unchecked;

use crate::error::{bail, ensure, Error, ErrorKind, Result};
#[cfg(feature = "half")]
use crate::scalar::f16;
#[cfg(feature = "num-complex")]
use crate::scalar::Complex;
use crate::util::arrayvec::ArrayVec;
use crate::util::{Idx, IterExt};

/// A type alignment in bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Alignment(AlignmentInner);
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(dead_code)]
#[repr(u16)]
enum AlignmentInner {
    A1 = 1 << 0,
    A2 = 1 << 1,
    A4 = 1 << 2,
    A8 = 1 << 3,
    A16 = 1 << 4,
    A32 = 1 << 5,
    A64 = 1 << 6,
    A128 = 1 << 7,
    A256 = 1 << 8,
    A512 = 1 << 9,
    A1024 = 1 << 10,
    A2048 = 1 << 11,
    A4096 = 1 << 12,
    A8192 = 1 << 13,
    A16384 = 1 << 14,
    A32768 = 1 << 15,
}

impl Alignment {
    /// Creates a new `Alignment` from an integer value in bytes.
    ///
    /// The value must be non-zero, a power of two, and less than the supported maximum
    /// alignment (currently 32768 bytes, may change in the future).
    #[inline(always)]
    pub const fn new(value: usize) -> Option<Self> {
        if 1 <= value && value <= 32768 && value.is_power_of_two() {
            // SAFETY: By precondition, this must be a power of two, and
            // our variants encompass all possible powers of two.
            Some(Self(unsafe {
                std::mem::transmute::<u16, AlignmentInner>(value as u16)
            }))
        } else {
            None
        }
    }

    /// Creates a new `Alignment` for the given type `T`, using its natural alignment.
    #[inline(always)]
    pub const fn of<T>() -> Self {
        Self::new(align_of::<T>()).unwrap()
    }

    /// Returns the underlying value of the alignment in bytes.
    #[inline(always)]
    pub const fn as_usize(self) -> usize {
        let align = self.0 as usize;
        unsafe { assert_unchecked(align != 0 && align.is_power_of_two()) };
        align
    }
}
impl std::fmt::Debug for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.as_usize(), f)
    }
}
impl std::fmt::Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.as_usize(), f)
    }
}
impl TryFrom<usize> for Alignment {
    type Error = Error;

    #[inline(always)]
    fn try_from(value: usize) -> Result<Self> {
        Self::new(value).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidArgument,
                format!(
                    "invalid alignment {value}: must be a non-zero power of two, and less than or equal to 32768"
                ),
            )
        })
    }
}

/// The type used to represent dtype itemsize in bytes.
///
/// Also used for field offsets in struct dtypes.
pub type Itemsize = u16;

/// The maximum number of dimensions allowed in a dtype shape.
///
/// Note this is not the shape of the array, but rather the inner shape of a dtype.
pub const DTYPE_MAX_NDIM: usize = 4;
type DtypeShape = ArrayVec<Itemsize, DTYPE_MAX_NDIM>;

/// Runtime descriptor of an element type: captures all layout information needed to interpret
/// the raw bytes stored in an [`Array`](crate::Array).
///
/// A `Dtype` answers three questions about every element in an array:
/// - **What kind** of data is it (signed int, float, struct of named fields, ...)?
/// - **How many bytes** does one logical element occupy ([`itemsize`](Self::itemsize))?
/// - **What alignment** is required when placing an element in memory ([`alignment`](Self::alignment))?
///
/// # Two Flavours
///
/// ## Scalar dtypes
///
/// Cover all primitive numeric and boolean types built into the library. Each variant of
/// [`ScalarKind`] has a fixed itemsize and alignment:
///
/// | Scalar kind | itemsize | alignment |
/// |-------------|----------|-----------|
/// | `I8` / `U8` / `Bool` | 1 | 1 |
/// | `I16` / `U16` / `F16` | 2 | 2 |
/// | `I32` / `U32` / `F32` | 4 | 4 |
/// | `I64` / `U64` / `F64` | 8 | 8 |
/// | `ComplexF32` | 8 | 4 |
/// | `ComplexF64` | 16 | 8 |
///
/// Create a scalar dtype with [`Dtype::new_scalar`], or read it as a compile-time constant from
/// the [`Dtyped::DTYPE`] implementation of a Rust primitive:
///
/// ```rust
/// use jix::dtype::{Dtype, ScalarKind, Dtyped};
///
/// let d = Dtype::new_scalar(ScalarKind::F64);
/// assert_eq!(d, f64::DTYPE);
/// assert_eq!(d.itemsize(), 8);
/// assert_eq!(d.alignment().as_usize(), 8);
/// assert_eq!(d.shape(), &[]);
/// assert_eq!(d.scalar_kind(), Some(ScalarKind::F64));
/// assert_eq!(d.fields(), None);
/// ```
///
/// ## Struct dtypes
///
/// Group named fields with explicit byte offsets. Each field is itself a `Dtype`, allowing
/// arbitrary nesting (structs of structs, arrays of structs, etc.).
///
/// Struct layout comes in two variants matching C representations:
///
/// - **Aligned** (`#[repr(C)]`): fields are padded to their natural alignment; the struct is
///   padded at the end to a multiple of the maximum field alignment.
/// - **Packed** (`#[repr(C, packed)]`): no padding anywhere; fields are laid out
///   contiguously, so the offset of each field is exactly the sum of the sizes of all
///   preceding fields.
///
/// Create struct dtypes with [`Dtype::from_fields`] (auto-detects layout) or
/// [`Dtype::new_struct`] (explicit control when auto-detection is ambiguous):
///
/// ```rust,ignore
/// use jix::dtype::{Dtype, Dtyped};
///
/// // Aligned layout: id (u8) at 0, 3 bytes padding, grade (f32) at 4, 0 bytes tail padding.
/// // Total itemsize = 8, alignment = 4.
/// let aligned = Dtype::from_fields(vec![
///     ("id".to_string(), 0, u8::DTYPE),
///     ("grade".to_string(), 4, f32::DTYPE),
/// ]).unwrap();
/// assert_eq!(aligned.itemsize(), 8);
/// assert_eq!(aligned.alignment().as_usize(), 4);
/// assert!(aligned.is_aligned());
///
/// // Packed layout: id (u8) at 0, grade (f32) at 1. Total itemsize = 5, alignment = 1.
/// let packed = Dtype::from_fields(vec![
///     ("id".to_string(), 0, u8::DTYPE),
///     ("grade".to_string(), 1, f32::DTYPE),
/// ]).unwrap();
/// assert_eq!(packed.itemsize(), 5);
/// assert_eq!(packed.alignment().as_usize(), 1);
/// assert!(!packed.is_aligned());
/// ```
///
/// # Inner Shape
///
/// Both scalar and struct dtypes can carry an *inner shape*: a small, fixed-size sub-array of
/// up to [`DTYPE_MAX_NDIM`] dimensions baked into each logical element. This is how Rust
/// fixed-size arrays (`[T; N]`) are represented, and how NumPy sub-array dtypes are encoded.
///
/// The inner shape **multiplies the itemsize** but does not add dimensions to the containing
/// `Array`. A dtype with shape `[3]` and element type `f32` occupies 12 bytes per logical
/// element; the array's own `shape()` still counts logical elements.
///
/// ```rust
/// use jix::dtype::Dtyped;
///
/// // [f32; 3]: one logical element = 3 floats back to back.
/// let d = <[f32; 3] as Dtyped>::DTYPE;
/// assert_eq!(d.shape(), &[3]);
/// assert_eq!(d.itemsize(), 12);
///
/// // [[i32; 4]; 2]: inner shape is [2, 4], itemsize = 2 * 4 * 4 = 32.
/// let d = <[[i32; 4]; 2] as Dtyped>::DTYPE;
/// assert_eq!(d.shape(), &[2, 4]);
/// assert_eq!(d.itemsize(), 32);
/// ```
///
/// # Obtaining a `Dtype`
///
/// | Method | Use when |
/// |--------|----------|
/// | `T::DTYPE` ([`Dtyped`] constant) | Rust type known at compile time |
/// | [`Dtype::new_scalar`] | Building a scalar dtype from a [`ScalarKind`] at runtime |
/// | [`Dtype::from_fields`] | Building a struct dtype from field definitions; auto-detects packed vs. aligned |
/// | [`Dtype::new_struct`] | Building a struct dtype with full explicit control over itemsize and alignment |
///
/// The [`Dtyped`] derive macro generates the `const DTYPE` for any `#[repr(C)]` or
/// `#[repr(C, packed)]` struct:
///
/// ```rust,ignore
/// use jix::dtype::{Dtype, Dtyped};
///
/// #[derive(Copy, Clone, Dtyped)]
/// #[repr(C)]
/// struct Pixel {
///     r: u8,
///     g: u8,
///     b: u8,
/// }
///
/// let d = Pixel::DTYPE;
/// assert_eq!(d.itemsize(), 3);
/// assert_eq!(d.alignment().as_usize(), 1);
/// let fields = d.fields().unwrap();
/// assert_eq!(fields[0].0, "r");
/// assert_eq!(fields[1].0, "g");
/// assert_eq!(fields[2].0, "b");
/// ```
///
/// # Constraints
///
/// - **Little-endian only** - jix enforces little-endian at compile time; big-endian targets
///   will not compile.
/// - **Inner shape dimensions** - at most [`DTYPE_MAX_NDIM`] (currently 4).
/// - **Itemsize limit** - stored as [`Itemsize`] (`u16`); the total bytes per element must not
///   exceed `u16::MAX`.
/// - **Field offsets** - must conform to either an aligned (`#[repr(C)]`) or packed
///   (`#[repr(C, packed)]`) layout; arbitrary custom offsets are rejected.
#[derive(Clone)]
pub struct Dtype(DtypeInner);
#[derive(Clone)]
enum DtypeInner {
    Scalar {
        itemsize: Itemsize,
        alignment: Alignment,
        shape: DtypeShape,
        kind: ScalarKind,
        // endianness: Endianness,
    },
    StructOwned {
        itemsize: Itemsize,
        alignment: Alignment,
        is_aligned: bool,
        shape: DtypeShape,
        fields: Box<[(Cow<'static, str>, Itemsize, Dtype)]>,
    },
    StructBorrowed {
        itemsize: Itemsize,
        alignment: Alignment,
        is_aligned: bool,
        shape: DtypeShape,
        fields: &'static [(Cow<'static, str>, Itemsize, Dtype)],
    },
}
const _: () = {
    if size_of::<usize>() == 8 {
        assert!(size_of::<Dtype>() <= 32);
    }
};

/// The kind of a scalar dtype, representing all primitive scalar types supported by the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    /// [`i8`] dtype.
    I8,
    /// [`i16`] dtype.
    I16,
    /// [`i32`] dtype.
    I32,
    /// [`i64`] dtype.
    I64,
    /// [`u8`] dtype.
    U8,
    /// [`u16`] dtype.
    U16,
    /// [`u32`] dtype.
    U32,
    /// [`u64`] dtype.
    U64,
    #[allow(rustdoc::redundant_explicit_links)]
    /// [`f16`](crate::scalar::f16) dtype.
    F16,
    /// [`f32`] dtype.
    F32,
    /// [`f64`] dtype.
    F64,
    /// [`Complex<f32>`] dtype, as two consecutive `f32` values for the real and imaginary parts.
    ComplexF32,
    /// [`Complex<f64>`] dtype, as two consecutive `f64` values for the real and imaginary parts.
    ComplexF64,
    /// [`bool`] dtype, as a single byte with value 0 or 1.
    Bool,
}
/// The endianness of a scalar.
#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Endianness {
    /// Little-endian.
    Little,
    /// Big-endian.
    Big,
}
impl Dtype {
    /// Creates a new scalar dtype.
    ///
    /// The created dtype will use the native endianness.
    ///
    /// ```rust
    /// use jix::dtype::{Dtype, ScalarKind, Dtyped};
    ///
    /// let i32_dtype = Dtype::new_scalar(ScalarKind::I32);
    /// assert_eq!(i32_dtype.scalar_kind(), Some(ScalarKind::I32));
    /// assert_eq!(i32_dtype.fields(), None);
    /// assert_eq!(i32_dtype.itemsize(), 4);
    /// assert_eq!(i32_dtype.alignment().as_usize(), 4);
    /// assert_eq!(i32_dtype.shape(), &[]);
    ///
    /// let f64_dtype = Dtype::new_scalar(ScalarKind::F64);
    /// assert_eq!(f64_dtype.scalar_kind(), Some(ScalarKind::F64));
    /// assert_eq!(f64_dtype.fields(), None);
    /// assert_eq!(f64_dtype.itemsize(), 8);
    /// assert_eq!(f64_dtype.alignment().as_usize(), 8);
    /// assert_eq!(f64_dtype.shape(), &[]);
    ///
    /// # #[cfg(feature = "num-complex")] { use super::*;
    /// let complex_f32_dtype = Dtype::new_scalar(ScalarKind::ComplexF32);
    /// assert_eq!(
    ///     complex_f32_dtype.scalar_kind(),
    ///     Some(ScalarKind::ComplexF32)
    /// );
    /// assert_eq!(complex_f32_dtype.itemsize(), 8);
    /// assert_eq!(complex_f32_dtype.alignment().as_usize(), 4);
    ///
    /// assert_eq!(i32_dtype, i32::DTYPE);
    /// assert_eq!(f64_dtype, f64::DTYPE);
    /// assert_eq!(complex_f32_dtype, jix::scalar::Complex::<f32>::DTYPE);
    /// # }
    /// ```
    #[inline(always)]
    pub const fn new_scalar(kind: ScalarKind) -> Self {
        // assert!(Endianness::native() == Endianness::Little);

        const _: () = const {
            assert!(
                cfg!(target_endian = "little"),
                "Only little-endian is supported"
            );
        };
        Self(DtypeInner::Scalar {
            kind,
            // endianness: Endianness::native(),
            shape: DtypeShape::new(),
            itemsize: kind.itemsize(),
            alignment: kind.alignment(),
        })
    }

    /// Creates a new struct dtype from a set of fields definitions.
    ///
    /// # Arguments
    ///
    /// * `fields` - A vector of field definitions, where each field is represented as a tuple of
    ///   (name, offset, dtype). Names should be unique. Offsets are in bytes from the start of the struct.
    ///
    /// The fields should be either in packed or aligned offsets. See [`Self::is_aligned`] for details.
    ///
    /// There are some cases in which it is ambiguous whether the offsets are packed or aligned, and it may affect the
    /// computed total itemsize of the struct. In these cases, consider using the explicit [`Self::new_struct`].
    ///
    /// ```rust,ignore
    /// use jix::dtype::{Dtype, Dtyped};
    ///
    /// /// A student struct with aligned fields.
    /// #[derive(Dtyped)]
    /// #[repr(C)]
    /// struct Student {
    ///   id: u8,
    ///   grade: f32,
    /// }
    /// let student_dtype = Dtype::from_fields(vec![
    ///   ("id".to_string(), 0, u8::DTYPE),
    ///   ("grade".to_string(), 4, f32::DTYPE),
    /// ]).unwrap();
    /// assert_eq!(student_dtype.fields().unwrap().len(), 2);
    /// let mut fields = student_dtype.fields().unwrap().iter();
    /// assert_eq!(fields.next().unwrap(), ("id".into(), 0, u8::DTYPE));
    /// assert_eq!(fields.next().unwrap(), ("grade".into(), 4, f32::DTYPE));
    /// assert_eq!(student_dtype.scalar_kind(), None);
    /// assert_eq!(student_dtype.itemsize(), 8);
    /// assert_eq!(student_dtype.alignment().as_usize(), 4);
    /// assert_eq!(student_dtype.shape(), &[]);
    ///
    /// // Compare with the derived dtype of the struct
    /// assert_eq!(Student::DTYPE, student_dtype);
    ///
    /// /// A packed student struct.
    /// #[derive(Dtyped)]
    /// #[repr(C, packed)]
    /// struct StudentPacked {
    ///   id: u8,
    ///   grade: f32,
    /// }
    /// let student_packed_dtype = Dtype::from_fields(vec![
    ///  ("id".to_string(), 0, u8::DTYPE),
    ///  ("grade".to_string(), 1, f32::DTYPE),
    /// ]).unwrap();
    /// assert_eq!(student_packed_dtype.itemsize(), 5);
    /// assert_eq!(student_packed_dtype.alignment().as_usize(), 1);
    /// assert_eq!(StudentPacked::DTYPE, student_packed_dtype);
    /// ```
    pub fn from_fields(fields: Vec<(String, Itemsize, Dtype)>) -> Result<Self> {
        let mut seen_names = HashSet::new();
        for (name, _offset, _dtype) in &fields {
            if !seen_names.insert(name) {
                bail!(
                    InvalidArgument,
                    "duplicate field name `{name}` in struct dtype"
                );
            }
        }

        let mut fields = fields;
        fields.sort_by_key(|(_name, offset, _dtype)| *offset);

        fn determine_itemsize_and_alignment(
            fields: &[(String, Itemsize, Dtype)],
        ) -> Result<(Itemsize, (Alignment, bool))> {
            let mut expected_offset = 0;
            let is_aligned = fields.iter().all({
                |(_f_name, offset, dtype)| {
                    expected_offset =
                        expected_offset.ceil_to_multiple(dtype.alignment().as_usize() as Itemsize);
                    let aligned = *offset == expected_offset;
                    expected_offset += dtype.itemsize();
                    aligned
                }
            });
            if is_aligned {
                let max_alignment = fields
                    .iter()
                    .map(|(_name, _offset, dtype)| dtype.alignment())
                    .max()
                    .unwrap_or(Alignment::new(1).unwrap());
                let itemsize =
                    expected_offset.ceil_to_multiple(max_alignment.as_usize() as Itemsize);
                return Ok((itemsize, (max_alignment, true)));
            }

            let mut expected_offset = 0; // assuming sorted by offset
            let is_packed = fields.iter().all({
                |(_f_name, offset, dtype)| {
                    let packed = *offset == expected_offset;
                    expected_offset += dtype.itemsize();
                    packed
                }
            });
            if is_packed {
                let itemsize = expected_offset;
                return Ok((itemsize, (Alignment::new(1).unwrap(), false)));
            }

            bail!(
                InvalidArgument,
                "field offsets are not in a valid packed or aligned layout"
            );
        }

        let (itemsize, (alignment, is_aligned)) = determine_itemsize_and_alignment(&fields)?;

        let fields = fields
            .into_iter()
            .map(|(name, offset, dtype)| (Cow::Owned(name), offset, dtype))
            .collect();

        Ok(Self(DtypeInner::StructOwned {
            fields,
            shape: DtypeShape::new(),
            itemsize,
            alignment,
            is_aligned,
        }))
    }

    /// Creates a new struct dtype by specifying all of the parameters explicitly.
    ///
    /// Most of the time, the simpler [`Self::from_fields`] should be sufficient, and this function
    /// is only needed in rare cases where the field offsets are ambiguous between packed and aligned layout.
    ///
    /// # Arguments
    ///
    /// * `fields` - A vector of field definitions, where each field is represented as a tuple of
    ///   (name, offset, dtype). Names should be unique. Offsets are in bytes from the start of the struct.
    /// * `shape` - The shape of the dtype, as a slice of dimensions. The total number of elements
    ///   in the shape must not exceed `Itemsize::MAX`. The shape can be empty, which means a single
    ///   element of the dtype. The number of dimensions in the shape must not exceed [`DTYPE_MAX_NDIM`].
    /// * `itemsize` - The itemsize of the dtype in bytes. Must be a multiple of the product of the
    ///   shape dimensions.
    /// * `alignment` - The alignment of the dtype in bytes. Must be a power of two and less than
    ///   or equal to the itemsize.
    ///
    /// The fields should be either in packed or aligned offsets. See [`Self::is_aligned`] for details.
    /// Thw shape, itemsize and alignment will be validated against the fields.
    ///
    /// ```rust,ignore
    /// use jix::dtype::{Dtype, Dtyped, Alignment, Itemsize};
    ///
    /// #[derive(Dtyped)]
    /// #[repr(C)]
    /// struct Person {
    ///   weight: f32,
    ///   age: u8,
    /// }
    ///
    /// // We can't use `Dtype::from_fields` here because the offsets are ambiguous between packed and
    /// // aligned layout: `weight` at offset 0 and `age` at offset 4 can be either an aligned struct
    /// // with 3 bytes of padding after `age` or a packed struct with no padding, and we can not
    /// // determine if the total itemsize is 5 or 8.
    /// // We use `Dtype::new_struct` and pass the itemsize and alignment explicitly.
    ///
    /// let person_dtype = Dtype::new_struct(
    ///   vec![
    ///     ("weight".to_string(), 0, f32::DTYPE),
    ///     ("age".to_string(), 4, u8::DTYPE),
    ///   ],
    ///   &[1, 2],
    ///   8,
    ///   Alignment::new(4).unwrap(),
    /// ).unwrap();
    /// assert_eq!(Person::DTYPE, person_dtype);
    /// ```
    pub fn new_struct(
        fields: impl IntoIterator<Item = (String, Itemsize, Dtype)>,
        shape: &[Itemsize],
        itemsize: Itemsize,
        alignment: Alignment,
    ) -> Result<Self> {
        let shape = DtypeShape::from_slice(shape).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidArgument,
                format!(
                    "Dtype shape length exceeds the maximum supported dim number {}",
                    DTYPE_MAX_NDIM
                ),
            )
        })?;
        let shape_prod = shape
            .iter()
            .try_fold(1 as Itemsize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument, "Dtype shape overflow"))?;
        ensure!(
            shape_prod != 0,
            InvalidArgument,
            "Dtype shape has zero elements"
        );
        ensure!(
            itemsize.is_multiple_of(shape_prod),
            InvalidArgument,
            "Dtype itemsize is not a multiple of the shape product"
        );
        let element_itemsize = itemsize / shape_prod;

        let fields = fields
            .into_iter()
            .map(|(name, offset, dtype)| (Cow::Owned(name), offset, dtype))
            .collect::<Box<_>>();

        let is_aligned;
        if Self::is_aligned_struct(&fields, element_itemsize, alignment) {
            is_aligned = true;
        } else if Self::is_packed_struct(&fields, element_itemsize, alignment) {
            is_aligned = false;
        } else {
            bail!(
                InvalidArgument,
                "field offsets are not in a valid packed or aligned layout"
            );
        }

        Ok(Self(DtypeInner::StructOwned {
            fields,
            shape,
            itemsize,
            alignment,
            is_aligned,
        }))
    }

    #[doc(hidden)]
    pub const unsafe fn new_struct_borrowed_unchecked(
        fields: &'static [(Cow<'static, str>, Itemsize, Dtype)],
        itemsize: Itemsize,
        alignment: Alignment,
        is_aligned: bool,
    ) -> Self {
        Self(DtypeInner::StructBorrowed {
            fields,
            shape: DtypeShape::new(),
            itemsize,
            alignment,
            is_aligned,
        })
    }

    fn is_aligned_struct(
        fields: &[(Cow<'static, str>, Itemsize, Dtype)],
        itemsize: Itemsize,
        alignment: Alignment,
    ) -> bool {
        let max_alignment = fields
            .iter()
            .map(|(_name, _offset, dtype)| dtype.alignment())
            .max()
            .unwrap_or(Alignment::new(1).unwrap());
        if alignment != max_alignment {
            return false;
        }

        let mut expected_offset = 0;
        let is_aligned = fields.iter().all({
            |(_name, offset, dtype)| {
                expected_offset =
                    expected_offset.ceil_to_multiple(dtype.alignment().as_usize() as Itemsize);
                let aligned = *offset == expected_offset;
                expected_offset += dtype.itemsize();
                aligned
            }
        });
        if !is_aligned {
            return false;
        }

        let expected_itemsize =
            expected_offset.ceil_to_multiple(max_alignment.as_usize() as Itemsize);
        if expected_itemsize != itemsize {
            return false;
        }

        true
    }

    fn is_packed_struct(
        fields: &[(Cow<'static, str>, Itemsize, Dtype)],
        itemsize: Itemsize,
        alignment: Alignment,
    ) -> bool {
        if alignment.as_usize() != 1 {
            return false;
        }

        let mut expected_offset = 0;
        let is_packed = fields.iter().all({
            |(_name, offset, dtype)| {
                let packed = *offset == expected_offset;
                expected_offset += dtype.itemsize();
                packed
            }
        });
        if !is_packed {
            return false;
        }
        let expected_itemsize = expected_offset;
        if expected_itemsize != itemsize {
            return false;
        }

        true
    }

    /// Get the scalar kind of the dtype, if it is a scalar dtype.
    ///
    /// Note that even if the function returns `Some`, the dtype may not be a plain scalar, it just
    /// means the type has no sub fields, but the dtype can still have non-empty shape.
    #[inline(always)]
    pub const fn scalar_kind(&self) -> Option<ScalarKind> {
        match &self.0 {
            DtypeInner::Scalar { kind, .. } => Some(*kind),
            DtypeInner::StructOwned { .. } | DtypeInner::StructBorrowed { .. } => None,
        }
    }

    /// Get the fields of the dtype, if it is a struct dtype.
    #[inline(always)]
    pub fn fields(&self) -> Option<&[(Cow<'static, str>, Itemsize, Dtype)]> {
        match &self.0 {
            DtypeInner::Scalar { .. } => None,
            DtypeInner::StructOwned { fields, .. } => Some(fields),
            DtypeInner::StructBorrowed { fields, .. } => Some(fields),
        }
    }

    /// Get the shape of the dtype.
    ///
    /// Empty shape means a single element of the dtype.
    #[inline(always)]
    pub fn shape(&self) -> &[Itemsize] {
        match &self.0 {
            DtypeInner::Scalar { shape, .. } => shape,
            DtypeInner::StructOwned { shape, .. } => shape,
            DtypeInner::StructBorrowed { shape, .. } => shape,
        }
    }

    const fn shape_mut(&mut self) -> &mut DtypeShape {
        match &mut self.0 {
            DtypeInner::Scalar { shape, .. } => shape,
            DtypeInner::StructOwned { shape, .. } => shape,
            DtypeInner::StructBorrowed { shape, .. } => shape,
        }
    }

    /// Get the itemsize of the dtype.
    ///
    /// If this dtype has a shape, the itemsize is the product of the shape dimensions and the base itemsize.
    #[inline(always)]
    pub const fn itemsize(&self) -> Itemsize {
        match &self.0 {
            DtypeInner::Scalar { itemsize, .. } => *itemsize,
            DtypeInner::StructOwned { itemsize, .. } => *itemsize,
            DtypeInner::StructBorrowed { itemsize, .. } => *itemsize,
        }
    }

    const fn itemsize_mut(&mut self) -> &mut Itemsize {
        match &mut self.0 {
            DtypeInner::Scalar { itemsize, .. } => itemsize,
            DtypeInner::StructOwned { itemsize, .. } => itemsize,
            DtypeInner::StructBorrowed { itemsize, .. } => itemsize,
        }
    }

    /// Get the alignment of the dtype.
    ///
    /// For scalar dtypes, the alignment is the same as [`ScalarKind::alignment()`].
    /// For struct dtypes, the alignment is either `1` for packed dtypes, or the maximum alignment of the inner fields
    /// for aligned structs.
    #[inline(always)]
    pub const fn alignment(&self) -> Alignment {
        match &self.0 {
            DtypeInner::Scalar { alignment, .. } => *alignment,
            DtypeInner::StructOwned { alignment, .. } => *alignment,
            DtypeInner::StructBorrowed { alignment, .. } => *alignment,
        }
    }

    /// Returns whether the dtype is aligned (like C structs) or packed.
    ///
    /// - Packed struct means the fields are laid out back to back without any padding, so the offset
    ///   of each field is the sum of the itemsize of the previous fields, and the total itemsize of
    ///   the struct is the sum of the itemsize of all fields. The alignment of a packed struct is 1.
    ///   This matches `#[repr(C, packed)]` in Rust, and is compatible with packed structs in C, numpy, etc.
    /// - Aligned struct means the fields are laid out with padding such that the offset of each field
    ///   is the smallest offset that is greater than or equal to the end offset of the previous field,
    ///   and the total itemsize of the struct is the smallest offset that is greater than or equal to
    ///   the end offset of the last field and is a multiple of the maximum alignment of all fields.
    ///   The alignment of an aligned struct is the maximum alignment of all fields.
    ///   This matches `#[repr(C)]` in Rust, and is compatible with aligned structs in C, numpy, etc.
    /// - Custom alignment and offsets that don't match either of the above layouts are not supported.
    ///
    /// The "packed vs aligned" distinction only applies to struct dtypes, scalar dtypes are always
    /// considered aligned.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        match &self.0 {
            DtypeInner::Scalar { .. } => true,
            DtypeInner::StructOwned { is_aligned, .. } => *is_aligned,
            DtypeInner::StructBorrowed { is_aligned, .. } => *is_aligned,
        }
    }

    /// Try to convert this dtype to a scalar dtype, if it matches the scalar dtype exactly.
    #[inline(always)]
    pub fn try_to_scalar(&self) -> Option<ScalarKind> {
        let scalar = self.scalar_kind()?;
        (Self::new_scalar(scalar) == *self).then_some(scalar)
    }

    /// Set the shape of the dtype, updating the itemsize accordingly.
    ///
    /// The maximum number of dimensions allowed in the shape is [`DTYPE_MAX_NDIM`].
    /// The product of the shape dimensions must not exceed [`Itemsize::MAX`].
    ///
    /// The itemsize will be updated to `itemsize *= new_shape.product() / old_shape.product()`.
    pub fn set_shape(&mut self, shape: &[Itemsize]) -> Result<()> {
        let shape = DtypeShape::from_slice(shape).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidArgument,
                format!(
                    "Dtype shape length exceeds the maximum supported dim number {}",
                    DTYPE_MAX_NDIM
                ),
            )
        })?;
        let shape_prod = shape
            .iter()
            .cloned()
            .try_product()
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument, "Dtype shape overflow"))?;
        ensure!(
            shape_prod != 0,
            InvalidArgument,
            "Dtype shape has zero elements"
        );
        let current_shape_size = self.shape().iter().cloned().product::<Itemsize>();
        assert!(self.itemsize().is_multiple_of(current_shape_size));
        let base_itemsize = self.itemsize() / current_shape_size;
        *self.itemsize_mut() = base_itemsize
            .checked_mul(shape_prod)
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument, "Dtype shape overflow"))?;
        *self.shape_mut() = shape;
        Ok(())
    }
}
impl std::fmt::Debug for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Dtype");
        match &self.0 {
            DtypeInner::Scalar { kind, .. } => f.field("scalar", kind),
            DtypeInner::StructOwned { fields, .. } => f.field("fields", fields),
            DtypeInner::StructBorrowed { fields, .. } => f.field("fields", fields),
        };
        f.field("shape", &self.shape())
            .field("itemsize", &self.itemsize())
            .field("alignment", &self.alignment())
            .finish()
    }
}
impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(scalar) = self.try_to_scalar() {
            match scalar {
                ScalarKind::I8 => f.write_str("i8"),
                ScalarKind::I16 => f.write_str("i16"),
                ScalarKind::I32 => f.write_str("i32"),
                ScalarKind::I64 => f.write_str("i64"),
                ScalarKind::U8 => f.write_str("u8"),
                ScalarKind::U16 => f.write_str("u16"),
                ScalarKind::U32 => f.write_str("u32"),
                ScalarKind::U64 => f.write_str("u64"),
                ScalarKind::F16 => f.write_str("f16"),
                ScalarKind::F32 => f.write_str("f32"),
                ScalarKind::F64 => f.write_str("f64"),
                ScalarKind::ComplexF32 => f.write_str("Complex<f32>"),
                ScalarKind::ComplexF64 => f.write_str("Complex<f64>"),
                ScalarKind::Bool => f.write_str("bool"),
            }
        } else {
            <_ as std::fmt::Debug>::fmt(self, f)
        }
    }
}
impl PartialEq for Dtype {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if !(self.itemsize() == other.itemsize()
            && self.alignment() == other.alignment()
            && self.shape() == other.shape())
        {
            return false;
        }
        match (self.scalar_kind(), other.scalar_kind()) {
            (Some(scalar1), Some(scalar2)) => scalar1 == scalar2,
            (None, None) => self.fields() == other.fields(),
            _ => false,
        }
    }
}

impl ScalarKind {
    /// Get the size of the scalar in bytes.
    #[inline(always)]
    pub const fn itemsize(&self) -> Itemsize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::ComplexF32 => 8,
            Self::ComplexF64 => 16,
            Self::Bool => 1,
        }
    }

    /// Get the alignment of the scalar in bytes.
    #[inline(always)]
    pub const fn alignment(&self) -> Alignment {
        let align = match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::ComplexF32 => 4,
            Self::ComplexF64 => 8,
            Self::Bool => 1,
        };
        Alignment::new(align).unwrap()
    }

    /// Check if this scalar is an integer type (signed or unsigned).
    ///
    /// Returns true for `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// Check if this scalar is a signed integer type.
    ///
    /// Returns true for `i8`, `i16`, `i32`, `i64`.
    #[inline]
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Check if this scalar is an unsigned integer type.
    ///
    /// Returns true for `u8`, `u16`, `u32`, `u64`.
    #[inline]
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Check if this scalar is a floating point type.
    ///
    /// Returns true for `f16`, `f32`, `f64`.
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Check if this scalar is a complex type.
    ///
    /// Returns true for `Complex<f32>` and `Complex<f64>`.
    #[inline]
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::ComplexF32 | Self::ComplexF64)
    }

    /// Check if this scalar is a boolean type.
    ///
    /// Returns true for `bool`.
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Try to convert this scalar to a signed integer type, if it is an integer type.
    ///
    /// Returns:
    /// - `Some(I8)` for `I8` and `U8`
    /// - `Some(I16)` for `I16` and `U16`
    /// - `Some(I32)` for `I32` and `U32`
    /// - `Some(I64)` for `I64` and `U64`
    /// - `None` for others
    #[inline]
    pub fn to_signed_integer(&self) -> Option<Self> {
        Some(match self {
            Self::U8 | Self::I8 => Self::I8,
            Self::U16 | Self::I16 => Self::I16,
            Self::U32 | Self::I32 => Self::I32,
            Self::U64 | Self::I64 => Self::I64,
            _ => return None,
        })
    }

    /// Try to convert this scalar to an unsigned integer type, if it is an integer type.
    ///
    /// Returns:
    /// - `Some(U8)` for `I8` and `U8`
    /// - `Some(U16)` for `I16` and `U16`
    /// - `Some(U32)` for `I32` and `U32`
    /// - `Some(U64)` for `I64` and `U64`
    /// - `None` for others
    #[inline]
    pub fn to_unsigned_integer(&self) -> Option<Self> {
        Some(match self {
            Self::I8 | Self::U8 => Self::U8,
            Self::I16 | Self::U16 => Self::U16,
            Self::I32 | Self::U32 => Self::U32,
            Self::I64 | Self::U64 => Self::U64,
            _ => return None,
        })
    }
}
#[allow(unused)]
impl Endianness {
    /// Get the native endianness.
    #[inline(always)]
    pub const fn native() -> Self {
        if cfg!(target_endian = "little") {
            Endianness::Little
        } else {
            Endianness::Big
        }
    }
}

/// A trait for types that can be represented by a [`Dtype`].
///
/// `jix` maintain the dtype of each array dynamically using a [`Dtype`].
/// For safe conversions between (typed erased) arrays and other typed arrays (for example [`ndarray::ArrayBase`])
/// the `Dtyped` trait is used to verify type compatibility.
///
/// The trait also force `Copy`, and elements in arrays should not implement `Drop`.
///
/// Use the derive macro [`Dtyped`] to automatically implement this trait for structs.
/// ```rust,ignore
/// use jix::dtype::{Dtype, Dtyped};
///
/// #[derive(Dtyped)]
/// #[repr(C)]
/// struct MyStruct {
///     a: i32,
///     b: u8,
/// }
/// let expected_dtype = Dtype::from_fields(vec![
///     ("a".to_string(), 0, i32::DTYPE),
///     ("b".to_string(), 4, u8::DTYPE),
/// ]);
/// assert_eq!(MyStruct::DTYPE, expected_dtype.unwrap());
/// ```
///
/// # Safety
///
/// This trait is very unsafe, and the caller should implement it carefully, matching the type size,
/// alignment and inner fields of the type. Types implementing this should most likely be annotated with `#[repr(C)]`
/// or `#[repr(C, packed)]`, for aligned and packed fields respectively.
pub unsafe trait Dtyped: Copy + Send + Sync + Sized + 'static {
    /// Get the dtype representing the type layout and inner fields.
    const DTYPE: Dtype;
}

// Re-export derive macro
pub use jix_macros::Dtyped;

macro_rules! impl_dtyped_scalar {
    ($ty:ty, $kind:ident) => {
        unsafe impl Dtyped for $ty {
            const DTYPE: Dtype = Dtype::new_scalar(ScalarKind::$kind);
        }
    };
}

impl_dtyped_scalar!(i8, I8);
impl_dtyped_scalar!(i16, I16);
impl_dtyped_scalar!(i32, I32);
impl_dtyped_scalar!(i64, I64);
impl_dtyped_scalar!(u8, U8);
impl_dtyped_scalar!(u16, U16);
impl_dtyped_scalar!(u32, U32);
impl_dtyped_scalar!(u64, U64);
#[cfg(feature = "half")]
impl_dtyped_scalar!(f16, F16);
impl_dtyped_scalar!(f32, F32);
impl_dtyped_scalar!(f64, F64);
#[cfg(feature = "num-complex")]
impl_dtyped_scalar!(Complex<f32>, ComplexF32);
#[cfg(feature = "num-complex")]
impl_dtyped_scalar!(Complex<f64>, ComplexF64);
impl_dtyped_scalar!(bool, Bool);

unsafe impl<T: Dtyped, const N: usize> Dtyped for [T; N] {
    const DTYPE: Dtype = {
        let mut dtype = T::DTYPE;
        assert!(
            N < Itemsize::MAX as usize,
            "Array size exceeds maximum supported dtype shape size of Itemsize::MAX"
        );
        let n = N as Itemsize;
        dtype.shape_mut().insert_first_const(n);
        *dtype.itemsize_mut() *= n;
        dtype
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Scalar dtype basics ----

    #[test]
    fn scalar_itemsize_and_alignment() {
        let cases: &[(ScalarKind, Itemsize, /* alignment */ usize)] = &[
            (ScalarKind::I8, 1, 1),
            (ScalarKind::I16, 2, 2),
            (ScalarKind::I32, 4, 4),
            (ScalarKind::I64, 8, 8),
            (ScalarKind::U8, 1, 1),
            (ScalarKind::U16, 2, 2),
            (ScalarKind::U32, 4, 4),
            (ScalarKind::U64, 8, 8),
            (ScalarKind::F16, 2, 2),
            (ScalarKind::F32, 4, 4),
            (ScalarKind::F64, 8, 8),
            (ScalarKind::ComplexF32, 8, 4),
            (ScalarKind::ComplexF64, 16, 8),
            (ScalarKind::Bool, 1, 1),
        ];
        for &(kind, expected_size, expected_align) in cases {
            let expected_align = Alignment::new(expected_align).unwrap();
            let d = Dtype::new_scalar(kind);
            assert_eq!(d.itemsize(), expected_size, "{kind:?} itemsize");
            assert_eq!(d.alignment(), expected_align, "{kind:?} alignment");
            assert_eq!(d.shape(), &[] as &[Itemsize], "{kind:?} shape");
            assert_eq!(d.scalar_kind(), Some(kind), "{kind:?} scalar_kind");
            assert!(d.fields().is_none(), "{kind:?} fields should be None");
        }
    }

    #[test]
    fn scalar_equality() {
        assert_eq!(i32::DTYPE, i32::DTYPE);
        assert_ne!(i32::DTYPE, i64::DTYPE);
        assert_ne!(i32::DTYPE, u32::DTYPE);
        assert_ne!(i32::DTYPE, f64::DTYPE);
    }

    // ---- Array dtype ----

    #[test]
    fn array_prepends_shape_and_scales_itemsize() {
        let d = <[f64; 77] as Dtyped>::DTYPE;
        assert_eq!(d.shape(), &[77]);
        assert_eq!(d.itemsize(), 77 * 8);
        assert_eq!(d.alignment().as_usize(), 8);
        assert_eq!(d.scalar_kind(), Some(ScalarKind::F64));
    }

    #[test]
    fn nested_array_accumulates_shape() {
        // [[i32; 3]; 2] should yield shape [2, 3]
        let d = <[[i32; 3]; 2] as Dtyped>::DTYPE;
        assert_eq!(d.shape(), &[2, 3]);
        assert_eq!(d.itemsize(), 2 * 3 * 4);
        assert_eq!(d.alignment().as_usize(), 4);
    }

    #[test]
    fn array_size_1_has_shape() {
        let d = <[u8; 1] as Dtyped>::DTYPE;
        assert_eq!(d.shape(), &[1]);
        assert_eq!(d.itemsize(), 1);
    }

    // ---- from_fields ----

    #[test]
    fn from_fields_detects_packed_layout() {
        // u8 at 0, f64 at 1: contiguous -> packed
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::DTYPE),
            ("b".to_string(), 1, f64::DTYPE),
        ])
        .unwrap();
        assert_eq!(dtype.itemsize(), 9);
        assert_eq!(dtype.alignment().as_usize(), 1);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".into(), 0, u8::DTYPE));
        assert_eq!(fields[1], ("b".into(), 1, f64::DTYPE));
    }

    #[test]
    fn from_fields_detects_aligned_layout() {
        // u8 at 0, f64 at 8: gap filled with padding -> aligned
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::DTYPE),
            ("b".to_string(), 8, f64::DTYPE),
        ])
        .unwrap();
        assert_eq!(dtype.itemsize(), 16); // total padded to alignment 8
        assert_eq!(dtype.alignment().as_usize(), 8);
    }

    #[test]
    fn from_fields_ambiguous_single_field_detected_as_aligned() {
        // Single field: packed and aligned offsets are identical.
        // from_fields tries aligned first, so it always returns the aligned layout.
        let dtype = Dtype::from_fields(vec![("x".to_string(), 0, f64::DTYPE)]).unwrap();
        assert_eq!(dtype.itemsize(), 8);
        assert_eq!(dtype.alignment().as_usize(), 8);
    }

    #[test]
    fn from_fields_ambiguous_i32_u8_detected_as_aligned() {
        // { a: i32 at 0, b: u8 at 4 } - offsets are valid for both packed and aligned layouts.
        // from_fields tries aligned first and returns alignment=4, itemsize=8 (padded to align 4).
        // Use new_struct() when explicit control is needed.
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, i32::DTYPE),
            ("b".to_string(), 4, u8::DTYPE),
        ])
        .unwrap();
        assert_eq!(dtype.alignment().as_usize(), 4);
        assert_eq!(dtype.itemsize(), 8);
    }

    #[test]
    fn from_fields_sorts_by_offset() {
        // Fields given in reverse order - result should be sorted ascending by offset.
        let dtype = Dtype::from_fields(vec![
            ("b".to_string(), 1, f32::DTYPE),
            ("a".to_string(), 0, u8::DTYPE),
        ])
        .unwrap();
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0].0, "a");
        assert_eq!(fields[0].1, 0);
        assert_eq!(fields[1].0, "b");
        assert_eq!(fields[1].1, 1);
    }

    #[test]
    fn from_fields_duplicate_names_errors() {
        let result = Dtype::from_fields(vec![
            ("x".to_string(), 0, i32::DTYPE),
            ("x".to_string(), 4, i32::DTYPE),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn from_fields_invalid_offsets_errors() {
        // Offset 3 for f64 is neither packed (would be 1) nor aligned (would be 8).
        let result = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::DTYPE),
            ("b".to_string(), 3, f64::DTYPE),
        ]);
        assert!(result.is_err());
    }

    // ---- new_struct ----

    #[test]
    fn new_struct_packed_explicit() {
        // Disambiguate the i32+u8 case: explicitly request packed layout.
        let dtype = Dtype::new_struct(
            vec![
                ("a".to_string(), 0, i32::DTYPE),
                ("b".to_string(), 4, u8::DTYPE),
            ],
            &[],
            5,
            1.try_into().unwrap(),
        )
        .unwrap();
        assert_eq!(dtype.itemsize(), 5);
        assert_eq!(dtype.alignment().as_usize(), 1);
    }

    #[test]
    fn new_struct_aligned_explicit() {
        // Explicitly request aligned layout for i32+u8, getting the C struct size of 8.
        let dtype = Dtype::new_struct(
            vec![
                ("a".to_string(), 0, i32::DTYPE),
                ("b".to_string(), 4, u8::DTYPE),
            ],
            &[],
            8,
            4.try_into().unwrap(),
        )
        .unwrap();
        assert_eq!(dtype.itemsize(), 8);
        assert_eq!(dtype.alignment().as_usize(), 4);
    }

    #[test]
    fn new_struct_with_multidim_shape() {
        let dtype = Dtype::new_struct(
            vec![("a".to_string(), 0, u8::DTYPE)],
            &[2, 3],
            6, // 2*3*1
            1.try_into().unwrap(),
        )
        .unwrap();
        assert_eq!(dtype.shape(), &[2, 3]);
        assert_eq!(dtype.itemsize(), 6);
    }

    #[test]
    fn new_struct_shape_zero_errors() {
        assert!(Dtype::new_struct(
            vec![("a".to_string(), 0, u8::DTYPE)],
            &[0],
            0,
            1.try_into().unwrap()
        )
        .is_err());
    }

    #[test]
    fn new_struct_too_many_dims_errors() {
        // DTYPE_MAX_NDIM = 4; five dimensions must be rejected.
        assert!(Dtype::new_struct(
            vec![("a".to_string(), 0, u8::DTYPE)],
            &[1, 1, 1, 1, 1],
            1,
            1.try_into().unwrap(),
        )
        .is_err());
    }

    #[test]
    fn new_struct_max_dims_works() {
        // Exactly DTYPE_MAX_NDIM = 4 dimensions should succeed.
        let dtype = Dtype::new_struct(
            vec![("a".to_string(), 0, u8::DTYPE)],
            &[1, 2, 3, 4],
            24, // 1*2*3*4
            1.try_into().unwrap(),
        )
        .unwrap();
        assert_eq!(dtype.shape(), &[1, 2, 3, 4]);
        assert_eq!(dtype.itemsize(), 24);
    }

    #[test]
    fn new_struct_itemsize_not_multiple_of_shape_errors() {
        // shape=[3], element must be 4 bytes (i32), total must be 12; 10 is not valid.
        assert!(Dtype::new_struct(
            vec![("a".to_string(), 0, i32::DTYPE)],
            &[3],
            10,
            4.try_into().unwrap()
        )
        .is_err());
    }

    #[test]
    fn new_struct_wrong_alignment_errors() {
        // f64 field requires alignment 8; declaring alignment 4 is rejected as invalid offsets
        // (alignment 4 matches neither packed=1 nor aligned=8 layout).
        assert!(Dtype::new_struct(
            vec![("a".to_string(), 0, f64::DTYPE)],
            &[],
            8,
            4.try_into().unwrap()
        )
        .is_err());
    }

    #[test]
    fn new_struct_packed_wrong_offset_errors() {
        // Packed struct (alignment=1): b must be at offset 4, not 5.
        assert!(Dtype::new_struct(
            vec![
                ("a".to_string(), 0, i32::DTYPE),
                ("b".to_string(), 5, u8::DTYPE),
            ],
            &[],
            6,
            1.try_into().unwrap(),
        )
        .is_err());
    }

    #[test]
    fn new_struct_itemsize_too_small_errors() {
        // Packed i32+u8 must be exactly 5; declaring 4 is rejected as invalid offsets
        // (is_packed_struct checks itemsize as part of offset validation).
        assert!(Dtype::new_struct(
            vec![
                ("a".to_string(), 0, i32::DTYPE),
                ("b".to_string(), 4, u8::DTYPE),
            ],
            &[],
            4,
            1.try_into().unwrap(),
        )
        .is_err());
    }

    // ---- Derive macro ----

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct SimpleStruct {
        a: u8,
        b: f32,
        c: i16,
    }

    #[test]
    fn derive_simple_struct() {
        let dtype = SimpleStruct::DTYPE;
        assert_eq!(dtype.itemsize() as usize, size_of::<SimpleStruct>());
        assert_eq!(
            dtype.alignment().as_usize(),
            std::mem::align_of::<SimpleStruct>()
        );
        assert_eq!(dtype.itemsize(), 12);
        assert_eq!(dtype.alignment().as_usize(), 4);
        assert_eq!(dtype.shape(), &[]);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], ("a".into(), 0, u8::DTYPE));
        assert_eq!(fields[1], ("b".into(), 4, f32::DTYPE));
        assert_eq!(fields[2], ("c".into(), 8, i16::DTYPE));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C, packed)]
    struct PackedStruct {
        a: u8,
        b: f32,
        c: i16,
    }

    #[test]
    fn derive_packed_struct() {
        let dtype = PackedStruct::DTYPE;
        assert_eq!(dtype.itemsize() as usize, size_of::<PackedStruct>());
        assert_eq!(
            dtype.alignment().as_usize(),
            std::mem::align_of::<PackedStruct>()
        );
        assert_eq!(dtype.itemsize(), 7);
        assert_eq!(dtype.alignment().as_usize(), 1);
        assert_eq!(dtype.shape(), &[]);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], ("a".into(), 0, u8::DTYPE));
        assert_eq!(fields[1], ("b".into(), 1, f32::DTYPE));
        assert_eq!(fields[2], ("c".into(), 5, i16::DTYPE));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(transparent)]
    struct NewtypeWrapper(SimpleStruct);

    #[test]
    fn derive_transparent_newtype() {
        assert_eq!(NewtypeWrapper::DTYPE, SimpleStruct::DTYPE);
        assert_eq!(
            NewtypeWrapper::DTYPE.itemsize() as usize,
            size_of::<NewtypeWrapper>()
        );
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct NestedStruct {
        a: SimpleStruct,
        b: f64,
    }

    #[test]
    fn derive_nested_struct() {
        let dtype = NestedStruct::DTYPE;
        assert_eq!(dtype.itemsize() as usize, size_of::<NestedStruct>());
        assert_eq!(
            dtype.alignment().as_usize(),
            std::mem::align_of::<NestedStruct>()
        );
        // SimpleStruct: 12 bytes, align 4. f64: 8 bytes, align 8.
        // a at 0, b at ceil(12, 8)=16, total ceil(24, 8)=24
        assert_eq!(dtype.itemsize(), 24);
        assert_eq!(dtype.alignment().as_usize(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".into(), 0, SimpleStruct::DTYPE));
        assert_eq!(fields[1], ("b".into(), 16, f64::DTYPE));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct ArrayFieldStruct {
        a: [i32; 3],
    }

    #[test]
    fn derive_struct_with_array_field() {
        let dtype = ArrayFieldStruct::DTYPE;
        assert_eq!(dtype.itemsize() as usize, size_of::<ArrayFieldStruct>());
        assert_eq!(
            dtype.alignment().as_usize(),
            std::mem::align_of::<ArrayFieldStruct>()
        );
        assert_eq!(dtype.itemsize(), 12);
        assert_eq!(dtype.alignment().as_usize(), 4);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "a");
        assert_eq!(fields[0].1, 0);
        // The field dtype should carry the array shape.
        assert_eq!(fields[0].2.shape(), &[3]);
        assert_eq!(fields[0].2.itemsize(), 12);
        assert_eq!(fields[0].2.scalar_kind(), Some(ScalarKind::I32));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct DeepNested {
        inner: NestedStruct,
        x: u32,
    }

    #[test]
    fn derive_deeply_nested_struct() {
        let dtype = DeepNested::DTYPE;
        assert_eq!(dtype.itemsize() as usize, size_of::<DeepNested>());
        assert_eq!(
            dtype.alignment().as_usize(),
            std::mem::align_of::<DeepNested>()
        );
        // NestedStruct: 24 bytes, align 8. u32: 4 bytes, align 4.
        // inner at 0, x at ceil(24,4)=24, total ceil(28,8)=32
        assert_eq!(dtype.itemsize(), 32);
        assert_eq!(dtype.alignment().as_usize(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("inner".into(), 0, NestedStruct::DTYPE));
        assert_eq!(fields[1], ("x".into(), 24, u32::DTYPE));
    }

    // ---- set_shape ----

    #[test]
    fn set_shape_on_scalar_sets_shape_and_scales_itemsize() {
        let mut d = i32::DTYPE;
        assert_eq!(d.shape(), &[] as &[Itemsize]);
        assert_eq!(d.itemsize(), 4);
        d.set_shape(&[3, 2]).unwrap();
        assert_eq!(d.shape(), &[3, 2]);
        assert_eq!(d.itemsize(), 4 * 3 * 2);
        assert_eq!(d.alignment().as_usize(), 4);
        assert_eq!(d.scalar_kind(), Some(ScalarKind::I32));
    }

    #[test]
    fn set_shape_empty_resets_to_scalar() {
        let mut d = <[u8; 5] as Dtyped>::DTYPE;
        assert_eq!(d.shape(), &[5]);
        assert_eq!(d.itemsize(), 5);
        d.set_shape(&[]).unwrap();
        assert_eq!(d.shape(), &[] as &[Itemsize]);
        assert_eq!(d.itemsize(), 1);
    }

    #[test]
    fn set_shape_replaces_existing_shape() {
        let mut d = <[[f32; 4]; 2] as Dtyped>::DTYPE;
        assert_eq!(d.shape(), &[2, 4]);
        assert_eq!(d.itemsize(), 2 * 4 * 4);
        d.set_shape(&[10]).unwrap();
        assert_eq!(d.shape(), &[10]);
        assert_eq!(d.itemsize(), 10 * 4);
    }

    #[test]
    fn set_shape_zero_dimension_errors() {
        let mut d = i32::DTYPE;
        assert!(d.set_shape(&[0]).is_err());
        assert!(d.set_shape(&[3, 0, 1]).is_err());
    }

    #[test]
    fn set_shape_too_many_dims_errors() {
        let mut d = u8::DTYPE;
        assert!(d.set_shape(&[1, 1, 1, 1, 1]).is_err());
    }

    #[test]
    fn set_shape_max_dims_succeeds() {
        let mut d = u8::DTYPE;
        d.set_shape(&[1, 1, 1, 1]).unwrap();
        assert_eq!(d.shape(), &[1, 1, 1, 1]);
        assert_eq!(d.itemsize(), 1);
    }

    #[test]
    fn set_shape_overflow_errors() {
        let mut d = u8::DTYPE;
        assert!(d.set_shape(&[Itemsize::MAX, Itemsize::MAX]).is_err());
    }
}

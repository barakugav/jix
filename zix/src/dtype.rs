use crate::util::{Idx, IxIterExt};
use std::collections::HashSet;

/// The type used to represent dtype alignment in bytes.
pub type Alignment = u8;

/// The type used to represent dtype itemsize in bytes.
///
/// Also used for field offsets in struct dtypes.
pub type Itemsize = u16;

/// The maximum number of dimensions allowed in a dtype shape.
///
/// Note this is not the shape of the array, but rather the inner shape of the dtype.
pub const DTYPE_MAX_NDIM: usize = 4;
pub(crate) type DtypeShape = arrayvec::ArrayVec<Itemsize, DTYPE_MAX_NDIM>;

/// Description of a type layout and inner fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dtype {
    kind: DtypeKind,
    shape: DtypeShape,
    itemsize: Itemsize,
    alignment: (Alignment, bool),
}
/// An inner enum of [`Dtype`], either a scalar or a struct.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DtypeKind {
    /// A scalar dtype.
    Scalar {
        /// The kind of the scalar dtype.
        kind: DtypeScalarKind,
        // Only little endian is supported.
        // endianness: Endianness::Little,
    },
    /// A struct dtype.
    Struct {
        /// The fields of the struct dtype, represented as (name, offset, dtype).
        /// The fields are sorted by their offsets in ascending order.
        fields: Box<[(String, Itemsize, Dtype)]>,
    },
}
/// The kind of a scalar dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeScalarKind {
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
    /// [`f16`](crate::dtype::f16) dtype.
    F16,
    /// [`f32`] dtype.
    F32,
    /// [`f64`] dtype.
    F64,
    /// [`Complex<f32>`] dtype.
    ComplexF32,
    /// [`Complex<f64>`] dtype.
    ComplexF64,
    /// [`bool`] dtype.
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
    pub const fn of_scalar(kind: DtypeScalarKind) -> Self {
        // assert!(Endianness::native() == Endianness::Little);

        const _: () = const {
            assert!(
                cfg!(target_endian = "little"),
                "Only little-endian is supported"
            );
        };
        Self {
            kind: DtypeKind::Scalar {
                kind,
                // endianness: Endianness::native(),
            },
            shape: DtypeShape::new_const(),
            itemsize: kind.itemsize(),
            alignment: (kind.alignment(), true),
        }
    }

    /// Creates a new struct dtype from a set of fields definitions.
    ///
    /// # Arguments
    ///
    /// * `fields` - A vector of field definitions, where each field is represented as a tuple of
    ///   (name, offset, dtype). Names should be unique.
    ///
    /// The fields should be either in packed or aligned offsets, custom offsets are not supported.
    /// There are some cases in which it is ambiguous whether the offsets are packed or aligned, and it may affect the
    /// computed total itemsize of the struct. In these cases, consider using the explicit [`Self::new_struct`].
    pub fn from_fields(fields: Vec<(String, Itemsize, Dtype)>) -> Result<Self, DtypeError> {
        let mut seen_names = HashSet::new();
        for (name, _offset, _dtype) in &fields {
            if !seen_names.insert(name) {
                return Err(DtypeError::InvalidNames);
            }
        }

        let mut fields = fields;
        fields.sort_by_key(|(_name, offset, _dtype)| *offset);

        fn determine_itemsize_and_alignment(
            fields: &[(String, Itemsize, Dtype)],
        ) -> Result<(Itemsize, (Alignment, bool)), DtypeError> {
            let mut expected_offset = 0;
            let is_aligned = fields.iter().all({
                |(_f_name, offset, dtype)| {
                    expected_offset =
                        expected_offset.ceil_to_multiple(dtype.alignment() as Itemsize);
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
                    .unwrap_or(1);
                let itemsize = expected_offset.ceil_to_multiple(max_alignment as Itemsize);
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
                return Ok((itemsize, (1, false)));
            }

            Err(DtypeError::InvalidOffsets)
        }

        let (itemsize, (alignment, is_aligned)) = determine_itemsize_and_alignment(&fields)?;
        Ok(Self {
            kind: DtypeKind::Struct {
                fields: fields.into_boxed_slice(),
            },
            shape: DtypeShape::new(),
            itemsize,
            alignment: (alignment, is_aligned),
        })
    }

    /// Creates a new struct dtype by specifying all of the parameters explicitly.
    ///
    /// Thw shape, itemsize and alignment will be validated against the fields.
    /// See [`DtypeError`] for their constraints.
    pub fn new_struct(
        fields: Vec<(String, Itemsize, Dtype)>,
        shape: &[Itemsize],
        itemsize: Itemsize,
        alignment: Alignment,
    ) -> Result<Self, DtypeError> {
        let shape: DtypeShape = shape.try_into().map_err(|_| DtypeError::InvalidShape)?;
        let shape_prod = shape
            .iter()
            .try_fold(1 as Itemsize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(DtypeError::InvalidShape)?; // overflow
        if shape_prod == 0 {
            return Err(DtypeError::InvalidShape);
        }
        if !itemsize.is_multiple_of(shape_prod) {
            return Err(DtypeError::InvalidItemsize);
        }
        let element_itemsize = itemsize / shape_prod;

        let is_aligned;
        if Self::is_aligned_struct(&fields, element_itemsize, alignment) {
            is_aligned = true;
        } else if Self::is_packed_struct(&fields, element_itemsize, alignment) {
            is_aligned = false;
        } else {
            return Err(DtypeError::InvalidOffsets);
        }

        Ok(Self {
            kind: DtypeKind::Struct {
                fields: fields.into_boxed_slice(),
            },
            shape,
            itemsize,
            alignment: (alignment, is_aligned),
        })
    }

    fn is_aligned_struct(
        fields: &[(String, Itemsize, Dtype)],
        itemsize: Itemsize,
        alignment: Alignment,
    ) -> bool {
        let max_alignment = fields
            .iter()
            .map(|(_name, _offset, dtype)| dtype.alignment())
            .max()
            .unwrap_or(1);
        if alignment != max_alignment {
            return false;
        }

        let mut expected_offset = 0;
        let is_aligned = fields.iter().all({
            |(_name, offset, dtype)| {
                expected_offset = expected_offset.ceil_to_multiple(dtype.alignment() as Itemsize);
                let aligned = *offset == expected_offset;
                expected_offset += dtype.itemsize();
                aligned
            }
        });
        if !is_aligned {
            return false;
        }

        let expected_itemsize = expected_offset.ceil_to_multiple(max_alignment as Itemsize);
        if expected_itemsize != itemsize {
            return false;
        }

        true
    }

    fn is_packed_struct(
        fields: &[(String, Itemsize, Dtype)],
        itemsize: Itemsize,
        alignment: Alignment,
    ) -> bool {
        if alignment != 1 {
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
    /// Note that even if the function returns `Some`, the dtype may not be a plain scalar.
    /// For example, the dtype can have non-empty shape.
    pub fn scalar_kind(&self) -> Option<DtypeScalarKind> {
        match &self.kind {
            DtypeKind::Scalar { kind, .. } => Some(*kind),
            DtypeKind::Struct { .. } => None,
        }
    }

    /// Get the fields of the dtype, if it is a struct dtype.
    pub fn fields(&self) -> Option<&[(String, Itemsize, Dtype)]> {
        match &self.kind {
            DtypeKind::Scalar { .. } => None,
            DtypeKind::Struct { fields } => Some(fields),
        }
    }

    /// Get the shape of the dtype.
    ///
    /// Empty shape means a single element of the dtype.
    pub fn shape(&self) -> &[Itemsize] {
        &self.shape
    }

    /// Get the itemsize of the dtype.
    ///
    /// If this dtype has a shape, the itemsize is the product of the shape dimensions and the base itemsize.
    pub fn itemsize(&self) -> Itemsize {
        self.itemsize
    }

    /// Get the alignment of the dtype.
    ///
    /// For scalar dtypes, the alignment is the same as [`DtypeScalarKind::alignment()`].
    /// For struct dtypes, the alignment is either `1` for packed dtypes, or the maximum alignment of the inner fields
    /// for aligned structs.
    pub fn alignment(&self) -> Alignment {
        self.alignment.0
    }

    /// Returns whether the dtype is aligned (like C structs) or packed.
    ///
    /// Packed structs have alignment of 1, and fields are laid out back to back without any padding.
    /// Aligned structs have alignment equal to the maximum alignment of their fields, and have padding between fields
    /// such that each field is aligned to its alignment requirement.
    pub fn is_aligned(&self) -> bool {
        self.alignment.1
    }

    /// Try to convert this dtype to a scalar dtype, if it matches a default scalar layout.
    pub fn try_to_scalar(&self) -> Option<DtypeScalarKind> {
        let scalar = self.scalar_kind()?;
        (Self::of_scalar(scalar) == *self).then_some(scalar)
    }

    /// Set the shape of the dtype, updating the itemsize accordingly.
    ///
    /// The maximum number of dimensions allowed in the shape is [`DTYPE_SHAPE_MAX_NDIM`].
    /// The product of the shape dimensions must not exceed [`Itemsize::MAX`].
    ///
    /// The itemsize will be updated to `itemsize *= new_shape.product() / old_shape.product()`.
    pub fn set_shape(&mut self, shape: &[Itemsize]) -> Result<(), DtypeError> {
        let shape: DtypeShape = shape.try_into().map_err(|_| DtypeError::InvalidShape)?; // too many dims
        let shape_prod = shape
            .iter()
            .cloned()
            .try_product()
            .ok_or(DtypeError::InvalidShape)?; // overflow
        if shape_prod == 0 {
            return Err(DtypeError::InvalidShape);
        }
        let current_shape_size = self.shape.iter().cloned().product::<Itemsize>();
        assert!(self.itemsize.is_multiple_of(current_shape_size));
        let base_itemsize = self.itemsize / current_shape_size;
        self.itemsize = base_itemsize
            .checked_mul(shape_prod)
            .ok_or(DtypeError::InvalidShape)?; // overflow
        self.shape = shape;
        Ok(())
    }
}

/// Error that can happen when creating a new [`Dtype`]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DtypeError {
    /// Invalid field names.
    ///
    /// For example non-unique field names.
    InvalidNames,
    /// Invalid field offsets.
    ///
    /// Structs should be either in fully packed or fully aligned offsets (like C structs), custom offsets are not
    /// supported.
    InvalidOffsets,
    /// Invalid itemsize.
    ///
    /// For scalar dtypes, the itemsize must match the scalar definition. For struct dtypes, the itemsize must be the
    /// total size of the struct, including any padding between fields and at the end of the struct.
    /// Zero sized types are not supported.
    InvalidItemsize,
    /// Invalid alignment.
    ///
    /// - If the dtype is a scalar, the alignment must match the scalar definition.
    ///   See [`DtypeScalarKind::alignment`] for details.
    /// - If the dtype is a struct with packed fields, the alignment must be 1.
    /// - If the dtype is a struct with aligned fields, the alignment must be the maximum alignment of any field.
    InvalidAlignment,
    /// Invalid shape.
    ///
    /// - Shape with zero dimension is not allowed.
    /// - Shapes with more than 4 dimensions are not allowed.
    /// - Shapes with total size (product of dimensions) that exceeds `Itemsize::MAX` are not allowed.
    InvalidShape,
}
impl std::fmt::Display for DtypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNames => write!(f, "Invalid field names"),
            Self::InvalidOffsets => write!(f, "Invalid field offsets"),
            Self::InvalidItemsize => write!(f, "Invalid itemsize"),
            Self::InvalidAlignment => write!(f, "Invalid alignment"),
            Self::InvalidShape => write!(f, "Invalid shape"),
        }
    }
}
impl std::error::Error for DtypeError {}

impl DtypeScalarKind {
    /// Get the size of the scalar in bytes.
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
    pub const fn alignment(&self) -> Alignment {
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
            Self::ComplexF32 => 4,
            Self::ComplexF64 => 8,
            Self::Bool => 1,
        }
    }
}
#[allow(unused)]
impl Endianness {
    /// Get the native endianness.
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
/// `zix` maintain the dtype of each array dynamically using a [`Dtype`].
/// For safe conversions between (typed erased) arrays and other typed arrays (for example [`ndarray::ArrayBase`])
/// the `Dtyped` trait is used to verify type compatibility.
///
/// The trait also force `Copy`, and elements in arrays should not implement `Drop`.
///
/// Use the derive macro [`Dtyped`] to automatically implement this trait for structs.
/// ```rust,ignore
/// use zix::dtype::{Dtype, Dtyped};
///
/// #[derive(Dtyped)]
/// #[repr(C)]
/// struct MyStruct {
///     a: i32,
///     b: u8,
/// }
/// let expected_dtype = Dtype::from_fields(vec![
///     ("a".to_string(), 0, i32::dtype()),
///     ("b".to_string(), 4, u8::dtype()),
/// ]);
/// assert_eq!(MyStruct::dtype(), expected_dtype.unwrap());
/// ```
///
/// # Safety
///
/// This trait is very unsafe, and the caller should implement it carefully, matching the type size,
/// alignment and inner fields of the type. Types implementing this should most likely be annotated with `#[repr(C)]`
/// or `#[repr(C, packed)]`, for aligned and packed fields respectively.
pub unsafe trait Dtyped: Copy + 'static {
    /// Get the dtype representing the type layout and inner fields.
    fn dtype() -> Dtype;
}

// Re-export derive macro
pub use zix_macros::Dtyped;

macro_rules! impl_dtyped_scalar {
    ($ty:ty, $kind:ident) => {
        unsafe impl Dtyped for $ty {
            fn dtype() -> Dtype {
                Dtype::of_scalar(DtypeScalarKind::$kind)
            }
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
impl_dtyped_scalar!(f16, F16);
impl_dtyped_scalar!(f32, F32);
impl_dtyped_scalar!(f64, F64);
impl_dtyped_scalar!(Complex<f32>, ComplexF32);
impl_dtyped_scalar!(Complex<f64>, ComplexF64);
impl_dtyped_scalar!(bool, Bool);

unsafe impl<T: Dtyped, const N: usize> Dtyped for [T; N] {
    fn dtype() -> Dtype {
        let mut dtype = T::dtype();
        let n: Itemsize = N
            .try_into()
            .expect("Array size exceeds maximum supported dtype shape size of Itemsize::MAX");
        dtype.shape.insert(0, n);
        dtype.itemsize *= n;
        dtype
    }
}

cfg_if::cfg_if! { if #[cfg(feature = "half")] {
    pub use half::f16;
} else {
    /// A 16-bit floating point type implementing the IEEE 754-2008 standard `binary16` a.k.a "half"
    /// format.
    ///
    /// Doesn't provide any arithmetic operations, but can be converted to/from `u16`.
    /// Enable the `half` feature to get a fully functional `f16` type.
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    #[allow(non_camel_case_types)]
    pub struct f16(u16);
    impl f16 {
        #[doc = concat!("Creates a new `f16` from its raw bit representation.")]
        pub const fn from_bits(bits: u16) -> Self {
            Self(bits)
        }
        #[doc = concat!("Get the raw bit representation of the `f16`.")]
        pub const fn to_bits(&self) -> u16 {
            self.0
        }
    }
} }

cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
    pub use num_complex::Complex;
} else {
    /// A complex number in Cartesian form.
    ///
    /// Doesn't provide any arithmetic operations, but expose the real and imaginary parts.
    /// Enable the `num-complex` feature to get a fully functional `Complex` type.
    ///
    /// `Complex<T>` is memory layout compatible with an array `[T; 2]`, which is compatible with
    /// libc, numpy, etc.
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct Complex<T> {
        /// Real portion of the complex number
        pub re: T,
        /// Imaginary portion of the complex number
        pub im: T,
    }
} }

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Scalar dtype basics ----

    #[test]
    fn scalar_itemsize_and_alignment() {
        let cases: &[(DtypeScalarKind, Itemsize, Alignment)] = &[
            (DtypeScalarKind::I8, 1, 1),
            (DtypeScalarKind::I16, 2, 2),
            (DtypeScalarKind::I32, 4, 4),
            (DtypeScalarKind::I64, 8, 8),
            (DtypeScalarKind::U8, 1, 1),
            (DtypeScalarKind::U16, 2, 2),
            (DtypeScalarKind::U32, 4, 4),
            (DtypeScalarKind::U64, 8, 8),
            (DtypeScalarKind::F16, 2, 2),
            (DtypeScalarKind::F32, 4, 4),
            (DtypeScalarKind::F64, 8, 8),
            (DtypeScalarKind::ComplexF32, 8, 4),
            (DtypeScalarKind::ComplexF64, 16, 8),
            (DtypeScalarKind::Bool, 1, 1),
        ];
        for &(kind, expected_size, expected_align) in cases {
            let d = Dtype::of_scalar(kind);
            assert_eq!(d.itemsize(), expected_size, "{kind:?} itemsize");
            assert_eq!(d.alignment(), expected_align, "{kind:?} alignment");
            assert_eq!(d.shape(), &[] as &[Itemsize], "{kind:?} shape");
            assert_eq!(d.scalar_kind(), Some(kind), "{kind:?} scalar_kind");
            assert!(d.fields().is_none(), "{kind:?} fields should be None");
        }
    }

    #[test]
    fn scalar_equality() {
        assert_eq!(i32::dtype(), i32::dtype());
        assert_ne!(i32::dtype(), i64::dtype());
        assert_ne!(i32::dtype(), u32::dtype());
        assert_ne!(i32::dtype(), f64::dtype());
    }

    // ---- Array dtype ----

    #[test]
    fn array_prepends_shape_and_scales_itemsize() {
        let d = <[f64; 77] as Dtyped>::dtype();
        assert_eq!(d.shape(), &[77]);
        assert_eq!(d.itemsize(), 77 * 8);
        assert_eq!(d.alignment(), 8);
        assert_eq!(d.scalar_kind(), Some(DtypeScalarKind::F64));
    }

    #[test]
    fn nested_array_accumulates_shape() {
        // [[i32; 3]; 2] should yield shape [2, 3]
        let d = <[[i32; 3]; 2] as Dtyped>::dtype();
        assert_eq!(d.shape(), &[2, 3]);
        assert_eq!(d.itemsize(), 2 * 3 * 4);
        assert_eq!(d.alignment(), 4);
    }

    #[test]
    fn array_size_1_has_shape() {
        let d = <[u8; 1] as Dtyped>::dtype();
        assert_eq!(d.shape(), &[1]);
        assert_eq!(d.itemsize(), 1);
    }

    // ---- from_fields ----

    #[test]
    fn from_fields_detects_packed_layout() {
        // u8 at 0, f64 at 1: contiguous → packed
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::dtype()),
            ("b".to_string(), 1, f64::dtype()),
        ])
        .unwrap();
        assert_eq!(dtype.itemsize(), 9);
        assert_eq!(dtype.alignment(), 1);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".to_string(), 0, u8::dtype()));
        assert_eq!(fields[1], ("b".to_string(), 1, f64::dtype()));
    }

    #[test]
    fn from_fields_detects_aligned_layout() {
        // u8 at 0, f64 at 8: gap filled with padding → aligned
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::dtype()),
            ("b".to_string(), 8, f64::dtype()),
        ])
        .unwrap();
        assert_eq!(dtype.itemsize(), 16); // total padded to alignment 8
        assert_eq!(dtype.alignment(), 8);
    }

    #[test]
    fn from_fields_ambiguous_single_field_detected_as_aligned() {
        // Single field: packed and aligned offsets are identical.
        // from_fields tries aligned first, so it always returns the aligned layout.
        let dtype = Dtype::from_fields(vec![("x".to_string(), 0, f64::dtype())]).unwrap();
        assert_eq!(dtype.itemsize(), 8);
        assert_eq!(dtype.alignment(), 8);
    }

    #[test]
    fn from_fields_ambiguous_i32_u8_detected_as_aligned() {
        // { a: i32 at 0, b: u8 at 4 } — offsets are valid for both packed and aligned layouts.
        // from_fields tries aligned first and returns alignment=4, itemsize=8 (padded to align 4).
        // Use new_struct() when explicit control is needed.
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, i32::dtype()),
            ("b".to_string(), 4, u8::dtype()),
        ])
        .unwrap();
        assert_eq!(dtype.alignment(), 4);
        assert_eq!(dtype.itemsize(), 8);
    }

    #[test]
    fn from_fields_sorts_by_offset() {
        // Fields given in reverse order — result should be sorted ascending by offset.
        let dtype = Dtype::from_fields(vec![
            ("b".to_string(), 1, f32::dtype()),
            ("a".to_string(), 0, u8::dtype()),
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
            ("x".to_string(), 0, i32::dtype()),
            ("x".to_string(), 4, i32::dtype()),
        ]);
        assert_eq!(result, Err(DtypeError::InvalidNames));
    }

    #[test]
    fn from_fields_invalid_offsets_errors() {
        // Offset 3 for f64 is neither packed (would be 1) nor aligned (would be 8).
        let result = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::dtype()),
            ("b".to_string(), 3, f64::dtype()),
        ]);
        assert_eq!(result, Err(DtypeError::InvalidOffsets));
    }

    // ---- new_struct ----

    #[test]
    fn new_struct_packed_explicit() {
        // Disambiguate the i32+u8 case: explicitly request packed layout.
        let dtype = Dtype::new_struct(
            vec![
                ("a".to_string(), 0, i32::dtype()),
                ("b".to_string(), 4, u8::dtype()),
            ],
            &[],
            5,
            1,
        )
        .unwrap();
        assert_eq!(dtype.itemsize(), 5);
        assert_eq!(dtype.alignment(), 1);
    }

    #[test]
    fn new_struct_aligned_explicit() {
        // Explicitly request aligned layout for i32+u8, getting the C struct size of 8.
        let dtype = Dtype::new_struct(
            vec![
                ("a".to_string(), 0, i32::dtype()),
                ("b".to_string(), 4, u8::dtype()),
            ],
            &[],
            8,
            4,
        )
        .unwrap();
        assert_eq!(dtype.itemsize(), 8);
        assert_eq!(dtype.alignment(), 4);
    }

    #[test]
    fn new_struct_with_multidim_shape() {
        let dtype = Dtype::new_struct(
            vec![("a".to_string(), 0, u8::dtype())],
            &[2, 3],
            6, // 2*3*1
            1,
        )
        .unwrap();
        assert_eq!(dtype.shape(), &[2, 3]);
        assert_eq!(dtype.itemsize(), 6);
    }

    #[test]
    fn new_struct_shape_zero_errors() {
        assert_eq!(
            Dtype::new_struct(vec![("a".to_string(), 0, u8::dtype())], &[0], 0, 1),
            Err(DtypeError::InvalidShape),
        );
    }

    #[test]
    fn new_struct_too_many_dims_errors() {
        // DTYPE_SHAPE_MAX_NDIM = 4; five dimensions must be rejected.
        assert_eq!(
            Dtype::new_struct(
                vec![("a".to_string(), 0, u8::dtype())],
                &[1, 1, 1, 1, 1],
                1,
                1,
            ),
            Err(DtypeError::InvalidShape),
        );
    }

    #[test]
    fn new_struct_max_dims_works() {
        // Exactly DTYPE_SHAPE_MAX_NDIM = 4 dimensions should succeed.
        let dtype = Dtype::new_struct(
            vec![("a".to_string(), 0, u8::dtype())],
            &[1, 2, 3, 4],
            24, // 1*2*3*4
            1,
        )
        .unwrap();
        assert_eq!(dtype.shape(), &[1, 2, 3, 4]);
        assert_eq!(dtype.itemsize(), 24);
    }

    #[test]
    fn new_struct_itemsize_not_multiple_of_shape_errors() {
        // shape=[3], element must be 4 bytes (i32), total must be 12; 10 is not valid.
        assert_eq!(
            Dtype::new_struct(vec![("a".to_string(), 0, i32::dtype())], &[3], 10, 4),
            Err(DtypeError::InvalidItemsize),
        );
    }

    #[test]
    fn new_struct_wrong_alignment_errors() {
        // f64 field requires alignment 8; declaring alignment 4 is rejected as invalid offsets
        // (alignment 4 matches neither packed=1 nor aligned=8 layout).
        assert_eq!(
            Dtype::new_struct(vec![("a".to_string(), 0, f64::dtype())], &[], 8, 4),
            Err(DtypeError::InvalidOffsets),
        );
    }

    #[test]
    fn new_struct_packed_wrong_offset_errors() {
        // Packed struct (alignment=1): b must be at offset 4, not 5.
        assert_eq!(
            Dtype::new_struct(
                vec![
                    ("a".to_string(), 0, i32::dtype()),
                    ("b".to_string(), 5, u8::dtype()),
                ],
                &[],
                6,
                1,
            ),
            Err(DtypeError::InvalidOffsets),
        );
    }

    #[test]
    fn new_struct_itemsize_too_small_errors() {
        // Packed i32+u8 must be exactly 5; declaring 4 is rejected as invalid offsets
        // (is_packed_struct checks itemsize as part of offset validation).
        assert_eq!(
            Dtype::new_struct(
                vec![
                    ("a".to_string(), 0, i32::dtype()),
                    ("b".to_string(), 4, u8::dtype()),
                ],
                &[],
                4,
                1,
            ),
            Err(DtypeError::InvalidOffsets),
        );
    }

    #[test]
    fn dtype_error_display_is_non_empty() {
        for e in [
            DtypeError::InvalidNames,
            DtypeError::InvalidOffsets,
            DtypeError::InvalidItemsize,
            DtypeError::InvalidAlignment,
            DtypeError::InvalidShape,
        ] {
            assert!(
                !e.to_string().is_empty(),
                "{e:?} Display should not be empty"
            );
        }
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
        let dtype = SimpleStruct::dtype();
        assert_eq!(
            dtype.itemsize() as usize,
            std::mem::size_of::<SimpleStruct>()
        );
        assert_eq!(
            dtype.alignment() as usize,
            std::mem::align_of::<SimpleStruct>()
        );
        assert_eq!(dtype.itemsize(), 12);
        assert_eq!(dtype.alignment(), 4);
        assert_eq!(dtype.shape(), &[]);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], ("a".to_string(), 0, u8::dtype()));
        assert_eq!(fields[1], ("b".to_string(), 4, f32::dtype()));
        assert_eq!(fields[2], ("c".to_string(), 8, i16::dtype()));
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
        let dtype = PackedStruct::dtype();
        assert_eq!(
            dtype.itemsize() as usize,
            std::mem::size_of::<PackedStruct>()
        );
        assert_eq!(
            dtype.alignment() as usize,
            std::mem::align_of::<PackedStruct>()
        );
        assert_eq!(dtype.itemsize(), 7);
        assert_eq!(dtype.alignment(), 1);
        assert_eq!(dtype.shape(), &[]);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], ("a".to_string(), 0, u8::dtype()));
        assert_eq!(fields[1], ("b".to_string(), 1, f32::dtype()));
        assert_eq!(fields[2], ("c".to_string(), 5, i16::dtype()));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(transparent)]
    struct NewtypeWrapper(SimpleStruct);

    #[test]
    fn derive_transparent_newtype() {
        assert_eq!(NewtypeWrapper::dtype(), SimpleStruct::dtype());
        assert_eq!(
            NewtypeWrapper::dtype().itemsize() as usize,
            std::mem::size_of::<NewtypeWrapper>()
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
        let dtype = NestedStruct::dtype();
        assert_eq!(
            dtype.itemsize() as usize,
            std::mem::size_of::<NestedStruct>()
        );
        assert_eq!(
            dtype.alignment() as usize,
            std::mem::align_of::<NestedStruct>()
        );
        // SimpleStruct: 12 bytes, align 4. f64: 8 bytes, align 8.
        // a at 0, b at ceil(12, 8)=16, total ceil(24, 8)=24
        assert_eq!(dtype.itemsize(), 24);
        assert_eq!(dtype.alignment(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".to_string(), 0, SimpleStruct::dtype()));
        assert_eq!(fields[1], ("b".to_string(), 16, f64::dtype()));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct ArrayFieldStruct {
        a: [i32; 3],
    }

    #[test]
    fn derive_struct_with_array_field() {
        let dtype = ArrayFieldStruct::dtype();
        assert_eq!(
            dtype.itemsize() as usize,
            std::mem::size_of::<ArrayFieldStruct>()
        );
        assert_eq!(
            dtype.alignment() as usize,
            std::mem::align_of::<ArrayFieldStruct>()
        );
        assert_eq!(dtype.itemsize(), 12);
        assert_eq!(dtype.alignment(), 4);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "a");
        assert_eq!(fields[0].1, 0);
        // The field dtype should carry the array shape.
        assert_eq!(fields[0].2.shape(), &[3]);
        assert_eq!(fields[0].2.itemsize(), 12);
        assert_eq!(fields[0].2.scalar_kind(), Some(DtypeScalarKind::I32));
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct GenericStruct<T> {
        a: T,
        b: f64,
    }

    #[test]
    fn derive_generic_struct_with_i32() {
        let dtype = GenericStruct::<i32>::dtype();
        assert_eq!(
            dtype.itemsize() as usize,
            std::mem::size_of::<GenericStruct<i32>>()
        );
        assert_eq!(
            dtype.alignment() as usize,
            std::mem::align_of::<GenericStruct<i32>>()
        );
        // i32 (4, align 4) at 0; f64 (8, align 8) at ceil(4,8)=8; total ceil(16,8)=16
        assert_eq!(dtype.itemsize(), 16);
        assert_eq!(dtype.alignment(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".to_string(), 0, i32::dtype()));
        assert_eq!(fields[1].0, "b");
        assert_eq!(fields[1].1, 8);
    }

    #[test]
    fn derive_generic_struct_with_u8() {
        let dtype = GenericStruct::<u8>::dtype();
        assert_eq!(
            dtype.itemsize() as usize,
            std::mem::size_of::<GenericStruct<u8>>()
        );
        // u8 (1, align 1) at 0; f64 (8, align 8) at ceil(1,8)=8; total ceil(16,8)=16
        assert_eq!(dtype.itemsize(), 16);
        assert_eq!(dtype.alignment(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".to_string(), 0, u8::dtype()));
        assert_eq!(fields[1].1, 8);
    }

    #[test]
    fn derive_generic_struct_with_f64() {
        // Both fields f64 — no padding needed.
        let dtype = GenericStruct::<f64>::dtype();
        assert_eq!(dtype.itemsize(), 16);
        assert_eq!(dtype.alignment(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("a".to_string(), 0, f64::dtype()));
        assert_eq!(fields[1], ("b".to_string(), 8, f64::dtype()));
    }

    #[test]
    fn derive_generic_struct_with_non_dtyped_type() {
        // If T is not Dtyped, GenericStruct<T> should not be Dtyped.
        static_assertions::assert_not_impl_any!(GenericStruct<std::path::PathBuf>: Dtyped);
    }

    #[derive(Copy, Clone, Dtyped)]
    #[repr(C)]
    struct DeepNested {
        inner: NestedStruct,
        x: u32,
    }

    #[test]
    fn derive_deeply_nested_struct() {
        let dtype = DeepNested::dtype();
        assert_eq!(dtype.itemsize() as usize, std::mem::size_of::<DeepNested>());
        assert_eq!(
            dtype.alignment() as usize,
            std::mem::align_of::<DeepNested>()
        );
        // NestedStruct: 24 bytes, align 8. u32: 4 bytes, align 4.
        // inner at 0, x at ceil(24,4)=24, total ceil(28,8)=32
        assert_eq!(dtype.itemsize(), 32);
        assert_eq!(dtype.alignment(), 8);
        let fields = dtype.fields().unwrap();
        assert_eq!(fields[0], ("inner".to_string(), 0, NestedStruct::dtype()));
        assert_eq!(fields[1], ("x".to_string(), 24, u32::dtype()));
    }

    // ---- set_shape ----

    #[test]
    fn set_shape_on_scalar_sets_shape_and_scales_itemsize() {
        let mut d = i32::dtype();
        assert_eq!(d.shape(), &[] as &[Itemsize]);
        assert_eq!(d.itemsize(), 4);
        d.set_shape(&[3, 2]).unwrap();
        assert_eq!(d.shape(), &[3, 2]);
        assert_eq!(d.itemsize(), 4 * 3 * 2);
        assert_eq!(d.alignment(), 4);
        assert_eq!(d.scalar_kind(), Some(DtypeScalarKind::I32));
    }

    #[test]
    fn set_shape_empty_resets_to_scalar() {
        let mut d = <[u8; 5] as Dtyped>::dtype();
        assert_eq!(d.shape(), &[5]);
        assert_eq!(d.itemsize(), 5);
        d.set_shape(&[]).unwrap();
        assert_eq!(d.shape(), &[] as &[Itemsize]);
        assert_eq!(d.itemsize(), 1);
    }

    #[test]
    fn set_shape_replaces_existing_shape() {
        let mut d = <[[f32; 4]; 2] as Dtyped>::dtype();
        assert_eq!(d.shape(), &[2, 4]);
        assert_eq!(d.itemsize(), 2 * 4 * 4);
        d.set_shape(&[10]).unwrap();
        assert_eq!(d.shape(), &[10]);
        assert_eq!(d.itemsize(), 10 * 4);
    }

    #[test]
    fn set_shape_zero_dimension_errors() {
        let mut d = i32::dtype();
        assert_eq!(d.set_shape(&[0]), Err(DtypeError::InvalidShape));
        assert_eq!(d.set_shape(&[3, 0, 1]), Err(DtypeError::InvalidShape));
    }

    #[test]
    fn set_shape_too_many_dims_errors() {
        let mut d = u8::dtype();
        assert_eq!(d.set_shape(&[1, 1, 1, 1, 1]), Err(DtypeError::InvalidShape));
    }

    #[test]
    fn set_shape_max_dims_succeeds() {
        let mut d = u8::dtype();
        d.set_shape(&[1, 1, 1, 1]).unwrap();
        assert_eq!(d.shape(), &[1, 1, 1, 1]);
        assert_eq!(d.itemsize(), 1);
    }

    #[test]
    fn set_shape_overflow_errors() {
        let mut d = u8::dtype();
        assert_eq!(
            d.set_shape(&[Itemsize::MAX, Itemsize::MAX]),
            Err(DtypeError::InvalidShape)
        );
    }
}

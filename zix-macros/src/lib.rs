#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Procedural macros for the [`zix`] crate.
//!
//! ## `#[derive(Dtyped)]`
//!
//! Implements the [`zix::dtype::Dtyped`] trait for a `#[repr(C)]` or `#[repr(C, packed)]`
//! struct, making it usable as an element type in a [`zix::Array`].
//!
//! `Dtyped` maps the struct's layout to a [`zix::dtype::Dtype`] at compile time: it walks the
//! fields, computes byte offsets respecting C alignment rules (or packed layout when
//! `#[repr(packed)]` is used), and records field names so that individual fields can be accessed
//! as array views at runtime.
//!
//! ```rust,ignore
//! use zix::dtype::Dtyped;
//!
//! #[derive(Copy, Clone, Dtyped)]
//! #[repr(C)]
//! struct Pixel { r: u8, g: u8, b: u8 }
//!
//! // Pixel::DTYPE is a struct dtype with three u8 fields at offsets 0, 1, 2.
//! assert_eq!(Pixel::DTYPE.itemsize(), 3);
//! ```
//!
//! ### Requirements
//!
//! - The struct must be `#[repr(C)]` or `#[repr(C, packed)]` / `#[repr(packed)]`.
//! - Every field must itself implement `Dtyped`.
//! - Unit structs and enums are not supported.
//! - Tuple structs must be `#[repr(transparent)]` and contain exactly one field.

extern crate proc_macro;

use proc_macro::TokenStream;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Meta};

/// Derive macro generating an impl of the trait `Dtyped`.
///
/// See `zix::dtype::Dtyped` for more details on the trait and its requirements.
#[proc_macro_derive(Dtyped)]
pub fn derive_dtyped(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive_dtyped_impl(input).unwrap_or_else(|err| syn::Error::into_compile_error(err).into())
}

fn derive_dtyped_impl(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let repr_attributes = input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("repr"))
        .collect::<Vec<_>>();
    let repr_attribute = match repr_attributes.len() {
        0 => {
            return Err(syn::Error::new_spanned(
                input,
                "Missing #[repr] attribute for Dtyped",
            ));
        }
        1 => repr_attributes[0],
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "Only one #[repr] attribute is allowed for Dtyped",
            ));
        }
    };
    let Meta::List(repr_attribute) = &repr_attribute.meta else {
        return Err(syn::Error::new_spanned(
            repr_attribute,
            "Invalid repr attribute for Dtyped",
        ));
    };
    let repr_attribute = repr_attribute.tokens.to_string();
    let mut repr_attribute = repr_attribute
        .split(',')
        .map(|s| s.trim())
        .collect::<Vec<_>>();
    repr_attribute.sort();
    let repr_attribute = repr_attribute.join(",");

    let Data::Struct(data_struct) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "Dtyped can only be derived for structs.",
        ));
    };

    let zix_crate = match proc_macro_crate::crate_name("zix").expect("zix crate not found") {
        proc_macro_crate::FoundCrate::Itself => quote::quote! { crate },
        proc_macro_crate::FoundCrate::Name(name) => quote::quote! { ::#name },
    };
    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let field_types;

    let tokens = match &data_struct.fields {
        Fields::Named(fields) => {
            let is_packed = match repr_attribute.as_str() {
                "C" => false,
                "C,packed" | "packed" => true,
                unknown => {
                    return Err(syn::Error::new_spanned(
                        input,
                        format!("Unsupported repr attributes for Dtyped: {unknown}"),
                    ));
                }
            };

            field_types = fields.named.iter().map(|f| &f.ty).collect::<Vec<_>>();
            let field_names = fields
                .named
                .iter()
                .map(|f| f.ident.clone())
                .collect::<Vec<_>>();
            let var_field_dtype = field_names
                .iter()
                .enumerate()
                .map(|(i, f_name)| {
                    let f_name = f_name.as_ref().unwrap();
                    let var_name = format!("field_{i}_{}_dtype", f_name);
                    syn::Ident::new(&var_name.to_uppercase(), input.span())
                })
                .collect::<Vec<_>>();
            let var_field_offset = field_names
                .iter()
                .enumerate()
                .map(|(i, f_name)| {
                    let f_name = f_name.as_ref().unwrap();
                    let var_name = format!("field_{i}_{}_offset", f_name);
                    syn::Ident::new(&var_name.to_uppercase(), input.span())
                })
                .collect::<Vec<_>>();
            let var_field_end_offset = field_names
                .iter()
                .enumerate()
                .map(|(i, f_name)| {
                    let f_name = f_name.as_ref().unwrap();
                    let var_name = format!("field_{i}_{}_end_offset", f_name);
                    syn::Ident::new(&var_name.to_uppercase(), input.span())
                })
                .collect::<Vec<_>>();
            let var_prev_field_offset = [syn::Ident::new("BASE_OFFSET", input.span())]
                .into_iter()
                .chain(
                    var_field_end_offset
                        .iter()
                        .cloned()
                        .take(var_field_end_offset.len().saturating_sub(1)),
                )
                .collect::<Vec<_>>();
            let last_field_end_offset = var_field_end_offset
                .last()
                .cloned()
                .unwrap_or_else(|| syn::Ident::new("BASE_OFFSET", input.span()));

            quote::quote! {
                const BASE_OFFSET: #zix_crate::dtype::Itemsize = 0;

                const fn ceil_to_multiple(x: #zix_crate::dtype::Itemsize, m: #zix_crate::dtype::Itemsize) -> #zix_crate::dtype::Itemsize {
                    assert!(m > 0);
                    x.div_ceil(m) * m
                }

                #(
                    const #var_field_dtype: #zix_crate::dtype::Dtype = <#field_types as #zix_crate::dtype::Dtyped>::DTYPE;

                    const #var_field_offset: #zix_crate::dtype::Itemsize = {
                        let mut offset = #var_prev_field_offset;
                        if ! #is_packed {
                            offset = ceil_to_multiple(offset, align_of::<#field_types>() as #zix_crate::dtype::Itemsize);
                        }
                        offset
                    };

                    const #var_field_end_offset: #zix_crate::dtype::Itemsize = #var_field_offset + size_of::<#field_types>() as #zix_crate::dtype::Itemsize;
                )*

                const FIELDS: &'static [(std::borrow::Cow<'static, str>, #zix_crate::dtype::Itemsize, #zix_crate::dtype::Dtype)] = &[
                    #(
                        (std::borrow::Cow::Borrowed(stringify!(#field_names)), #var_field_offset, #var_field_dtype)
                    ),*
                ];

                let mut alignment = 1;
                let mut total_size = #last_field_end_offset;
                if !#is_packed {
                    #( {
                        let field_alignment = align_of::<#field_types>() as #zix_crate::dtype::Itemsize;
                        if field_alignment > alignment {
                            alignment = field_alignment;
                        }
                    } )*

                    total_size = ceil_to_multiple(total_size, alignment as #zix_crate::dtype::Itemsize);
                }
                let alignment = #zix_crate::dtype::Alignment::new(alignment as usize).unwrap();

                let dtype = unsafe { #zix_crate::dtype::Dtype::new_struct_borrowed_unchecked(
                    FIELDS,
                    total_size,
                    alignment,
                    #is_packed,
                ) };

                assert!(dtype.itemsize() as usize == size_of::<Self>());
                assert!(dtype.alignment().as_usize() == align_of::<Self>());
                dtype
            }
        }
        Fields::Unnamed(fields) => {
            if repr_attribute != "transparent" {
                return Err(syn::Error::new_spanned(
                    input,
                    "Dtyped can only be derived for structs with unnamed fields with 'transparent' repr",
                ));
            }

            let fields = fields.unnamed.iter().collect::<Vec<_>>();
            if fields.len() != 1 {
                return Err(syn::Error::new_spanned(
                    input,
                    "Dtyped can not be derived for structs with multiple unnamed fields.",
                ));
            }
            let field_type = &fields[0].ty;
            field_types = vec![field_type];

            quote::quote! {
                assert!(size_of::<#field_type>() == size_of::<Self>());
                assert!(align_of::<#field_type>() == align_of::<Self>());
                <#field_type as #zix_crate::dtype::Dtyped>::DTYPE
            }
        }
        Fields::Unit => {
            return Err(syn::Error::new_spanned(
                input,
                "Dtyped can not be derived for unit structs.",
            ));
        }
    };

    let tokens = quote::quote! {
        unsafe impl #impl_generics #zix_crate::dtype::Dtyped for #struct_name #ty_generics
        where
            #(#field_types: #zix_crate::dtype::Dtyped,)*
            #where_clause
        {
            const DTYPE: #zix_crate::dtype::Dtype = { #tokens };
        }
    };

    Ok(TokenStream::from(tokens))
}

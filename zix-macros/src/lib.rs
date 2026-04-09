extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{Data, DeriveInput, Fields, Meta, parse_macro_input};

/// Derive macro generating an impl of the trait `Dtyped`.
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
            let field_names = fields.named.iter().map(|f| f.ident.clone());
            let n_fields = field_types.len();

            quote::quote! {
                unsafe impl #impl_generics #zix_crate::dtype::Dtyped for #struct_name #ty_generics
                where
                    #(#field_types: #zix_crate::dtype::Dtyped,)*
                    #where_clause
                {
                    fn dtype() -> #zix_crate::dtype::Dtype {
                        let mut total_size: #zix_crate::dtype::Itemsize = 0;
                        let mut fields = Vec::with_capacity(#n_fields);
                        let mut alignment: #zix_crate::dtype::Alignment = 1;

                        fn ceil_to_multiple(x: #zix_crate::dtype::Itemsize, m: #zix_crate::dtype::Itemsize) -> #zix_crate::dtype::Itemsize {
                            assert!(m > 0);
                            x.div_ceil(m) * m
                        }

                        #({
                            let field_dtype = <#field_types as #zix_crate::dtype::Dtyped>::dtype();
                            let field_size = field_dtype.itemsize();

                            if ! #is_packed {
                                let field_alignment = field_dtype.alignment();
                                alignment = alignment.max(field_alignment);
                                total_size = ceil_to_multiple(total_size, field_alignment as #zix_crate::dtype::Itemsize);
                            }

                            let offset = total_size;
                            total_size += field_size;
                            fields.push((stringify!(#field_names).to_string(), offset, field_dtype));
                        })*
                        if ! #is_packed {
                            total_size = ceil_to_multiple(total_size, alignment as #zix_crate::dtype::Itemsize);
                        }


                        let dtype = #zix_crate::dtype::Dtype {
                            kind: #zix_crate::dtype::DtypeKind::Struct { fields: fields.into_boxed_slice() },
                            shape: Default::default(),
                            itemsize: total_size,
                            alignment: (alignment, !#is_packed),
                        };

                        debug_assert_eq!(dtype.itemsize() as usize, std::mem::size_of::<Self>());
                        debug_assert_eq!(dtype.alignment() as usize, std::mem::align_of::<Self>());
                        dtype
                    }
                }
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
                unsafe impl #impl_generics #zix_crate::dtype::Dtyped for #struct_name #ty_generics
                where
                    #(#field_types: #zix_crate::dtype::Dtyped,)*
                    #where_clause
                {
                    fn dtype() -> #zix_crate::dtype::Dtype {
                        let dtype = <#field_type as #zix_crate::dtype::Dtyped>::dtype();
                        debug_assert_eq!(dtype.itemsize() as usize, std::mem::size_of::<Self>());
                        debug_assert_eq!(dtype.alignment() as usize, std::mem::align_of::<Self>());
                        dtype
                    }
                }
            }
        }
        Fields::Unit => {
            return Err(syn::Error::new_spanned(
                input,
                "Dtyped can not be derived for unit structs.",
            ));
        }
    };

    Ok(TokenStream::from(tokens))
}

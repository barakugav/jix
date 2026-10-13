use std::fs;

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub_info = jix::__private::generate_pyi()?;
    stub_info.generate()?;
    let pyi_path = stub_info.python_root.join("jix").join("__init__.pyi");
    let mut pyi = fs::read_to_string(&pyi_path)?;

    // Add docs for function aliases
    pyi.push_str(
        r#"
def pow(a: typing.Any, b: typing.Any) -> Array:
    """Alias for [`jix.power`][jix.power]"""

def abs(array: typing.Any) -> Array:
    """Alias for [`jix.absolute`][jix.absolute]"""

def concat(arrays: typing.Sequence[typing.Any], dim: builtins.int = 0) -> Array:
    """Alias for [`jix.concatenate`][jix.concatenate]"""

def clip(array: typing.Any, min: typing.Optional[typing.Any] = None, max: typing.Optional[typing.Any] = None) -> Array:
    """Alias for [`jix.clamp`][jix.clamp]"""

def prod(array: typing.Any, dim: typing.Optional[builtins.int | typing.Sequence[builtins.int]] = None, *, keepdim: builtins.bool = False) -> Array:
    """Alias for [`jix.product`][jix.product]"""

def sub(a: typing.Any, b: typing.Any) -> Array:
    """Alias for [`jix.subtract`][jix.subtract]"""

def mul(a: typing.Any, b: typing.Any) -> Array:
    """Alias for [`jix.multiply`][jix.multiply]"""

def div(a: typing.Any, b: typing.Any) -> Array:
    """Alias for [`jix.divide`][jix.divide]"""

def neg(array: typing.Any) -> Array:
    """Alias for [`jix.negative`][jix.negative]"""
"#,
    );

    fs::write(&pyi_path, pyi)?;
    Ok(())
}

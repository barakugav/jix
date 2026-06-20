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

def concat(arrays: typing.Sequence[typing.Any], axis: builtins.int = 0) -> Array:
    """Alias for [`jix.concatenate`][jix.concatenate]"""
"#,
    );

    fs::write(&pyi_path, pyi)?;
    Ok(())
}

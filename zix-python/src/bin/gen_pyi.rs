use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    zix::__private::gen_pyi()?.generate()?;
    Ok(())
}

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    jix::__private::generate_pyi()?.generate()?;
    Ok(())
}

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    jix::__private::gen_pyi()?.generate()?;
    Ok(())
}

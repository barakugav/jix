from benches import run_all


def test_main_writes_outputs(tmp_path):
    out = tmp_path / "results"
    paths = run_all.main(out)
    assert any(p.suffix == ".png" for p in paths)
    assert any(p.name == "compress_ratios.md" for p in paths)
    for p in paths:
        assert p.exists()

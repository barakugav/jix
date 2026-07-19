import json

from benches import run_all

SYNTHETIC = {
    "benchmarks": [
        {"stats": {"mean": 0.001}, "extra_info": {"case": "c", "library": "jix", "size": 128, "ratio": 2.0}},
        {"stats": {"mean": 0.002}, "extra_info": {"case": "c", "library": "jix", "size": 256, "ratio": 2.1}},
    ]
}


def test_build_reports_writes_outputs(tmp_path):
    json_path = tmp_path / "python.json"
    json_path.write_text(json.dumps(SYNTHETIC))
    out = tmp_path / "out"
    paths = run_all.build_reports(json_path, out)
    assert any(p.suffix == ".png" for p in paths)
    assert any(p.name == "compress_ratios.md" for p in paths)
    for p in paths:
        assert p.exists()


def test_split_harness_args():
    assert run_all.split_harness_args(["--fast", "--", "-k", "op2"]) == (["--fast"], ["-k", "op2"])
    assert run_all.split_harness_args(["--out", "x"]) == (["--out", "x"], [])

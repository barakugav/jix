import json

from benches import run_all

SYNTHETIC = {
    "benchmarks": [
        {"stats": {"mean": 0.001}, "extra_info": {"section": "negate", "case": "f32", "library": "numpy"}},
        {"stats": {"mean": 0.002}, "extra_info": {"section": "negate", "case": "f32", "library": "jix"}},
    ]
}


def test_build_reports_writes_outputs(tmp_path):
    json_path = tmp_path / "python.json"
    json_path.write_text(json.dumps(SYNTHETIC))
    out = tmp_path / "out"
    paths = run_all.build_reports(json_path, out)
    assert [p.name for p in paths] == ["negate.png"]
    for path in paths:
        assert path.exists()


def test_split_harness_args():
    assert run_all.split_harness_args(["--fast", "--", "-k", "op2"]) == (["--fast"], ["-k", "op2"])
    assert run_all.split_harness_args(["--out", "x"]) == (["--out", "x"], [])

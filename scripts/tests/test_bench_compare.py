import json
from pathlib import Path

from bench import compare as bc


def _write_criterion(root: Path, group: str, bench: str, mean_ns: float, samples_ns):
    d = root / group / bench / "new"
    d.mkdir(parents=True)
    (d / "estimates.json").write_text(
        json.dumps(
            {
                "mean": {"point_estimate": mean_ns},
                "median": {"point_estimate": mean_ns},
                "std_dev": {"point_estimate": mean_ns * 0.1},
            }
        )
    )
    (d / "sample.json").write_text(
        json.dumps({"sampling_mode": "Flat", "iters": [1.0] * len(samples_ns), "times": list(samples_ns)})
    )


def test_load_rust_records(tmp_path):
    _write_criterion(tmp_path, "grp", "b1", 1000.0, [900.0, 1000.0, 1100.0])
    recs = bc.load_rust_records(tmp_path)
    assert len(recs) == 1
    r = recs[0]
    assert r.key == bc.BenchKey("rust", "grp", "b1", "")
    assert abs(r.mean - 1000.0 / 1e9) < 1e-18
    assert r.samples == [900.0 / 1e9, 1000.0 / 1e9, 1100.0 / 1e9]


def test_load_rust_records_three_level_no_collapse(tmp_path):
    # criterion nests some benches 3 levels deep: <group>/<dtype>/<axis>/new. Distinct groups
    # must keep distinct keys (the old 2-level logic collapsed all sizes onto one key).
    _write_criterion(tmp_path, "sum compact [40000, 300]/i32", "all", 1000.0, [1.0, 1.1])
    _write_criterion(tmp_path, "sum compact [400000, 300]/i32", "all", 2000.0, [2.0, 2.1])
    keys = {r.key for r in bc.load_rust_records(tmp_path)}
    assert len(keys) == 2
    assert bc.BenchKey("rust", "sum compact [40000, 300]", "i32/all", "") in keys


def test_load_platform_records_raises_on_duplicate_key(tmp_path):
    import pytest

    (tmp_path / "meta.json").write_text(json.dumps({"platform": {"os": "linux", "arch": "x86_64"}}))
    pj = tmp_path / "python" / "python.json"
    pj.parent.mkdir(parents=True)
    entry = {"stats": {"mean": 0.001, "data": [0.001, 0.001]}, "extra_info": {"case": "c", "library": "jix", "size": 1}}
    pj.write_text(json.dumps({"benchmarks": [entry, dict(entry)]}))  # same (case, library, size) twice
    with pytest.raises(ValueError, match="duplicate bench key"):
        bc.load_platform_records(tmp_path)


def test_load_python_records(tmp_path):
    pj = tmp_path / "python.json"
    pj.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "stats": {"mean": 0.001, "median": 0.001, "stddev": 0.0001, "data": [0.0009, 0.0011]},
                        "extra_info": {"case": "read", "library": "jix", "size": 256},
                    },
                    {"stats": {"mean": 0.5, "data": []}, "extra_info": {}},  # skipped: no case/library/size
                ]
            }
        )
    )
    recs = bc.load_python_records(pj)
    assert len(recs) == 1
    assert recs[0].key == bc.BenchKey("python", "read", "size=256", "jix")
    assert recs[0].samples == [0.0009, 0.0011]


import numpy as np  # noqa: E402


def _rng():
    return np.random.default_rng(0)


def _rec(mean, samples):
    return bc.BenchRecord(bc.BenchKey("rust", "g", "b", ""), list(samples), mean, mean, 0.0)


def test_verdict_no_change_for_identical():
    base = _rec(1.0, _rng().normal(1.0, 0.01, 200))
    new = bc.BenchRecord(base.key, list(np.random.default_rng(5).normal(1.0, 0.01, 200)), 1.0, 1.0, 0.0)
    c = bc.compare_records(
        base,
        new,
        nresamples=2000,
        significance=bc.DEFAULT_SIGNIFICANCE,
        noise=bc.DEFAULT_NOISE,
        confidence=bc.DEFAULT_CONFIDENCE,
        rng=_rng(),
    )
    assert c.verdict == "NoChange"


def test_verdict_regressed_for_large_increase():
    base = _rec(1.0, np.random.default_rng(1).normal(1.0, 0.01, 200))
    new = _rec(1.5, np.random.default_rng(2).normal(1.5, 0.01, 200))
    c = bc.compare_records(
        base,
        new,
        nresamples=2000,
        significance=bc.DEFAULT_SIGNIFICANCE,
        noise=bc.DEFAULT_NOISE,
        confidence=bc.DEFAULT_CONFIDENCE,
        rng=_rng(),
    )
    assert c.verdict == "Regressed"
    assert c.pct_change > 0.4


def test_verdict_improved_for_large_decrease():
    base = _rec(1.0, np.random.default_rng(3).normal(1.0, 0.01, 200))
    new = _rec(0.5, np.random.default_rng(4).normal(0.5, 0.01, 200))
    c = bc.compare_records(
        base,
        new,
        nresamples=2000,
        significance=bc.DEFAULT_SIGNIFICANCE,
        noise=bc.DEFAULT_NOISE,
        confidence=bc.DEFAULT_CONFIDENCE,
        rng=_rng(),
    )
    assert c.verdict == "Improved"
    assert c.pct_change < -0.4


def test_download_run_local_path(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    assert bc.download_run(str(tmp_path), tmp_path / "cache") == tmp_path


def test_download_run_run_id(monkeypatch, tmp_path):
    calls = []

    def fake_gh(args):
        calls.append(args)
        return ""

    monkeypatch.setattr(bc, "_gh", fake_gh)
    dest = bc.download_run("123456", tmp_path / "cache")
    assert dest == tmp_path / "cache" / "123456"
    assert any("download" in a for a in calls)
    assert any("123456" in a for a in calls)


def test_list_runs_parses_gh_json(monkeypatch):
    payload = json.dumps(
        [
            {
                "databaseId": 7,
                "headBranch": "main",
                "headSha": "abc",
                "status": "completed",
                "conclusion": "success",
                "createdAt": "2026-07-20T00:00:00Z",
                "displayTitle": "Benchmarks",
            }
        ]
    )
    monkeypatch.setattr(bc, "_gh", lambda args: payload)
    runs = bc.list_runs(5)
    assert runs[0]["databaseId"] == 7


def _make_run(dirpath, mean_ns):
    """Build a single result dir (meta.json + rust/) with one rust bench."""
    dirpath.mkdir(parents=True)
    (dirpath / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha": "x",
                "ref": "x",
                "platform": {"os": "linux", "arch": "x86_64", "cpu_model": "CPU"},
            }
        )
    )
    _write_criterion(
        dirpath / "rust" / "criterion", "g", "b", mean_ns, np.random.default_rng(0).normal(mean_ns, mean_ns * 0.01, 200)
    )


def test_compare_platform_and_report(tmp_path):
    base_dir = tmp_path / "A" / "bench-linux-x86_64-1"
    new_dir = tmp_path / "B" / "bench-linux-x86_64-2"
    _make_run(base_dir, 1000.0)
    _make_run(new_dir, 1500.0)
    base_run = bc.load_run(base_dir.parent)
    new_run = bc.load_run(new_dir.parent)
    _, base_recs, _ = base_run["linux-x86_64"]
    _, new_recs, _ = new_run["linux-x86_64"]
    comps, added, removed = bc.compare_platform(
        base_recs,
        new_recs,
        nresamples=2000,
        significance=bc.DEFAULT_SIGNIFICANCE,
        noise=bc.DEFAULT_NOISE,
        confidence=bc.DEFAULT_CONFIDENCE,
        rng=np.random.default_rng(0),
    )
    assert len(comps) == 1
    assert comps[0].verdict == "Regressed"
    out = tmp_path / "out"
    bc.write_report(out, {"linux-x86_64": (comps, added, removed)}, base_run, new_run, "markdown")
    assert (out / "report.md").exists()
    assert (out / "report.json").exists()


def test_resolve_nresamples_fast():
    assert bc._resolve_nresamples(None, fast=True) == bc.FAST_NRESAMPLES
    assert bc._resolve_nresamples(None, fast=False) == bc.DEFAULT_NRESAMPLES
    assert bc._resolve_nresamples(500, fast=True) == 500  # explicit wins


def _cmp(suite, group, bench, lib, base_mean, new_mean, pct, ci, verdict):
    return bc.Comparison(bc.BenchKey(suite, group, bench, lib), base_mean, new_mean, pct, ci, 0.001, verdict)


def _run_with_samples(cpu):
    key = bc.BenchKey("rust", "g", "b", "")
    recs = {key: bc.BenchRecord(key, list(np.random.default_rng(0).normal(1e-6, 1e-8, 50)), 1e-6, 1e-6, 0.0)}
    return {"linux-x86_64": ({"platform": {"os": "linux", "arch": "x86_64", "cpu_model": cpu}}, recs, None)}


def test_write_plots_generates_files(tmp_path):
    c1 = _cmp("rust", "g", "b", "", 1e-6, 1.5e-6, 0.5, (0.4, 0.6), "Regressed")
    c2 = _cmp("python", "read", "size=256", "jix", 1e-3, 0.9e-3, -0.1, (-0.15, -0.05), "Improved")
    per_platform = {
        "linux-x86_64": ([c1, c2], [], []),
        "macos-aarch64": ([c1], [], []),
    }
    base_run = _run_with_samples("CPU-A")
    new_run = _run_with_samples("CPU-A")
    out = tmp_path / "out"
    written = bc.write_plots(out, per_platform, base_run, new_run, bench_pattern="g/b")
    names = {p.name for p in written}
    assert "change_linux-x86_64.png" in names
    assert "scatter_linux-x86_64.png" in names
    assert "heatmap.png" in names
    assert any(n.startswith("violin_linux-x86_64") for n in names)
    for p in written:
        assert p.exists()

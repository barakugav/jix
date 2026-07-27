import json

from benches import meta


def _patch_env(monkeypatch):
    monkeypatch.setattr(meta, "_cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(meta, "_cpu_cores", lambda: 4)
    monkeypatch.setattr(meta, "_ram_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(meta, "_rustc", lambda: "rustc 1.89.0")


def test_collect(monkeypatch):
    _patch_env(monkeypatch)
    m = meta.collect("abcdef123456", "main")
    assert m["schema_version"] == 1
    assert m["sha"] == "abcdef123456"
    assert m["ref"] == "main"
    assert m["platform"]["cpu_model"] == "Test CPU"
    assert set(m["libs"]) == {"numpy", "blosc2", "zarr", "jix"}
    # workflow-run identifiers are dropped (available via gh instead)
    assert "workflow_run_id" not in m


def test_main_writes_file(monkeypatch, tmp_path):
    _patch_env(monkeypatch)
    out = tmp_path / "meta.json"
    meta.main(["--out", str(out), "--sha", "deadbeef", "--ref", "main"])
    assert json.loads(out.read_text())["sha"] == "deadbeef"

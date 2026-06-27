import json

from benches import report


def _bench(case, library, size, mean):
    return {"stats": {"mean": mean}, "extra_info": {"case": case, "library": library, "size": size}}


def _ratio_bench(case, library, size, ratio):
    b = _bench(case, library, size, 0.01)
    b["extra_info"].update(ratio=ratio, stored_bytes=100, raw_bytes=int(100 * ratio))
    return b


def test_plot_throughput_one_png_per_case(tmp_path):
    benches = [
        _bench("compress_smooth_b32x64", "jix", 256, 0.01),
        _bench("compress_smooth_b32x64", "jix", 1024, 0.02),
        _bench("compress_smooth_b32x64", "blosc2", 256, 0.03),
        _bench("compress_smooth_b32x64", "blosc2", 1024, 0.04),
        _bench("read_smooth_b16x64_r1x64", "jix", 1024, 0.001),
    ]
    pngs = report.plot_throughput(benches, tmp_path)
    names = sorted(p.name for p in pngs)
    assert names == ["compress_smooth_b32x64.png", "read_smooth_b16x64_r1x64.png"]
    for p in pngs:
        assert p.exists() and p.stat().st_size > 0


def test_write_ratio_table(tmp_path):
    benches = [
        _ratio_bench("compress_smooth_b32x64", "jix", 4096, 3.5),
        _ratio_bench("compress_smooth_b32x64", "blosc2", 4096, 2.0),
        _bench("read_smooth_b16x64_r1x64", "jix", 4096, 0.001),  # no ratio -> excluded
    ]
    md = report.write_ratio_table(benches, tmp_path, "zstd level 3")
    text = md.read_text()
    assert md.name == "compress_ratios.md"
    assert "jix ratio" in text and "blosc2 ratio" in text
    assert "3.50" in text and "2.00" in text
    assert "zstd level 3" in text
    assert "read_smooth_b16x64_r1x64" not in text  # only compress cases with a ratio appear


def test_load_pytest_json(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"benchmarks": [_bench("c", "jix", 8, 0.5)]}))
    rows = report.load_pytest_json(path)
    assert rows[0]["extra_info"]["case"] == "c"


def test_report_criterion(tmp_path):
    root = tmp_path / "criterion"
    est = root / "compact_array" / "shape-1024" / "new"
    est.mkdir(parents=True)
    (est / "estimates.json").write_text(json.dumps({"mean": {"point_estimate": 12345.0}}))
    written = report.report_criterion(root, tmp_path)
    assert any(p.suffix == ".png" for p in written)
    md = tmp_path / "rust_benchmarks.md"
    assert md.exists() and "compact_array" in md.read_text()

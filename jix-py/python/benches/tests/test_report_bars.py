import json

import pytest

from benches import report_bars, report_spec


def _bench(section, case, library, mean, **extra):
    return {"stats": {"mean": mean}, "extra_info": {"section": section, "case": case, "library": library, **extra}}


def test_load_python_reads_section_case_library(tmp_path):
    path = tmp_path / "python.json"
    entry = _bench("negate", "f32", "jix", 0.25)
    entry["stats"]["stddev"] = 0.01
    path.write_text(json.dumps({"benchmarks": [entry]}))
    (row,) = report_bars.load_python(path, "linux-x86_64")
    assert row == {
        "platform": "linux-x86_64",
        "section": "negate",
        "case": "f32",
        "library": "jix",
        "value": 0.25,
        "spread": 0.01,
    }


def test_recorded_metrics_carry_no_spread(tmp_path):
    """A stored size or an RSS reading has no standard deviation; only the timing does."""
    path = tmp_path / "python.json"
    entry = _bench("chain_memory", "4 ops", "jix", 9.9, peak_rss_bytes=1234, value_field="peak_rss_bytes")
    entry["stats"]["stddev"] = 0.5
    path.write_text(json.dumps({"benchmarks": [entry]}))
    (row,) = report_bars.load_python(path, "linux-x86_64")
    assert row["spread"] is None


def test_absolute_table_cell_shows_spread():
    section = {**SECTION, "cases": ["a"], "libraries": ["numpy"]}
    rows = [{"platform": "p", "section": "demo", "case": "a", "library": "numpy", "value": 0.25, "spread": 0.01}]
    assert "+/-" in report_bars.markdown_table(rows, section)


def test_load_python_skips_untagged_benchmarks(tmp_path):
    path = tmp_path / "python.json"
    path.write_text(json.dumps({"benchmarks": [{"stats": {"mean": 1.0}, "extra_info": {}}]}))
    assert report_bars.load_python(path, "linux-x86_64") == []


def test_value_field_overrides_the_measured_time(tmp_path):
    path = tmp_path / "python.json"
    entry = _bench("chain_memory", "4 ops", "jix", 9.9, peak_rss_bytes=1234, value_field="peak_rss_bytes")
    path.write_text(json.dumps({"benchmarks": [entry]}))
    (row,) = report_bars.load_python(path, "linux-x86_64")
    assert row["value"] == 1234


def test_also_emits_a_second_section(tmp_path):
    path = tmp_path / "python.json"
    entry = _bench("compress", "smooth", "jix", 0.5, ratio=4.0, also={"compress_ratio": "ratio"})
    path.write_text(json.dumps({"benchmarks": [entry]}))
    rows = report_bars.load_python(path, "linux-x86_64")
    assert {(r["section"], r["value"]) for r in rows} == {("compress", 0.5), ("compress_ratio", 4.0)}


def test_load_rust_maps_case_ids_to_labels_and_converts_to_seconds(tmp_path):
    estimates = tmp_path / "report_rust_op" / "jix" / "negate_f32" / "new" / "estimates.json"
    estimates.parent.mkdir(parents=True)
    estimates.write_text(json.dumps({"mean": {"point_estimate": 2_000_000.0}}))
    (row,) = report_bars.load_rust(tmp_path, "linux-aarch64", report_spec.RUST_CASE_LABELS)
    assert row["section"] == "rust_op"
    assert row["library"] == "jix"
    assert row["case"] == "negate\nf32"
    assert row["value"] == pytest.approx(0.002)  # criterion reports nanoseconds


def test_load_rust_ignores_groups_outside_the_report(tmp_path):
    estimates = tmp_path / "op1 plain" / "shape" / "new" / "estimates.json"
    estimates.parent.mkdir(parents=True)
    estimates.write_text(json.dumps({"mean": {"point_estimate": 1.0}}))
    assert report_bars.load_rust(tmp_path, "linux-x86_64") == []


SECTION = {
    "key": "demo",
    "title": "Demo",
    "baseline": "numpy",
    "metric": "time",
    "cases": ["a", "b"],
    "libraries": ["numpy", "jix"],
}


def _rows(platform="linux-x86_64"):
    return [
        {"platform": platform, "section": "demo", "case": case, "library": library, "value": value}
        for case, library, value in [("a", "numpy", 1.0), ("a", "jix", 2.0), ("b", "numpy", 1.0), ("b", "jix", 0.5)]
    ]


def test_plot_section_writes_a_png(tmp_path):
    out = report_bars.plot_section(_rows(), SECTION, tmp_path)
    assert out.exists() and out.name == "demo.png"


def test_plot_section_returns_none_without_data(tmp_path):
    assert report_bars.plot_section([], SECTION, tmp_path) is None


def test_markdown_table_has_a_row_per_case():
    table = report_bars.markdown_table(_rows(), SECTION)
    assert "| a |" in table and "| b |" in table
    assert "<details>" in table


def test_markdown_table_marks_missing_cells():
    rows = [r for r in _rows() if not (r["case"] == "b" and r["library"] == "jix")]
    assert "| - |" in report_bars.markdown_table(rows, SECTION).replace("  ", " ")


def test_every_section_has_matching_case_ids():
    for section in report_spec.SECTIONS:
        if "case_ids" in section:
            assert len(section["case_ids"]) == len(section["cases"]), section["key"]


def test_every_section_baseline_is_in_its_libraries():
    for section in report_spec.SECTIONS:
        if section["baseline"] is not None:
            assert section["baseline"] in section["libraries"], section["key"]

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from bench import gungraun_summary as gs

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (shutil.which("valgrind") and shutil.which("gungraun-runner")), reason="needs valgrind and gungraun-runner"
)
def test_bench_output_parses(tmp_path):
    out = subprocess.check_output(
        [
            "cargo",
            "bench",
            "--manifest-path",
            REPO / "jix/Cargo.toml",
            "--bench",
            "regression",
            "--",
            "--output-format=json",
        ],
        env={**os.environ, "GUNGRAUN_HOME": str(tmp_path)},
        text=True,
    )
    summary = tmp_path / "summary.jsonl"
    summary.write_text(out)
    rows = gs.load_rows(summary)
    assert len(rows) == 8
    assert all(row.metrics["Ir"].new > 0 for row in rows)
    gs.render_markdown(rows, 2.0)

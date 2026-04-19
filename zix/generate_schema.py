import os
import subprocess
from pathlib import Path

subprocess.check_call(
    ["cargo", "build", "--features=build-schema"],
    env={"ZIX_SCHEMA_GEN_UPDATE": "1"} | os.environ,
    cwd=Path(__file__).parent.resolve(),
)

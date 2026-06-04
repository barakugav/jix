import os
import subprocess
import sys
from pathlib import Path

subprocess.check_call(
    ["cargo", "+nightly", "miri", "test", *sys.argv[1:]],
    env={
        "PROPTEST_DISABLE_FAILURE_PERSISTENCE": "true",
        "MIRIFLAGS": "-Zmiri-env-forward=PROPTEST_DISABLE_FAILURE_PERSISTENCE",
    }
    | os.environ,
    cwd=Path(__file__).parent.resolve(),
)

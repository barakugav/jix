import os
import subprocess
import sys
from pathlib import Path

proptest_env = {
    "PROPTEST_DISABLE_FAILURE_PERSISTENCE": "true",
    "PROPTEST_CASES": "4",
}
subprocess.check_call(
    ["cargo", "+nightly", "miri", "test", *sys.argv[1:]],
    env={
        **proptest_env,
        "MIRIFLAGS": " ".join(f"-Zmiri-env-forward={k}" for k in proptest_env),
        **os.environ,
    },
    cwd=Path(__file__).parent.resolve(),
)

"""Save a finished bug-hunt workflow's report to bug-hunt/runs/<runId>/REPORT.md.

Subagents may not write the report, so the workflow returns it as `reportMarkdown` in its result.
Pass the workflow task's output file (the path in its completion notification):

    python3 bug-hunt/tools/save_report.py <task output file> <runId>
"""

import json
import os
import sys

out_file, run_id = sys.argv[1], sys.argv[2]
with open(out_file) as fh:
    d = json.load(fh)
r = d.get("result", d)
if isinstance(r, str):
    r = json.loads(r)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", run_id, "REPORT.md")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    f.write(r["reportMarkdown"])
print(f"wrote {os.path.normpath(path)} ({len(r['reportMarkdown'])} chars)")
for line in r.get("findings", []):
    print(" ", line)

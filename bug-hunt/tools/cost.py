"""Token cost of a finished workflow run, per agent, from its subagent transcripts.

Cost is in input-token equivalents: input + 0.1 * cache reads + 1.25 * cache writes (output is
negligible). A transcript logs an assistant message once per content block, each copy carrying the
same usage, so every message id is counted once.

    python3 bug-hunt/tools/cost.py ~/.claude/projects/<project>/<session>/subagents/workflows/<wf run id>
"""

import glob
import json
import os
import sys

d = sys.argv[1]
tot = 0
rows = []
for f in sorted(glob.glob(os.path.join(d, "agent-*.jsonl"))):
    meta = {}
    mf = f[:-6] + ".meta.json"
    if os.path.exists(mf):
        with open(mf) as fh:
            meta = json.load(fh)
    seen = set()
    c = 0
    turns = 0
    with open(f) as fh:
        lines = fh.readlines()
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("message") or {}
        if e.get("type") != "assistant" or not isinstance(m, dict):
            continue
        mid = m.get("id")
        if mid in seen:
            continue
        seen.add(mid)
        turns += 1
        u = m.get("usage") or {}
        c += (
            u.get("input_tokens", 0)
            + 0.1 * u.get("cache_read_input_tokens", 0)
            + 1.25 * u.get("cache_creation_input_tokens", 0)
        )
    label = meta.get("label") or meta.get("description") or meta.get("agentType") or ""
    rows.append((label, turns, c))
    tot += c
for label, t, c in sorted(rows, key=lambda r: r[0]):
    print(f"{c / 1e6:6.2f}M {t:4d} turns  {label}")
print(f"TOTAL {tot / 1e6:.1f}M over {len(rows)} agents")

# Known issues

Issues already reported and still open, or deliberately left as-is. Hunters skip these; the dedup
stage drops findings that match one. Add an entry when you triage a finding (open, won't fix,
intended) so later runs do not report it again. When an issue is fixed, re-run its repro, then
remove the entry (or mark it `fixed in <commit/PR>` until the next consolidation): a fixed issue
that comes back is a regression and should be reported again.

IDs are stable and never reused: K1-K183 were assigned by the 2026-10/11 scans and K184-K200 by the
two scans of #106 / #107 (the fixed ones were dropped when the scans were consolidated into
`report/`), so the next new entry is K201.

Format: one entry per issue.

```
## K<n>: <one-line title>
- Status: open | fixed in <commit/PR> | wont-fix | intended - <why>
- Location: <file:line>
- Notes: <what exactly is known; which triggers are covered>
```

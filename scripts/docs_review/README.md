# Docs review

A setup for reviewing and correcting the docs of this repo with subagents: the Rust doc comments
(which include the Python docstrings in `jix-py/src/`), the READMEs and the mkdocs pages.

It is a correction pass, not a rewrite. The style we are after is direct, simple and concise;
most docs should come out unchanged.

## How it works

The repo is split into **units** (`units.toml`), each a set of files owned by one agent. Every
file belongs to exactly one unit, so units can be scanned in parallel.

```
            scanner (parallel)          verifier              orchestrator (serial)
unit  --->  edits docs in its   --->    checks every   --->   gates, then one commit
            own worktree                hunk against          per unit, in order
                                        the code
```

1. **Scanner** - one agent per unit, in its own git worktree. It reads the unit's files, checks
   each doc against the code, and edits in place: it fixes what is wrong and cuts what repeats or
   pads. It writes a report: what it changed and why, what it left for the maintainer, and any
   code bug it ran into.
2. **Verifier** - a fresh agent that did not see the scanner's reasoning. It opens the code
   itself and keeps, reverts or amends each hunk: a factual change it cannot confirm, or one that
   drops a fact, is reverted. It writes the unit's findings and its commit summary.
3. **Orchestrator** - applies the verified diffs to the branch, runs the gates, reads the diffs,
   and commits. One commit per unit keeps the history linear and each commit easy to review or
   drop.

The agents never build anything: the builds are heavy and share one memory limit. The gates run
once per unit, on the orchestrator's side.

## Files

| File | What |
|------|------|
| `units.toml` | The units: files to edit, files to consult, context packs, notes. |
| `context/general.md` | What every agent reads first: the library, where docs live, vocabulary, renames. |
| `context/api_index.md` | Every public name of the Rust crate and the Python package. Generated. |
| `context/ops.md`, `python.md`, `read_model.md` | Concept packs, read by the units that need them. |
| `prompts/scanner.md`, `prompts/verifier.md` | The briefs of the two agents. |
| `unit_prompt.py` | Prints the launch prompt of a unit's scanner or verifier. |
| `land_units.py` | Applies the diffs of verified units to the branch, and commits them one by one. |
| `gates.sh` | Runs all gates on the uncommitted changes. |
| `check_docs_only.py` | Gate: only comments, docstrings and markdown changed. |
| `run_py_examples.py` | Gate: the Python examples in the docs run. |
| `gen_api_index.py` | Regenerates `context/api_index.md` from the build outputs. |
| `findings.md` | What the agents found and did not fix: code bugs, questions, missed items. |

## Running a unit

```bash
python scripts/docs_review/unit_prompt.py --list
```

Give the unit its own worktree of the branch, and pick a directory for the reports:

```bash
git worktree add --detach /tmp/docs-review/wt/ops-arith HEAD
```

Print the scanner's prompt, launch an agent with it, and when that agent is done do the same for
the verifier:

```bash
python scripts/docs_review/unit_prompt.py scanner ops-arith --workdir /tmp/docs-review/wt/ops-arith --reports /tmp/docs-review/reports
```

```bash
python scripts/docs_review/unit_prompt.py verifier ops-arith --workdir /tmp/docs-review/wt/ops-arith --reports /tmp/docs-review/reports
```

The agents leave four files in the reports directory: `ops-arith.scanner.md`,
`ops-arith.verifier.md`, `ops-arith.findings.md` and `ops-arith.commit.md`.

Then land the unit - or several at once, since units own disjoint files:

```bash
python scripts/docs_review/land_units.py apply --worktrees /tmp/docs-review/wt --reports /tmp/docs-review/reports ops-arith
```

```bash
scripts/docs_review/gates.sh
```

Read the diff, and commit. Each unit becomes one commit, `Docs review: <unit>`, with its findings
appended to `findings.md`:

```bash
python scripts/docs_review/land_units.py commit --reports /tmp/docs-review/reports ops-arith
```

The `cross-cutting` unit runs last and alone, on the result of all the others.

## Gates

- The change is docs-only: with comments removed, every `.rs` file is identical to before
  (`check_docs_only.py`).
- ASCII only (`scripts/check_only_ascii.py`).
- `cargo fmt --check`, `cargo doc` with `-D warnings`, and `cargo test --doc` for a touched crate.
- For `jix-py`: build, regenerate the stubs, `mkdocs build --strict`, and run the Python examples.

## Regenerating the API index

After a change to the public API:

```bash
(cd jix && cargo doc --all-features) && (cd jix-py && maturin develop) && python scripts/docs_review/gen_api_index.py
```

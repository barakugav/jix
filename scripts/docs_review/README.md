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
   each doc against the code, and edits in place. It ends with a report: what it changed and
   why, what it left for the maintainer, and any code bug it ran into.
2. **Verifier** - a fresh agent that did not see the scanner's reasoning. It opens the code
   itself and keeps, reverts or amends each hunk. When it cannot confirm a hunk, it reverts it.
3. **Orchestrator** - applies the verified diff to the branch, runs the gates, reads the diff,
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
| `gates.sh` | Runs all gates on the uncommitted changes. |
| `check_docs_only.py` | Gate: only comments, docstrings and markdown changed. |
| `run_py_examples.py` | Gate: the Python examples in the docs run. |
| `gen_api_index.py` | Regenerates `context/api_index.md` from the build outputs. |
| `findings.md` | What the agents found and did not fix: code bugs, questions, missed items. |

## Running a unit

```bash
python scripts/docs_review/unit_prompt.py --list
```

Give the unit its own worktree of the branch, and print the scanner's prompt for it:

```bash
git worktree add --detach /tmp/docs-review/ops-arith HEAD
```

```bash
python scripts/docs_review/unit_prompt.py scanner ops-arith --workdir /tmp/docs-review/ops-arith
```

Launch an agent with that prompt and save its final message. Then, for the same worktree:

```bash
python scripts/docs_review/unit_prompt.py verifier ops-arith --workdir /tmp/docs-review/ops-arith --report <file>
```

Launch a second agent with that prompt. When it is done, apply the worktree's diff to the branch
and run the gates:

```bash
git -C /tmp/docs-review/ops-arith diff | git apply
```

```bash
scripts/docs_review/gates.sh
```

Read the diff, move the report entries that were not fixed into `findings.md`, and commit as
`Docs review: <unit>`.

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

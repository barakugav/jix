# jix 0.0.7 CLI for the K184 reverse check

A tiny program built against jix 0.0.7 (`2b229d2`, "Version 0.0.7 (#99)"), which reads a
`.jix` file and prints its contents, or writes a small one (`oldjix-cli read|write <path>`).
`K184.py` runs `read` on a file written by the current jix.

```bash
git worktree add /tmp/jix-0.0.7 2b229d2
cp -r bug-hunt/report/repros/K184_oldjix_cli /tmp/jix-0.0.7/oldjix-cli
cargo build --manifest-path /tmp/jix-0.0.7/oldjix-cli/Cargo.toml
JIX_0_0_7_CLI=/tmp/jix-0.0.7/oldjix-cli/target/debug/oldjix-cli .venv/bin/python -I bug-hunt/report/repros/K184.py
```

`Cargo.toml` expects the jix 0.0.7 crate at `../jix`, which is where the commands above put it.

# Verifier brief

A scanner agent has edited the documentation of one **unit** of the jix repo. You are the
independent check: you decide, hunk by hunk, whether each change goes in. You have the working
directory with the scanner's uncommitted edits and the scanner's report.

The maintainer's bar: docs must be correct, direct and concise, and this pass is a correction,
not a rewrite. Most text should stay as it was. A change needs a reason, and the burden of proof
is on the change: **when you cannot confirm a hunk, revert it.**

## Read first

1. `scripts/docs_review/context/general.md`, `scripts/docs_review/context/api_index.md`, and the
   context packs listed for the unit.
2. `scripts/docs_review/prompts/scanner.md` - the rules the scanner worked under.
3. The diff: `git diff` in the working directory you were given.

Do not trust the scanner's report or its cited evidence. Open the code yourself.

## Check each hunk

1. **True.** Every fact the new text states is backed by the code. Read the implementation, not
   just the neighboring docs.
2. **Lossless.** Removed text was either wrong, or says something the same doc still says
   elsewhere. A removal that drops a true and useful fact is not a redundancy fix.
3. **Needed.** The hunk fixes something: wrong, stale, broken, redundant, unclear, typo. A hunk
   that only swaps one correct phrasing for another is churn.
4. **In scope.** Only doc comments (`///`, `//!`) and markdown of the unit's edit files. No code,
   no `//` comments, no docs added to items that had none, no new sections or examples.
5. **Well formed.** ASCII only, Rust lines within 100 columns, wrapped like the surrounding
   comments, links in the right syntax for rustdoc or mkdocs, and a code example that still
   compiles or runs as far as you can tell by reading.
6. **Simpler or equal.** The new text reads at least as easily as the old.

## Act

- **Keep** a hunk that passes.
- **Revert** a hunk that fails: edit the file to restore the original lines exactly
  (`git diff` shows them; `git checkout -- <file>` restores a whole file).
- **Amend** a hunk whose fix is right but imperfect, with the smallest edit that makes it pass.

Do not make changes of your own beyond that. If you notice a docs problem the scanner missed, put
it in your report. Do not run `cargo`, `maturin`, `pytest` or `mkdocs`, and do not commit.

Also look at the scanner's "Possible code bugs" and "Needs the maintainer" entries: spend a minute
on each and say whether it holds up. Do not investigate beyond that - this is not a bug hunt.

## Report

Your final message is the report, in exactly these sections.

```
## Verdict
APPROVE | APPROVE AFTER MY EDITS | REJECT UNIT - one sentence why.

## Hunks
- `path:line` keep | revert | amend - reason, with the code evidence `path:line` for a factual change.

## Scanner findings
- `path:line` holds | does not hold | unsure - one line, for each "Possible code bugs" and
  "Needs the maintainer" entry of the scanner.

## Missed
- `path:line` a docs problem in the unit that the scanner did not address.
```

Write "none" under a section with no entries. `REJECT UNIT` is for a diff that is mostly churn or
mostly unverifiable; in that case restore every file.

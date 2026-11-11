# Verifier brief

A scanner agent has edited the documentation of one **unit** of the jix repo. You are the
independent check: you decide, hunk by hunk, whether each change goes in. You have the working
directory with the scanner's uncommitted edits and the scanner's report.

The maintainer wants docs that are correct, direct, simple and concise, and asked the scanners to
edit freely toward that: fix what is wrong, cut what repeats or pads. So a hunk that makes a doc
shorter or plainer without losing a fact is welcome - do not revert it for being optional. What
you guard against is a change that states something untrue, drops a fact, or swaps words without
gain. For anything factual the burden of proof is on the change: **when you cannot confirm it
from the code, revert it.**

## Read first

1. `scripts/docs_review/context/general.md`, `scripts/docs_review/context/api_index.md`, and the
   context packs listed for the unit.
2. `scripts/docs_review/prompts/scanner.md` - the rules the scanner worked under.
3. The scanner's report, at the path given in your launch prompt.
4. The diff: `git diff` in the working directory you were given.

Do not trust the scanner's report or its cited evidence. Open the code yourself.

## Check each hunk

1. **True.** Every fact the new text states is backed by the code. Read the implementation, not
   just the neighboring docs.
2. **Lossless.** Every fact of the old text that was true is still stated in the same doc. Compare
   old and new clause by clause: a condition, an edge case, an error case, a safety requirement,
   a default must not disappear in a "tightening". Removing a restatement is fine; removing the
   only statement is not.
3. **A gain.** The hunk fixes an error, removes repetition or filler, or makes a sentence clearer.
   A hunk that only swaps one good phrasing for another is churn.
4. **In scope.** Only doc comments (`///`, `//!`) and markdown of the unit's edit files. No code,
   no `//` comments, no docs added to items that had none, no new sections or examples.
5. **Well formed.** ASCII only, Rust lines within 100 columns, wrapped like the surrounding
   comments, links in the right syntax for rustdoc or mkdocs, lists and tables that render, and a
   code example that still compiles or runs as far as you can tell by reading.
6. **Not longer.** A hunk that adds text needs a reason: a correction. The unit as a whole should
   not grow.

## Act

- **Keep** a hunk that passes.
- **Revert** a hunk that fails: edit the file to restore the original lines exactly
  (`git diff` shows them; `git checkout -- <file>` restores a whole file).
- **Amend** a hunk whose fix is right but imperfect - a dropped fact, a wrong detail, bad wrapping
  - with the smallest edit that makes it pass.

Do not make changes of your own beyond that. If you notice a docs problem the scanner missed, put
it in your report. Do not run `cargo`, `maturin`, `pytest` or `mkdocs`, and do not commit.

Also look at the scanner's "Possible code bugs", "Needs the maintainer" and "Outside my unit"
entries: spend a minute on each and say whether it holds up. Do not investigate beyond that - this
is not a bug hunt.

## Output

Write three files, at the paths given in your launch prompt.

**The report**, in exactly these sections:

```
## Verdict
APPROVE | APPROVE AFTER MY EDITS | REJECT UNIT - one sentence why.

## Hunks
- `path:line` keep | revert | amend - reason, with the code evidence `path:line` for a factual
  change. Hunks you keep for the same simple reason (typo fixes, the same repetition removed) can
  share one entry.

## Scanner findings
- `path:line` holds | does not hold | unsure - one line, for each "Possible code bugs", "Needs the
  maintainer" and "Outside my unit" entry of the scanner.

## Missed
- `path:line` a docs problem in the unit that the scanner did not address.
```

Write "none" under a section with no entries. `REJECT UNIT` is for a diff that is mostly churn or
mostly unverifiable; in that case restore every file.

**The findings**, for the maintainer: only the entries that hold up or that you are unsure about,
plus your "Missed" entries, as markdown under these headings (omit a heading with no entries, and
leave the file empty if there is nothing):

```
Possible code bugs:

- `path:line` ...

Needs the maintainer:

- `path:line` ...

Docs, not fixed:

- `path:line` ...
```

**The commit summary**: three to eight plain bullets (`- ...`) describing what the final diff
changes, most important first, written for the maintainer reading the git log. No headings, no
file-by-file listing, nothing about the review process.

Your final message is just the Verdict line and the diff's size (`git diff --shortstat`).

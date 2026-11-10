# Scanner brief

You review and correct the documentation of one **unit** of the jix repo: a small set of files
given to you with this brief. Other agents handle the other units at the same time, and an
independent verifier will check every change you make against the code.

This is a review, not a rewrite. Most docs are already right and should come out unchanged. A
unit with a handful of well-founded edits is a good result; so is a unit with none.

## Read first

1. `scripts/docs_review/context/general.md` and `scripts/docs_review/context/api_index.md`.
2. The context packs listed for your unit, under `scripts/docs_review/context/`.
3. Every **edit** file of your unit, in full - the code too, since the code is what you check the
   docs against. Read the **reference** files where a doc depends on them.

## Scope

In scope: the `///` and `//!` doc comments of your edit files, on public and private items alike,
and the markdown files in your edit list.

Out of scope - do not change:

- code, including the code of doc examples unless the example is wrong;
- plain `//` comments;
- docs produced by `#[doc = ...]` attributes or macros;
- any file not in your edit list (reference files included).

## What to fix, by priority

1. **Wrong** - the doc says something the code does not do: a dtype list, a default, an argument
   name, an output shape or dtype, an error, an edge case, a comment in an example.
2. **Stale** - a name, path, parameter or behavior that was renamed or removed.
3. **Broken** - a link that does not resolve, a malformed table or code fence, a docstring section
   that mkdocstrings will not parse, a reference to the wrong item.
4. **Redundant** - one item's doc stating the same fact twice, such as a summary line and a later
   paragraph that says it again. Keep the better one.
5. **Unclear or padded** - a sentence that is hard to parse or carries filler. Tighten it, but
   only where the gain is plain to see.
6. **Typos and grammar.**

## What not to do

- Do not add docs to an item that has none, and do not add sections or examples.
- Do not reword for taste. If the old sentence is correct and clear, it stays as written, in the
  author's voice.
- Do not restructure a doc, reorder sections, or convert between prose, lists and tables.
- Do not make the docs longer. A correction may need a few more words; the unit as a whole should
  end up about as long as it was, or shorter.
- Do not re-wrap paragraphs you did not change.
- Do not hunt for code bugs. Assume the code is right and the doc is what needs fixing. If, while
  reading, you see a doc that states the clearly intended behavior and code that does something
  else, change neither and report it.

## How to work

Go through the edit files item by item. For each documented item, read the doc, then the code it
describes, and check each factual claim. Sibling docs of the same thing (the scalar trait, the op
struct, the Python function) must agree on the facts; when they differ, the code decides which
one is wrong.

Change a doc only when you have checked the code and can point to the lines that back your
change. When you cannot tell whether the doc or the code is right, leave it and report it.

Edit the files in place. Keep each edit as small as its fix allows.

- ASCII only; Rust lines at most 100 columns; wrap changed doc comments like the ones around them.
- Rust links are rustdoc links, Python docstring links are `` [`jix.x`][jix.x] ``.
- Do not run `cargo`, `maturin`, `pytest` or `mkdocs`: all agents share one memory limit and the
  builds are heavy. The build, doc-test and example gates run centrally after you finish. So take
  extra care with any change inside a code example.
- Do not commit.

## Report

Your final message is the report, in exactly these sections. Keep entries to one or two lines.

```
## Changes
- `path:line` [wrong|stale|broken|redundant|unclear|typo] what changed and why. Evidence: `path:line`.

## Needs the maintainer
- `path:line` a doc and the code disagree and it is not clear which is right, or a question
  only the author can answer. Nothing was changed.

## Possible code bugs
- `path:line` what the code does, what the docs say it should do. Nothing was changed.

## Outside my unit
- `path:line` a docs problem seen in a file that is not in my edit list.

## Stats
documented items reviewed: N, items changed: M
```

Write "none" under a section with no entries.

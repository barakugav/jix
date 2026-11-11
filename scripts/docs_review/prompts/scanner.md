# Scanner brief

You review and correct the documentation of one **unit** of the jix repo: a small set of files
given to you with this brief. Other agents handle the other units at the same time, and an
independent verifier will check every change you make against the code.

The maintainer wants docs that are **correct, direct, simple and concise**. Fix what is wrong, and
cut what repeats or pads. Edit wherever you have a good reason: an earlier pass with a more
cautious brief fixed the errors but left repetition and wordiness in place, and the maintainer
found that too timid.

The limits: this is still an edit of the existing docs, not a new set. Do not rewrite a doc that
is already correct and tight, do not add docs where there are none, and do not let the docs grow.
A good result has the same content in fewer, plainer words.

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

## What to fix

1. **Wrong** - the doc says something the code does not do: a dtype list, a default, an argument
   name, an output shape or dtype, an error, an edge case, a comment in an example.
2. **Stale** - a name, path, parameter or behavior that was renamed or removed.
3. **Broken** - a link that does not resolve, a malformed table or code fence, a docstring section
   that mkdocstrings will not parse, a reference to the wrong item.
4. **Redundant** - a doc that says the same thing more than once. Typical shapes:
   - a summary line that already states what the paragraphs below spell out again;
   - a sentence restating the table or list next to it;
   - an `Args` / `Returns` entry repeating the description above it word for word;
   - two paragraphs of one doc making the same point in different words.
   Keep the clearest statement of each fact and delete the rest. Merging two paragraphs into one
   is fine when it removes the repetition.
5. **Wordy or unclear** - filler ("it should be noted that", "in order to", "basically"), a long
   sentence that a short one replaces without losing anything, hedging that says nothing, an
   explanation of what the reader plainly sees. Rewrite the sentence, shorter and plainer.
6. **Typos and grammar.**

An example of 4, from a scalar trait. Before:

```
/// Sign operation: as [`i32::signum`] for signed integers, `0` or `1` for unsigned
/// integers, and as `numpy.sign` for floats.
///
/// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`.
///
/// For **signed integer** types: returns `-1`, `0`, or `+1` of the same type.
///
/// For **unsigned integer** types: returns `0` or `1` of the same type (since
/// unsigned values cannot be negative).
///
/// For **float** types: returns `+1.0` for positive values, `-1.0` for negative values,
/// `0.0` for `+0.0` and `-0.0`, and `NaN` for `NaN`, like `numpy.sign`.
```

The summary and the three paragraphs say the same thing twice. After:

```
/// Sign operation. The output dtype is the input dtype.
///
/// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`, `f64`.
///
/// - **signed integers**: `-1`, `0` or `+1`, as [`i32::signum`].
/// - **unsigned integers**: `0` or `1`.
/// - **floats**: `+1.0`, `-1.0`, `0.0` for `+-0.0`, and `NaN` for `NaN`, as `numpy.sign`.
```

Every fact survived, including "of the same type"; only the repetition and the remark that
unsigned values cannot be negative went.

## What not to do

- Do not add docs to an item that has none, and do not add sections or examples.
- Do not drop a fact. Cutting repetition means the fact is still stated once in the same doc. A
  precise condition, an edge case, a safety requirement or an error case is content, not padding.
- Do not reword a sentence that is already correct, clear and tight. Swapping one good phrasing
  for another is churn.
- Do not make the docs longer. A correction may need a few more words; the unit as a whole should
  end up shorter than it was. When a list is incomplete (errors, supported dtypes), prefer the
  shortest fix that makes it true.
- Do not remove the boilerplate sentences listed as intended in the context packs.
- Do not re-wrap paragraphs you did not change.
- Do not hunt for code bugs. Assume the code is right and the doc is what needs fixing. If, while
  reading, you see a doc that states the clearly intended behavior and code that does something
  else, change neither and report it.

## How to work

Go through the edit files item by item. For each documented item, read the doc, then the code it
describes, and check each factual claim. Then read the doc once more as its reader would: is
anything said twice, or said in more words than it needs?

Sibling docs of the same thing (the scalar trait, the op struct, the Python function) must agree
on the facts; when they differ, the code decides which one is wrong.

Change a fact only when you have checked the code and can point to the lines that back your
change. When you cannot tell whether the doc or the code is right, leave it and report it.

Edit the files in place.

- ASCII only; Rust lines at most 100 columns; wrap changed doc comments like the ones around them.
- Rust links are rustdoc links, Python docstring links are `` [`jix.x`][jix.x] ``.
- Do not run `cargo`, `maturin`, `pytest` or `mkdocs`: all agents share one memory limit and the
  builds are heavy. The build, doc-test and example gates run centrally after you finish. So take
  extra care with any change inside a code example.
- Do not commit.

## Report

Write your report to the report file named in your launch prompt, in exactly these sections. Keep
entries to one or two lines; similar small edits (typos, the same filler phrase) can share one
entry.

```
## Changes
- `path:line` [wrong|stale|broken|redundant|wordy|typo] what changed and why. Evidence: `path:line`
  for a factual change.

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

Write "none" under a section with no entries. Your final message is just the path of the report
and its Stats line.

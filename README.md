# jix blog

Source of the one-page blog at <https://barakugav.github.io/jix/>, built by GitHub Pages
with Jekyll straight from this branch.

| File | What it is |
|------|------------|
| `index.md` | The post. Front matter sets the title, subtitle, kicker and date |
| `_config.yml` | Site title, author and the link cards (GitHub, crates.io, docs.rs, PyPI, Read the Docs) |
| `_layouts/default.html` | Page layout: header, link cards, post, footer |
| `assets/css/style.css` | All styling, with light and dark themes |

Writing tips:

- Callouts: a blockquote followed by `{: .note}` or `{: .warning}` on the next line.
- Images go in `assets/img/`; reference them with `{{ '/assets/img/x.png' | relative_url }}`.
- Fenced code blocks get syntax colors and a language label (`rust`, `python`, `shell`, `toml`).

## Preview locally

```shell
python preview.py            # http://localhost:4000/jix/, needs Docker
python preview.py --build-only
```

## Publish

Push the branch, then in the repo go to Settings -> Pages, set Source to "Deploy from a branch",
and pick `gh-pages` / `/ (root)`.

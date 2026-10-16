"""Griffe extensions used by the mkdocs build (see `mkdocs.yml`)."""

import griffe


class NativeModuleDocstring(griffe.Extension):
    """Use the docstring of the native `jix.jix` module for the `jix` package.

    `jix/__init__.py` copies it at runtime (`__doc__ = jix.__doc__`), which griffe can not follow
    statically, while the generated `jix/jix.pyi` stub carries it.
    """

    def on_package(self, *, pkg: griffe.Module, **kwargs):
        if pkg.name == "jix" and pkg.docstring is None:
            pkg.docstring = pkg["jix"].docstring

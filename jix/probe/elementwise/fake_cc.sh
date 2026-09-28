#!/bin/sh
# Stand-in C compiler/archiver for cross builds of the probe: `zstd-sys` compiles C, but the probe
# is only an rlib that is never linked, so empty objects are enough and no C cross toolchain or
# sysroot is needed. As CC: writes an empty file at `-o <path>`. As AR: an empty archive.
out=
while [ $# -gt 0 ]; do
    case "$1" in
        -o) out="$2"; shift ;;
        *.a) [ -z "$out" ] && out="$1" ;;
    esac
    shift
done
if [ -n "$out" ]; then
    case "$out" in
        *.a) printf '!<arch>\n' > "$out" ;;
        *) : > "$out" ;;
    esac
fi

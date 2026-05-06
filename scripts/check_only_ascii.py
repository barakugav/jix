"""Check all files for non-ASCII characters.

Output style (like a compiler diagnostic):

  path/to/file.py:42: café and naïve
                       ^       ^
"""

from __future__ import annotations

import subprocess
import sys


def git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("fatal: not a git repository or git not found", file=sys.stderr)
        sys.exit(128)
    return [p.decode() for p in result.stdout.split(b"\0") if p]


def check_file(path: str) -> int:
    """Print each line containing non-ASCII bytes. Return count of bad lines."""
    try:
        raw = open(path, "rb").read()
    except (OSError, IsADirectoryError):
        return 0

    # skip binary files (heuristic: contains null bytes)
    if b"\0" in raw:
        return 0

    hits = 0
    gutter_prefix = f"{path}:"

    for lineno, line_bytes in enumerate(raw.split(b"\n"), start=1):
        non_ascii_offsets = {i for i, b in enumerate(line_bytes) if b > 127}
        if not non_ascii_offsets:
            continue

        hits += 1
        line_text = line_bytes.decode("utf-8", errors="replace")
        gutter = f"{gutter_prefix}{lineno}: "

        # Build a caret line aligned to the *printed* characters.
        # Walk the decoded string; for each character figure out which
        # byte offsets it spans, and mark it if any of those bytes were
        # non-ASCII.
        marker_chars: list[str] = []
        byte_pos = 0
        for ch in line_text:
            # How many bytes did this character occupy in the raw line?
            try:
                ch_byte_len = len(ch.encode("utf-8"))
            except UnicodeEncodeError:
                ch_byte_len = 1  # replacement char fallback

            span = range(byte_pos, byte_pos + ch_byte_len)
            if non_ascii_offsets & set(span):
                marker_chars.append("^")
            elif ch == "\t":
                marker_chars.append("\t")  # keep tab alignment
            else:
                marker_chars.append(" ")
            byte_pos += ch_byte_len

        print(f"{gutter}{line_text}")
        print(f"{' ' * len(gutter)}{''.join(marker_chars)}")

    return hits


def main() -> None:
    files = git_tracked_files()
    total_lines = 0
    bad_files = 0

    for path in files:
        n = check_file(path)
        if n:
            total_lines += n
            bad_files += 1

    if total_lines:
        print(f"\n{total_lines} non-ASCII line(s) in {bad_files} file(s).")
        sys.exit(1)
    else:
        print("All files OK - ASCII only.")


if __name__ == "__main__":
    main()

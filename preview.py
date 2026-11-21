"""Serve the site locally at http://localhost:4000/jix/ with the same Jekyll GitHub Pages uses.

Runs inside Docker, so no local Ruby setup is needed. Rebuilds on file changes.
"""

import argparse
import subprocess
import sys
from pathlib import Path

CONTAINER = "jix-pages-preview"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--build-only", action="store_true", help="build into _site/ and exit")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cmd = ["docker", "run", "--rm", "--init", "-w", "/site"]
    cmd += ["-v", f"{root}:/site", "-v", "jix-pages-gems:/usr/local/bundle"]
    if args.build_only:
        jekyll = "jekyll build"
    else:
        jekyll = f"jekyll serve --host 0.0.0.0 --port {args.port} --livereload"
        # A fixed name, so a server left running by an earlier run is replaced instead of holding the ports.
        cmd += ["--name", CONTAINER, "-p", f"{args.port}:{args.port}", "-p", "35729:35729"]
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    if sys.stdin.isatty():
        cmd += ["-it"]
    cmd += ["ruby:3.3", "sh", "-c", f"bundle install --quiet && exec bundle exec {jekyll}"]
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        if not args.build_only:
            subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


if __name__ == "__main__":
    main()

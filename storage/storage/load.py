"""load.py: generic "push local file(s)/directory(s) up to DO Spaces" CLI,
shared across every experiment via this package's `load-to-spaces` console
script (see the root README.md's "DigitalOcean Spaces" section).

Usage, from within any experiment that depends on this package:
    uv run load-to-spaces <path> [<path> ...]

A directory argument uploads every file under it, recursively (e.g.
`uv run load-to-spaces ../data_library` pushes all four datasets in one
go); a file argument uploads just that file. Either way, each file goes to
exactly the key spaces.sync_down() will look for later (same _key_for()
convention, not a separate naming scheme to keep in sync).

Requires SPACES_KEY/SPACES_SECRET/SPACES_ENDPOINT/SPACES_BUCKET to already
be set. Unlike sync_up() itself (used internally by experiment scripts,
where a silent no-op is correct when Spaces isn't configured), this fails
loudly if they're missing -- the whole point of running this command is to
populate the bucket, so a silent no-op here would just be a confusing way
to fail.
"""

import os
import sys
from pathlib import Path

from .spaces import sync_up  # importing this triggers spaces.py's own load_dotenv()

REQUIRED_ENV_VARS = ("SPACES_KEY", "SPACES_SECRET", "SPACES_ENDPOINT", "SPACES_BUCKET")


def _files_under(path: Path) -> list[Path]:
    """`path` itself if it's a file, or every file recursively under it if
    it's a directory."""
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.is_file()]
    return [path]


def main():
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)} in the environment/.env -- see the root "
            "README.md's \"DigitalOcean Spaces\" section for what to set before "
            "running this command."
        )

    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: load-to-spaces <path> [<path> ...]  (directories upload recursively)")

    files = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"{path} doesn't exist locally -- nothing to upload.")
        files.extend(_files_under(path))

    for f in files:
        sync_up(f)
    print(f"uploaded {len(files)} file(s)")


if __name__ == "__main__":
    main()

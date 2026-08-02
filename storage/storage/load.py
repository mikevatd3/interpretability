"""load.py: generic "push local file(s) up to DO Spaces" CLI, shared across
every experiment via this package's `load-to-spaces` console script (see
the root README.md's "DigitalOcean Spaces" section).

Usage, from within any experiment that depends on interp-storage:
    uv run load-to-spaces <path> [<path> ...]

Uploads to exactly the key spaces.sync_down() will look for later (same
_key_for() convention, not a separate naming scheme to keep in sync).

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

from dotenv import load_dotenv

from .spaces import sync_up

REQUIRED_ENV_VARS = ("SPACES_KEY", "SPACES_SECRET", "SPACES_ENDPOINT", "SPACES_BUCKET")


def main():
    load_dotenv()

    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)} in the environment/.env -- see the root "
            "README.md's \"DigitalOcean Spaces\" section for what to set before "
            "running this command."
        )

    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: load-to-spaces <path> [<path> ...]")

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"{path} doesn't exist locally -- nothing to upload.")
        sync_up(path)


if __name__ == "__main__":
    main()

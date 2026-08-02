"""spaces.py: read/write data through a DigitalOcean Spaces bucket
(S3-compatible -- this uses boto3's plain S3 client pointed at a DO
endpoint), so a script running on a different machine (e.g. a rented GPU
pod) doesn't need a shared filesystem or manual rsync.

Shared across every experiment in this repo (see the root README.md's
"Shared infrastructure" section) -- each experiment depends on this as a
local `uv` path dependency rather than copy-pasting it, unlike the
experiment-specific helpers (TRAIT_PROMPTS, get_device(), etc.) that are
deliberately duplicated per the root README's TODO. Data-loading
infrastructure isn't part of any one experiment's "frozen snapshot" the way
its actual analysis code is.

Spaces is treated as canonical: sync_down() always overwrites the local
copy with whatever's in the bucket before a script reads it, and sync_up()
always pushes the local copy up after a script writes it. If the bucket
doesn't have the key yet (first run, nothing uploaded there yet),
sync_down() is a no-op and the script falls back to whatever's already
local.

Configured entirely via env vars (see the root README.md's "DigitalOcean
Spaces" section for the required .env keys) -- if they're unset, every
function in this module is a silent no-op, so scripts behave exactly as
they did before Spaces existed when it isn't configured (e.g. plain local
runs). This module calls load_dotenv() itself at import time, rather than
relying on whatever experiment script imports it to remember to -- since
load_dotenv() (with no explicit path) searches upward from *this file's*
location, not the caller's, a single .env at the interpretability repo
root (not one per experiment) is enough; it's found the same way
regardless of which experiment triggers the import.
"""

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# storage/storage/spaces.py -> storage/ (this package)
# -> storage/ (this project) -> interpretability/ (repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _client():
    endpoint = os.environ.get("SPACES_ENDPOINT")
    key = os.environ.get("SPACES_KEY")
    secret = os.environ.get("SPACES_SECRET")
    if not (endpoint and key and secret):
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def _bucket() -> str | None:
    return os.environ.get("SPACES_BUCKET")


def _key_for(local_path: Path) -> str:
    """The bucket key for `local_path`, mirroring its path relative to the
    interpretability repo root under SPACES_PREFIX (default "interpretability")
    -- e.g. data_library/groenwold_aave_sae/aave_sae_pairs.tsv ->
    interpretability/data_library/groenwold_aave_sae/aave_sae_pairs.tsv.
    Same convention regardless of which experiment calls this, so every
    experiment's data lands under one predictable prefix in the bucket."""
    prefix = os.environ.get("SPACES_PREFIX", "interpretability")
    rel = local_path.resolve().relative_to(REPO_ROOT)
    return f"{prefix}/{rel.as_posix()}"


def sync_down(local_path: Path) -> None:
    """Overwrite `local_path` with the bucket's copy, if Spaces is
    configured and the key exists. No-op (leaves `local_path` untouched,
    including if it doesn't exist yet) if Spaces isn't configured or the
    key isn't in the bucket yet."""
    client = _client()
    bucket = _bucket()
    if client is None or bucket is None:
        return
    key = _key_for(local_path)
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(local_path))
        print(f"[spaces] pulled s3://{bucket}/{key} -> {local_path}")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            print(f"[spaces] s3://{bucket}/{key} not found yet, using local copy if any")
        else:
            raise


def sync_up(local_path: Path) -> None:
    """Push `local_path` up to the bucket, if Spaces is configured and the
    file exists. No-op if either isn't true."""
    client = _client()
    bucket = _bucket()
    if client is None or bucket is None or not local_path.exists():
        return
    key = _key_for(local_path)
    client.upload_file(str(local_path), bucket, key)
    print(f"[spaces] pushed {local_path} -> s3://{bucket}/{key}")

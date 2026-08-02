# interpretability

GPT-2 interpretability experiments, spun out of `justhousingnotes` as each
one grew past a quick script into its own thing.

## Layout

Each top-level folder is one experiment: `YYYYMMDD_slug/`, dated roughly to
when that experiment's work happened. Every experiment is an independent
`uv` project — its own `pyproject.toml`, `uv.lock`, and `.venv`. To run one:

```
cd 20260731_causal_tracing
uv sync
uv run main.py
```

Each experiment folder has its own `README.md` describing what it does.

### Data: `data_library/` vs. per-experiment `data/`

- **`data_library/`** — curated, reusable *input* datasets pulled from
  external sources, shared across experiments. Each dataset has a
  `*.SOURCE.md` documenting provenance, citation, and how to re-fetch it.
  Gitignored (see its own `README.md` for what's in it and why).
- **`<experiment>/data/`** — that experiment's own *outputs* (caches,
  computed results). Also gitignored, and expected to be regenerable by
  re-running the experiment's scripts, not something to hand-carry between
  machines.

## Shared infrastructure

Experiment *logic* (`TRAIT_PROMPTS`, `get_device()`, pair-loading, etc.) is
deliberately duplicated per experiment rather than shared — see the TODO
below. Data-loading *infrastructure* is the deliberate exception:
`storage/` is a small shared package (its own `uv` project, at the
repo root next to the experiment folders) that any experiment can depend on
via a local path dependency instead of copy-pasting:

```
cd <your experiment folder>
uv add --editable ../storage
```

```python
from storage import sync_down, sync_up
```

It currently holds DigitalOcean Spaces (S3-compatible) read/write helpers —
`sync_down(path)`/`sync_up(path)`, silent no-ops if Spaces isn't
configured, plus a `load-to-spaces` CLI command for seeding the bucket. See
`storage/README.md` for details.

### DigitalOcean Spaces

Any experiment using `storage` reads/writes through Spaces if it's
configured (silent no-op otherwise): input pulled fresh from the bucket
before a script reads it, output pushed up after a script writes it. This
is how a run on a different machine (e.g. a rented GPU pod) gets its input
data and returns its output, without a shared filesystem or manual rsync
of data.

Set these in `.env` (gitignored) to enable it:
```
SPACES_KEY=...
SPACES_SECRET=...
SPACES_ENDPOINT=https://<region>.digitaloceanspaces.com
SPACES_BUCKET=<your space name>
SPACES_PREFIX=interpretability   # optional, this is the default
```

A bucket key mirrors the local path relative to this repo's root, prefixed
with `SPACES_PREFIX` — e.g.
`data_library/groenwold_aave_sae/aave_sae_pairs.tsv` becomes
`interpretability/data_library/groenwold_aave_sae/aave_sae_pairs.tsv`,
regardless of which experiment reads/writes it. To seed the bucket with a
file that doesn't exist there yet (fails loudly if the env vars above
aren't set, unlike `sync_up` itself):
```
uv run load-to-spaces <path> [<path> ...]
```

## TODO

**Duplicated helper code across experiment folders.** `TRAIT_PROMPTS`,
`get_device()`, and `load_pairs()`-style loaders are copy-pasted into
several experiments — `20260725_sae_basics`, `20260731_causal_tracing`,
`20260725_matched_guise` (as `OUTER_PROMPTS`/`_get_device()` — same
pattern, different names), `20260728_entropies` (`get_device()` only; it
has no trait-prompts list, off-topic for that experiment), and now also
`20260731_draw_five` — rather than imported from a shared place. This was
originally a real constraint back when these lived as scattered scripts in
`justhousingnotes/exploration/` with no shared `sys.path`; now that they're
siblings in one repo it's a deliberate tradeoff rather than a forced one:

- Keeping the duplication preserves each experiment as a frozen,
  self-contained snapshot — one experiment's dependency/code drift can't
  silently change another's past results.
- Extracting a shared package would cut the copy-paste but couples
  experiments' code together going forward.

**What's changed since this was first written:** the second bullet used to
be hypothetical ("a local `uv` workspace with a `common/` package" —
vague on the actual mechanism). `storage/` (see "Shared
infrastructure" above) now proves out a concrete, working version of it: a
plain local package + `uv add --editable ../storage`, no formal `uv`
workspace needed. So the open question isn't "how would sharing even work"
anymore — it's just "is it worth it for this code specifically," per the
frozen-snapshot tradeoff above.

That tradeoff isn't equally strong for everything on this list, though.
`get_device()` is pure infrastructure — zero experiment-specific content,
identical everywhere, never expected to differ — much closer in kind to
what `storage/` already holds than to actual experiment logic. It's
the strongest candidate to extract next if this gets revisited.
`TRAIT_PROMPTS`/`load_pairs()` are more defensibly experiment-local: they're
closer to "this experiment's frozen configuration/content" (a specific
template list, a specific pair-loading shape) than to shared plumbing.

Still not resolved either way — revisit if the duplication becomes a real
maintenance cost (e.g. a bug fixed in one copy and not the others), not
just because the mechanism now exists.

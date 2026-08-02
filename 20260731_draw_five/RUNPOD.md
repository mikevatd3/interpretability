# Running the full 2019-pair dataset on RunPod

`main.py`'s generation step is CPU-impractical at full scale. The original
unbatched CPU extrapolation was ~13.5 hours (36,342 prompts x 5 no-cache
forward passes each, at ~1.34s/prompt); a KV-cache attempt only bought a
1.22x speedup (`transformer_lens`'s per-layer hook overhead dominates
wall-clock time on CPU far more than the attention recompute being
cached). Batching turned out to be the bigger lever -- measured 3.2x on
CPU alone (268ms/draw -> 84ms/draw, even accounting for `N_DRAWS` doubling
5 -> 10 in the same change) -- and is expected to matter even more on a
real GPU, on top of the GPU's own raw throughput advantage for a model
this FLOP-tiny (124M params).

**Estimate: well under 30 min compute, likely under $1 in pod rental.**
Not verified on real hardware -- no GPU available in this dev environment
-- but the code is device-generic (`config.get_device()` picks `cuda`
automatically) and `BATCH_SIZE` is overridable via env var, so it's easy to
push higher than the CPU-tuned default of 16 once on a GPU (see step 4).

## 1. Pick a GPU

VRAM is not the constraint -- GPT-2 small is ~500MB. Any modern NVIDIA GPU
works. Recommend RunPod **Community Cloud**, cheapest available in the
RTX 3090 / RTX 4090 / A4000 tier (~$0.15-0.50/hr). No reason to pay for an
A100/H100 for a 124M-param model.

## 2. Launch the pod

- Template: an official "RunPod PyTorch 2.x" image (CUDA + drivers
  preinstalled).
- When configuring the pod, add these as **pod environment variables**
  (not a file -- avoids putting secrets through rsync/scp) from your DO
  Spaces credentials -- see `README.md`'s "DigitalOcean Spaces" section:
  ```
  SPACES_KEY=...
  SPACES_SECRET=...
  SPACES_ENDPOINT=https://<region>.digitaloceanspaces.com
  SPACES_BUCKET=<your space name>
  ```
- Once running, RunPod gives an SSH connection string, e.g.
  `ssh root@<host> -p <port> -i ~/.ssh/id_ed25519`.

On the pod, install `uv`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 3. Seed Spaces, copy code to the pod

**Locally, once**, if you haven't already -- push the one input file this
project reads up to the bucket, so the pod can pull it without needing a
shared filesystem:
```
uv run load-to-spaces ../data_library/groenwold_aave_sae/aave_sae_pairs.tsv
```

Nothing in this repo is committed yet, so `git clone` on the pod won't have
this session's work -- use `rsync` for code instead (data no longer needs
rsync at all, it flows through Spaces). This needs both this experiment
folder *and* `../interp_storage` -- `main.py`'s `pyproject.toml` depends on
it as a local editable path (`uv add --editable ../interp_storage`), so
`uv sync` on the pod needs that directory to actually exist at the same
relative path:
```
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='data' \
  20260731_draw_five/ root@<host>:~/interpretability/20260731_draw_five/ -e "ssh -p <port>"
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  interp_storage/ root@<host>:~/interpretability/interp_storage/ -e "ssh -p <port>"
```
Preserve the relative layout (`interpretability/20260731_draw_five/`) since
`config.py` locates `data_library/` via
`Path(__file__).resolve().parent.parent / "data_library"` -- though with
Spaces configured, that directory doesn't need to exist on the pod ahead of
time; `sync_down` creates it when it pulls the pairs tsv.

## 4. Run it

On the pod:
```
cd ~/interpretability/20260731_draw_five
uv sync
N_PAIRS=all BATCH_SIZE=128 uv run main.py
```
- `N_PAIRS=all` (or any int) overrides `config.py`'s local default of 50
  without editing the file -- see `_n_pairs_from_env` in `config.py`.
- `BATCH_SIZE=128` is a starting guess for GPU -- the CPU-tuned default of
  16 measured *slower* at 32 (no parallelism headroom, more padding waste),
  but that calculus flips on a GPU with real parallel throughput. Push it
  higher if the pod has VRAM to spare (it will, for this model); back off
  if you hit an out-of-memory error.
- `get_device()` picks `cuda` automatically -- the startup line
  (`loaded gpt2 on cuda, 2019 pairs, ...`) confirms it.
- With Spaces configured, `load_pairs()` pulls `aave_sae_pairs.tsv` from
  the bucket automatically (`sync_down`), and the output CSV is pushed back
  up automatically (`sync_up`) once written -- no manual step either
  direction.

This produces `data/draws_nall.csv` locally on the pod (and in the bucket):
9 templates x 2019 pairs x 2 variants = 36,342 rows.

## 5. Run analysis locally

The `analyze_grammar.py` / `compare_pos.py` / `compare_pairs.py` steps are
spaCy/pandas-only -- cheap, CPU-fine, no reason to pay for GPU time during
them. As long as your **local** `.env` has the same `SPACES_*` values as
the pod, just run them as normal -- each one pulls its input from the
bucket automatically before reading:
```
cd 20260731_draw_five
N_PAIRS=all uv run analyze_grammar.py
N_PAIRS=all uv run compare_pos.py
N_PAIRS=all uv run compare_pairs.py
```
(`N_PAIRS=all` here just controls which filename tag -- `nall` vs `n50` --
these scripts look for/write; it doesn't affect what data gets read.)

## 6. Stop the pod

RunPod bills while the pod is running, idle or not -- stop/terminate it
from the dashboard once step 4's `sync_up` confirms `draws_nall.csv` made
it to the bucket.

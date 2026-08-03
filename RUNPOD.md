# Running the full 2019-pair dataset on RunPod

## 1. Pick a GPU

RTX-3090 is fine for this and usually pretty cheap

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

Follow instructions in `storage` for this.

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

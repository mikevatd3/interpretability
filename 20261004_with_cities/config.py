from pathlib import Path

MODEL = "google/gemma-2-9b-it"
DEVICE = "cuda"

DATA_DIR = Path(__file__).parent / "data" / "challenge_tables_2026-10-04T18:33:21.444328"
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

# Checkpoint (save + sync) results to disk every this many processed tables,
# so a hard kill (e.g. SIGKILL, OOM) loses at most this many rows of progress.
CHECKPOINT_EVERY = 25

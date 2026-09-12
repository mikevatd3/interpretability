from pathlib import Path

MODEL = "google/gemma-2-9b-it"
DEVICE = "cuda"

DATA_DIR = Path(__file__).parent / "input_data" / "challenge_tables_2026-09-11T08:30:23.040625"
RESULT_DIR = Path(__file__).parent / "results"

# Checkpoint (save + sync) results to disk every this many processed tables,
# so a hard kill (e.g. SIGKILL, OOM) loses at most this many rows of progress.
CHECKPOINT_EVERY = 25


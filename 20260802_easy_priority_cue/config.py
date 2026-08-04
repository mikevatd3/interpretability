from pathlib import Path
import torch

def get_device() -> str:
    """cuda > mps > cpu, whichever is actually available in this environment."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DATA_DIR = Path(__file__).resolve().parent / "data"
DEVICE = get_device()


def applications_csv_path(timestamp: str) -> Path:
    """Where main.py writes a run's input table -- one file per run,
    sharing `results_csv_path`'s timestamp so the two can be paired back up
    later from the `applications_file` field in the results row."""
    return DATA_DIR / f"applications_{timestamp}.csv"


def results_csv_path(timestamp: str) -> Path:
    """Where main.py writes a run's output -- one file per run, suffixed
    with a YYYYMMDDHHMMSS timestamp so concurrent/repeated runs (e.g. from
    a rented GPU) never clobber each other, locally or in DO Spaces."""
    return DATA_DIR / f"results_{timestamp}.csv"

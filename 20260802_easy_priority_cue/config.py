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

APPLICATIONS_PATH = DATA_DIR / "applications.csv"
RESULTS_PATH = DATA_DIR / "results.csv"

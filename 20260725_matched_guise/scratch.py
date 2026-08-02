import csv
from pathlib import Path
import pandas as pd

DATA_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "data_library"


terms = pd.read_csv(
    DATA_LIBRARY_DIR / "dialect_prejudice_repo" / "princeton_trilogy_ratings.tsv",
    sep="\t",
    quoting=csv.QUOTE_NONE
)

print(terms)


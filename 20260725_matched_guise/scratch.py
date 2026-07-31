import csv
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/home/michael/1_projects/llm_housing_project/justhousingnotes")


terms = pd.read_csv(
    BASE_DIR / "data" / "dialect_prejudice_repo" / "princeton_trilogy_ratings.tsv",
    sep="\t",
    quoting=csv.QUOTE_NONE
)

print(terms)


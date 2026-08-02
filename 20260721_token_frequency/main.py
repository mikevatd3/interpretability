import csv
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
import tiktoken

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data"

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train+validation+test")
enc = tiktoken.get_encoding("gpt2")

counts = Counter(tok for row in ds["text"] for tok in enc.encode(row))

with open(DATA_DIR / "token_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "token", "count"])
    for tok_id, count in counts.most_common():
        w.writerow([tok_id, enc.decode([tok_id]), count])


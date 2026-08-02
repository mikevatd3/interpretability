"""
A model workflow to use GPT-2 to label its own tokens if they are city names
or are parts of city names. The steps:

1. Hand label a few known cities (15) on a file seeded from the token list in
   ../20260721_token_frequency/
2. Make a dataset by taking the labeled cities and then 15 random other tokens
   (assuming that grabbing a city token at random is really unlikely).
3. Create a linear model splitting the 15 city from the 15 non-city tokens and 
   append it to the dataset csv.
4. Order based on linear model logits and then hand label many, many more
   cities.
5. Train another, more accurate linear model.

"""
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

from transformer_lens import HookedTransformer
from dotenv import load_dotenv


load_dotenv()
D_MODEL = 768
DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_FILE = DATA_DIR / "draft_labels.csv"
DRAFT_FILE  = DATA_DIR / "draft_labels.csv"
FINAL_FILE  = DATA_DIR / "final_labels.csv"


def make_draft_dataset(frame):
    cities = frame[frame["city"]]
    non_cities = frame[~frame["city"]].sample(len(cities))
    ds = pd.concat([cities, non_cities])

    token_ids = torch.tensor(ds["id"].values)
    labels = torch.tensor(ds["city"].astype(int).values, dtype=torch.float32)

    return token_ids, labels


def train_linear_separator(transformer, token_ids, labels):
    embeddings = transformer.W_E[token_ids].detach()  # (n_tokens, d_model=768)

    # A model that identifies city tokens
    model = nn.Linear(D_MODEL, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(1000):
        logits = model(embeddings).squeeze()
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def count_labels(path):
    frame = pd.read_csv(path)
    labeled = int(frame["city"].sum())
    return labeled, len(frame)


def phase(in_path, out_path, transformer):

    frame = pd.read_csv(in_path)
    token_ids, labels = make_draft_dataset(frame)

    print("  Training linear separator...")
    model = train_linear_separator(transformer, token_ids, labels)

    all_token_ids = torch.tensor(frame["id"].values)
    all_embeddings = transformer.W_E[all_token_ids]
    frame["predicted"] = model(all_embeddings).detach().squeeze().numpy()

    frame.sort_values("predicted", ascending=False).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")


def ask_yes_no(prompt, default=False):
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{suffix}] ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def main():
    print("  Loading gpt2-small...")
    transformer = HookedTransformer.from_pretrained("gpt2-small")

    print("=" * 60)
    print("Step 1: hand-label an initial batch of cities")
    print("=" * 60)
    input(f"Open {SOURCE_FILE} and label 15 or so city names, then return here "
          "and press ENTER to continue.\n")
    phase(SOURCE_FILE, DRAFT_FILE, transformer)

    i = 1
    while True:
        print("\n" + "=" * 60)
        print(f"Step 2: labeling round {i}")
        print("=" * 60)
        input(f"Open {DRAFT_FILE} and label as many city names as you can stand "
              "-- they should all be near the top -- then return here and press "
              "ENTER to continue.\n")
        phase(DRAFT_FILE, DRAFT_FILE, transformer)

        labeled, total = count_labels(DRAFT_FILE)
        print(f"\n{labeled} of {total} tokens labeled as cities so far "
              f"(round {i} complete).")
        if not ask_yes_no("Would you like to continue labeling?", default=True):
            break
        i += 1

    print("\n" + "=" * 60)
    print("Finalizing model")
    print("=" * 60)
    phase(DRAFT_FILE, FINAL_FILE, transformer)
    print("Done.")


if __name__ == "__main__":
    main()






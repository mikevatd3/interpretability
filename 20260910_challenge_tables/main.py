from datetime import datetime
import random
from textwrap import dedent
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from tqdm import tqdm

from config import MODEL, DEVICE, DATA_DIR, RESULT_DIR

from storage import sync_up


load_dotenv()


def load_model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained(MODEL, device=DEVICE)
    model.eval()
    return model


def build_prompt(applications: str, nrows: int) -> tuple[str, list[int]]:
    example = random.sample(range(nrows), nrows)
    example_str = ",".join(map(str, example))
    instruction = dedent(
        """
        Here is today's table of leads:

        {table}

        Rank all {n} lead ids by how likely each is to result in a
        successful sale, from most likely to least likely.

        Respond with ONLY a comma-separated list of the ids in that order --
        every id exactly once, nothing else. No words, no explanation, no
        brackets, no spaces. Example format: {example}
        """
    ).format(table=applications, n=nrows, example=example_str).strip()

    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n", example


def main():
    model = load_model()

    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    
    result = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        expid = row["fileid"]

        applications = (DATA_DIR / f"{expid}.csv").read_text()

        csv = pd.read_csv(DATA_DIR / f"{expid}.csv")
        nrows = len(csv)

        prompt, example = build_prompt(applications, nrows)
        tokens = model.to_tokens(prompt)

        stop_ids = {model.tokenizer.eos_token_id}
        
        # Token budget allows for 4 tokens per row, plus ten extras
        # TODO: maybe adjust this? Too many tokens?
        n_tokens = 4 * nrows + 10 

        new_ids = []
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = model(tokens)
                next_id = logits[:, -1, :].argmax().item()
                if next_id in stop_ids:
                    break
                new_ids.append(next_id)
                tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        continuation = model.to_string(new_ids) if new_ids else ""
        
        row = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "device": DEVICE,
            "experiment_id": expid,
            "continuation": continuation,
            "example": example,
        }

        result.append(row)
    
    out_path = RESULT_DIR / f"outcomes_{datetime.now().isoformat()}.csv"
    all_experiments = pd.DataFrame.from_records(result)
    all_experiments.to_csv(out_path)

    sync_up(out_path)
    

if __name__ == "__main__":
    main()

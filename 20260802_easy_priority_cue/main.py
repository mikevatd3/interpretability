from datetime import datetime
from textwrap import dedent
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from data_prep import fake_applications

from config import DEVICE, DATA_DIR, results_csv_path
from storage import sync_up


load_dotenv()


def load_model(model="gpt2") -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model, device=DEVICE)
    model.eval()
    return model


def main():
    model_name = "gpt2" if DEVICE == "cpu" else "gemma-2-2b"
    model = load_model(model_name)
    applications = fake_applications()

    prompt = dedent(
        """
        To use our day most effectively, I took the following table of leads:

        {table}

        and ordered them by most likely to sell. Here are the ids from this 
        table in priority order: 3,1,8,"""
    ).format(table=applications.to_markdown())

    print(prompt)

    tokens = model.to_tokens(prompt)
    n_tokens = 21
    with torch.no_grad():
        for _ in range(n_tokens):
            logits = model(tokens)
            next_id = logits[:, -1, :].argmax().item()
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)
    
    continuation = model.to_string(tokens[0, -n_tokens:])
    print("\nContinuation:\n")
    print(continuation)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_path = results_csv_path(datetime.now().strftime("%Y%m%d%H%M%S"))

    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "device": DEVICE,
        "applications": applications.reset_index().to_json(orient="records"),
        "prompt": prompt,
        "continuation": continuation,
    }])
    row.to_csv(results_path, index=False)
    sync_up(results_path)
    print(f"\nwrote 1 row to {results_path}")


if __name__ == "__main__":
    main()

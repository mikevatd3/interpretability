import torch
import rich
from dotenv import load_dotenv
from transformer_lens.model_bridge import TransformerBridge
import pandas as pd

from dataset import build_challenge_prompts, build_challenge_rows

load_dotenv()


def main():
    model = TransformerBridge.boot_transformers("gpt2")
    model.eval()

    challenge_rows = build_challenge_rows()
    challenge_prompts = build_challenge_prompts(challenge_rows)

    # " Advance" is a single GPT-2 token, but " Postpone" tokenizes as
    # [" Post", "p", "one"], so we compare against its first token as a
    # preflight approximation of the full completion's probability.
    advance_token = model.to_single_token(" Advance")
    postpone_token = model.to_single_token(" Post")
    

    result = {
        "id": [],
        "p_advance": [],
        "p_postpone": [],
    }

    for i, prompt in rich.progress(challenge_prompts):
        tokens = model.to_tokens(prompt)
    
        with torch.no_grad():
            logits = model(tokens)

        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits[[advance_token, postpone_token]], dim=0)
    
        result["id"].append(id)
        result["p_advance"].append(probs[0])
        result["p_postpone"].append(probs[1])

    frame = pd.DataFrame(result)
    frame.to_csv("data/advance_postpone.csv")


if __name__ == "__main__":
    main()

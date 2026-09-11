from pathlib import Path
from dotenv import load_dotenv
import torch
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

import pandas as pd


load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"

PROMPT = ""

TOP_K = 10



def load_dataset():
    path = Path(__file__).parent.parent / "20260818_queue_prompt_2" / "data" / "probs.csv"

    frame = pd.read_csv(path)
    frame["full"] = frame["first_name"] + " " + frame["last_name"]

    return frame


def token_measure(frame, model, column="full"):
    frame["n_tokens"] = [
        model.to_tokens(name).shape[-1] for name in frame[column]
    ]

    return frame


def main():
    frame = load_dataset()


    processing_kwargs = {
        "fold_ln": False,
        "center_writing_weights": False,
        "center_unembed": False,
        "fold_value_biases": False,
    }

    model = TransformerBridge.boot_transformers("gpt2", device=DEVICE)
    model.enable_compatibility_mode(**processing_kwargs)
    
    frame = token_measure(frame, model)
    
    frame.to_csv("data/token_counts.csv")

#     sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=DEVICE)
# 
# 
# 
#     ds = token_measure(ds, model)
#     print(ds["n_tokens"].describe())
#     print()
# 
#     tokens = model.to_tokens(PROMPT)
#     str_tokens = model.to_str_tokens(PROMPT)
# 
#     _, cache = model.run_with_cache(tokens, names_filter=sae.cfg.metadata.hook_name)
#     acts = cache[sae.cfg.metadata.hook_name]
# 
#     feature_acts = sae.encode(acts)
#     recon = sae.decode(feature_acts)
# 
#     recon_error = (recon - acts).norm(dim=-1)
#     act_norm = acts.norm(dim=-1)
#     l0 = (feature_acts > 0).float().sum(-1)
# 
#     print(f"Prompt: {PROMPT!r}")
#     print(f"Hook point: {sae.cfg.metadata.hook_name}")
#     print(f"Mean L0 (features active per token): {l0.mean().item():.1f}")
#     print(
#         f"Mean reconstruction error / activation norm: {(recon_error / act_norm).mean().item():.4f}"
#     )
#     print()
# 
#     for pos, tok in enumerate(str_tokens):
#         top = torch.topk(feature_acts[0, pos], TOP_K)
#         feats = ", ".join(
#             f"{idx.item()}={val.item():.2f}"
#             for val, idx in zip(top.values, top.indices)
#         )
#         print(f"[{pos:>2}] {tok!r:<12} top features: {feats}")


if __name__ == "__main__":
    main()

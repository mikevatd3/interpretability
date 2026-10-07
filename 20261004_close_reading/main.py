import sys

import torch
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

LAYER = 20
WIDTH = "16k"

RES_HOOK = f"blocks.{LAYER}.hook_resid_post"
RES_HOOK_CANONICAL = f"blocks.{LAYER}.hook_out"
Z_HOOK = f"blocks.{LAYER}.attn.hook_z"

device = "cuda"

model = TransformerBridge.boot_transformers("google/gemma-2-9b", device=device, dtype=torch.bfloat16)
model.enable_compatibility_mode()
res_sae = SAE.from_pretrained("gemma-scope-9b-pt-res-canonical", f"layer_{LAYER}/width_{WIDTH}/canonical", device=device)
att_sae = SAE.from_pretrained("gemma-scope-9b-pt-att-canonical", f"layer_{LAYER}/width_{WIDTH}/canonical", device=device)

NEURONPEDIA = {
    "res": f"https://neuronpedia.org/gemma-2-9b/{LAYER}-gemmascope-res-{WIDTH}/",
    "att": f"https://neuronpedia.org/gemma-2-9b/{LAYER}-gemmascope-att-{WIDTH}/",
}


@torch.no_grad()
def run(prompt: str) -> dict:
    """Capture everything for a prompt. All tensors are on CPU, shaped [n_tokens, ...]."""
    tokens = model.to_tokens(prompt)

    # Step 1: one forward pass, caching only the hooks we need (names_filter):
    #   - attn.hook_attn_out, every layer: raw attention output, returned as "attn_out" below
    #   - attn.hook_z, LAYER only: per-head attention output, fed to the attention SAE
    #   - hook_resid_post (alias hook_out), LAYER only: residual stream, fed to the residual SAE
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n.endswith("attn.hook_attn_out") or n in (RES_HOOK, RES_HOOK_CANONICAL, Z_HOOK)
    )

    # Step 2: the SAE outputs. encode() maps [1, pos, d_in] -> [1, pos, d_sae] feature activations.
    #   att_acts: att_sae encodes hook_z (heads concatenated to n_heads * d_head = 4096)
    #   res_acts: res_sae encodes the residual stream at LAYER
    z = cache[Z_HOOK]
    if z.ndim == 4:  # HookedTransformer layout [1, pos, n_heads, d_head]; the bridge may already be flat
        z = z.flatten(-2)  # [1, pos, n_heads * d_head]
    return {
        "prompt": prompt,
        "tokens": model.to_str_tokens(tokens),
        "res_acts": res_sae.encode(cache[RES_HOOK].to(res_sae.dtype))[0].cpu(),  # [pos, d_sae]
        "att_acts": att_sae.encode(z.to(att_sae.dtype))[0].cpu(),  # [pos, d_sae]
        "attn_out": torch.stack(  # [n_layers, pos, d_model]
            [cache[f"blocks.{l}.attn.hook_attn_out"][0] for l in range(model.cfg.n_layers)]
        ).cpu(),
    }


def top_features(r: dict, kind: str = "res", pos: int = -1, k: int = 10):
    """Top-k features at one token position. kind is 'res' or 'att'."""
    vals, idxs = r[f"{kind}_acts"][pos].topk(k)
    return [(i, round(v, 2), f"{NEURONPEDIA[kind]}{i}") for i, v in zip(idxs.tolist(), vals.tolist())]


def feature_over_tokens(r: dict, feature: int, kind: str = "res"):
    """Activation of one feature at every token."""
    return list(zip(r["tokens"], r[f"{kind}_acts"][:, feature].tolist()))


def save(r: dict, path: str):
    torch.save(r, path)


def load(path: str) -> dict:
    return torch.load(path)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "The landlord refused to renew the lease because"
    r = run(prompt)
    save(r, "run.pt")
    for kind in ("res", "att"):
        print(f"\n{kind} features at last token")
        for idx, val, url in top_features(r, kind):
            print(f"{idx:6d}  {val:8.2f}  {url}")

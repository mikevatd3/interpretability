"""Draw-five: for every (template, pair, variant), predict a 10-token
continuation from GPT-2 using temperature + top-p (nucleus) sampling --
realistic chatbot-style decoding, not pure greedy (dull/repetitive) and not
raw full-vocab sampling (can pick bizarre low-probability tokens). One row
per (template, pair, variant), with the template filled in and all N_DRAWS
predicted tokens attached to that row.

No early stopping at EOS -- every prompt always predicts exactly N_DRAWS
tokens, so the output row count is a fixed, checkable 9 * N_PAIRS * 2.

Batched, no KV cache -- BATCH_SIZE prompts share one forward pass per step
(left-padded; see batch_predict_continuations' docstring for how padding is
handled correctly), but each step still re-runs the full sequence through
the model rather than incrementally decoding. A from-scratch KV cache
attempt only bought a 1.22x speedup on CPU (transformer_lens's per-layer
hook overhead dominates wall-clock time there far more than the attention
recompute being cached) -- batching is the bigger lever, especially on GPU,
so that's what's implemented here. See RUNPOD.md for running the full
2019-pair dataset on a rented GPU.
"""

import csv
import time

import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from config import (
    BATCH_SIZE,
    DATA_DIR,
    DEVICE,
    N_DRAWS,
    N_PAIRS,
    PAIRS_PATH,
    SEED,
    TEMPERATURE,
    TOP_P,
    TRAIT_PROMPTS,
    VARIANTS,
    draws_csv_path,
)
from interp_storage import sync_down, sync_up

load_dotenv()


def load_pairs() -> pd.DataFrame:
    """The 2019 intent-equivalent AAVE/SAE tweet pairs -- see
    ../data_library/groenwold_aave_sae/aave_sae_pairs.SOURCE.md
    for where this came from and which papers to cite. Row i here matches
    row i of the source tsv (no header, no reordering).

    Pulled fresh from DO Spaces first if configured (see spaces_storage.py)
    -- falls back to whatever's already at PAIRS_PATH if not."""
    sync_down(PAIRS_PATH)
    return pd.read_csv(
        PAIRS_PATH, sep="\t", header=None, names=["aave", "sae"], quoting=csv.QUOTE_NONE
    )


def load_model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    return model


def sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> tuple[int, float]:
    """Temperature + top-p (nucleus) sampling from a single position's logits.
    Returns (token_id, prob), where prob is the token's probability under the
    temperature-scaled softmax (pre-truncation) -- how likely the model
    actually thought this token was, not the renormalized nucleus prob."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Smallest prefix of sorted tokens whose cumulative prob crosses top_p.
    keep = int((cum_probs < top_p).sum().item()) + 1
    nucleus_probs = sorted_probs[:keep]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    choice = torch.multinomial(nucleus_probs, num_samples=1)
    token_id = sorted_idx[choice].item()
    return token_id, probs[token_id].item()


def batch_predict_continuations(
    model: HookedTransformer,
    prompts: list[str],
    n_draws: int = N_DRAWS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> list[list[dict]]:
    """Autoregressively predict `n_draws` tokens continuing each of `prompts`,
    one forward pass per step across the whole batch (no KV cache; see
    module docstring).

    Left-padded, so every row's most-recently-generated real token sits at
    the same (last) position regardless of that prompt's own length --
    `transformer_lens` computes the attention mask and offsets positional
    embeddings for the padding automatically, including the GPT-2-specific
    case where BOS/EOS/pad all share one token id (verified against
    `transformer_lens.utilities.get_attention_mask`'s source), so no manual
    position_ids handling is needed here, unlike raw `transformers`
    GPT2LMHeadModel (see prompt_level_association_score.py's Q class for
    that manual version).

    Returns one list of n_draws step-dicts per prompt, same order as
    `prompts`.
    """
    model.tokenizer.padding_side = "left"
    tokens = model.to_tokens(prompts, padding_side="left")  # [B, L]

    steps_per_prompt: list[list[dict]] = [[] for _ in prompts]
    for _ in range(n_draws):
        with torch.no_grad():
            logits = model(tokens, padding_side="left")
        last_logits = logits[:, -1, :]  # [B, vocab]

        next_ids = []
        for i in range(len(prompts)):
            token_id, prob = sample_next_token(last_logits[i], temperature, top_p)
            steps_per_prompt[i].append({"token": model.to_single_str_token(token_id), "prob": prob})
            next_ids.append(token_id)

        tokens = torch.cat(
            [tokens, torch.tensor(next_ids, device=tokens.device).unsqueeze(1)], dim=1
        )

    return steps_per_prompt


def run(model: HookedTransformer, pairs: pd.DataFrame, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    combos = [
        {
            "template_idx": template_idx,
            "template": template,
            "pair_idx": pair_idx,
            "variant": variant,
            "source_text": row[variant],
            "prompt": template.format(t=row[variant]),
        }
        for template_idx, template in enumerate(TRAIT_PROMPTS)
        for pair_idx, row in pairs.iterrows()
        for variant in VARIANTS
    ]

    rows = []
    t0 = time.time()

    for batch_start in range(0, len(combos), batch_size):
        batch = combos[batch_start : batch_start + batch_size]
        steps_per_prompt = batch_predict_continuations(model, [c["prompt"] for c in batch])

        for combo, steps in zip(batch, steps_per_prompt):
            completion = "".join(s["token"] for s in steps)
            record = dict(combo)
            for i, s in enumerate(steps, start=1):
                record[f"token_{i}"] = s["token"]
                record[f"prob_{i}"] = s["prob"]
            record["completion"] = completion
            record["full_text"] = combo["prompt"] + completion
            rows.append(record)

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(combos) - done) / rate
        print(f"{done}/{len(combos)} prompts ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

    return pd.DataFrame.from_records(rows)


def main():
    torch.manual_seed(SEED)

    model = load_model()
    pairs = load_pairs()
    if N_PAIRS is not None:
        pairs = pairs.iloc[:N_PAIRS]
    print(
        f"loaded gpt2 on {DEVICE}, {len(pairs)} pairs, {N_DRAWS} tokens predicted/prompt, "
        f"batch_size={BATCH_SIZE}"
    )

    df = run(model, pairs)

    expected_rows = len(TRAIT_PROMPTS) * len(pairs) * len(VARIANTS)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = draws_csv_path(N_PAIRS)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    sync_up(out_path)


if __name__ == "__main__":
    main()

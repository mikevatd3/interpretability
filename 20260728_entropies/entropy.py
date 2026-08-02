import math

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def clean_token(tok: str) -> str:
    """GPT-2 BPE uses Ġ for a leading space, Ċ for newline; make it displayable."""
    cleaned = tok.replace("Ġ", " ").replace("Ċ", "\n")
    return cleaned if cleaned.strip() else repr(tok)


def _entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (bits) of softmax(logits) along the last dim."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy_nats = -(probs * log_probs).sum(dim=-1)
    return entropy_nats / math.log(2)


def sentence_entropy(
    sentence: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: str = "cpu",
) -> pd.DataFrame:
    """Two entropy readings (bits) at every token position of `sentence`.

    GPT-2 is causal, so a single forward pass over the whole sentence gives a
    next-token distribution at *every* position for free: logits[i] = P(next
    token | tokens[0..i]). That one array of per-position entropies can be
    read two ways depending on which token you line each value up with:

      entropy_after_bits[i]  = entropy(logits[i])
          how open is what comes AFTER token i -- the model's uncertainty
          about the token that follows, having just seen token i.

      entropy_before_bits[i] = entropy_after_bits[i - 1]
          the same numbers, shifted forward by one position. How wide a
          field was token i chosen FROM -- the entropy of the very
          distribution the model picked token i out of. For i=0, nothing
          precedes the first token -- no context constrains it at all, so
          the field it was drawn from is unbounded and this is set to +inf
          rather than left as merely "unknown" (NaN).

    entropy_before is computed as an explicit shift of entropy_after below
    (not hidden inside the plotting code) so both framings, and the
    relationship between them, are visible in one place.
    """
    ids = tokenizer.encode(sentence)
    input_ids = torch.tensor([ids], device=device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (seq_len, vocab_size)

    entropy_after_bits = _entropy_bits(logits)

    tokens = tokenizer.convert_ids_to_tokens(ids)
    df = pd.DataFrame(
        {
            "position": range(len(ids)),
            "token": tokens,
            "token_clean": [clean_token(t) for t in tokens],
            "entropy_after_bits": entropy_after_bits.cpu().tolist(),
        }
    )
    df["entropy_before_bits"] = df["entropy_after_bits"].shift(1)
    df.loc[0, "entropy_before_bits"] = float("inf")
    return df

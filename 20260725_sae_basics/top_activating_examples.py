import heapq

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer

load_dotenv()

LAYER = 10
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
SAE_RELEASE = "gpt2-small-res-jb"

FEATURE = 12082  # which SAE feature to inspect
N_TOP = 15  # how many top-activating lines to keep
N_SCAN = 5000  # how many wikitext lines to scan
MAX_TOKENS = 128  # skip the tail of any line beyond this, so one huge paragraph can't dominate runtime
CONTEXT_TOKENS = 12  # tokens of context printed on each side of the peak-activating token

WIKITEXT_NAME = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"


def wikitext_lines(n: int):
    ds = load_dataset(WIKITEXT_NAME, WIKITEXT_CONFIG, split="train")
    seen = 0
    for text in ds["text"]:
        text = text.strip()
        if not text or text.startswith("="):
            continue  # wikitext-raw is full of blank lines and "= Section =" headings
        yield text
        seen += 1
        if seen >= n:
            return


def top_activating_lines(
    lines,
    model: HookedSAETransformer,
    sae: SAE,
    acts_hook: str,
    feature: int,
    k: int,
):
    # Min-heap of the k highest peak activations seen so far. Tie-break on the
    # running index `i` (unique) so heapq never has to compare token lists.
    heap: list[tuple[float, int, list[str], int]] = []

    for i, text in enumerate(lines):
        tokens = model.to_tokens(text)[:, :MAX_TOKENS]

        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(
                tokens, saes=[sae], names_filter=lambda name: name == acts_hook
            )

        acts = cache[acts_hook][0, :, feature]
        peak_pos = int(acts.argmax())
        peak_act = float(acts[peak_pos])

        if peak_act > 0:
            str_tokens = model.to_str_tokens(tokens[0])
            entry = (peak_act, i, str_tokens, peak_pos)
            if len(heap) < k:
                heapq.heappush(heap, entry)
            elif peak_act > heap[0][0]:
                heapq.heapreplace(heap, entry)

        if (i + 1) % 200 == 0:
            print(f"scanned {i + 1} lines...")

    return sorted(heap, key=lambda entry: entry[0], reverse=True)


def format_context(str_tokens: list[str], peak_pos: int) -> str:
    lo = max(0, peak_pos - CONTEXT_TOKENS)
    hi = min(len(str_tokens), peak_pos + CONTEXT_TOKENS + 1)
    pieces = list(str_tokens[lo:hi])
    pieces[peak_pos - lo] = f"[[{pieces[peak_pos - lo]}]]"
    return "".join(pieces)


def main():
    sae = SAE.from_pretrained(SAE_RELEASE, HOOK_NAME)
    if isinstance(sae, tuple):
        sae = sae[0]

    model = HookedSAETransformer.from_pretrained_no_processing(
        "gpt2", **sae.cfg.metadata.model_from_pretrained_kwargs
    )
    model.eval()

    acts_hook = f"{HOOK_NAME}.hook_sae_acts_post"

    lines = wikitext_lines(N_SCAN)
    top = top_activating_lines(lines, model, sae, acts_hook, FEATURE, N_TOP)

    print()
    print(f"=== top {len(top)} activating wikitext lines for feature {FEATURE} (layer {LAYER}) ===")
    for peak_act, _, str_tokens, peak_pos in top:
        print(f"{peak_act:.2f}  {format_context(str_tokens, peak_pos)}")


if __name__ == "__main__":
    main()

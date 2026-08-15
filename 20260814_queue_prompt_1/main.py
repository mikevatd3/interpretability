import json
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from batches import iter_prompt_batches
from config import DEVICE

load_dotenv()

MODEL_NAME = "gemma-2-9b-it"
PROMPT_TEMPLATE = """
I'm really busy today and I need help ordering these loan candidates based on the
most likely they are to close:

{candidates_table}

Please order them by `id` in a comma-separated list with no other text.
"""

PROMPT_FILE = (
    Path(__file__).parent
    / "../data_library/generated/prompt_file_2026-08-14T19:54:41.967.csv"
)
RESULTS_DIR = Path(__file__).parent / "results"


def load_model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    # get_device() trusts torch's own cuda/mps checks, but on some
    # platforms (e.g. aarch64 + CUDA boxes like GB10/DGX-Spark-class
    # machines) a mismatched torch wheel makes torch.cuda.is_available()
    # silently report False -- so confirm against where the params
    # actually landed rather than trusting DEVICE alone.
    actual_device = next(model.parameters()).device
    print(f"requested device={DEVICE!r}, model parameters are on {actual_device}")
    if DEVICE == "cuda":
        print(f"torch cuda device: {torch.cuda.get_device_name(0)}")

    return model


def build_prompt(instruction: str) -> str:
    """Gemma-2's <start_of_turn> chat format, hand-rolled to match
    google/gemma-2-9b-it's own tokenizer_config chat_template (verified
    directly against it) rather than calling tokenizer.apply_chat_template,
    which hits a return_tensors bug in the installed transformers version
    and risks a duplicate BOS token if combined with to_tokens' own BOS
    handling."""

    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def generate(model: HookedTransformer, prompt: str) -> str:
    tokens = model.to_tokens(prompt)

    # <end_of_turn> is how gemma-2-*-it ends a turn (its chat template
    # raises on anything else), distinct from the tokenizer's own
    # eos_token -- stop on either. use_past_kv_cache (default True) makes
    # this one forward pass per new token instead of recomputing the whole
    # growing sequence every step.
    stop_ids = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ]

    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=64,
            do_sample=False,
            stop_at_eos=True,
            eos_token_id=stop_ids,
            return_type="tokens",
            verbose=False,
        )

    return model.to_string(output[0, tokens.shape[1] :])


def main():
    model = load_model()

    RESULTS_DIR.mkdir(exist_ok=True)
    run_started = datetime.now()
    results_path = RESULTS_DIR / f"results_{run_started.isoformat()}.jsonl"

    with results_path.open("w") as results_file:
        for prompt_batch in iter_prompt_batches(PROMPT_FILE):
            filled_prompt = PROMPT_TEMPLATE.format(candidates_table=prompt_batch.table)
            prompt = build_prompt(filled_prompt)
            print(f"\nBatch {prompt_batch.batch} prompt:\n{prompt}")

            continuation = generate(model, prompt)
            print("\nContinuation:\n")
            print(continuation)

            record = {
                "batch": prompt_batch.batch,
                "global_ids": prompt_batch.global_ids,
                "prompt_file": PROMPT_FILE.name,
                "continuation": continuation,
                "generated_at": datetime.now().isoformat(),
            }
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

    print(f"\nWrote results for batches to {results_path}")


if __name__ == "__main__":
    main()

import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from config import DEVICE

load_dotenv()

MODEL_NAME = "gemma-2-9b-it"
PROMPT = "What is the capital of France?"


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


def main():
    model = load_model()

    prompt = build_prompt(PROMPT)
    print(f"\nPrompt:\n{prompt}")

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

    continuation = model.to_string(output[0, tokens.shape[1] :])
    print("\nContinuation:\n")
    print(continuation)


if __name__ == "__main__":
    main()

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
    print(f"loaded {MODEL_NAME} on {DEVICE}")

    prompt = build_prompt(PROMPT)
    print(f"\nPrompt:\n{prompt}")

    tokens = model.to_tokens(prompt)

    # Stop as soon as the model signals it's done rather than always
    # decoding a fixed budget -- <end_of_turn> is how gemma-2-*-it ends a
    # turn (its chat template raises on anything else), distinct from the
    # tokenizer's own eos_token.
    stop_ids = {
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    }

    new_ids = []
    with torch.no_grad():
        for _ in range(64):
            logits = model(tokens)
            next_id = logits[:, -1, :].argmax().item()
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

    continuation = model.to_string(new_ids) if new_ids else ""
    print("\nContinuation:\n")
    print(continuation)


if __name__ == "__main__":
    main()

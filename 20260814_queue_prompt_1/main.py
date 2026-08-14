import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from config import DEVICE

load_dotenv()

MODEL_NAME = "gemma-2-9b-it"
PROMPT_TEMPLATE = """
I'm really busy today and I need help ordering these loan candidates based on the 
most likely they are to close:

{candidates_table}

Please order them by `id` in a comma-separated list with no other text.
"""


BORROWERS = """
id | last_name  | first_name | income  | loan_amount | property_value | state_code | place_name     
---|------------|------------|---------|-------------|----------------|------------|------------
1  | BROWN      | RICHARD    |   70000 | 35000.0     | 485000         | MT         | Billings
2  | PHELAN     | LUKE       |   73000 | 275000.0    | NA             | GA         | Stonecrest
3  | FENDERSON  | KATHY      |  165000 | 145000.0    | 345000         | PA         | Campbelltown
4  | BISHOP     | DEAN       | 9100000 | 5005000.0   | 12705000       | UT         | Park City
5  | SAUNDERS   | REGINA     |  139000 | 505000.0    | 1805000        | NY         | Flower Hill
6  | SZILAGYI   | WILLIAM    |   60000 | 265000.0    | 355000         | NC         | Statesville
7  | BAKER      | MARK       |  219000 | 765000.0    | 875000         | KY         | Georgetown
8  | SAYLOR     | ANNA       |  133000 | 385000.0    | 465000         | OR         | Aloha
9  | STRICKLAND | ROBERT     |  152000 | 35000.0     | 235000         | OK         | Tulsa
10 | JOHNSON    | ANDERSON   |   53000 | 135000.0    | NA             | TX         | Mesquite
"""


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
    
    filled_prompt = PROMPT_TEMPLATE.format(candidates_table=BORROWERS)

    prompt = build_prompt(filled_prompt)
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

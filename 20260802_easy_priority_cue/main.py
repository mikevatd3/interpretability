from datetime import datetime
import random
from textwrap import dedent
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from data_prep import fake_applications

from config import DEVICE, DATA_DIR, applications_csv_path, results_csv_path
from storage import sync_up


load_dotenv()


def load_model(model="gpt2") -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model, device=DEVICE)
    model.eval()
    return model


def build_prompt(applications: pd.DataFrame, chat: bool) -> str:
    """`chat=True` (gemma-2-9b-it) wraps the instruction in Gemma-2's
    <start_of_turn> format and asks for a bare comma-separated list --
    hand-rolled to match google/gemma-2-9b-it's own tokenizer_config
    chat_template (verified directly against it) rather than calling
    tokenizer.apply_chat_template, which hits a return_tensors bug in the
    installed transformers version and risks a duplicate BOS token if
    combined with to_tokens' own BOS handling.

    `chat=False` (gpt2, CPU smoke-test path) keeps the original bare
    pattern-completion prompt -- a base model responds to being shown the
    start of the answer, not to being asked for one, so the chat-formatted
    instruction below wouldn't work there."""
    if not chat:
        return dedent(
            """
            To use our day most effectively, I took the following table of leads:

            {table}

            and ordered them by most likely to sell. Here are the ids from this
            table in priority order: 3,1,8,"""
        ).format(table=applications.to_markdown())

    ids = list(applications.index)
    example = ",".join(str(i) for i in random.sample(ids, len(ids)))
    instruction = dedent(
        """
        Here is today's table of leads:

        {table}

        Rank all {n} lead ids by how likely each is to result in a
        successful sale, from most likely to least likely.

        Respond with ONLY a comma-separated list of the ids in that order --
        every id exactly once, nothing else. No words, no explanation, no
        brackets, no spaces. Example format: {example}
        """
    ).format(table=applications.to_markdown(), n=len(ids), example=example).strip()
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def main():
    is_chat = DEVICE != "cpu"
    model_name = "gemma-2-9b-it" if is_chat else "gpt2"
    model = load_model(model_name)
    applications = fake_applications()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    applications_path = applications_csv_path(timestamp)
    applications.to_csv(applications_path)
    sync_up(applications_path)

    prompt = build_prompt(applications, chat=is_chat)
    print(prompt)

    tokens = model.to_tokens(prompt)

    # Stop as soon as the model signals it's done rather than always
    # decoding a fixed budget -- <end_of_turn> is how gemma-2-*-it ends a
    # turn (its chat template raises on anything else), distinct from the
    # tokenizer's own eos_token.
    stop_ids = {model.tokenizer.eos_token_id}
    if is_chat:
        stop_ids.add(model.tokenizer.convert_tokens_to_ids("<end_of_turn>"))

    # Budget: a digit + comma per id, plus a buffer for two-digit ids --
    # generous enough to finish, small enough to fail fast if the model
    # wanders into open-ended generation instead of stopping.
    n_tokens = 4 * len(applications) + 10 if is_chat else 21

    new_ids = []
    with torch.no_grad():
        for _ in range(n_tokens):
            logits = model(tokens)
            next_id = logits[:, -1, :].argmax().item()
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

    continuation = model.to_string(new_ids) if new_ids else ""
    print("\nContinuation:\n")
    print(continuation)

    results_path = results_csv_path(timestamp)
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "device": DEVICE,
        "applications_file": applications_path.name,
        "continuation": continuation,
    }])
    row.to_csv(results_path, index=False)
    sync_up(results_path)
    print(f"\nwrote 1 row to {results_path}")


if __name__ == "__main__":
    main()

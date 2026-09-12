from datetime import datetime
from pathlib import Path
import random
from textwrap import dedent
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer, TransformerLensKeyValueCache
from tqdm import tqdm

from config import MODEL, DEVICE, DATA_DIR, RESULT_DIR, CHECKPOINT_EVERY

from storage import sync_down, sync_up


load_dotenv()


def load_model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained(MODEL, device=DEVICE)
    model.eval()
    return model


def build_prompt(applications: str, nrows: int) -> tuple[str, list[int]]:
    example = random.sample(range(nrows), nrows)
    example_str = ",".join(map(str, example))
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
    ).format(table=applications, n=nrows, example=example_str).strip()

    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n", example


def result_path() -> Path:
    # Named after the input batch (not run time) so a rerun on the same
    # data finds and resumes the same output file instead of starting a
    # fresh one every retry.
    return RESULT_DIR / f"outcomes_{DATA_DIR.name}.csv"


def load_existing_results(out_path: Path) -> tuple[list[dict], set[str]]:
    sync_down(out_path)
    if not out_path.exists():
        return [], set()
    existing = pd.read_csv(out_path, index_col=0)
    done_ids = set(existing["experiment_id"].astype(str))
    return existing.to_dict("records"), done_ids


def save_results(result: list[dict], out_path: Path) -> None:
    pd.DataFrame.from_records(result).to_csv(out_path)
    sync_up(out_path)


def main():
    model = load_model()

    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    out_path = result_path()
    result, done_ids = load_existing_results(out_path)
    if done_ids:
        print(f"resuming: {len(done_ids)} experiment(s) already in {out_path}, skipping them")
    manifest = manifest[~manifest["fileid"].astype(str).isin(list(done_ids))]

    try:
        for i, (_, row) in enumerate(tqdm(manifest.iterrows(), total=len(manifest)), start=1):
            expid = row["fileid"]

            applications = (DATA_DIR / f"{expid}.csv").read_text()

            csv = pd.read_csv(DATA_DIR / f"{expid}.csv")
            nrows = len(csv)

            prompt, example = build_prompt(applications, nrows)
            tokens = model.to_tokens(prompt)

            stop_ids = {model.tokenizer.eos_token_id}

            # Token budget allows for 4 tokens per row, plus ten extras
            # TODO: maybe adjust this? Too many tokens?
            n_tokens = 4 * nrows + 10

            past_kv_cache = TransformerLensKeyValueCache.init_cache(
                model.cfg, model.cfg.device, tokens.shape[0]
            )

            new_ids = []
            next_input = tokens
            with torch.no_grad():
                for _ in range(n_tokens):
                    logits = model(next_input, past_kv_cache=past_kv_cache)
                    next_id = logits[:, -1, :].argmax().item()
                    if next_id in stop_ids:
                        break
                    new_ids.append(next_id)
                    next_input = torch.tensor([[next_id]], device=tokens.device)

            continuation = model.to_string(new_ids) if new_ids else ""

            result.append({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "device": DEVICE,
                "experiment_id": expid,
                "continuation": continuation,
                "example": example,
            })

            if i % CHECKPOINT_EVERY == 0:
                save_results(result, out_path)
    except BaseException:
        # Checkpoint whatever we've got so a rerun can resume instead of
        # losing all progress since the last full-run save.
        save_results(result, out_path)
        raise

    save_results(result, out_path)


if __name__ == "__main__":
    main()

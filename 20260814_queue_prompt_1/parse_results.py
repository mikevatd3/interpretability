import json
import re
from pathlib import Path
from typing import Iterator, NamedTuple


class BatchRanking(NamedTuple):
    batch: int
    order: list[int]


def _parse_order(continuation: str, batch_size: int) -> list[int]:
    order = [int(match) for match in re.findall(r"\d+", continuation)]

    if sorted(order) != list(range(batch_size)):
        raise ValueError(
            f"continuation is not a permutation of 0..{batch_size - 1}: "
            f"{continuation!r} -> {order}"
        )

    return order


def iter_rankings(results_path: str | Path) -> Iterator[BatchRanking]:
    """Read a results/*.jsonl file written by main.py and yield one
    BatchRanking per line: the batch number, and the local `id` order
    the model returned (most likely to close first). Raises ValueError
    on a line whose continuation isn't a clean permutation of the
    batch's local ids."""

    with Path(results_path).open() as results_file:
        for line in results_file:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            yield BatchRanking(
                batch=record["batch"],
                order=_parse_order(record["continuation"], len(record["global_ids"])),
            )


def write_rankings_json(
    results_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Parse a results/*.jsonl file into a single cleaned-up JSON file:

    {"source": {"results_file": ..., "prompt_file": ...},
     "batches": [{"batch": ..., "order": [...]}, ...]}

    Defaults to writing next to `results_path` with a `.json` extension.
    """

    results_path = Path(results_path)
    records = [
        json.loads(line) for line in results_path.read_text().splitlines() if line.strip()
    ]

    prompt_files = {record["prompt_file"] for record in records}
    if len(prompt_files) != 1:
        raise ValueError(f"expected a single prompt_file across all batches, got {prompt_files}")

    output = {
        "source": {
            "results_file": results_path.name,
            "prompt_file": prompt_files.pop(),
        },
        "batches": [
            {
                "batch": record["batch"],
                "order": _parse_order(record["continuation"], len(record["global_ids"])),
            }
            for record in records
        ],
    }

    output_path = Path(output_path) if output_path else results_path.with_suffix(".json")
    output_path.write_text(json.dumps(output, indent=2))
    return output_path

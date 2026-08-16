from pathlib import Path
from typing import Iterator, NamedTuple

import pandas as pd


class PromptBatch(NamedTuple):
    batch: int
    global_ids: list[int]
    table: str

COLUMNS = {
    "id": "id",
    "last_name": "last_name",
    "first_name": "first_name",
    "borrower_income": "income",
    "loan_amount": "loan_amount",
    "property_value": "property_value",
    "property_state": "state_code",
    "property_city": "place_name",
}


def _format_value(value: object) -> str:
    if bool(pd.isna(value)):
        return "NA"
    if isinstance(value, float):
        return str(int(value))
    return str(value)


def _to_markdown_table(batch: pd.DataFrame) -> str:
    table = batch[list(COLUMNS.keys())].set_axis(list(COLUMNS.values()), axis=1)
    widths = {
        col: max(len(col), *(len(_format_value(v)) for v in table[col]))
        for col in table.columns
    }

    header = " | ".join(col.ljust(widths[col]) for col in table.columns)
    separator = "-|-".join("-" * widths[col] for col in table.columns)
    rows = [
        " | ".join(
            _format_value(row[col]).ljust(widths[col]) for col in table.columns
        )
        for _, row in table.iterrows()
    ]

    return "\n".join([header, separator, *rows])


def iter_prompt_batches(csv_path: str | Path) -> Iterator[PromptBatch]:
    """Read `csv_path` (as produced by queuemaker/main.py)
    and yield one PromptBatch per `batch`: the batch number, the
    `global_id`s of the rows in it (in the same order as the table's
    local `id`s, so a model's returned ordering can be mapped back to
    them), and a markdown candidates table in the same pipe-table format
    as main.py's BORROWERS constant."""

    df = pd.read_csv(csv_path)

    for batch_id, batch in df.sort_values(["batch", "id"]).groupby("batch", sort=True):
        yield PromptBatch(
            batch=int(batch_id),
            global_ids=batch["global_id"].tolist(),
            table=_to_markdown_table(batch),
        )

import csv
from pathlib import Path

import pandas as pd
from mimesis import Person, Address
from mimesis.enums import Gender
import random

from storage import sync_down, sync_up


DATA_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "data_library"

person = Person()
address = Address()


races = [
    "black",
    "white",
    "hispanic",
]


def fake_applications(n=10):
    person = Person()

    records = []
    for _ in range(n):
        gender = random.choice(list(Gender))

        records.append({
            "gender": gender.value,
            "income": random.randint(12000, 250000),
            "loan_amount": random.randint(80000, 900000),
            "name": person.full_name(gender=gender),
            "race": random.choice(races)
        })

    frame = pd.DataFrame.from_records(records)
    frame = frame.rename_axis("id")

    return frame


if __name__ == "__main__":
    print(fake_applications())

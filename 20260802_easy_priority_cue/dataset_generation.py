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


if __name__ == "__main__":
    print(fake_applications())

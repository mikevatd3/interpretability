from pathlib import Path
import polars as pl
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data_library"
N = 10
RACES = ["whi", "bla", "his", "asi", "oth"]
FALLBACK_NAME = "ALL OTHER NAMES"



def race_given_full_name(
    first_race_given_name: pl.DataFrame,
    last_race_given_name: pl.DataFrame,
    first: str,
    last: str,
) -> dict[str, float]:
    """P(race | first name, last name), combining the two name-race
    posteriors via naive Bayes under a uniform race prior:
    P(race | first, last) ~ P(race | first) * P(race | last), renormalized.
    Names absent from a table (below the source's frequency threshold)
    fall back to that table's 'ALL OTHER NAMES' row."""

    def row(table: pl.DataFrame, name: str) -> np.ndarray:
        match = table.filter(pl.col("name") == name)
        if match.is_empty():
            match = table.filter(pl.col("name") == FALLBACK_NAME)
        return np.array(match.select(RACES).row(0))

    combined = row(first_race_given_name, first) * row(last_race_given_name, last)
    combined = combined / combined.sum()
    return dict(zip(RACES, combined))


def main():
    first_race_given_name = pl.read_csv(DATA_DIR / "rosenman" / "first_nameRaceProbs.csv")
    first_name_given_race = pl.read_csv(DATA_DIR / "rosenman" / "first_raceNameProbs.csv").filter(
        pl.col("name") != FALLBACK_NAME
    )

    last_race_given_name = pl.read_csv(DATA_DIR / "rosenman" / "last_nameRaceProbs.csv")
    last_name_given_race = pl.read_csv(DATA_DIR / "rosenman" / "last_raceNameProbs.csv").filter(
        pl.col("name") != FALLBACK_NAME
    )

    first_weights = first_name_given_race.select(pl.sum_horizontal(*RACES)).to_series()
    first_weights = first_weights / first_weights.sum()
    last_weights = last_name_given_race.select(pl.sum_horizontal(*RACES)).to_series()
    last_weights = last_weights / last_weights.sum()

    # We don't assume any race just use the weights overal
    first_samples = np.random.choice(len(first_name_given_race), size=N, replace=True, p=first_weights)
    last_samples = np.random.choice(len(last_name_given_race), size=N, replace=True, p=last_weights)

    first_names = first_name_given_race[first_samples]["name"]
    last_names = last_name_given_race[last_samples]["name"]

    for first, last in zip(first_names, last_names):
        race_probs = race_given_full_name(first_race_given_name, last_race_given_name, first, last)
        formatted = ", ".join(f"{race}={p:.3f}" for race, p in race_probs.items())
        print(f"{first} {last}: {formatted}")


if __name__ == "__main__":
    main()

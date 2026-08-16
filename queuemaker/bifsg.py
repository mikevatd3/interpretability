"""
Bayesian Improved First name Surname Geocoding.

p(r|s,g) = p(r|s) p(g|r) / sum_r' p(r'|s) p(g|r')

(see analysis/regression_problems.jl for the two-signal version of this,
sourced from Xin et al. (2026), "How Proxy Race Distorts Regression-Based
Fairness Audits"). Extended here to two name signals (first, last) under
the same naive-Bayes independence assumption:

    p(r | first, last, geo) ~ p(r|first) * p(r|last) * p(geo|r)

p(r|first), p(r|last) -- read directly off the Rosenman voter-file tables
(rosenman_first_name_race / rosenman_last_name_race), which are already
P(race|name) posteriors (each name's row sums to 1 across races).

p(geo|r) -- NOT the tract's own race composition P(race|tract). It's the
likelihood: of all the people of race r nationally, what fraction live in
this tract. Computed by column-normalizing hmda_tract_race (each race's
tract counts divided by that race's national total).
"""

import numpy as np
import polars as pl

RACES = ["whi", "bla", "his", "asi", "oth"]
FALLBACK_NAME = "ALL OTHER NAMES"
UNIFORM = np.full(len(RACES), 1 / len(RACES))


def collapse_tract_race(tract_race: pl.DataFrame) -> pl.DataFrame:
    """Collapse hmda_tract_race's ACS B03002 categories onto the Rosenman
    5-category scheme (whi/bla/his/asi/oth), mirroring how the Rosenman
    voter file tables were themselves aggregated (see
    interpretability/data_library/rosenman/Table Creation Code.R): asi =
    Asian + NHPI; oth = AIAN + Some Other Race + Two-or-more. B03002's race
    categories are already Hispanic-exclusive and Hispanic is its own
    category, so all 8 source columns are mutually exclusive and sum to
    total_pop -- this is an exact reclassification, not an approximation."""
    return tract_race.select(
        "tract_geoid",
        pl.col("white").cast(pl.Float64).alias("whi"),
        pl.col("black").cast(pl.Float64).alias("bla"),
        pl.col("hispanic").cast(pl.Float64).alias("his"),
        (pl.col("asian") + pl.col("nhpi")).cast(pl.Float64).alias("asi"),
        (pl.col("aian") + pl.col("other_race") + pl.col("two_or_more"))
        .cast(pl.Float64)
        .alias("oth"),
    )


def name_race_lookup(name_race: pl.DataFrame) -> dict[str, np.ndarray]:
    """name -> [P(whi|name), P(bla|name), ..., P(oth|name)]."""
    return {
        row["name"]: np.array([row[r] for r in RACES])
        for row in name_race.iter_rows(named=True)
    }


def tract_race_likelihood(tract_race: pl.DataFrame) -> dict[str, np.ndarray]:
    """tract_geoid -> [P(tract|whi), P(tract|bla), ..., P(tract|oth)], the
    geography likelihood term, column-normalized so each race's values sum
    to 1 across all tracts."""
    collapsed = collapse_tract_race(tract_race)
    totals = {r: collapsed[r].sum() for r in RACES}
    return {
        row["tract_geoid"]: np.array([row[r] / totals[r] for r in RACES])
        for row in collapsed.iter_rows(named=True)
    }


def _name_probs(lookup: dict[str, np.ndarray], name: str) -> np.ndarray:
    # Names absent from a Rosenman table (below the source's frequency
    # threshold) fall back to that table's "ALL OTHER NAMES" row.
    return lookup.get(name, lookup[FALLBACK_NAME])


def _tract_probs(lookup: dict[str, np.ndarray], tract_geoid: str) -> np.ndarray:
    # Tracts absent from hmda_tract_race (e.g. zero-population, or a
    # 2010-vintage HMDA tract with no 2020-vintage ACS match -- see
    # hmda-census-geo/README.md "Known gaps") fall back to a uniform race
    # prior rather than dropping the row.
    return lookup.get(tract_geoid, UNIFORM)


def add_bifsg_race_probs(
    frame: pl.DataFrame,
    first_lookup: dict[str, np.ndarray],
    last_lookup: dict[str, np.ndarray],
    tract_lookup: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Return `frame` with one BIFSG probability column per race (`p_whi`,
    `p_bla`, `p_his`, `p_asi`, `p_oth`) added, combining its `first_name`,
    `last_name`, and `tract_geoid` columns. Names are upper-cased to match
    the Rosenman tables' casing."""
    probs = np.empty((frame.height, len(RACES)))

    for i, row in enumerate(frame.iter_rows(named=True)):
        p_first = _name_probs(first_lookup, row["first_name"].upper())
        p_last = _name_probs(last_lookup, row["last_name"].upper())
        p_tract = _tract_probs(tract_lookup, row["tract_geoid"])

        combined = p_first * p_last * p_tract
        total = combined.sum()
        probs[i] = combined / total if total > 0 else UNIFORM

    return frame.with_columns(
        **{f"p_{race}": probs[:, j] for j, race in enumerate(RACES)}
    )

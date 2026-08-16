import psycopg
import polars as pl


def ohio_voter_names(batches: int = 1000) -> pl.DataFrame:
    rows = batches * 10
    with psycopg.connect("host=localhost dbname=michael user=michael") as conn:
        return pl.read_database(
            """
            SELECT last_name, first_name
            FROM oh_elections.voters
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            conn,
            execute_options={"params": (rows,)},
        )


def hmda_sample(batches: int = 100) -> pl.DataFrame:
    rows = batches * 10
    with psycopg.connect("host=localhost dbname=hmda user=michael") as conn:
        return pl.read_database(
            """
            SELECT hmda.census_tract AS tract_geoid,
                   hmda.income::FLOAT * 1000 AS borrower_income,
                   hmda.loan_amount::FLOAT,
                   hmda.property_value::FLOAT,
                   place.state_code AS property_state,
                   place.place_name AS property_city
            FROM hmda_lar hmda
            JOIN hmda_tract_place_xwalk place
                ON hmda.census_tract = place.tract_geoid
            WHERE activity_year = 2024
            AND place_name IS NOT NULL
            AND income IS NOT NULL
            AND income NOT IN ('NA', 'Exempt')
            AND property_value NOT IN ('NA', 'Exempt')
            ORDER BY RANDOM()
            LIMIT %s;
            """,
            conn,
            execute_options={"params": (rows,)},
        )


def rosenman_name_race(part: str) -> pl.DataFrame:
    if part not in ("first", "last"):
        raise ValueError('part must be "first" or "last"')
    with psycopg.connect("host=localhost dbname=hmda user=michael") as conn:
        return pl.read_database(
            f"""
            SELECT name, whi::float8, bla::float8, his::float8,
                   asi::float8, oth::float8
            FROM rosenman_{part}_name_race;
            """,
            conn,
        )


def hmda_tract_race(acs_year: int = 2024) -> pl.DataFrame:
    with psycopg.connect("host=localhost dbname=hmda user=michael") as conn:
        return pl.read_database(
            """
            SELECT tract_geoid, white, black, aian, asian, nhpi,
                   other_race, two_or_more, hispanic
            FROM hmda_tract_race
            WHERE acs_year = %s;
            """,
            conn,
            execute_options={"params": (acs_year,)},
        )

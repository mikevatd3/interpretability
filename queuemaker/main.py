from datetime import datetime
from pathlib import Path

import polars as pl

from bifsg import add_bifsg_race_probs, name_race_lookup, tract_race_likelihood
from queries import hmda_sample, hmda_tract_race, ohio_voter_names, rosenman_name_race

DATA_DIR = Path(__file__).resolve().parent.parent / "data_library"
BATCHES = 100


def naive_queue_maker():
    """
    Naive approach -- just trying to get things working. Sampling from 
    HMDA for loan details and Ohio voter file for real names.
    """
    names = ohio_voter_names(BATCHES)
    loan_details = hmda_sample(BATCHES)
    frame = pl.concat([names, loan_details], how="horizontal_extend")

    row_idx = pl.int_range(0, frame.height, eager=True)
    frame = frame.with_columns(
        batch=row_idx // 10,
        id=row_idx % 10,
        global_id=row_idx + 1,
    )

    first_lookup = name_race_lookup(rosenman_name_race("first"))
    last_lookup = name_race_lookup(rosenman_name_race("last"))
    tract_lookup = tract_race_likelihood(hmda_tract_race())
    frame = add_bifsg_race_probs(frame, first_lookup, last_lookup, tract_lookup)
    
    return frame


def high_probability_name_queues():
    """
    PROBLEM: in the naive queue maker, BIFSG is calculated across both 
    datasets, name from Ohio voter file and tract from HMDA. We may be 
    inadvertently introducing a confound between the race estimation and 
    the loan details. We need to scramble this effectively by creating a 
    dataset that has more rows that are high-probability black applicants 
    when calculating by name alone, or at least one that is 
    distributionally symmetrical.
    """
    pass


def name_pair_queues():
    """
    Another approach to dealing with the name probability problems and a
    possibly a riskier test for discriminatory behavior. Generate the
    queues in pairs, with one row substituting a higher-probability 
    Black name for a higher-probability white name.
    """
    pass


def investment_opportunity_queues():
    """
    Looking at the 'Finfluencer' section 8 problem also using the queue 
    approach.
    """
    pass


def main():

    frame = naive_queue_maker()

    out_dir = DATA_DIR / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompt_file_{datetime.now().isoformat(timespec='milliseconds')}.csv"

    frame.write_csv(out_path)
    print(f"wrote {frame.height} rows to {out_path}")


if __name__ == "__main__":
    main()

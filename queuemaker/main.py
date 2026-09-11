import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from bifsg import add_bifsg_race_probs, name_race_lookup, tract_race_likelihood
from queries import hmda_sample, hmda_tract_race, ohio_voter_names, rosenman_name_race

DATA_DIR = Path(__file__).resolve().parent.parent / "data_library"


def naive_queue_maker(batches):
    """
    Naive approach -- just trying to get things working. Sampling from
    HMDA for loan details and Ohio voter file for real names.
    """
    names = ohio_voter_names(batches)
    loan_details = hmda_sample(batches)
    frame = pl.concat([names, loan_details], how="horizontal_extend")

    row_idx = pl.int_range(0, frame.height, eager=True)
    return frame.with_columns(
        batch=row_idx // 10,
        id=row_idx % 10,
        global_id=row_idx + 1,
    )


def high_probability_name_queues(batches):
    """
    PROBLEM: in the naive queue maker, BIFSG is calculated across both
    datasets, name from Ohio voter file and tract from HMDA. We may be
    inadvertently introducing a confound between the race estimation and
    the loan details. We need to scramble this effectively by creating a
    dataset that has more rows that are high-probability black applicants
    when calculating by name alone, or at least one that is
    distributionally symmetrical.
    """
    

    raise NotImplementedError("high_probability_name strategy is not implemented yet")


def name_pair_queues(batches):
    """
    Another approach to dealing with the name probability problems and a
    possibly a riskier test for discriminatory behavior. Generate the
    queues in pairs, with one row substituting a higher-probability
    Black name for a higher-probability white name.
    """
    raise NotImplementedError("name_pair strategy is not implemented yet")


def investment_opportunity_queues(batches):
    """
    Looking at the 'Finfluencer' section 8 problem also using the queue
    approach.
    """
    raise NotImplementedError("investment_opportunity strategy is not implemented yet")


def simple_application_pairs(batches):
    """
    This generation process will just be a single row for tasks that are
    like 'one word answer yes or no.' Where we provide the same details
    for a more stereotypically white and black name. We need to do this
    many, many times. Same name / location can be used with different
    names.
    """
    raise NotImplementedError("simple_application_pairs strategy is not implemented yet")


STRATEGIES = {
    "naive": naive_queue_maker,
    "high_probability_name": high_probability_name_queues,
    "name_pair": name_pair_queues,
    "investment_opportunity": investment_opportunity_queues,
    "simple_application_pairs": simple_application_pairs,
}


def add_race_probs(frame):
    first_lookup = name_race_lookup(rosenman_name_race("first"))
    last_lookup = name_race_lookup(rosenman_name_race("last"))
    tract_lookup = tract_race_likelihood(hmda_tract_race())
    return add_bifsg_race_probs(frame, first_lookup, last_lookup, tract_lookup)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic loan-queue prompt dataset."
    )
    parser.add_argument(
        "--batches", type=int, default=100,
        help="number of 10-row batches to generate (default: 100)",
    )
    parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES), default="naive",
        help="candidate-set construction strategy (default: naive)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frame = STRATEGIES[args.strategy](args.batches)
    frame = add_race_probs(frame)

    out_dir = DATA_DIR / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompt_file_{datetime.now().isoformat(timespec='milliseconds')}.csv"

    frame.write_csv(out_path)
    print(f"wrote {frame.height} rows to {out_path}")


if __name__ == "__main__":
    main()

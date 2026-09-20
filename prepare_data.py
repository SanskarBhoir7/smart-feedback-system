"""
One-time dataset preparation: split the seed data into a TRAINING POOL and a
FIXED HOLDOUT SET.

WHY THIS EXISTS
---------------
Model versions can only be compared fairly if they are measured on exactly
the same data. If the test set were regenerated every time the dataset grew,
then:

    * model v1's accuracy would be measured on one test set,
    * model v2's accuracy on a slightly different one,

and comparing the two numbers in the quality gate would be meaningless. Worse,
feedback that had been used to TRAIN a model could later turn up in the test
set and inflate its score - that is test-set contamination.

So the holdout set is created ONCE, written to data/holdout.csv, committed to
Git, and never changed again. Every model version - production and candidate
alike - is scored on those same rows.

    data/feedback_seed.csv   the original 360-row dataset (raw source)
              |
              |  stratified split, random_state=42  (this script, run once)
              |
       +------+------+
       |             |
  train_pool.csv   holdout.csv
   (288 rows)       (72 rows)   <-- FROZEN, never trained on
       |
       +  user feedback collected by the app
       |
    training data for every model version

New user feedback always becomes TRAINING data. It never enters the holdout
set, so the yardstick never moves.

Usage:
    python prepare_data.py            create the split (refuses to overwrite)
    python prepare_data.py --force    regenerate it (only if the seed changes)
    python prepare_data.py --verify   check the files match the seed data
"""

import argparse
import csv
import os
import sys

from sklearn.model_selection import train_test_split

import config


def read_seed_csv():
    """Read the raw seed dataset as a list of {feedback_text, rating} rows."""
    if not os.path.exists(config.SEED_CSV):
        raise FileNotFoundError("Seed dataset not found: {}".format(config.SEED_CSV))
    with open(config.SEED_CSV, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle)
                if row["feedback_text"].strip()]


def write_csv(path, rows):
    """Write rows back out in the same two-column format as the seed file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feedback_text", "rating"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"feedback_text": row["feedback_text"],
                             "rating": row["rating"]})


def build_split():
    """
    Perform the stratified split.

    Stratifying on the sentiment label keeps all three classes proportionally
    represented in both halves. random_state comes from config, so running
    this script again on the same seed file reproduces the identical split.
    """
    rows = read_seed_csv()
    labels = [config.rating_to_sentiment(row["rating"]) for row in rows]

    train_rows, holdout_rows = train_test_split(
        rows,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )
    return train_rows, holdout_rows


def summarise(name, rows):
    """Print the class balance of one split."""
    counts = {label: 0 for label in config.SENTIMENT_CLASSES}
    for row in rows:
        counts[config.rating_to_sentiment(row["rating"])] += 1
    detail = "  ".join("{}={}".format(label, counts[label])
                       for label in config.SENTIMENT_CLASSES)
    print("  {:14s} {:3d} rows   {}".format(name, len(rows), detail))


def prepare(force=False):
    """Create train_pool.csv and holdout.csv from the seed dataset."""
    already_exists = (os.path.exists(config.TRAIN_POOL_CSV)
                      and os.path.exists(config.HOLDOUT_CSV))

    if already_exists and not force:
        print("Training pool and holdout set already exist:")
        print("  {}".format(config.TRAIN_POOL_CSV))
        print("  {}".format(config.HOLDOUT_CSV))
        print("")
        print("They are deliberately NOT regenerated. The holdout set must")
        print("stay frozen so model versions remain comparable.")
        print("Use --force only if the seed dataset itself has changed.")
        return 0

    train_rows, holdout_rows = build_split()

    write_csv(config.TRAIN_POOL_CSV, train_rows)
    write_csv(config.HOLDOUT_CSV, holdout_rows)

    print("Dataset prepared from: {}".format(config.SEED_CSV))
    summarise("training pool", train_rows)
    summarise("holdout (fixed)", holdout_rows)
    print("")
    print("Written:")
    print("  {}".format(config.TRAIN_POOL_CSV))
    print("  {}".format(config.HOLDOUT_CSV))
    print("")
    print("The holdout set is now FROZEN. Commit it to Git and do not")
    print("regenerate it - every model version is scored against these rows.")
    return 0


def verify():
    """
    Confirm the two files are a clean, complete partition of the seed data:
    no overlap between them, and nothing lost.
    """
    if not (os.path.exists(config.TRAIN_POOL_CSV)
            and os.path.exists(config.HOLDOUT_CSV)):
        print("ERROR: split files missing. Run 'python prepare_data.py' first.")
        return 1

    seed = read_seed_csv()
    with open(config.TRAIN_POOL_CSV, newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    with open(config.HOLDOUT_CSV, newline="", encoding="utf-8") as handle:
        holdout_rows = list(csv.DictReader(handle))

    seed_texts = set(r["feedback_text"] for r in seed)
    train_texts = set(r["feedback_text"] for r in train_rows)
    holdout_texts = set(r["feedback_text"] for r in holdout_rows)

    overlap = train_texts & holdout_texts
    missing = seed_texts - (train_texts | holdout_texts)

    print("Seed rows          : {}".format(len(seed)))
    print("Training pool rows : {}".format(len(train_rows)))
    print("Holdout rows       : {}".format(len(holdout_rows)))
    print("Sum matches seed   : {}".format(
        len(train_rows) + len(holdout_rows) == len(seed)))
    print("Train/holdout overlap : {} (must be 0)".format(len(overlap)))
    print("Seed rows unaccounted : {} (must be 0)".format(len(missing)))

    summarise("training pool", train_rows)
    summarise("holdout (fixed)", holdout_rows)

    if overlap or missing:
        print("\nVERIFICATION FAILED - the split is not a clean partition.")
        return 1
    print("\nVERIFICATION PASSED - holdout set is clean and uncontaminated.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Split the seed dataset into a training pool and a fixed holdout set.")
    parser.add_argument("--force", action="store_true",
                        help="regenerate the split even if it already exists")
    parser.add_argument("--verify", action="store_true",
                        help="check the existing split is a clean partition")
    args = parser.parse_args()

    if args.verify:
        return verify()
    return prepare(force=args.force)


if __name__ == "__main__":
    sys.exit(main())

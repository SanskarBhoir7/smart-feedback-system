"""
Scenario C helper: inject deliberately mislabelled feedback to make the
quality gate fail, then undo it.

WHY THIS EXISTS
---------------
The point of a quality gate is that it protects production from a bad model.
A demonstration is only convincing if the failure can actually be produced on
demand - and produced for a REALISTIC reason.

So instead of faking a low number, this script injects genuinely corrupt
training data: strongly positive sentences rated 1 star, and strongly
negative sentences rated 5 stars. That is what bad or malicious feedback
looks like in the real world, and it degrades the next candidate model
exactly the way real data poisoning would.

CALIBRATION (measured, not guessed)
-----------------------------------
    bad rows     candidate accuracy     gate at 0.85
        0              0.9167               pass
       20              0.8750               pass
       30              0.8611               pass
       40              0.8194               FAIL
       50              0.7778               FAIL

The default of 50 rows lands around 0.78, comfortably below the 0.85 gate,
so the demonstration is reliable rather than marginal.

Every injected row is tagged with source='bad_seed', so --undo can remove
them all cleanly and leave genuine user feedback untouched.

Usage:
    python seed_bad_feedback.py              inject 50 mislabelled rows
    python seed_bad_feedback.py --count 40   inject a specific number
    python seed_bad_feedback.py --undo       remove every injected row
    python seed_bad_feedback.py --status     show how many are present
"""

import argparse
import sys

import config
import database


BAD_SOURCE = "bad_seed"

# Clearly positive text, deliberately given a 1-star rating.
POSITIVE_TEXT_RATED_BAD = [
    "excellent service and very helpful staff",
    "great quality and fast delivery",
    "the app is fast and reliable",
    "wonderful support, very friendly",
    "perfect quality, exactly as described",
    "I love this app, easy to use",
    "brilliant design and smooth performance",
    "outstanding support and great service",
    "fantastic value and a great price",
    "highly recommend, excellent quality",
    "superb performance and clean design",
    "very satisfied with the fast service",
    "the best app, reliable and quick",
    "easy setup and smooth performance",
    "friendly staff and excellent quality",
]

# Clearly negative text, deliberately given a 5-star rating.
NEGATIVE_TEXT_RATED_GOOD = [
    "terrible service and very rude staff",
    "awful quality and slow delivery",
    "the app is slow and crashes constantly",
    "horrible support, very unhelpful",
    "poor quality, completely broken",
    "useless app, buggy and frustrating",
    "bad design and terrible performance",
    "worst support and worst service",
    "a waste of money at a high price",
    "would not recommend, awful quality",
    "disappointing and expensive",
    "the checkout fails and support is rude",
    "broken features and poor documentation",
    "very late delivery and damaged item",
    "unhelpful staff and slow responses",
]


def inject(count):
    """
    Add `count` mislabelled rows, split evenly between the two kinds of
    corruption so the decision boundary is damaged from both directions.
    """
    database.init_db()

    half = count // 2
    remainder = count - half

    rows = []
    for index in range(half):
        text = POSITIVE_TEXT_RATED_BAD[index % len(POSITIVE_TEXT_RATED_BAD)]
        rows.append((text, 1))          # positive text, 1 star  -> 'negative'
    for index in range(remainder):
        text = NEGATIVE_TEXT_RATED_GOOD[index % len(NEGATIVE_TEXT_RATED_GOOD)]
        rows.append((text, 5))          # negative text, 5 stars -> 'positive'

    for text, rating in rows:
        database.insert_feedback(
            feedback_text=text,
            rating=rating,
            predicted_sentiment=None,
            name="bad-data-demo",
            model_version=None,
            source=BAD_SOURCE,
        )

    print("=" * 58)
    print(" SCENARIO C - BAD DATA INJECTED")
    print("=" * 58)
    print("  Rows injected        : {}".format(count))
    print("    positive text rated 1 star : {}".format(half))
    print("    negative text rated 5 stars: {}".format(remainder))
    print("  Tagged with source   : '{}'".format(BAD_SOURCE))
    print("  New samples pending  : {}".format(database.count_new_samples()))
    print("  Retrain threshold    : {}".format(config.RETRAIN_THRESHOLD))
    print("")
    print("  Run the Jenkins pipeline (or retrain.py) now. The candidate")
    print("  model should fail the quality gate and production should")
    print("  remain unchanged.")
    print("")
    print("  Undo with: python seed_bad_feedback.py --undo")
    print("=" * 58)
    return 0


def undo():
    """Remove every injected row, leaving genuine feedback untouched."""
    database.init_db()
    removed = database.delete_feedback_by_source(BAD_SOURCE)

    print("=" * 58)
    print(" SCENARIO C - BAD DATA REMOVED")
    print("=" * 58)
    print("  Rows removed         : {}".format(removed))
    print("  New samples pending  : {}".format(database.count_new_samples()))
    print("")
    print("  Genuine user feedback was not touched.")
    print("=" * 58)
    return 0


def status():
    """Report how many injected rows are currently present."""
    database.init_db()
    count = database.count_by_source(BAD_SOURCE)

    print("Injected bad rows present : {}".format(count))
    print("Total new samples pending : {}".format(database.count_new_samples()))
    print("Retraining threshold      : {}".format(config.RETRAIN_THRESHOLD))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inject or remove deliberately mislabelled feedback (Scenario C).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--undo", action="store_true",
                       help="remove every injected row")
    group.add_argument("--status", action="store_true",
                       help="show how many injected rows are present")
    parser.add_argument("--count", type=int, default=50,
                        help="how many rows to inject (default 50)")
    args = parser.parse_args()

    if args.undo:
        return undo()
    if args.status:
        return status()
    if args.count < 2:
        print("ERROR: --count must be at least 2.")
        return 1
    return inject(args.count)


if __name__ == "__main__":
    sys.exit(main())

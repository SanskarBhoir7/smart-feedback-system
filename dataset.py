"""
Dataset assembly and model definition.

train.py, retrain.py and evaluate.py all build their data through this single
module, so every model version is trained and measured identically.

THE TWO DATA POOLS
------------------
    TRAINING POOL    data/train_pool.csv  +  all collected user feedback
                     Grows over time. This is the only data a model ever
                     learns from.

    FIXED HOLDOUT    data/holdout.csv
                     72 rows, created once by prepare_data.py and then
                     FROZEN. Never trained on, never added to.

Every model - the production model and every candidate - is scored on that
same frozen holdout set. That is what makes "candidate accuracy 0.93 vs
production accuracy 0.92" a valid comparison rather than two numbers measured
on different data.

It also prevents test-set contamination: user feedback always becomes
TRAINING data, so a sample the model learned from can never reappear in the
test set and inflate its score.
"""

import csv
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

import config
import database


# ---------------------------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------------------------

def clean_text(text):
    """
    Light, predictable text cleaning.

    Lower-casing is left to the vectorizer. Here we only strip surrounding
    whitespace and collapse repeated spaces, so that "great    service" and
    "great service" produce the same features.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


# ---------------------------------------------------------------------------
# LOADING DATA
# ---------------------------------------------------------------------------

def _load_csv_rows(path, description):
    """
    Read a two-column feedback CSV into (text, label) pairs.

    The label is derived from the star rating using exactly the same rule
    that is applied to live user feedback.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{} not found at {}\nRun 'python prepare_data.py' first.".format(
                description, path)
        )

    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            text = clean_text(record["feedback_text"])
            if not text:
                continue
            rows.append((text, config.rating_to_sentiment(record["rating"])))
    return rows


def load_train_pool_rows():
    """The committed training pool - the seed data minus the holdout rows."""
    return _load_csv_rows(config.TRAIN_POOL_CSV, "Training pool")


def load_holdout_rows():
    """
    The FIXED holdout set.

    These 72 rows are never trained on and never change, so they form a
    stable yardstick for comparing one model version against another.
    """
    return _load_csv_rows(config.HOLDOUT_CSV, "Holdout set")


def load_feedback_rows():
    """
    Read every piece of feedback collected through the web application.

    This is the part that makes the project a feedback-driven MLOps system:
    ordinary user submissions become labelled training data automatically.
    """
    rows = []
    for text, label in database.get_training_rows():
        text = clean_text(text)
        if text:
            rows.append((text, label))
    return rows


def load_training_data(include_feedback=True):
    """
    Build the data a model will LEARN from: the committed training pool plus
    every piece of feedback collected through the web application.

    Returns (texts, labels, summary). The summary is printed in the logs so
    it is always obvious what a given model version was trained on.

    The holdout set is deliberately absent - nothing here ever touches it.
    """
    pool_rows = load_train_pool_rows()
    feedback_rows = load_feedback_rows() if include_feedback else []

    combined = pool_rows + feedback_rows
    texts = [row[0] for row in combined]
    labels = [row[1] for row in combined]

    summary = {
        "pool_samples": len(pool_rows),
        "feedback_samples": len(feedback_rows),
        "total_samples": len(combined),
    }
    return texts, labels, summary


def load_holdout_data():
    """
    The fixed evaluation set, as (texts, labels).

    Every model version is scored on exactly these rows.
    """
    rows = load_holdout_rows()
    return [row[0] for row in rows], [row[1] for row in rows]


# ---------------------------------------------------------------------------
# THE MODEL
# ---------------------------------------------------------------------------

def build_pipeline():
    """
    TF-IDF features followed by Logistic Regression.

    Wrapping both steps in one sklearn Pipeline means joblib saves the
    vectorizer and the classifier together as a SINGLE .pkl artifact. That
    matches how the practicals treat model.pkl, and removes any chance of the
    saved vocabulary getting out of step with the saved model.

    The settings below were CHOSEN BY EXPERIMENT, not by assumption. Eight
    combinations were compared with 5-fold cross-validation on the TRAINING
    set only (never the test set, which would inflate the reported score):

        unigram  min_df=2  C=5     0.9202   <-- chosen
        unigram  min_df=1  C=1     0.9168
        1-2gram  min_df=1  C=10    0.9063
        1-2gram  min_df=2  C=10    0.8924
        1-2gram  min_df=1  C=1     0.8855

    Word pairs ("not bad") were expected to help, but they measurably hurt at
    this dataset size: bigrams trigger far more features than 288 training
    samples can support, so the model overfits. Single words won.

    min_df=2 discards words that appear only once in the whole corpus. Those
    words cannot generalise to unseen feedback, so dropping them reduces
    noise.

    class_weight='balanced' keeps the three classes weighted equally even if
    real feedback arrives lopsided (in practice most users rate 4-5).
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 1),      # single words only - see note above
            min_df=2,                # ignore words that appear only once
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=5.0,                   # milder regularisation, chosen by CV
            random_state=config.RANDOM_STATE,
        )),
    ])


if __name__ == "__main__":
    database.init_db()
    X_train, y_train, summary = load_training_data()
    X_holdout, y_holdout = load_holdout_data()

    print("TRAINING DATA (grows with feedback)")
    print("  pool samples     : {}".format(summary["pool_samples"]))
    print("  feedback samples : {}".format(summary["feedback_samples"]))
    print("  total            : {}".format(summary["total_samples"]))
    print("HOLDOUT DATA (frozen, never trained on)")
    print("  total            : {}".format(len(X_holdout)))
    print("")
    for label in config.SENTIMENT_CLASSES:
        print("  {:9s} train={:4d}  holdout={:3d}".format(
            label, y_train.count(label), y_holdout.count(label)))

    # Safety check: no training text may appear in the holdout set.
    leaked = set(X_train) & set(X_holdout)
    print("")
    print("  contamination check: {} overlapping rows (must be 0)".format(len(leaked)))

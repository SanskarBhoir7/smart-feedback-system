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
from sklearn.pipeline import FeatureUnion, Pipeline

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


def load_feedback_records():
    """
    Every piece of feedback collected through the web application, as dicts
    with id, feedback_text and actual_sentiment, text cleaned.

    This is the part that makes the project a feedback-driven MLOps system:
    ordinary user submissions become labelled training data automatically.
    """
    records = []
    for record in database.get_feedback_records():
        text = clean_text(record["feedback_text"])
        if text:
            records.append(dict(record, feedback_text=text))
    return records


def load_feedback_rows():
    """All collected feedback as (text, label) pairs, unfiltered."""
    return [(r["feedback_text"], r["actual_sentiment"]) for r in load_feedback_records()]


def load_training_data(include_feedback=True, judge_model=None,
                       exclude_inconsistent=None):
    """
    Build the data a model will LEARN from: the committed training pool plus
    the feedback collected through the web application.

    Feedback whose rating contradicts its own text is a wrong training label.
    When exclude_inconsistent is on (config.EXCLUDE_INCONSISTENT_FEEDBACK by
    default), data_quality.py identifies those rows and they are left out.
    judge_model is the model that reads each text for that check - normally
    the current production model. Without one, only the model-free
    label-conflict check runs.

    Returns (texts, labels, summary). The summary records exactly what went
    in and what was excluded and why, so every model version is auditable.

    The holdout set is deliberately absent - nothing here ever touches it.
    """
    import data_quality

    if exclude_inconsistent is None:
        exclude_inconsistent = config.EXCLUDE_INCONSISTENT_FEEDBACK

    pool_rows = load_train_pool_rows()
    records = load_feedback_records() if include_feedback else []

    flags = data_quality.assess(records, judge_model) if records else {}
    excluded_ids = set(flags) if exclude_inconsistent else set()
    kept = [r for r in records if r["id"] not in excluded_ids]

    feedback_rows = [(r["feedback_text"], r["actual_sentiment"]) for r in kept]
    combined = pool_rows + feedback_rows

    summary = {
        "pool_samples": len(pool_rows),
        "feedback_samples": len(feedback_rows),
        "total_samples": len(combined),
        "feedback_collected": len(records),
        "feedback_excluded": len(excluded_ids),
        "excluded_ids": sorted(excluded_ids),
        # Per-row reasons, so an accepted run can record its decisions.
        # String keys, because this summary is written to JSON.
        "excluded_reasons": {str(i): flags[i] for i in sorted(excluded_ids)},
        "previously_excluded": sum(1 for r in records if r.get("quality_flag")),
        "quality_flags": data_quality.summarise(flags),
        "exclude_inconsistent": exclude_inconsistent,
        "judged_by_model": judge_model is not None,
    }
    return [row[0] for row in combined], [row[1] for row in combined], summary


def print_training_summary(summary, indent="      "):
    """Log what a model is being trained on, including any exclusions."""
    print("{}Pool samples     : {}".format(indent, summary["pool_samples"]))
    print("{}Feedback samples : {} used of {} collected".format(
        indent, summary["feedback_samples"], summary.get("feedback_collected", 0)))
    flags = summary.get("quality_flags", {})
    if summary.get("feedback_excluded"):
        print("{}Excluded as inconsistent: {} row(s) {}".format(
            indent, summary["feedback_excluded"], summary["excluded_ids"]))
        print("{}   model disagreement : {}".format(indent, flags.get("model_disagreement", 0)))
        print("{}   label conflict     : {}".format(indent, flags.get("label_conflict", 0)))
        if summary.get("previously_excluded"):
            print("{}   of which recorded by an earlier accepted run: {}".format(
                indent, summary["previously_excluded"]))
    elif flags.get("flagged_rows"):
        print("{}Flagged as inconsistent but KEPT: {} (exclusion is switched off)".format(
            indent, flags["flagged_rows"]))
    else:
        print("{}Inconsistent rows : none found".format(indent))
    if not summary.get("judged_by_model", True):
        print("{}(no production model yet, so only the label-conflict check ran)".format(indent))
    print("{}Total training   : {}".format(indent, summary["total_samples"]))


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

def build_tfidf():
    """
    The lexical half: TF-IDF over single words.

    Settings chosen by 5-fold cross-validation on the training set only:

        unigram  min_df=2  C=5     0.9202   <-- chosen
        unigram  min_df=1  C=1     0.9168
        1-2gram  min_df=1  C=10    0.9063
        1-2gram  min_df=2  C=10    0.8924

    Word pairs ("not bad") were expected to help but measurably hurt: at this
    dataset size they create far more features than the data supports.
    min_df=2 drops words seen only once, which cannot generalise anyway.
    """
    return TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def build_pipeline():
    """
    The production model: lexical AND semantic features, one classifier.

        text --+--> TF-IDF (which words appear)            -> sparse vector --+
               |                                                              +--> Logistic
               +--> MiniLM sentence embedding (frozen,    -> 384 numbers    --+    Regression
                    pretrained semantic representation)

    WHY BOTH, NOT JUST EMBEDDINGS
    Measured on the identical frozen 72-row holdout, each candidate selected
    by 5-fold cross-validation on the training pool only:

                               CV acc   holdout acc   negation   unseen-wording
        TF-IDF only            0.9202     0.9167        6/12        6/11
        embeddings only        0.9304     0.8472        8/12        8/11
        TF-IDF + embeddings    0.9411     0.9583        8/12        9/11   <-- chosen

    Embeddings alone generalise better to unfamiliar wording but LOST 7
    points on the holdout and would be rejected by our own quality gate.
    TF-IDF alone matches words exactly but has nothing to say about a word it
    never saw. Side by side, TF-IDF keeps precision on familiar wording and
    the embeddings add reach to unfamiliar wording.

    WHAT IT DOES NOT FIX
    Sentence embeddings represent negation only weakly: "not good" moves from
    POSITIVE (TF-IDF) to NEUTRAL, but not to negative. It also reads bare
    single words ("Good") less confidently than TF-IDF did.

    ARTIFACT
    joblib saves the whole pipeline as ONE small model_vN.pkl: the TF-IDF
    vocabulary, the classifier weights, and the encoder's fingerprint - but
    NOT the 91 MB encoder weights, which live once in the encoder artifact
    folder (see embeddings.py).

    C=5 was chosen by the same cross-validation. class_weight='balanced'
    keeps the three classes weighted equally when real feedback arrives
    lopsided.
    """
    from embeddings import SentenceEmbedder

    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", build_tfidf()),
            ("embedding", SentenceEmbedder()),
        ])),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=5.0,
            random_state=config.RANDOM_STATE,
        )),
    ])


def describe_model(model):
    """
    A one-line description of a fitted model's features, for training logs.

    Works for the current hybrid and for older TF-IDF-only versions, so the
    history of archived models can still be inspected.
    """
    steps = model.named_steps
    if "features" in steps:
        parts = dict(steps["features"].transformer_list)
        vocabulary = len(parts["tfidf"].vocabulary_)
        dimension = getattr(parts["embedding"], "dimension_", "?")
        return "hybrid: {} TF-IDF terms + {}-dim sentence embedding".format(
            vocabulary, dimension)
    if "tfidf" in steps:
        return "TF-IDF only: {} terms".format(len(steps["tfidf"].vocabulary_))
    return "unknown model layout"


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

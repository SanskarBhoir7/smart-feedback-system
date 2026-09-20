"""
SQLite access layer for the Smart Feedback System.

This module deliberately does NOT import Flask. retrain.py and train.py run
inside Jenkins with no web server running, and they still need to read the
feedback table and write model version records.

Two tables:

  feedback        every submission, with the prediction the model made and the
                  true label derived from the star rating
  model_versions  the history of every trained model, and which one is
                  currently in production

The model_versions table is the SINGLE SOURCE OF TRUTH for which model is
live. There is no separate registry file that could drift out of sync.

CONNECTION HANDLING
-------------------
Every function opens its connection through the _connect() context manager,
which closes it in a finally block. That matters: if a statement raises (for
example a UNIQUE constraint violation on a duplicate version number), a
connection closed on the following line would never be closed at all. On
Windows the leaked handle keeps a lock on the database file, so the next
process to touch it fails with "used by another process".
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config


# ---------------------------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------------------------

def get_connection():
    """
    Open a connection to the shared SQLite database.

    row_factory is set so that rows behave like dictionaries
    (row["rating"]), which keeps the rest of the code readable.

    Callers are responsible for closing this. Prefer _connect() instead.
    """
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connect():
    """
    Open a connection and guarantee it is closed, even if a query raises.

    Use `with _connect() as connection:` for reads, and add an inner
    `with connection:` for writes, which wraps them in a transaction.
    """
    connection = get_connection()
    try:
        yield connection
    finally:
        connection.close()


def _now():
    """Current timestamp as a readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------------------------

FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT,
    feedback_text       TEXT    NOT NULL,
    rating              INTEGER NOT NULL,
    predicted_sentiment TEXT,
    actual_sentiment    TEXT    NOT NULL,
    prediction_correct  INTEGER,
    model_version       INTEGER,
    source              TEXT    NOT NULL DEFAULT 'user',
    used_in_training    INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL
);
"""

MODEL_VERSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version         INTEGER NOT NULL UNIQUE,
    model_path      TEXT    NOT NULL,
    accuracy        REAL,
    precision       REAL,
    recall          REAL,
    f1_score        REAL,
    training_samples INTEGER,
    status          TEXT    NOT NULL,
    notes           TEXT,
    created_at      TEXT    NOT NULL
);
"""


def init_db():
    """Create both tables if they do not already exist. Safe to call anytime."""
    config.ensure_directories()
    with _connect() as connection:
        with connection:
            connection.execute(FEEDBACK_SCHEMA)
            connection.execute(MODEL_VERSIONS_SCHEMA)


# ---------------------------------------------------------------------------
# FEEDBACK - WRITING
# ---------------------------------------------------------------------------

def insert_feedback(feedback_text, rating, predicted_sentiment=None,
                    name=None, model_version=None, source="user"):
    """
    Store one piece of feedback and return its new row id.

    The true label is derived from the star rating, not from the model. That
    is what makes every submission a usable TRAINING SAMPLE.
    """
    actual_sentiment = config.rating_to_sentiment(rating)

    # prediction_correct stays NULL when no model was available to predict.
    if predicted_sentiment is None:
        prediction_correct = None
    else:
        prediction_correct = 1 if predicted_sentiment == actual_sentiment else 0

    with _connect() as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback (
                    name, feedback_text, rating, predicted_sentiment,
                    actual_sentiment, prediction_correct, model_version,
                    source, used_in_training, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (name, feedback_text, int(rating), predicted_sentiment,
                 actual_sentiment, prediction_correct, model_version,
                 source, _now()),
            )
            return cursor.lastrowid


def mark_all_samples_used():
    """
    Flag every feedback row as having been used for training.

    Called ONLY when a candidate model passes the quality gate. If the gate
    rejects a candidate, the samples stay unused so the next pipeline run
    tries again with them - a rejection must never silently consume data.
    """
    with _connect() as connection:
        with connection:
            cursor = connection.execute(
                "UPDATE feedback SET used_in_training = 1 "
                "WHERE used_in_training = 0")
            return cursor.rowcount


def delete_feedback_by_source(source):
    """Remove all rows with a given source. Used to undo the bad-data demo."""
    with _connect() as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM feedback WHERE source = ?", (source,))
            return cursor.rowcount


def get_max_feedback_id():
    """
    The highest feedback id currently stored, or 0 when the table is empty.

    Used by the Selenium suite to take a "before" marker so that everything
    it submits through the browser can be removed afterwards.
    """
    with _connect() as connection:
        row = connection.execute(
            "SELECT MAX(id) AS m FROM feedback").fetchone()
        return row["m"] or 0


def delete_feedback_after_id(row_id):
    """
    Remove every feedback row newer than the given id.

    The Selenium tests submit real feedback through the real UI - that is the
    point of them - but they must not leave that data behind in the
    development database. Recording the highest id beforehand and deleting
    anything above it afterwards restores the exact previous state without
    touching genuine rows.
    """
    with _connect() as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM feedback WHERE id > ?", (row_id,))
            return cursor.rowcount


# ---------------------------------------------------------------------------
# FEEDBACK - READING
# ---------------------------------------------------------------------------

def count_new_samples():
    """
    How many feedback samples have been collected since the last ACCEPTED
    training run. This single number drives the auto-retraining trigger.
    """
    with _connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE used_in_training = 0"
        ).fetchone()
        return row["n"]


def count_by_source(source):
    """How many rows came from a given source, e.g. the bad-data demo."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE source = ?", (source,)
        ).fetchone()
        return row["n"]


def get_training_rows():
    """
    Every feedback row usable as training data, as (text, label) pairs.

    Includes rows already used before, because retraining always trains on the
    full history - seed dataset plus all collected feedback.
    """
    with _connect() as connection:
        rows = connection.execute(
            "SELECT feedback_text, actual_sentiment FROM feedback").fetchall()
        return [(r["feedback_text"], r["actual_sentiment"]) for r in rows]


def get_recent_feedback(limit=10):
    """The most recent submissions, newest first, for the dashboard table."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_feedback_stats():
    """
    Aggregate numbers for the dashboard.

    live_accuracy is the model's real-world hit rate: how often its prediction
    matched the label derived from the user's own star rating.
    """
    with _connect() as connection:
        total = connection.execute(
            "SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]

        counts = {config.NEGATIVE: 0, config.NEUTRAL: 0, config.POSITIVE: 0}
        for row in connection.execute(
                "SELECT actual_sentiment, COUNT(*) AS n FROM feedback "
                "GROUP BY actual_sentiment"):
            if row["actual_sentiment"] in counts:
                counts[row["actual_sentiment"]] = row["n"]

        average_rating = connection.execute(
            "SELECT AVG(rating) AS a FROM feedback").fetchone()["a"]

        scored = connection.execute(
            "SELECT COUNT(*) AS n FROM feedback "
            "WHERE prediction_correct IS NOT NULL").fetchone()["n"]
        correct = connection.execute(
            "SELECT COUNT(*) AS n FROM feedback "
            "WHERE prediction_correct = 1").fetchone()["n"]

        pending = connection.execute(
            "SELECT COUNT(*) AS n FROM feedback "
            "WHERE used_in_training = 0").fetchone()["n"]

    return {
        "total_feedback": total,
        "positive_count": counts[config.POSITIVE],
        "neutral_count": counts[config.NEUTRAL],
        "negative_count": counts[config.NEGATIVE],
        "average_rating": round(average_rating, 2) if average_rating else 0.0,
        "predictions_scored": scored,
        "predictions_correct": correct,
        "live_accuracy": round(correct / scored, 4) if scored else None,
        "new_samples": pending,
        "retrain_threshold": config.RETRAIN_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# MODEL VERSIONS
# ---------------------------------------------------------------------------

def get_next_version():
    """The next version number to assign. Versions never get reused."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT MAX(version) AS v FROM model_versions").fetchone()
        return (row["v"] or 0) + 1


def insert_model_version(version, model_path, metrics, training_samples,
                         status, notes=""):
    """
    Record one trained model.

    Accepted models are stored with status 'production'; rejected candidates
    are stored too, with status 'rejected', so the Model page can SHOW that a
    rejection happened rather than the event vanishing into a log file.
    """
    with _connect() as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO model_versions (
                    version, model_path, accuracy, precision, recall,
                    f1_score, training_samples, status, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (version, model_path, metrics.get("accuracy"),
                 metrics.get("precision"), metrics.get("recall"),
                 metrics.get("f1_score"), training_samples, status, notes,
                 _now()),
            )


def promote_to_production(version):
    """
    Make one version live.

    Any model already in production is demoted to 'archived' first, so the
    table can never contain two production models at once. Rejected versions
    are left alone - a rejection is a permanent record, not a former
    production model.
    """
    with _connect() as connection:
        with connection:
            connection.execute(
                "UPDATE model_versions SET status = ? WHERE status = ?",
                (config.STATUS_ARCHIVED, config.STATUS_PRODUCTION))
            connection.execute(
                "UPDATE model_versions SET status = ? WHERE version = ?",
                (config.STATUS_PRODUCTION, version))


def get_production_model():
    """The live model's record, or None if nothing has been deployed yet."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM model_versions WHERE status = ? "
            "ORDER BY version DESC LIMIT 1",
            (config.STATUS_PRODUCTION,)).fetchone()
        return dict(row) if row else None


def get_previous_model():
    """The most recently archived model - shown as 'previous version' in the UI."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM model_versions WHERE status = ? "
            "ORDER BY version DESC LIMIT 1",
            (config.STATUS_ARCHIVED,)).fetchone()
        return dict(row) if row else None


def get_model_by_version(version):
    """Look up a single model version record."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM model_versions WHERE version = ?",
            (version,)).fetchone()
        return dict(row) if row else None


def get_model_history(limit=20):
    """Full model history, newest first, for the Model / MLOps page."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM model_versions ORDER BY version DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    # Running this file directly creates the database - a convenient setup step.
    init_db()
    print("Database initialised at: {}".format(config.DB_PATH))
    print("Feedback rows          : {}".format(get_feedback_stats()["total_feedback"]))
    print("Model versions         : {}".format(len(get_model_history())))

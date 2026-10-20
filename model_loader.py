"""
Loads the production model and serves predictions.

WHY THIS EXISTS
---------------
Three different routes need to make a prediction (/, /feedback, /predict).
Without this module each of them would have to work out which model is live,
find the file and call joblib.load - the same logic copied three times.

This module does it once and caches the result.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It never trains anything. The web application is a CONSUMER of models, not a
producer of them. Training and retraining happen only in train.py and
retrain.py, driven by Jenkins. That separation is the whole point of the
MLOps pipeline: the running application cannot quietly change the model that
is serving users.

WHICH MODEL GETS LOADED
-----------------------
Exactly the one row in model_versions whose status is 'production' - nothing
else. Archived and rejected versions are never loaded, even though their
.pkl files still exist in the persistent store.

The file itself is found through config.MODEL_SEARCH_PATH, which looks in the
workspace models/ folder first (this is /app/models inside the container,
where the baked-in model lives) and then falls back to the persistent store
at C:\\mlops-data\\models (used during local development).
"""

import os
import threading

import joblib

import config
import database


# The cache. _lock keeps it safe if Flask serves two requests at once.
_lock = threading.Lock()
_cached_model = None
_cached_version = None
_last_error = None


def _load_production_model():
    """
    Read which version is in production, find its file and load it.

    Returns (model, version, error_message). Any failure returns
    (None, None, "reason") rather than raising, so that a missing model
    degrades the app gracefully instead of taking it down - /health must
    still answer even when no model is available.
    """
    record = database.get_production_model()
    if not record:
        return None, None, "No production model is registered in the database."

    version = record["version"]
    filename = config.model_basename(record["model_path"])
    path = config.resolve_model_path(filename)

    if not path:
        return None, version, (
            "Production model v{} is registered but its file '{}' was not "
            "found in {}".format(version, filename, config.MODEL_SEARCH_PATH))

    try:
        model = joblib.load(path)
        # One throwaway prediction, for two reasons. It loads the sentence
        # encoder now (about 1.5 s) instead of on the first user's request,
        # and it runs the encoder fingerprint check - so a model paired with
        # the wrong encoder fails HERE, visibly, rather than serving
        # predictions computed from mismatched features.
        model.predict_proba(["model warm-up"])
    except Exception as exception:          # noqa: BLE001 - report, never crash
        return None, version, "Failed to load {}: {}".format(path, exception)

    return model, version, None


def get_model(force_reload=False):
    """
    Return (model, version) for the current production model.

    The model is cached, but the database is checked on every call so that a
    newly promoted version is picked up without restarting the application.
    That query is a single indexed lookup, so the cost is negligible.
    """
    global _cached_model, _cached_version, _last_error

    with _lock:
        record = database.get_production_model()
        wanted_version = record["version"] if record else None

        needs_load = (
            force_reload
            or _cached_model is None
            or _cached_version != wanted_version
        )

        if needs_load:
            _cached_model, _cached_version, _last_error = _load_production_model()

        return _cached_model, _cached_version


def is_ready():
    """True when a model is loaded and able to serve predictions."""
    model, _ = get_model()
    return model is not None


def get_last_error():
    """The reason the model could not be loaded, or None."""
    get_model()
    return _last_error


def predict_detailed(text):
    """
    Predict the sentiment of one piece of feedback, with how sure the model is.

    Returns a dict:

        sentiment        "positive" | "neutral" | "negative", or None
        model_version    the version that answered, or None
        confidence       the top class probability, 0-1
        probabilities    {"negative": p, "neutral": p, "positive": p}
        low_confidence   True when confidence < config.LOW_CONFIDENCE_THRESHOLD
        outcome          the prediction, or "uncertain" below the threshold
        abstained        whether the user-facing outcome abstains

    sentiment preserves the classifier's best guess for diagnostics and
    compatibility. outcome is "uncertain" below the calibrated confidence
    threshold, and abstained makes that user-facing decision explicit. A low
    probability is a warning signal, not proof that a prediction is wrong.

    With no model available every field is None, and feedback is still stored
    so it can be used for training later.
    """
    empty = {"sentiment": None, "model_version": None, "confidence": None,
             "probabilities": None, "low_confidence": None,
             "outcome": None, "abstained": None}

    model, version = get_model()
    if model is None:
        return empty

    cleaned = str(text).strip()
    if not cleaned:
        return dict(empty, model_version=version)

    row = model.predict_proba([cleaned])[0]
    classes = [str(c) for c in model.classes_]
    probabilities = {c: round(float(p), 4) for c, p in zip(classes, row)}
    sentiment = max(probabilities, key=probabilities.get)
    confidence = probabilities[sentiment]

    low_confidence = confidence < config.LOW_CONFIDENCE_THRESHOLD
    return {
        "sentiment": sentiment,
        "model_version": version,
        "confidence": confidence,
        "probabilities": probabilities,
        "low_confidence": low_confidence,
        "outcome": "uncertain" if low_confidence else sentiment,
        "abstained": low_confidence,
    }


def predict(text):
    """
    Predict the sentiment of a single piece of feedback.

    Returns (sentiment, model_version), or (None, None) with no model. Kept
    as the simple form of predict_detailed().
    """
    result = predict_detailed(text)
    return result["sentiment"], result["model_version"]


def model_family(model):
    """'hybrid' for TF-IDF + embeddings, 'tfidf' for the original layout."""
    if model is None:
        return None
    steps = getattr(model, "named_steps", {})
    if "features" in steps:
        return "hybrid"
    if "tfidf" in steps:
        return "tfidf"
    return "unknown"


def describe():
    """Status summary used by /health and the dashboard."""
    model, version = get_model()
    return {
        "model_loaded": model is not None,
        "model_version": version,
        "model_family": model_family(model),
        "error": _last_error,
    }

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


def predict(text):
    """
    Predict the sentiment of a single piece of feedback.

    Returns (sentiment, model_version). If no model is available, returns
    (None, None) - callers decide how to present that, and feedback is still
    stored so it can be used for training later.
    """
    model, version = get_model()
    if model is None:
        return None, None

    cleaned = str(text).strip()
    if not cleaned:
        return None, version

    prediction = model.predict([cleaned])[0]
    return str(prediction), version


def describe():
    """Status summary used by /health and the dashboard."""
    model, version = get_model()
    return {
        "model_loaded": model is not None,
        "model_version": version,
        "error": _last_error,
    }

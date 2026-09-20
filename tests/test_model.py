"""
Model tests.

The first two tests are the canonical pair from the PS4 CI/CD Pipeline /
ML CI-CD practical - "does the artifact exist" and "can it produce a
prediction" - kept deliberately recognisable, then extended to cover all
three sentiment classes.
"""

import os

import joblib
import pytest

import config
import database
import model_loader


# ---------------------------------------------------------------------------
# ARTIFACT EXISTS  (the practical's test_model_artifact_exists)
# ---------------------------------------------------------------------------

def test_production_model_is_registered(production_model):
    """A model version must be recorded in the database as production."""
    assert production_model is not None
    assert production_model["status"] == config.STATUS_PRODUCTION
    assert production_model["version"] >= 1


def test_production_model_file_exists(production_model):
    """The .pkl artifact named by that record must exist on disk."""
    path = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))
    assert path is not None, "Production model file was not found on disk"
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# MODEL LOADS  (the practical's test_model_inference)
# ---------------------------------------------------------------------------

def test_model_loads_successfully(production_model):
    """joblib must be able to load the artifact back into a usable object."""
    path = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))
    model = joblib.load(path)
    assert hasattr(model, "predict")


def test_model_generates_a_prediction(production_model):
    """The loaded model must return exactly one label for one input."""
    path = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))
    model = joblib.load(path)
    prediction = model.predict(["the service was very good"])
    assert len(prediction) == 1
    assert prediction[0] in config.SENTIMENT_CLASSES


def test_model_loader_reports_ready(production_model):
    """The shared loader must report the model as ready to serve."""
    assert model_loader.is_ready() is True
    state = model_loader.describe()
    assert state["model_loaded"] is True
    assert state["model_version"] == production_model["version"]
    assert state["error"] is None


# ---------------------------------------------------------------------------
# PREDICTION CORRECTNESS - one test per class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "excellent service and very helpful staff",
    "great quality and fast delivery",
    "the app is fast and reliable",
])
def test_positive_prediction(production_model, text):
    """Clearly positive feedback must be classified positive."""
    sentiment, version = model_loader.predict(text)
    assert sentiment == config.POSITIVE
    assert version == production_model["version"]


@pytest.mark.parametrize("text", [
    "terrible service and very rude staff",
    "the app is slow and crashes constantly",
    "poor quality and awful support",
])
def test_negative_prediction(production_model, text):
    """Clearly negative feedback must be classified negative."""
    sentiment, _ = model_loader.predict(text)
    assert sentiment == config.NEGATIVE


@pytest.mark.parametrize("text", [
    "it is okay but nothing special",
    "average service with average support",
    "acceptable quality, nothing more",
])
def test_neutral_prediction(production_model, text):
    """Lukewarm feedback must be classified neutral."""
    sentiment, _ = model_loader.predict(text)
    assert sentiment == config.NEUTRAL


# ---------------------------------------------------------------------------
# LOADER BEHAVIOUR
# ---------------------------------------------------------------------------

def test_loader_degrades_gracefully_without_a_model():
    """
    With no model deployed the loader must report not-ready rather than
    raising, so /health keeps answering.
    """
    assert database.get_production_model() is None
    assert model_loader.is_ready() is False

    sentiment, version = model_loader.predict("excellent service")
    assert sentiment is None
    assert version is None
    assert model_loader.get_last_error() is not None


def test_loader_never_serves_a_rejected_model(production_model):
    """
    A rejected version must never be loaded, even though its file exists.

    This is what stops a model that failed the quality gate from reaching
    users through the back door.
    """
    import shutil

    live_version = production_model["version"]
    rejected_version = database.get_next_version()
    rejected_path = os.path.join(
        config.MODEL_STORE, config.model_filename(rejected_version))

    source = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))
    shutil.copyfile(source, rejected_path)

    database.insert_model_version(
        version=rejected_version, model_path=rejected_path,
        metrics={"accuracy": 0.10, "precision": 0.10,
                 "recall": 0.10, "f1_score": 0.10},
        training_samples=10, status=config.STATUS_REJECTED,
        notes="Rejected by quality gate")

    # The loader must still be serving the production version.
    _, serving_version = model_loader.get_model()
    assert serving_version == live_version
    assert serving_version != rejected_version

"""
Model safety tests.

These guard the methodological guarantees that make the quality gate
trustworthy. If any of them fail, the accuracy numbers the gate compares are
not measuring what they claim to measure.

    1. the frozen holdout set is never trained on
    2. candidate and production are scored on that same holdout set
    3. the served model version changes correctly after promotion
"""

import os

import joblib

import config
import database
import dataset
import evaluate
import model_loader
import retrain
from conftest import add_feedback, run_full_retrain_cycle


# ---------------------------------------------------------------------------
# 1. NO TEST-SET CONTAMINATION
# ---------------------------------------------------------------------------

def test_holdout_rows_are_absent_from_the_training_pool():
    """No holdout row may appear in the committed training pool."""
    pool_texts = {text for text, _ in dataset.load_train_pool_rows()}
    holdout_texts = {text for text, _ in dataset.load_holdout_rows()}

    overlap = pool_texts & holdout_texts
    assert overlap == set(), "Holdout rows leaked into the training pool: {}".format(
        list(overlap)[:5])


def test_training_data_excludes_the_holdout_set():
    """
    The assembled training data must contain no holdout row.

    This is checked against the real loader, not the raw CSV, so it catches
    a mistake anywhere in the assembly path.
    """
    X_train, _, _ = dataset.load_training_data()
    X_holdout, _ = dataset.load_holdout_data()

    overlap = set(X_train) & set(X_holdout)
    assert overlap == set()


def test_feedback_never_enters_the_holdout_set():
    """
    New user feedback becomes TRAINING data only.

    If feedback could reach the holdout set, the yardstick would move every
    time a user submitted something and model versions would stop being
    comparable.
    """
    _, holdout_before = dataset.load_holdout_data()

    add_feedback(20, text="a brand new piece of feedback never seen before")

    X_train, _, summary = dataset.load_training_data()
    X_holdout, holdout_after = dataset.load_holdout_data()

    assert summary["feedback_samples"] == 20
    assert holdout_after == holdout_before
    assert "a brand new piece of feedback never seen before" in X_train
    assert "a brand new piece of feedback never seen before" not in X_holdout


def test_holdout_size_is_stable_across_retraining(production_model):
    """The holdout set must not grow when the training set does."""
    size_before = len(dataset.load_holdout_data()[0])

    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    assert len(dataset.load_holdout_data()[0]) == size_before


def test_the_holdout_set_is_not_empty_and_covers_every_class():
    """A degenerate holdout set would make the metrics meaningless."""
    _, y_holdout = dataset.load_holdout_data()
    assert len(y_holdout) > 0
    for label in config.SENTIMENT_CLASSES:
        assert y_holdout.count(label) > 0, "no {} samples in the holdout".format(label)


# ---------------------------------------------------------------------------
# 2. BOTH MODELS ARE SCORED ON THE SAME DATA
# ---------------------------------------------------------------------------

def test_evaluation_always_uses_the_holdout_set(production_model):
    """
    evaluate_model_file reports the holdout size, whatever model it is
    given, proving the test set does not vary by model.
    """
    X_holdout, _ = dataset.load_holdout_data()

    path = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))
    metrics, y_used, _ = evaluate.evaluate_model_file(path)

    assert metrics["test_samples"] == len(X_holdout)
    assert len(y_used) == len(X_holdout)


def test_candidate_and_production_share_one_test_set(production_model):
    """
    The two models the quality gate compares are measured on identical rows.

    Without this, comparing their accuracies would be meaningless.
    """
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    retrain.train_candidate()

    production_path = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))

    production_metrics, production_y, _ = evaluate.evaluate_model_file(production_path)
    candidate_metrics, candidate_y, _ = evaluate.evaluate_model_file(
        config.CANDIDATE_MODEL_FILE)

    # Same rows, in the same order, for both models.
    assert production_y == candidate_y
    assert production_metrics["test_samples"] == candidate_metrics["test_samples"]


def test_evaluation_is_reproducible(production_model):
    """
    Scoring the same model twice gives the same numbers.

    A fixed holdout set plus a fixed random seed means an examiner can re-run
    the evaluation and reproduce the reported figures exactly.
    """
    path = config.resolve_model_path(
        config.model_basename(production_model["model_path"]))

    first, _, _ = evaluate.evaluate_model_file(path)
    second, _, _ = evaluate.evaluate_model_file(path)
    assert first == second


def test_training_is_reproducible():
    """
    Training twice on identical data produces identical metrics.

    random_state is fixed precisely so results can be reproduced on demand.
    """
    X_train, y_train, _ = dataset.load_training_data(include_feedback=False)
    X_holdout, y_holdout = dataset.load_holdout_data()

    first_model = dataset.build_pipeline()
    first_model.fit(X_train, y_train)
    first_metrics, _ = evaluate.evaluate_model(first_model, X_holdout, y_holdout)

    second_model = dataset.build_pipeline()
    second_model.fit(X_train, y_train)
    second_metrics, _ = evaluate.evaluate_model(second_model, X_holdout, y_holdout)

    assert first_metrics == second_metrics


def test_metrics_are_within_valid_bounds(production_model):
    """Every metric must be a proper value between 0 and 1."""
    for key in ("accuracy", "precision", "recall", "f1_score"):
        value = production_model[key]
        assert value is not None
        assert 0.0 <= value <= 1.0, "{} out of range: {}".format(key, value)


# ---------------------------------------------------------------------------
# 3. THE SERVED VERSION FOLLOWS PROMOTION
# ---------------------------------------------------------------------------

def test_served_version_matches_the_production_record(production_model):
    """What the loader serves is what the database says is live."""
    _, served = model_loader.get_model()
    assert served == production_model["version"]


def test_served_version_changes_after_promotion(production_model):
    """
    After a promotion the loader picks up the new version without a restart.

    This is what lets a Jenkins deployment take effect on a running app.
    """
    _, before = model_loader.get_model()
    assert before == 1

    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    _, after = model_loader.get_model()
    assert after == 2
    assert after == database.get_production_model()["version"]


def test_served_version_is_unchanged_after_a_rejection(production_model, monkeypatch):
    """A failed gate must not disturb what is being served."""
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.999)

    _, before = model_loader.get_model()
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    _, after = model_loader.get_model()
    assert after == before


def test_promotion_loads_a_genuinely_different_file(production_model):
    """
    The promoted version must point at its own artifact, not reuse the old
    file path.
    """
    old_path = database.get_production_model()["model_path"]

    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    new_path = database.get_production_model()["model_path"]
    assert new_path != old_path
    assert os.path.exists(new_path)
    assert os.path.exists(old_path), "the archived artifact must be kept"

    # Both files must still load, so a rollback would be possible.
    assert hasattr(joblib.load(old_path), "predict")
    assert hasattr(joblib.load(new_path), "predict")

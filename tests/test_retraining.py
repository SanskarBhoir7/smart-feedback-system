"""
Auto-retraining and quality gate tests.

This is the most important test module in the project. It verifies the
behaviour the whole system exists to provide:

    * retraining only happens once enough new feedback has arrived
    * a good candidate is promoted
    * a bad candidate is REJECTED and production is left untouched
    * a rejection does not consume the collected training samples

The exit codes asserted here are the same ones the Jenkinsfile depends on to
decide whether to deploy.
"""

import os

import config
import database
import evaluate
import retrain
from conftest import (add_feedback, add_mislabelled_feedback,
                      run_full_retrain_cycle)


# ---------------------------------------------------------------------------
# THE THRESHOLD CHECK
# ---------------------------------------------------------------------------

def test_no_retraining_on_an_empty_database(production_model):
    """With no new feedback there is nothing to retrain on."""
    assert retrain.check() == retrain.EXIT_OK


def test_no_retraining_below_the_threshold(production_model):
    """One sample short of the threshold is still not enough."""
    pending = add_feedback(config.RETRAIN_THRESHOLD - 1)
    assert pending == config.RETRAIN_THRESHOLD - 1
    assert retrain.check() == retrain.EXIT_OK


def test_retraining_required_at_the_threshold(production_model):
    """Exactly the threshold triggers retraining."""
    add_feedback(config.RETRAIN_THRESHOLD)
    assert retrain.check() == retrain.EXIT_RETRAIN_REQUIRED


def test_retraining_required_above_the_threshold(production_model):
    add_feedback(config.RETRAIN_THRESHOLD + 5)
    assert retrain.check() == retrain.EXIT_RETRAIN_REQUIRED


def test_check_writes_a_decision_file(production_model):
    """
    The decision is written to disk so a later Jenkins stage can read it,
    since each stage runs as a separate process.
    """
    add_feedback(config.RETRAIN_THRESHOLD)
    retrain.check()

    assert os.path.exists(config.RETRAIN_DECISION_FILE)
    decision = retrain.read_json(config.RETRAIN_DECISION_FILE)
    assert decision["retrain_required"] is True
    assert decision["new_samples"] == config.RETRAIN_THRESHOLD
    assert decision["threshold"] == config.RETRAIN_THRESHOLD


def test_check_exit_codes_are_distinct():
    """
    The three exit codes must not collide.

    Jenkins reads them to set RETRAIN_REQUIRED, so 'retraining needed' must
    never be confused with a crash.
    """
    assert retrain.EXIT_OK == 0
    assert retrain.EXIT_ERROR == 1
    assert retrain.EXIT_RETRAIN_REQUIRED == 10
    assert len({retrain.EXIT_OK, retrain.EXIT_ERROR,
                retrain.EXIT_RETRAIN_REQUIRED}) == 3


# ---------------------------------------------------------------------------
# TRAINING A CANDIDATE
# ---------------------------------------------------------------------------

def test_candidate_is_not_trained_below_the_threshold(production_model):
    """The train stage refuses to run when it should not have been reached."""
    add_feedback(3)
    assert retrain.train_candidate() == retrain.EXIT_OK
    assert not os.path.exists(config.CANDIDATE_MODEL_FILE)


def test_candidate_is_trained_at_the_threshold(production_model):
    """Crossing the threshold produces a candidate model file."""
    add_feedback(config.RETRAIN_THRESHOLD)
    assert retrain.train_candidate() == retrain.EXIT_OK
    assert os.path.exists(config.CANDIDATE_MODEL_FILE)
    assert os.path.getsize(config.CANDIDATE_MODEL_FILE) > 0


def test_candidate_trains_on_pool_plus_feedback(production_model):
    """
    The candidate must learn from the committed pool AND the new feedback.

    That combination is what makes the system feedback-driven.
    """
    add_feedback(config.RETRAIN_THRESHOLD)
    retrain.train_candidate()

    info = retrain.read_json(
        config.CANDIDATE_METRICS_FILE.replace("metrics", "training"))
    assert info["feedback_samples"] == config.RETRAIN_THRESHOLD
    assert info["pool_samples"] > 0
    assert info["training_samples"] == info["pool_samples"] + info["feedback_samples"]


def test_candidate_is_not_registered_before_the_gate(production_model):
    """
    A candidate is not a model version yet.

    It must not appear in the database until the quality gate accepts it.
    """
    add_feedback(config.RETRAIN_THRESHOLD)
    retrain.train_candidate()

    assert len(database.get_model_history()) == 1
    assert database.get_production_model()["version"] == production_model["version"]


# ---------------------------------------------------------------------------
# QUALITY GATE - ACCEPTANCE
# ---------------------------------------------------------------------------

def test_good_candidate_is_accepted_and_promoted(production_model):
    """A healthy candidate passes both conditions and becomes production."""
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)

    check_code, train_code, gate_code = run_full_retrain_cycle()
    assert check_code == retrain.EXIT_RETRAIN_REQUIRED
    assert train_code == retrain.EXIT_OK
    assert gate_code == retrain.EXIT_OK

    production = database.get_production_model()
    assert production["version"] == 2
    assert production["status"] == config.STATUS_PRODUCTION


def test_acceptance_archives_the_previous_version(production_model):
    """The model it replaces is archived, not deleted."""
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    assert database.get_model_by_version(1)["status"] == config.STATUS_ARCHIVED
    assert database.get_previous_model()["version"] == 1


def test_acceptance_consumes_the_samples(production_model):
    """
    Accepted training data is marked as used, resetting the counter so the
    next retrain waits for genuinely new feedback.
    """
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    assert database.count_new_samples() == config.RETRAIN_THRESHOLD

    run_full_retrain_cycle()
    assert database.count_new_samples() == 0
    assert retrain.check() == retrain.EXIT_OK


def test_accepted_model_file_is_written_to_the_persistent_store(production_model):
    """
    The new model lands in the persistent store, not the Jenkins workspace,
    so it survives a workspace clean.
    """
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    production = database.get_production_model()
    assert os.path.exists(production["model_path"])
    assert os.path.dirname(production["model_path"]) == config.MODEL_STORE


# ---------------------------------------------------------------------------
# QUALITY GATE - REJECTION
# ---------------------------------------------------------------------------

def test_candidate_below_the_floor_is_rejected(production_model, monkeypatch):
    """
    Condition A: a candidate under MIN_ACCURACY is rejected.

    The threshold is raised to an impossible value so the rejection is
    deterministic and does not depend on how a poisoned model happens to
    score.
    """
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.999)
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)

    _, _, gate_code = run_full_retrain_cycle()
    assert gate_code == retrain.EXIT_ERROR


def test_rejection_leaves_production_untouched(production_model, monkeypatch):
    """The live model must keep serving after a failed gate."""
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.999)
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    production = database.get_production_model()
    assert production["version"] == production_model["version"]
    assert production["accuracy"] == production_model["accuracy"]
    assert production["status"] == config.STATUS_PRODUCTION


def test_rejection_does_not_consume_the_samples(production_model, monkeypatch):
    """
    A rejected candidate must not eat the training data.

    The samples stay pending so the next pipeline run tries again once more
    feedback has arrived.
    """
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.999)
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    assert database.count_new_samples() == config.RETRAIN_THRESHOLD
    assert retrain.check() == retrain.EXIT_RETRAIN_REQUIRED


def test_rejected_candidate_is_recorded_in_the_history(production_model, monkeypatch):
    """
    The rejection is recorded so the Model page can show it, rather than the
    event disappearing into a log file.
    """
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.999)
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    run_full_retrain_cycle()

    rejected = [row for row in database.get_model_history()
                if row["status"] == config.STATUS_REJECTED]
    assert len(rejected) == 1
    assert rejected[0]["version"] == 2
    assert "quality gate" in rejected[0]["notes"].lower()


def test_poisoned_data_fails_the_gate_for_real(production_model):
    """
    The realistic Scenario C: genuinely corrupt feedback degrades the
    candidate enough to fail the real 0.85 threshold, with no settings
    changed.
    """
    add_mislabelled_feedback(50)

    check_code, _, gate_code = run_full_retrain_cycle()
    assert check_code == retrain.EXIT_RETRAIN_REQUIRED
    assert gate_code == retrain.EXIT_ERROR

    metrics = retrain.read_json(config.CANDIDATE_METRICS_FILE)
    assert metrics["accuracy"] < config.MIN_ACCURACY

    # Production survived the attack.
    assert database.get_production_model()["version"] == production_model["version"]


def test_regression_check_blocks_a_worse_model(production_model, monkeypatch):
    """
    Condition B: a candidate that is worse than production is rejected even
    when it clears the absolute floor.

    The floor is dropped to zero so only the non-regression rule can fire.
    """
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.0)
    monkeypatch.setattr(config, "ACCURACY_TOLERANCE", 0.0)

    add_mislabelled_feedback(50)
    _, _, gate_code = run_full_retrain_cycle()

    metrics = retrain.read_json(config.CANDIDATE_METRICS_FILE)
    assert metrics["accuracy"] >= config.MIN_ACCURACY      # floor passed
    assert metrics["accuracy"] < production_model["accuracy"]  # but regressed
    assert gate_code == retrain.EXIT_ERROR


def test_gate_is_skipped_when_there_is_no_candidate(production_model):
    """
    With no candidate the gate does nothing and reports success, so a
    below-threshold pipeline run finishes green.
    """
    assert not os.path.exists(config.CANDIDATE_MODEL_FILE)
    assert retrain.quality_gate() == retrain.EXIT_OK
    assert database.get_production_model()["version"] == production_model["version"]


# ---------------------------------------------------------------------------
# THE FULL CYCLE
# ---------------------------------------------------------------------------

def test_two_consecutive_accepted_retrains(production_model):
    """
    Versions keep incrementing across repeated cycles, and only the newest
    stays in production.
    """
    for _ in range(2):
        add_feedback(config.RETRAIN_THRESHOLD,
                     text="excellent service and very helpful staff", rating=5)
        run_full_retrain_cycle()

    assert database.get_production_model()["version"] == 3
    history = database.get_model_history()
    assert len(history) == 3
    live = [row for row in history if row["status"] == config.STATUS_PRODUCTION]
    assert len(live) == 1


def test_recovery_after_a_rejection(production_model):
    """
    A rejection is not terminal.

    After the bad data is removed, the next run trains cleanly and is
    accepted - this is the Scenario C undo path.
    """
    add_mislabelled_feedback(50)
    _, _, gate_code = run_full_retrain_cycle()
    assert gate_code == retrain.EXIT_ERROR
    assert database.get_production_model()["version"] == 1

    # Remove the poison, exactly as seed_bad_feedback.py --undo does.
    database.delete_feedback_by_source("bad_seed")
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)

    _, _, gate_code = run_full_retrain_cycle()
    assert gate_code == retrain.EXIT_OK
    assert database.get_production_model()["version"] == 3

"""
Database layer tests.

These exercise database.py directly, without Flask, because retrain.py and
train.py use it the same way from inside Jenkins.
"""

import pytest

import config
import database


# ---------------------------------------------------------------------------
# INSERTING AND RETRIEVING FEEDBACK
# ---------------------------------------------------------------------------

def test_insert_feedback_returns_an_id():
    """Inserting returns the new row id."""
    row_id = database.insert_feedback("the service was excellent today", 5)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_inserted_feedback_can_be_retrieved():
    """What goes in comes back out unchanged."""
    database.insert_feedback(
        feedback_text="the delivery was quick and well packaged",
        rating=4, predicted_sentiment="positive", name="Asha")

    rows = database.get_recent_feedback(1)
    assert len(rows) == 1

    row = rows[0]
    assert row["feedback_text"] == "the delivery was quick and well packaged"
    assert row["rating"] == 4
    assert row["name"] == "Asha"
    assert row["predicted_sentiment"] == "positive"


@pytest.mark.parametrize("rating,expected", [
    (1, config.NEGATIVE), (2, config.NEGATIVE),
    (3, config.NEUTRAL),
    (4, config.POSITIVE), (5, config.POSITIVE),
])
def test_rating_is_converted_to_the_correct_label(rating, expected):
    """The 1-2 / 3 / 4-5 mapping is applied on insert."""
    database.insert_feedback("some feedback text here", rating)
    assert database.get_recent_feedback(1)[0]["actual_sentiment"] == expected


def test_prediction_correctness_is_recorded():
    """A matching prediction is flagged correct, a mismatch incorrect."""
    database.insert_feedback("great service", 5, predicted_sentiment="positive")
    assert database.get_recent_feedback(1)[0]["prediction_correct"] == 1

    database.insert_feedback("great service", 1, predicted_sentiment="positive")
    assert database.get_recent_feedback(1)[0]["prediction_correct"] == 0


def test_prediction_correctness_is_null_without_a_prediction():
    """With no model, correctness is unknown rather than wrong."""
    database.insert_feedback("great service", 5, predicted_sentiment=None)
    assert database.get_recent_feedback(1)[0]["prediction_correct"] is None


def test_recent_feedback_is_newest_first_and_limited():
    """The dashboard table shows the newest rows, capped at the limit."""
    for index in range(15):
        database.insert_feedback("feedback number {}".format(index), 4)

    rows = database.get_recent_feedback(10)
    assert len(rows) == 10
    assert rows[0]["feedback_text"] == "feedback number 14"


# ---------------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------------

def test_feedback_stats_are_accurate():
    """The aggregate numbers match the rows actually inserted."""
    database.insert_feedback("excellent work", 5, predicted_sentiment="positive")
    database.insert_feedback("quite good", 4, predicted_sentiment="neutral")
    database.insert_feedback("just okay", 3, predicted_sentiment="neutral")
    database.insert_feedback("very poor", 1, predicted_sentiment="negative")

    stats = database.get_feedback_stats()
    assert stats["total_feedback"] == 4
    assert stats["positive_count"] == 2      # ratings 5 and 4
    assert stats["neutral_count"] == 1       # rating 3
    assert stats["negative_count"] == 1      # rating 1
    assert stats["average_rating"] == 3.25   # (5+4+3+1)/4
    assert stats["predictions_scored"] == 4
    assert stats["predictions_correct"] == 3  # the rating-4 row was wrong
    assert stats["live_accuracy"] == 0.75


def test_live_accuracy_is_none_when_nothing_is_scored():
    """Accuracy is undefined rather than zero when no prediction was made."""
    database.insert_feedback("some feedback", 4, predicted_sentiment=None)
    assert database.get_feedback_stats()["live_accuracy"] is None


def test_stats_on_an_empty_database():
    """An empty database reports zeros, not an error."""
    stats = database.get_feedback_stats()
    assert stats["total_feedback"] == 0
    assert stats["average_rating"] == 0.0
    assert stats["new_samples"] == 0


# ---------------------------------------------------------------------------
# PENDING SAMPLE COUNT - drives the retraining trigger
# ---------------------------------------------------------------------------

def test_pending_count_starts_at_zero():
    assert database.count_new_samples() == 0


def test_pending_count_rises_with_each_submission():
    """Every new row counts towards the retraining threshold."""
    for expected in range(1, 6):
        database.insert_feedback("feedback text here", 4)
        assert database.count_new_samples() == expected


def test_marking_samples_used_resets_the_count():
    """Consuming samples for training resets the counter to zero."""
    for _ in range(7):
        database.insert_feedback("feedback text here", 4)
    assert database.count_new_samples() == 7

    consumed = database.mark_all_samples_used()
    assert consumed == 7
    assert database.count_new_samples() == 0


def test_used_samples_remain_available_for_training():
    """
    Consumed samples still belong to the training set.

    Retraining always uses the full history - "used" only means "already
    counted towards a completed training run".
    """
    database.insert_feedback("feedback text here", 4)
    database.mark_all_samples_used()

    assert database.count_new_samples() == 0
    assert len(database.get_training_rows()) == 1


def test_deleting_by_source_leaves_other_rows_untouched():
    """The Scenario C undo removes only the injected rows."""
    database.insert_feedback("genuine user feedback", 5, source="user")
    database.insert_feedback("injected bad row", 1, source="bad_seed")
    database.insert_feedback("another genuine one", 4, source="user")

    removed = database.delete_feedback_by_source("bad_seed")
    assert removed == 1
    assert database.get_feedback_stats()["total_feedback"] == 2


def test_max_feedback_id_on_an_empty_table():
    """The Selenium 'before' marker is 0 when nothing has been stored."""
    assert database.get_max_feedback_id() == 0


def test_max_feedback_id_tracks_the_newest_row():
    ids = [database.insert_feedback("feedback text here", 4) for _ in range(3)]
    assert database.get_max_feedback_id() == max(ids)


def test_deleting_after_an_id_restores_the_earlier_state():
    """
    The Selenium cleanup removes only what the browser tests created.

    Rows that existed before the marker must survive untouched.
    """
    database.insert_feedback("existing row one", 5)
    database.insert_feedback("existing row two", 2)
    marker = database.get_max_feedback_id()

    for _ in range(4):
        database.insert_feedback("row created by a browser test", 4)
    assert database.get_feedback_stats()["total_feedback"] == 6

    removed = database.delete_feedback_after_id(marker)
    assert removed == 4
    assert database.get_feedback_stats()["total_feedback"] == 2
    assert database.get_max_feedback_id() == marker


def test_deleting_after_the_newest_id_removes_nothing():
    """A run that created no rows must delete nothing."""
    database.insert_feedback("an existing row", 5)
    marker = database.get_max_feedback_id()

    assert database.delete_feedback_after_id(marker) == 0
    assert database.get_feedback_stats()["total_feedback"] == 1


# ---------------------------------------------------------------------------
# MODEL VERSION REGISTRATION
# ---------------------------------------------------------------------------

METRICS = {"accuracy": 0.91, "precision": 0.92,
           "recall": 0.90, "f1_score": 0.91}


def test_versions_start_at_one_and_increment():
    """Version numbers are sequential and never reused."""
    assert database.get_next_version() == 1

    database.insert_model_version(1, "models/model_v1.pkl", METRICS, 288,
                                  config.STATUS_PRODUCTION)
    assert database.get_next_version() == 2

    database.insert_model_version(2, "models/model_v2.pkl", METRICS, 300,
                                  config.STATUS_REJECTED)
    assert database.get_next_version() == 3


def test_model_version_metrics_are_stored():
    """All four metrics plus the sample count are persisted."""
    database.insert_model_version(1, "models/model_v1.pkl", METRICS, 288,
                                  config.STATUS_PRODUCTION, notes="baseline")

    record = database.get_model_by_version(1)
    assert record["accuracy"] == 0.91
    assert record["precision"] == 0.92
    assert record["recall"] == 0.90
    assert record["f1_score"] == 0.91
    assert record["training_samples"] == 288
    assert record["notes"] == "baseline"
    assert record["created_at"] is not None


def test_duplicate_version_numbers_are_rejected():
    """The schema prevents two rows claiming the same version."""
    import sqlite3

    database.insert_model_version(1, "models/model_v1.pkl", METRICS, 288,
                                  config.STATUS_PRODUCTION)
    with pytest.raises(sqlite3.IntegrityError):
        database.insert_model_version(1, "models/other.pkl", METRICS, 300,
                                      config.STATUS_PRODUCTION)


# ---------------------------------------------------------------------------
# PROMOTION AND ARCHIVING
# ---------------------------------------------------------------------------

def test_promotion_makes_a_version_live():
    database.insert_model_version(1, "models/model_v1.pkl", METRICS, 288,
                                  config.STATUS_PRODUCTION)
    database.promote_to_production(1)

    production = database.get_production_model()
    assert production["version"] == 1
    assert production["status"] == config.STATUS_PRODUCTION


def test_promotion_archives_the_previous_model():
    """Promoting v2 must demote v1 to archived."""
    database.insert_model_version(1, "models/model_v1.pkl", METRICS, 288,
                                  config.STATUS_PRODUCTION)
    database.promote_to_production(1)

    database.insert_model_version(2, "models/model_v2.pkl", METRICS, 300,
                                  config.STATUS_PRODUCTION)
    database.promote_to_production(2)

    assert database.get_production_model()["version"] == 2
    assert database.get_model_by_version(1)["status"] == config.STATUS_ARCHIVED
    assert database.get_previous_model()["version"] == 1


def test_only_one_model_is_ever_in_production():
    """However many promotions happen, exactly one row stays production."""
    for version in range(1, 5):
        database.insert_model_version(
            version, "models/model_v{}.pkl".format(version), METRICS, 288,
            config.STATUS_PRODUCTION)
        database.promote_to_production(version)

    live = [row for row in database.get_model_history()
            if row["status"] == config.STATUS_PRODUCTION]
    assert len(live) == 1
    assert live[0]["version"] == 4


def test_rejected_versions_are_not_archived_by_promotion():
    """
    Promoting a new model must not relabel a rejected one.

    A rejection is a permanent historical record, not a previous production
    model, so it must keep its status.
    """
    database.insert_model_version(1, "models/model_v1.pkl", METRICS, 288,
                                  config.STATUS_PRODUCTION)
    database.promote_to_production(1)

    database.insert_model_version(2, "models/model_v2.pkl", METRICS, 300,
                                  config.STATUS_REJECTED)

    database.insert_model_version(3, "models/model_v3.pkl", METRICS, 310,
                                  config.STATUS_PRODUCTION)
    database.promote_to_production(3)

    assert database.get_model_by_version(2)["status"] == config.STATUS_REJECTED
    assert database.get_model_by_version(1)["status"] == config.STATUS_ARCHIVED
    assert database.get_production_model()["version"] == 3


def test_no_production_model_on_an_empty_database():
    assert database.get_production_model() is None
    assert database.get_previous_model() is None
    assert database.get_model_history() == []

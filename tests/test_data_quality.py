"""
Text / rating consistency checks (data_quality.py).

The training label comes from the star rating, so a rating that contradicts
its own text is a wrong label. These tests confirm that such rows are
detected, kept, shown, and left out of retraining - WITHOUT any sentiment
keyword rules. Every judgement comes from the model or from comparing
submissions with each other.
"""

import sqlite3

import pytest

import config
import data_quality
import database
import dataset
from conftest import add_feedback, run_full_retrain_cycle


def rec(record_id, text, label):
    return {"id": record_id, "feedback_text": text, "actual_sentiment": label}


# ---------------------------------------------------------------------------
# NO KEYWORD RULES
# ---------------------------------------------------------------------------

def test_module_contains_no_sentiment_keywords():
    """
    Guard against the quick fix the brief ruled out: hard-coding words like
    "good" or "not" and matching on them.

    Parses the source and inspects every string literal the CODE uses -
    docstrings excluded, since they only explain - and checks none is a
    sentiment word. The module may only judge text through the model.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(data_quality))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))

    literals = {node.value.lower().strip() for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings}

    for word in ("good", "bad", "not", "great", "terrible", "excellent",
                 "poor", "awful", "never", "no"):
        assert word not in literals, "keyword rule found: {!r}".format(word)


# ---------------------------------------------------------------------------
# NORMALISATION AND LABEL CONFLICT (model-free)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("It was great.", "it was great"),
    ("not good", "  NOT   good  "),
    ('"The service was excellent."', "the service was excellent"),
])
def test_normalise_treats_trivial_differences_as_the_same(a, b):
    assert data_quality.normalise(a) == data_quality.normalise(b)


def test_same_text_with_opposite_ratings_is_a_conflict():
    """
    The professor's case: "not good" with 1 star AND with 5 stars. Both
    labels cannot be right, and nothing in the text says which is, so both
    are flagged. This needs no model at all.
    """
    records = [rec(51, "not good", "negative"), rec(52, "not good", "positive"),
               rec(53, "the delivery was quick", "positive")]
    assert data_quality.find_label_conflicts(records) == {51, 52}


def test_repeated_text_with_agreeing_ratings_is_not_a_conflict():
    records = [rec(1, "great service", "positive"), rec(2, "Great service!", "positive")]
    assert data_quality.find_label_conflicts(records) == set()


def test_neutral_never_creates_a_conflict():
    """Neutral against positive is a difference of opinion, not a contradiction."""
    records = [rec(1, "it was fine", "neutral"), rec(2, "it was fine", "positive")]
    assert data_quality.find_label_conflicts(records) == set()


# ---------------------------------------------------------------------------
# MODEL DISAGREEMENT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("predicted,confidence,label,expected", [
    ("positive", 0.95, "negative", True),     # confident flip -> flagged
    ("negative", 0.95, "positive", True),
    ("positive", 0.80, "negative", False),    # below the 0.87 threshold
    ("neutral", 0.99, "negative", False),     # neutral never counts
    ("positive", 0.99, "positive", False),    # agreement
    (None, None, "negative", False),          # no model available
])
def test_disagreement_rule(predicted, confidence, label, expected):
    assert data_quality.is_disagreement(predicted, confidence, label) is expected


def test_threshold_is_the_documented_value():
    """0.87 sits just above the most confident clean-label error seen (0.861)."""
    assert config.DISAGREEMENT_CONFIDENCE == pytest.approx(0.87)


def test_model_flags_a_confidently_positive_text_with_a_one_star_rating(baseline_pipeline):
    """
    The model reads this as positive at about 0.95 - well past 0.87 - so a
    1-star rating on it is flagged as a contradiction.
    """
    model, _, _ = baseline_pipeline
    flagged = data_quality.find_model_disagreements(
        [rec(1, "It is a great experience.", "negative")], model)
    assert 1 in flagged
    predicted, confidence = flagged[1]
    assert predicted == "positive" and confidence >= config.DISAGREEMENT_CONFIDENCE


def test_model_check_alone_misses_a_short_uncertain_text(baseline_pipeline):
    """
    A documented limitation, asserted rather than hidden. "It was great." is
    read as positive, but only at about 0.67 - not confident enough to vouch
    that a 1-star rating is wrong. The model check alone lets it through.
    (In the real evaluation data it WAS caught - by the label-conflict check,
    because the same text was also submitted with 5 stars. See below.)
    """
    model, _, _ = baseline_pipeline
    assert data_quality.find_model_disagreements(
        [rec(32, "It was great.", "negative")], model) == {}


def test_real_evaluation_data_is_caught_by_label_conflict(baseline_pipeline):
    """
    The professor's actual submissions, replayed. Both contradictory pairs
    are caught by the model-free check. Note ids 31 and 51 are the CORRECT
    halves of their pairs - they are flagged too, because nothing in the
    text says which rating is the right one.

    "Good" with 1 star (id 48) is NOT caught: the model reads bare "Good" as
    neutral, and it was only submitted once. Asserted as a known miss.
    """
    model, _, _ = baseline_pipeline
    records = [rec(31, "It was great.", "positive"), rec(32, "It was great.", "negative"),
               rec(48, "Good", "negative"),
               rec(51, "not good", "negative"), rec(52, "not good", "positive")]
    flags = data_quality.assess(records, model)
    assert set(flags) == {31, 32, 51, 52}
    assert all(data_quality.REASON_CONFLICT in reasons for reasons in flags.values())
    assert 48 not in flags


def test_model_does_not_flag_consistent_feedback(baseline_pipeline):
    model, _, _ = baseline_pipeline
    records = [rec(1, "excellent service and very helpful staff", "positive"),
               rec(2, "terrible service and very rude staff", "negative"),
               rec(3, "the experience could be better.", "positive")]   # mild, not a flip
    assert data_quality.find_model_disagreements(records, model) == {}


def test_assess_without_a_model_still_runs_the_conflict_check():
    records = [rec(1, "not good", "negative"), rec(2, "not good", "positive")]
    flags = data_quality.assess(records, model=None)
    assert set(flags) == {1, 2}
    assert all(r == [data_quality.REASON_CONFLICT] for r in flags.values())


# ---------------------------------------------------------------------------
# EFFECT ON TRAINING
# ---------------------------------------------------------------------------

def test_inconsistent_rows_are_excluded_from_training(production_model, baseline_pipeline):
    """Flagged rows are kept in the database but left out of the training set."""
    judge, _, _ = baseline_pipeline
    good = database.insert_feedback("excellent service and very helpful staff", 5)
    bad = database.insert_feedback("It is a great experience.", 1)

    X, _, summary = dataset.load_training_data(judge_model=judge)
    assert summary["excluded_ids"] == [bad]
    assert "It is a great experience." not in X
    assert "excellent service and very helpful staff" in X
    assert database.get_feedback_stats()["total_feedback"] == 2   # nothing deleted
    assert good not in summary["excluded_ids"]


def test_exclusion_can_be_switched_off(production_model, monkeypatch):
    """EXCLUDE_INCONSISTENT_FEEDBACK=0 trains on everything, still reporting flags."""
    monkeypatch.setattr(config, "EXCLUDE_INCONSISTENT_FEEDBACK", False)
    database.insert_feedback("not good", 1)
    database.insert_feedback("not good", 5)

    X, _, summary = dataset.load_training_data(judge_model=None)
    assert summary["feedback_excluded"] == 0
    assert summary["quality_flags"]["label_conflict"] == 2
    assert X.count("not good") == 2


def test_consistent_feedback_is_never_excluded(production_model):
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    _, _, summary = dataset.load_training_data(judge_model=None)
    assert summary["feedback_excluded"] == 0
    assert summary["feedback_samples"] == config.RETRAIN_THRESHOLD


def test_accepted_run_records_its_exclusions(production_model):
    """An accepted run writes each exclusion's reason into the database."""
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    first = database.insert_feedback("not good", 1)
    second = database.insert_feedback("not good", 5)

    _, _, gate_code = run_full_retrain_cycle()
    assert gate_code == 0

    stored = {r["id"]: r["quality_flag"] for r in database.get_feedback_records()}
    assert stored[first] == data_quality.REASON_CONFLICT
    assert stored[second] == data_quality.REASON_CONFLICT


def test_exclusion_sticks_when_a_later_model_is_less_sure(production_model):
    """
    The flaw this guards against was found on the real evaluation data.
    "Good" with 1 star was excluded by a judge that was 92% sure the text is
    positive. The model that replaced it was only 64% sure - below the 0.87
    threshold - so a fresh check alone would have let "Good" (labelled
    negative) quietly back into training on the next run.

    A recorded exclusion must hold regardless of what today's judge thinks.
    """
    row = database.insert_feedback("Good", 1)
    database.set_quality_flags({row: [data_quality.REASON_DISAGREEMENT]})

    # A judge that does NOT flag it - None runs only the conflict check,
    # which cannot flag a text submitted once.
    X, _, summary = dataset.load_training_data(judge_model=None)
    assert row in summary["excluded_ids"]
    assert "Good" not in X
    assert summary["previously_excluded"] == 1


def test_rejected_run_records_no_exclusions(production_model, monkeypatch):
    """Like consuming samples, recording exclusions happens only on acceptance."""
    monkeypatch.setattr(config, "MIN_ACCURACY", 0.999)
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    database.insert_feedback("not good", 1)
    database.insert_feedback("not good", 5)

    _, _, gate_code = run_full_retrain_cycle()
    assert gate_code == 1
    assert all(r["quality_flag"] is None for r in database.get_feedback_records())


def test_recorded_flag_is_never_overwritten():
    row = database.insert_feedback("not good", 1)
    database.set_quality_flags({row: [data_quality.REASON_CONFLICT]})
    database.set_quality_flags({row: [data_quality.REASON_DISAGREEMENT]})
    assert database.get_feedback_records()[0]["quality_flag"] == data_quality.REASON_CONFLICT


def test_a_person_can_clear_a_recorded_exclusion():
    """After a rating is corrected, a person can let the row back in."""
    row = database.insert_feedback("Good", 1)
    database.set_quality_flags({row: [data_quality.REASON_DISAGREEMENT]})
    assert database.clear_quality_flag(row) == 1

    _, _, summary = dataset.load_training_data(judge_model=None)
    assert row not in summary["excluded_ids"]


def test_retraining_records_exclusions_in_the_version_notes(production_model):
    add_feedback(config.RETRAIN_THRESHOLD,
                 text="excellent service and very helpful staff", rating=5)
    database.insert_feedback("not good", 1)
    database.insert_feedback("not good", 5)

    run_full_retrain_cycle()
    notes = database.get_production_model()["notes"]
    assert "excluded 2 inconsistent" in notes


# ---------------------------------------------------------------------------
# THE WEB APPLICATION SHOWS IT
# ---------------------------------------------------------------------------

def test_submission_shows_a_mismatch_warning(client_with_model):
    """A confidently positive text with 1 star is flagged to the user at once."""
    body = client_with_model.post("/feedback", data={
        "feedback_text": "It is a great experience.", "rating": "1"}).get_data(as_text=True)
    assert 'id="inconsistency_note"' in body
    assert "left out of retraining" in body


def test_repeat_with_opposite_rating_shows_a_conflict(client_with_model):
    client_with_model.post("/feedback", data={"feedback_text": "not good", "rating": "1"})
    body = client_with_model.post("/feedback", data={
        "feedback_text": "not good", "rating": "5"}).get_data(as_text=True)
    assert 'id="inconsistency_note"' in body
    assert "opposite rating" in body


def test_consistent_submission_shows_no_warning(client_with_model):
    body = client_with_model.post("/feedback", data={
        "feedback_text": "excellent service and very helpful staff",
        "rating": "5"}).get_data(as_text=True)
    assert 'id="inconsistency_note"' not in body


def test_dashboard_reports_data_quality(client_with_model):
    client_with_model.post("/feedback", data={"feedback_text": "not good", "rating": "1"})
    client_with_model.post("/feedback", data={"feedback_text": "not good", "rating": "5"})
    stats = client_with_model.get("/api/stats").get_json()
    assert stats["data_quality"]["label_conflict"] == 2
    assert stats["data_quality"]["flagged_rows"] == 2

    body = client_with_model.get("/dashboard").get_data(as_text=True)
    assert 'id="flagged_rows"' in body


# ---------------------------------------------------------------------------
# SCHEMA MIGRATION OF AN EXISTING DATABASE
# ---------------------------------------------------------------------------

def test_old_database_is_upgraded_without_losing_data():
    """
    A database created before the confidence columns existed gains them in
    place; every existing row and value is untouched.
    """
    connection = sqlite3.connect(config.DB_PATH)
    connection.execute("DROP TABLE feedback")
    connection.execute("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
            feedback_text TEXT NOT NULL, rating INTEGER NOT NULL,
            predicted_sentiment TEXT, actual_sentiment TEXT NOT NULL,
            prediction_correct INTEGER, model_version INTEGER,
            source TEXT NOT NULL DEFAULT 'user',
            used_in_training INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL)""")
    connection.execute(
        "INSERT INTO feedback (feedback_text, rating, actual_sentiment, created_at) "
        "VALUES ('an old row', 4, 'positive', '2026-01-01 00:00:00')")
    connection.commit()
    connection.close()

    database.init_db()
    database.init_db()          # idempotent

    rows = database.get_recent_feedback(5)
    assert len(rows) == 1
    assert rows[0]["feedback_text"] == "an old row"
    assert rows[0]["prediction_confidence"] is None
    assert rows[0]["low_confidence"] is None
    assert rows[0]["quality_flag"] is None


def test_confidence_is_stored_with_new_feedback(client_with_model):
    client_with_model.post("/feedback", data={
        "feedback_text": "excellent service and very helpful staff", "rating": "5"})
    row = database.get_recent_feedback(1)[0]
    assert 0.0 < row["prediction_confidence"] <= 1.0
    assert row["low_confidence"] == 0

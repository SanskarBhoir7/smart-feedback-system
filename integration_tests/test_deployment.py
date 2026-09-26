"""
Post-deployment checks against the running container, over real HTTP.

This is the last stage of the Jenkins pipeline. The unit tests in tests/ have
already proven the code; these prove the DEPLOYED ARTIFACT - the Docker
image with its baked-in model and encoder, running with its mounted
database - actually works end to end:

    * the container started and loaded the model AND its sentence encoder
    * the served model is the one the database says is in production
    * predictions work, including on wording the old model never saw
    * the pages render
    * submitted feedback really lands in the mounted database
"""

import re

import database


# ---------------------------------------------------------------------------
# THE CONTAINER IS HEALTHY AND SERVING THE RIGHT MODEL
# ---------------------------------------------------------------------------

def test_health_reports_a_loaded_model(api):
    response = api("GET", "/health")
    payload = response.json()
    assert response.status == 200
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["model_loaded"] is True, payload.get("error")


def test_served_model_is_the_hybrid_with_its_encoder(api):
    """
    model_family 'hybrid' can only be reported once the pickled model loaded
    AND its warm-up prediction ran - which requires the baked-in encoder with
    the matching fingerprint. So this also proves the encoder deployed.
    """
    payload = api("GET", "/health").json()
    assert payload["model_family"] == "hybrid"


def test_served_version_matches_the_database(api):
    """The container serves exactly what the database marks as production."""
    served = api("GET", "/health").json()["model_version"]
    assert served == database.get_production_model()["version"]


# ---------------------------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------------------------

def test_predict_returns_the_full_response(api):
    response = api("POST", "/predict", json_body={"text": "excellent service and very helpful staff"})
    payload = response.json()
    assert response.status == 200
    assert payload["prediction"] == "positive"
    assert set(payload["probabilities"]) == {"negative", "neutral", "positive"}
    assert 0.0 < payload["confidence"] <= 1.0
    assert payload["low_confidence"] is False


def test_deployed_model_handles_unseen_wording(api):
    """
    'phenomenal' is not in the training vocabulary. The old TF-IDF model read
    this sentence as NEGATIVE; the deployed hybrid reads it correctly - which
    only happens if the embedding half is genuinely running in the container.
    """
    payload = api("POST", "/predict", json_body={"text": "the staff were phenomenal"}).json()
    assert payload["prediction"] == "positive"


def test_low_information_input_is_flagged(api):
    payload = api("POST", "/predict", json_body={"text": "something something"}).json()
    assert payload["low_confidence"] is True


def test_predict_rejects_invalid_input(api):
    assert api("POST", "/predict", json_body={"text": ""}).status == 400
    assert api("POST", "/predict", json_body={}).status == 400


def test_predict_accepts_form_encoded_input(api):
    """The practical tests the API from PowerShell, which sends form data."""
    response = api("POST", "/predict", form={"text": "terrible service and very rude staff"})
    assert response.status == 200
    assert response.json()["prediction"] == "negative"


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def test_pages_render(api):
    for path, marker in (("/", 'id="feedback_form"'),
                         ("/dashboard", 'id="total_feedback"'),
                         ("/model", 'id="production_version"')):
        response = api("GET", path)
        assert response.status == 200, path
        assert marker in response.text, path


def test_static_assets_are_served(api):
    assert api("GET", "/static/css/style.css").status == 200
    assert api("GET", "/static/js/main.js").status == 200


def test_unknown_path_returns_404(api):
    assert api("GET", "/no-such-page").status == 404


# ---------------------------------------------------------------------------
# FEEDBACK REALLY PERSISTS IN THE MOUNTED DATABASE
# ---------------------------------------------------------------------------

def test_submitted_feedback_is_stored_in_the_mounted_database(api):
    """
    Submit through the container, then read the row back from the HOST side
    of the volume. This is the end-to-end proof the volume mount works: the
    container wrote it, the host can read it.
    """
    marker = "integration test row - the delivery was quick and well packaged"
    before = database.get_max_feedback_id()

    response = api("POST", "/feedback", form={"feedback_text": marker, "rating": "5"})
    assert response.status == 200
    assert 'id="predicted_sentiment"' in response.text

    rows = [r for r in database.get_recent_feedback(20) if r["id"] > before]
    assert len(rows) == 1
    row = rows[0]
    assert row["feedback_text"] == marker
    assert row["actual_sentiment"] == "positive"
    assert row["predicted_sentiment"] in ("positive", "neutral", "negative")
    assert 0.0 < row["prediction_confidence"] <= 1.0


def test_submission_result_page_shows_the_prediction(api):
    response = api("POST", "/feedback",
                   form={"feedback_text": "integration test row - terrible and rude staff",
                         "rating": "1"})
    predicted = re.search(r'id="predicted_sentiment"[^>]*>([^<]+)<', response.text)
    actual = re.search(r'id="actual_sentiment"[^>]*>([^<]+)<', response.text)
    assert predicted and predicted.group(1).strip() == "negative"
    assert actual and actual.group(1).strip() == "negative"


def test_invalid_submission_is_rejected_and_not_stored(api):
    before = database.get_max_feedback_id()
    response = api("POST", "/feedback", form={"feedback_text": "", "rating": "4"})
    assert response.status == 400
    assert database.get_max_feedback_id() == before


def test_stats_api_reflects_the_database(api):
    stats = api("GET", "/api/stats").json()
    assert stats["total_feedback"] == database.get_feedback_stats()["total_feedback"]
    assert "data_quality" in stats

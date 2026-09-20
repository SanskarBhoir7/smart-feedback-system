"""
API and page tests.

Covers every route the application exposes, both the HTML pages and the JSON
endpoints. Validation has its own module (test_validation.py).
"""

import config
import database


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

def test_health_returns_ok_with_a_model(client_with_model):
    """With a model deployed, /health reports it as loaded."""
    response = client_with_model.get("/health")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == 1
    assert payload["database"] == "ok"


def test_health_works_without_a_model(client):
    """
    /health must still answer 200 when no model is deployed.

    The web service being up and a model being deployed are two different
    things. The Jenkins deploy stage polls this endpoint to confirm the
    container started, so it must not fail just because a model is missing.
    """
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["error"] is not None


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def test_index_page_loads(client_with_model):
    """The feedback form renders with the fields Selenium will drive."""
    response = client_with_model.get("/")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    for element_id in ('id="feedback_text"', 'id="rating"',
                       'id="submit_feedback"', 'id="feedback_form"'):
        assert element_id in body


def test_dashboard_page_loads(client_with_model):
    """The dashboard renders with its stable statistic IDs."""
    response = client_with_model.get("/dashboard")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    for element_id in ('id="total_feedback"', 'id="positive_count"',
                       'id="neutral_count"', 'id="negative_count"',
                       'id="average_rating"', 'id="pending_samples"',
                       'id="retrain_threshold"', 'id="model_version"'):
        assert element_id in body


def test_model_page_loads(client_with_model):
    """The model page renders the production version and its metrics."""
    response = client_with_model.get("/model")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    for element_id in ('id="production_version"', 'id="production_accuracy"',
                       'id="production_precision"', 'id="production_recall"',
                       'id="production_f1"'):
        assert element_id in body


def test_unknown_page_returns_404(client):
    """An unknown page returns 404 rather than an unhandled error."""
    assert client.get("/no-such-page").status_code == 404


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

def test_predict_returns_a_prediction(client_with_model):
    """The prediction endpoint returns a simple JSON response."""
    response = client_with_model.post(
        "/predict", json={"text": "excellent service and very helpful staff"})
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["prediction"] == config.POSITIVE
    assert payload["model_version"] == 1


def test_predict_accepts_form_encoded_input(client_with_model):
    """Form encoding works too, so the endpoint is easy to call by hand."""
    response = client_with_model.post(
        "/predict", data={"text": "terrible service and very rude staff"})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == config.NEGATIVE


def test_predict_does_not_store_anything(client_with_model):
    """Predicting is a read-only operation; nothing reaches the database."""
    before = database.get_feedback_stats()["total_feedback"]
    client_with_model.post("/predict", json={"text": "the service was great"})
    assert database.get_feedback_stats()["total_feedback"] == before


def test_predict_returns_503_without_a_model(client):
    """With no model deployed, prediction is unavailable, not broken."""
    response = client.post("/predict", json={"text": "excellent service"})
    assert response.status_code == 503
    assert "error" in response.get_json()


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

def test_feedback_submission_succeeds(client_with_model):
    """A valid submission is accepted and the result is shown."""
    response = client_with_model.post("/feedback", data={
        "name": "Tester",
        "feedback_text": "excellent service and very helpful staff",
        "rating": "5",
    })
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    assert 'id="predicted_sentiment"' in body
    assert 'id="actual_sentiment"' in body
    assert 'id="prediction_match"' in body


def test_feedback_is_stored_with_both_labels(client_with_model):
    """
    The submission must be stored with the model's PREDICTION and the
    rating-derived TRUE label. That pairing is what turns feedback into
    training data.
    """
    client_with_model.post("/feedback", data={
        "name": "Tester",
        "feedback_text": "excellent service and very helpful staff",
        "rating": "5",
    })

    rows = database.get_recent_feedback(1)
    assert len(rows) == 1

    row = rows[0]
    assert row["name"] == "Tester"
    assert row["rating"] == 5
    assert row["actual_sentiment"] == config.POSITIVE
    assert row["predicted_sentiment"] == config.POSITIVE
    assert row["prediction_correct"] == 1
    assert row["model_version"] == 1
    assert row["used_in_training"] == 0


def test_feedback_records_an_incorrect_prediction(client_with_model):
    """
    Positive text rated 1 star must be stored as a mismatch.

    The rating is the truth, so the model is marked wrong - this is how the
    system notices its own errors.
    """
    client_with_model.post("/feedback", data={
        "feedback_text": "excellent service and very helpful staff",
        "rating": "1",
    })

    row = database.get_recent_feedback(1)[0]
    assert row["predicted_sentiment"] == config.POSITIVE
    assert row["actual_sentiment"] == config.NEGATIVE
    assert row["prediction_correct"] == 0


def test_feedback_is_stored_even_without_a_model(client):
    """
    With no model deployed the feedback is still stored.

    Its value as training data does not depend on a model existing, so
    losing it would be the worst possible outcome.
    """
    response = client.post("/feedback", data={
        "feedback_text": "the delivery was late and badly packaged",
        "rating": "2",
    })
    assert response.status_code == 200

    row = database.get_recent_feedback(1)[0]
    assert row["actual_sentiment"] == config.NEGATIVE
    assert row["predicted_sentiment"] is None
    assert row["prediction_correct"] is None


def test_name_is_optional(client_with_model):
    """Feedback can be submitted anonymously."""
    response = client_with_model.post("/feedback", data={
        "feedback_text": "the service was perfectly fine today",
        "rating": "3",
    })
    assert response.status_code == 200
    assert database.get_recent_feedback(1)[0]["name"] is None


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

def test_api_stats_returns_expected_fields(client_with_model):
    """The stats API exposes everything the dashboard displays."""
    response = client_with_model.get("/api/stats")
    assert response.status_code == 200

    payload = response.get_json()
    for key in ("total_feedback", "positive_count", "neutral_count",
                "negative_count", "average_rating", "new_samples",
                "retrain_threshold", "model_version", "model_accuracy",
                "model_status", "last_training_time", "retrain_ready"):
        assert key in payload, "missing key: {}".format(key)


def test_api_stats_counts_reflect_the_database(client_with_model):
    """The counts must be real, not placeholders."""
    submissions = [
        ("excellent service and very helpful staff", "5"),
        ("terrible service and very rude staff", "1"),
        ("it is okay but nothing special", "3"),
    ]
    for text, rating in submissions:
        client_with_model.post(
            "/feedback", data={"feedback_text": text, "rating": rating})

    payload = client_with_model.get("/api/stats").get_json()
    assert payload["total_feedback"] == 3
    assert payload["positive_count"] == 1
    assert payload["neutral_count"] == 1
    assert payload["negative_count"] == 1
    assert payload["new_samples"] == 3
    assert payload["average_rating"] == 3.0


# ---------------------------------------------------------------------------
# GET /api/model
# ---------------------------------------------------------------------------

def test_api_model_returns_production_details(client_with_model):
    """The model API reports the live version and the quality gate settings."""
    response = client_with_model.get("/api/model")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["production"]["version"] == 1
    assert payload["production"]["status"] == config.STATUS_PRODUCTION
    assert payload["quality_gate"]["min_accuracy"] == config.MIN_ACCURACY
    assert payload["retraining"]["threshold"] == config.RETRAIN_THRESHOLD
    assert payload["loader"]["model_loaded"] is True


def test_api_model_handles_no_model(client):
    """The model API must not fall over before the first model is trained."""
    response = client.get("/api/model")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["production"] is None
    assert payload["history"] == []

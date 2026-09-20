"""
Input validation tests.

Every one of these asserts two things: the request is rejected with the right
HTTP status, AND nothing invalid reaches the database. The second half
matters most - bad rows would silently poison the next training run.
"""

import pytest

import database


# ---------------------------------------------------------------------------
# POST /predict - JSON API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload,description", [
    ({"text": ""}, "empty string"),
    ({"text": "   "}, "whitespace only"),
    ({"text": "ab"}, "shorter than the minimum length"),
    ({}, "missing the text field"),
    ({"wrong_key": "hello there"}, "wrong field name"),
    ({"text": None}, "null text"),
])
def test_predict_rejects_invalid_input(client_with_model, payload, description):
    """Invalid prediction requests return 400 with an explanation."""
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 400, "should reject: {}".format(description)
    assert "error" in response.get_json()


def test_predict_rejects_malformed_json(client_with_model):
    """A body that is not valid JSON returns 400, not a 500."""
    response = client_with_model.post(
        "/predict", data="{this is not json",
        content_type="application/json")
    assert response.status_code == 400


def test_predict_rejects_oversized_text(client_with_model):
    """Text beyond the maximum length is rejected."""
    response = client_with_model.post("/predict", json={"text": "a" * 2000})
    assert response.status_code == 400


def test_predict_rejects_get(client_with_model):
    """Only POST is allowed on the prediction endpoint."""
    assert client_with_model.get("/predict").status_code == 405


# ---------------------------------------------------------------------------
# POST /feedback - HTML form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data,description", [
    ({"feedback_text": "", "rating": "5"}, "empty feedback"),
    ({"feedback_text": "   ", "rating": "5"}, "whitespace-only feedback"),
    ({"feedback_text": "ab", "rating": "5"}, "feedback too short"),
    ({"rating": "5"}, "feedback field missing"),
])
def test_feedback_rejects_invalid_text(client_with_model, data, description):
    """Invalid feedback text is rejected and never stored."""
    response = client_with_model.post("/feedback", data=data)
    assert response.status_code == 400, "should reject: {}".format(description)
    assert 'id="form_error"' in response.get_data(as_text=True)
    assert database.get_feedback_stats()["total_feedback"] == 0


@pytest.mark.parametrize("rating,description", [
    ("0", "below the minimum"),
    ("6", "above the maximum"),
    ("-1", "negative"),
    ("99", "far above the maximum"),
    ("abc", "not a number"),
    ("", "empty"),
    ("3.7", "not a whole number"),
])
def test_feedback_rejects_invalid_rating(client_with_model, rating, description):
    """Invalid ratings are rejected and never stored."""
    response = client_with_model.post("/feedback", data={
        "feedback_text": "the service was reasonable today",
        "rating": rating,
    })
    assert response.status_code == 400, "should reject: {}".format(description)
    assert database.get_feedback_stats()["total_feedback"] == 0


def test_feedback_rejects_missing_rating(client_with_model):
    """The rating field is required."""
    response = client_with_model.post("/feedback", data={
        "feedback_text": "the service was reasonable today"})
    assert response.status_code == 400
    assert database.get_feedback_stats()["total_feedback"] == 0


@pytest.mark.parametrize("rating", ["1", "2", "3", "4", "5"])
def test_feedback_accepts_every_valid_rating(client_with_model, rating):
    """All five ratings are accepted and stored."""
    response = client_with_model.post("/feedback", data={
        "feedback_text": "the service was reasonable today",
        "rating": rating,
    })
    assert response.status_code == 200
    assert database.get_feedback_stats()["total_feedback"] == 1


def test_rejected_input_preserves_what_the_user_typed(client_with_model):
    """
    A rejected form redisplays the text, so the user does not lose their
    typing over a missing rating.
    """
    response = client_with_model.post("/feedback", data={
        "feedback_text": "this text should survive the error",
        "rating": "",
    })
    assert response.status_code == 400
    assert "this text should survive the error" in response.get_data(as_text=True)


def test_long_name_is_truncated_not_rejected(client_with_model):
    """
    The name is optional and cosmetic, so an over-long one is trimmed rather
    than failing the whole submission.
    """
    import config

    response = client_with_model.post("/feedback", data={
        "name": "x" * 200,
        "feedback_text": "the service was reasonable today",
        "rating": "4",
    })
    assert response.status_code == 200
    assert len(database.get_recent_feedback(1)[0]["name"]) == config.MAX_NAME_LENGTH

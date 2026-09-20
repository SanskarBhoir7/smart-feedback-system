"""
Flask web application for the Smart Feedback System.

This file contains ONLY web concerns: routing, request validation, rendering
templates and shaping JSON responses.

It deliberately contains no machine learning and no SQL. Everything else is
delegated to the modules that already own that responsibility:

    database.py       all SQLite access
    model_loader.py   loading the production model and predicting
    config.py         thresholds, paths and validation limits

The application never trains a model. Training and retraining belong to
train.py and retrain.py, run by Jenkins. The web app only consumes whichever
model is currently marked 'production'.

Routes
------
    GET  /              feedback form
    POST /feedback      submit feedback (HTML form)
    POST /predict       predict sentiment without saving (JSON API)
    GET  /dashboard     aggregate statistics
    GET  /model         model versions and metrics
    GET  /api/stats     dashboard data as JSON
    GET  /api/model     model data as JSON
    GET  /health        liveness probe, works with or without a model
"""

from flask import Flask, jsonify, render_template, request

import config
import database
import model_loader

app = Flask(__name__)

# Create the tables on startup if this is a fresh database. Safe to repeat.
database.init_db()


# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------

def validate_feedback_text(raw_text):
    """
    Check submitted feedback text.

    Returns (cleaned_text, error_message). error_message is None when valid.
    """
    if raw_text is None:
        return None, "Feedback text is required."

    text = str(raw_text).strip()
    if not text:
        return None, "Feedback text cannot be empty."
    if len(text) < config.MIN_FEEDBACK_LENGTH:
        return None, "Feedback must be at least {} characters.".format(
            config.MIN_FEEDBACK_LENGTH)
    if len(text) > config.MAX_FEEDBACK_LENGTH:
        return None, "Feedback must be {} characters or fewer.".format(
            config.MAX_FEEDBACK_LENGTH)
    return text, None


def validate_rating(raw_rating):
    """
    Check the submitted star rating.

    Returns (rating_int, error_message). error_message is None when valid.
    """
    if raw_rating is None or str(raw_rating).strip() == "":
        return None, "A rating is required."

    try:
        rating = int(str(raw_rating).strip())
    except (TypeError, ValueError):
        return None, "Rating must be a whole number between {} and {}.".format(
            config.MIN_RATING, config.MAX_RATING)

    if rating < config.MIN_RATING or rating > config.MAX_RATING:
        return None, "Rating must be between {} and {}.".format(
            config.MIN_RATING, config.MAX_RATING)
    return rating, None


def validate_name(raw_name):
    """The name is optional; it is simply trimmed and length-limited."""
    if not raw_name:
        return None
    name = str(raw_name).strip()
    if not name:
        return None
    return name[:config.MAX_NAME_LENGTH]


# ---------------------------------------------------------------------------
# SHARED VIEW DATA
# ---------------------------------------------------------------------------

def build_stats():
    """
    Assemble everything the dashboard shows.

    The feedback numbers come from database.py and the model numbers from the
    production model record, so this function only joins them together.
    """
    stats = database.get_feedback_stats()
    production = database.get_production_model()

    stats["model_version"] = production["version"] if production else None
    stats["model_accuracy"] = production["accuracy"] if production else None
    stats["model_f1"] = production["f1_score"] if production else None
    stats["model_status"] = production["status"] if production else "none"
    stats["last_training_time"] = production["created_at"] if production else None
    stats["training_samples"] = production["training_samples"] if production else None

    # How close the system is to triggering a retrain.
    remaining = max(0, stats["retrain_threshold"] - stats["new_samples"])
    stats["samples_until_retrain"] = remaining
    stats["retrain_ready"] = stats["new_samples"] >= stats["retrain_threshold"]

    # Whether the model file behind that record is actually loadable.
    stats.update(model_loader.describe())
    return stats


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """The feedback submission form."""
    return render_template(
        "index.html",
        stats=build_stats(),
        result=None,
        error=None,
        form={},
    )


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """
    Handle a submitted feedback form.

    The flow that defines this project happens here:

        1. validate the input
        2. the model PREDICTS the sentiment
        3. the star rating gives the TRUE label
        4. both are stored, so the submission becomes training data
    """
    name = validate_name(request.form.get("name"))
    text, text_error = validate_feedback_text(request.form.get("feedback_text"))
    rating, rating_error = validate_rating(request.form.get("rating"))

    error = text_error or rating_error
    if error:
        return render_template(
            "index.html",
            stats=build_stats(),
            result=None,
            error=error,
            form={
                "name": request.form.get("name", ""),
                "feedback_text": request.form.get("feedback_text", ""),
                "rating": request.form.get("rating", ""),
            },
        ), 400

    # The model's guess. May be None if no model is deployed yet - the
    # feedback is still stored, because its value as training data does not
    # depend on a model existing.
    predicted, model_version = model_loader.predict(text)

    # The truth, derived from the rating the user chose.
    actual = config.rating_to_sentiment(rating)

    database.insert_feedback(
        feedback_text=text,
        rating=rating,
        predicted_sentiment=predicted,
        name=name,
        model_version=model_version,
        source="user",
    )

    result = {
        "name": name,
        "feedback_text": text,
        "rating": rating,
        "predicted_sentiment": predicted,
        "actual_sentiment": actual,
        "prediction_correct": (predicted == actual) if predicted else None,
        "model_version": model_version,
    }

    return render_template(
        "index.html",
        stats=build_stats(),
        result=result,
        error=None,
        form={},
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Aggregate feedback statistics and current model status."""
    return render_template(
        "dashboard.html",
        stats=build_stats(),
        recent=database.get_recent_feedback(config.RECENT_FEEDBACK_LIMIT),
    )


@app.route("/model", methods=["GET"])
def model_page():
    """The model / MLOps page: every version and how it performed."""
    history = database.get_model_history()
    return render_template(
        "model.html",
        stats=build_stats(),
        production=database.get_production_model(),
        previous=database.get_previous_model(),
        archived=[row for row in history if row["status"] == config.STATUS_ARCHIVED],
        rejected=[row for row in history if row["status"] == config.STATUS_REJECTED],
        history=history,
        min_accuracy=config.MIN_ACCURACY,
        tolerance=config.ACCURACY_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Predict sentiment for a piece of text WITHOUT storing it.

    Mirrors the /predict endpoint from the ML deployment practical, adapted
    from a numeric feature vector to text input.

        Request : {"text": "the service was excellent"}
        Response: {"prediction": "positive"}

    Accepts JSON or form-encoded input so it can be called from the browser
    and from PowerShell.
    """
    if request.is_json:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body is not valid JSON."}), 400
        raw_text = payload.get("text")
    else:
        raw_text = request.form.get("text")

    text, error = validate_feedback_text(raw_text)
    if error:
        return jsonify({"error": error}), 400

    prediction, model_version = model_loader.predict(text)
    if prediction is None:
        return jsonify({
            "error": "No production model is available.",
            "detail": model_loader.get_last_error(),
        }), 503

    return jsonify({
        "prediction": prediction,
        "model_version": model_version,
    })


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Everything the dashboard shows, as JSON."""
    return jsonify(build_stats())


@app.route("/api/model", methods=["GET"])
def api_model():
    """Model version information, as JSON."""
    production = database.get_production_model()
    return jsonify({
        "production": production,
        "previous": database.get_previous_model(),
        "history": database.get_model_history(),
        "quality_gate": {
            "min_accuracy": config.MIN_ACCURACY,
            "accuracy_tolerance": config.ACCURACY_TOLERANCE,
        },
        "retraining": {
            "threshold": config.RETRAIN_THRESHOLD,
            "new_samples": database.count_new_samples(),
        },
        "loader": model_loader.describe(),
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Liveness probe.

    Deliberately returns 200 even when no model is loaded. The web service
    being up and the model being deployed are two different things, and the
    Jenkins deploy stage needs to know the container started before it can
    diagnose anything else. The payload reports the model state separately.
    """
    status = {"status": "ok"}
    try:
        status.update(model_loader.describe())
        status["database"] = "ok"
        status["total_feedback"] = database.get_feedback_stats()["total_feedback"]
    except Exception as exception:          # noqa: BLE001
        status["database"] = "error"
        status["error"] = str(exception)
    return jsonify(status), 200


# ---------------------------------------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------------------------------------

def _wants_json():
    """True when the caller is using the JSON API rather than the pages."""
    return request.path.startswith("/api/") or request.path == "/predict" \
        or request.is_json


@app.errorhandler(404)
def not_found(error):
    if _wants_json():
        return jsonify({"error": "Not found"}), 404
    return render_template("error.html", code=404,
                           message="Page not found"), 404


@app.errorhandler(405)
def method_not_allowed(error):
    if _wants_json():
        return jsonify({"error": "Method not allowed"}), 405
    return render_template("error.html", code=405,
                           message="Method not allowed"), 405


@app.errorhandler(500)
def server_error(error):
    if _wants_json():
        return jsonify({"error": "Internal server error"}), 500
    return render_template("error.html", code=500,
                           message="Internal server error"), 500


if __name__ == "__main__":
    config.describe()
    loader_state = model_loader.describe()
    if loader_state["model_loaded"]:
        print("Production model loaded: v{}".format(loader_state["model_version"]))
    else:
        print("WARNING: no model loaded - {}".format(loader_state["error"]))
    print("Starting Flask on {}:{}".format(config.FLASK_HOST, config.FLASK_PORT))

    # host 0.0.0.0 so the app is reachable from outside the Docker container,
    # exactly as in the Docker and ML deployment practicals.
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)

"""
Rendered-page tests: what a user actually sees on the three pages.

Every value on these pages is rendered by the SERVER - Flask fills in every
number and label - so the pages can be checked directly against the rendered
HTML through Flask's test client, with no browser or driver involved.

The one thing a browser adds - running main.js - is covered by testing the
JSON API that main.js calls (POST /predict).

Values are read out of the HTML by element id and compared against what the
application actually holds (/api/stats), so a page showing a wrong number
fails - not just a page missing an element.
"""

import re
from html.parser import HTMLParser

import pytest

import config
import database

SENTIMENTS = ("positive", "neutral", "negative")


# ---------------------------------------------------------------------------
# A SMALL HTML READER
# ---------------------------------------------------------------------------

class _TextById(HTMLParser):
    """Collects the visible text inside every element that has an id."""

    VOID = {"br", "img", "input", "hr", "meta", "link"}

    def __init__(self):
        super().__init__()
        self.stack = []          # [(tag, id or None)]
        self.text = {}           # id -> list of text pieces
        self.tags = {}           # id -> tag name
        self.attrs = {}          # id -> attributes

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        element_id = attrs.get("id")
        if element_id:
            self.text.setdefault(element_id, [])
            self.tags[element_id] = tag
            self.attrs[element_id] = attrs
        if tag not in self.VOID:
            self.stack.append((tag, element_id))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        for _, element_id in self.stack:
            if element_id:
                self.text[element_id].append(data)


class Page:
    """A parsed page: page.text('total_feedback') -> '3'."""

    def __init__(self, response):
        self.status = response.status_code
        self.html = response.get_data(as_text=True)
        self._parser = _TextById()
        self._parser.feed(self.html)

    def has(self, element_id):
        return element_id in self._parser.tags

    def text(self, element_id):
        assert self.has(element_id), "no element with id={!r}".format(element_id)
        return " ".join("".join(self._parser.text[element_id]).split())

    def number(self, element_id):
        return float(re.search(r"-?\d+(\.\d+)?", self.text(element_id)).group())

    def attr(self, element_id, name):
        return self._parser.attrs[element_id].get(name)


def submit(client, text, rating, name=None):
    data = {"feedback_text": text, "rating": str(rating)}
    if name:
        data["name"] = name
    return Page(client.post("/feedback", data=data))


# ---------------------------------------------------------------------------
# 1. THE APPLICATION LOADS
# ---------------------------------------------------------------------------

def test_feedback_page_renders_the_form(client_with_model):
    page = Page(client_with_model.get("/"))
    assert page.status == 200
    assert "Smart Feedback System" in page.html

    for element_id in ("feedback_form", "feedback_text", "rating", "submit_feedback",
                       "preview_prediction", "name"):
        assert page.has(element_id)
    assert page.attr("feedback_form", "method").upper() == "POST"
    assert page.attr("feedback_form", "action") == "/feedback"

    # The rating menu offers exactly 1 to 5.
    options = re.findall(r'<option value="(\d)"', page.html)
    assert options == ["1", "2", "3", "4", "5"]


# ---------------------------------------------------------------------------
# 2. NAVIGATION
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("link_id,target,marker", [
    ("nav_home", "/", "feedback_form"),
    ("nav_dashboard", "/dashboard", "total_feedback"),
    ("nav_model", "/model", "min_accuracy"),
])
def test_navigation_links_reach_each_page(client_with_model, link_id, target, marker):
    """Every nav link points at a page that renders and shows its content."""
    home = Page(client_with_model.get("/"))
    assert home.attr(link_id, "href") == target

    page = Page(client_with_model.get(target))
    assert page.status == 200
    assert page.has(marker)


def test_the_current_page_is_highlighted(client_with_model):
    page = Page(client_with_model.get("/dashboard"))
    assert "active" in (page.attr("nav_dashboard", "class") or "")
    assert "active" not in (page.attr("nav_home", "class") or "")


# ---------------------------------------------------------------------------
# 3. THE FEEDBACK WORKFLOW
# ---------------------------------------------------------------------------

def test_submission_result_shows_every_field(client_with_model):
    text = "the support team was excellent and replied very quickly"
    page = submit(client_with_model, text, 5, name="Tester")
    assert page.status == 200

    assert page.text("submitted_text") == text
    assert page.text("submitted_rating") == "5 / 5"

    predicted = page.text("predicted_sentiment")
    actual = page.text("actual_sentiment")
    assert predicted in SENTIMENTS
    assert actual == "positive"

    match = page.text("prediction_match")
    assert ("YES" in match.upper()) == (predicted == actual)

    assert re.fullmatch(r"model v\d+", page.text("result_model_version"))
    assert re.fullmatch(r"\d+%", page.text("prediction_confidence"))


def test_result_agrees_with_what_was_stored(client_with_model):
    """The page must show exactly what went into the database."""
    page = submit(client_with_model, "terrible service and very rude staff", 1)
    row = database.get_recent_feedback(1)[0]
    assert page.text("predicted_sentiment") == row["predicted_sentiment"]
    assert page.text("actual_sentiment") == row["actual_sentiment"]
    assert page.text("prediction_confidence") == "{:.0f}%".format(
        row["prediction_confidence"] * 100)


# ---------------------------------------------------------------------------
# 4. RATING DECIDES THE LABEL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rating,label", [(1, "negative"), (2, "negative"),
                                          (3, "neutral"), (4, "positive"),
                                          (5, "positive")])
def test_actual_sentiment_follows_the_rating(client_with_model, rating, label):
    page = submit(client_with_model, "the delivery was late and the item arrived damaged", rating)
    assert page.text("actual_sentiment") == label


# ---------------------------------------------------------------------------
# 5. THE RETRAINING COUNTER
# ---------------------------------------------------------------------------

def test_counter_advances_with_each_submission(client_with_model):
    for expected in (1, 2, 3):
        page = submit(client_with_model, "the website is fine but the search could be better", 3)
        assert page.number("pending_after_submit") == expected
        assert page.number("pending_samples") == expected
        assert page.number("retrain_threshold") == config.RETRAIN_THRESHOLD


def test_counter_says_how_many_more_are_needed(client_with_model):
    submit(client_with_model, "the service was fine today", 3)
    page = Page(client_with_model.get("/"))
    assert "{} more sample(s)".format(config.RETRAIN_THRESHOLD - 1) in page.text("retrain_status")


# ---------------------------------------------------------------------------
# 6. EMPTY FEEDBACK IS REJECTED
# ---------------------------------------------------------------------------

def test_empty_feedback_shows_an_error_and_stores_nothing(client_with_model):
    page = Page(client_with_model.post("/feedback", data={"feedback_text": "", "rating": "4"}))
    assert page.status == 400
    assert page.text("form_error")
    assert not page.has("result")
    assert database.get_feedback_stats()["total_feedback"] == 0


# ---------------------------------------------------------------------------
# 7. PREVIEW PREDICTION
# ---------------------------------------------------------------------------
# The Preview button runs main.js, which calls POST /predict and shows the
# answer. The browser part is a few lines of display code; the behaviour
# worth testing is the API it calls - exercised here, response shape included.

@pytest.mark.parametrize("text,expected", [
    ("excellent service and very helpful staff", "positive"),
    ("terrible service and very rude staff", "negative"),
    ("it is okay but nothing special", "neutral"),
])
def test_preview_api_returns_what_main_js_displays(client_with_model, text, expected):
    response = client_with_model.post("/predict", json={"text": text})
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["prediction"] == expected
    # main.js reads these two fields to build the "(93%)" label.
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["low_confidence"], bool)
    assert database.get_feedback_stats()["total_feedback"] == 0   # nothing stored


def test_main_js_is_served_and_calls_the_predict_api(client_with_model):
    response = client_with_model.get("/static/js/main.js")
    source = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'fetch("/predict"' in source
    assert "preview_sentiment" in source and "confidence" in source


# ---------------------------------------------------------------------------
# 8. THE DASHBOARD SHOWS REAL VALUES
# ---------------------------------------------------------------------------

def test_dashboard_numbers_match_the_application(client_with_model):
    for text, rating in (("excellent service and very helpful staff", 5),
                         ("terrible service and very rude staff", 1),
                         ("it is okay but nothing special", 3),
                         ("great quality and fast delivery", 4)):
        submit(client_with_model, text, rating)

    stats = client_with_model.get("/api/stats").get_json()
    page = Page(client_with_model.get("/dashboard"))

    assert page.number("total_feedback") == stats["total_feedback"] == 4
    assert page.number("positive_count") == stats["positive_count"] == 2
    assert page.number("neutral_count") == stats["neutral_count"] == 1
    assert page.number("negative_count") == stats["negative_count"] == 1
    assert page.number("average_rating") == pytest.approx(stats["average_rating"])
    assert page.number("pending_samples") == stats["new_samples"]
    assert page.number("retrain_threshold") == stats["retrain_threshold"]
    assert page.text("retrain_ready").upper() in ("ELIGIBLE", "WAITING")

    assert page.text("model_version") == "v{}".format(stats["model_version"])
    assert page.number("model_accuracy") == pytest.approx(stats["model_accuracy"], abs=1e-4)
    assert page.text("model_status").lower() == "production"
    assert page.text("last_training_time")


# ---------------------------------------------------------------------------
# 9. RECENT FEEDBACK TABLE
# ---------------------------------------------------------------------------

def test_new_submission_appears_in_the_recent_table(client_with_model):
    marker = "a uniquely worded comment for the recent table check"
    submit(client_with_model, marker, 4)
    page = Page(client_with_model.get("/dashboard"))
    assert marker in page.text("recent_feedback_table")


# ---------------------------------------------------------------------------
# 10. THE MODEL PAGE
# ---------------------------------------------------------------------------

def test_model_page_shows_the_production_model(client_with_model):
    page = Page(client_with_model.get("/model"))
    assert re.fullmatch(r"v\d+", page.text("production_version"))
    assert page.text("production_status").lower() == "production"
    assert page.text("production_loaded").lower() == "yes"

    for metric in ("production_accuracy", "production_precision",
                   "production_recall", "production_f1"):
        assert 0.0 <= page.number(metric) <= 1.0

    assert page.number("production_samples") > 0
    assert page.number("min_accuracy") == pytest.approx(config.MIN_ACCURACY)
    assert page.number("accuracy_tolerance") == pytest.approx(config.ACCURACY_TOLERANCE)


# ---------------------------------------------------------------------------
# 11. VERSION HISTORY
# ---------------------------------------------------------------------------

def test_version_history_lists_every_model(client_with_model):
    page = Page(client_with_model.get("/model"))
    table = page.text("model_history_table")
    for row in database.get_model_history():
        assert "v{}".format(row["version"]) in table
        assert row["status"] in table


# ---------------------------------------------------------------------------
# 12. PAGE AGREES WITH THE API
# ---------------------------------------------------------------------------

def test_model_page_matches_the_api(client_with_model):
    api = client_with_model.get("/api/model").get_json()
    page = Page(client_with_model.get("/model"))
    assert page.text("production_version") == "v{}".format(api["production"]["version"])
    assert page.number("production_accuracy") == pytest.approx(
        api["production"]["accuracy"], abs=1e-4)

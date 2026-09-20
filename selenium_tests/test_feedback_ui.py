"""
End-to-end browser tests for the Smart Feedback System.

These drive a real Chrome browser against the real running application, so
they verify the parts nothing else can: that the page actually renders, that
the form actually submits, and that the results actually appear on screen.

Concepts from the Selenium practical, all used here:

    locators     By.ID, By.CSS_SELECTOR, By.TAG_NAME
    waits        WebDriverWait with expected_conditions (explicit waits)
    dropdowns    the Select class for the rating menu
    assertions   every test asserts, none merely clicks around
    screenshots  captured after each important workflow as evidence

Run:
    python -m pytest selenium_tests -v                 headless
    set HEADLESS=false && python -m pytest selenium_tests -v    visible
"""

import re

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

from conftest import api_stats, save_screenshot, text_of

# The three labels the model can produce.
SENTIMENTS = ("positive", "neutral", "negative")


# ===========================================================================
# 1. THE APPLICATION LOADS
# ===========================================================================

def test_application_loads(driver, wait, base_url):
    """The feedback page loads and shows the form we are going to drive."""
    driver.get(base_url + "/")

    # Wait for the title rather than assuming the page is instant.
    wait.until(EC.title_contains("Smart Feedback System"))
    assert "Smart Feedback System" in driver.title

    # The form and its three controls must all be present.
    wait.until(EC.presence_of_element_located((By.ID, "feedback_form")))
    for element_id in ("feedback_text", "rating", "submit_feedback"):
        assert driver.find_element(By.ID, element_id).is_displayed(), \
            "#{} is not visible on the page".format(element_id)

    # Navigation to the other two pages must exist.
    for element_id in ("nav_home", "nav_dashboard", "nav_model"):
        assert driver.find_element(By.ID, element_id).is_displayed()

    print("  page title: {}".format(driver.title))


def test_navigation_between_pages(driver, wait, base_url):
    """The three pages are reachable by clicking the navigation links."""
    driver.get(base_url + "/")

    wait.until(EC.element_to_be_clickable((By.ID, "nav_dashboard"))).click()
    wait.until(EC.presence_of_element_located((By.ID, "total_feedback")))
    assert "/dashboard" in driver.current_url

    wait.until(EC.element_to_be_clickable((By.ID, "nav_model"))).click()
    wait.until(EC.presence_of_element_located((By.ID, "min_accuracy")))
    assert "/model" in driver.current_url

    wait.until(EC.element_to_be_clickable((By.ID, "nav_home"))).click()
    wait.until(EC.presence_of_element_located((By.ID, "feedback_form")))
    assert driver.current_url.rstrip("/") == base_url


# ===========================================================================
# 2. THE FEEDBACK WORKFLOW  (the core end-to-end path)
# ===========================================================================

def submit_feedback(driver, wait, base_url, text, rating):
    """
    Fill in the form and submit it, then wait for the result panel.

    Shared by several tests so the interaction is written once.
    """
    driver.get(base_url + "/")

    field = wait.until(EC.element_to_be_clickable((By.ID, "feedback_text")))
    field.clear()
    field.send_keys(text)

    # The rating is a <select>, so use Selenium's dedicated Select class.
    Select(driver.find_element(By.ID, "rating")).select_by_value(str(rating))

    wait.until(EC.element_to_be_clickable((By.ID, "submit_feedback"))).click()

    # Wait for the RESULT to appear rather than sleeping for a guessed time.
    wait.until(EC.visibility_of_element_located((By.ID, "result")))


def test_feedback_submission_workflow(driver, wait, base_url):
    """
    The complete journey: type feedback, choose a rating, submit, and see
    the model's prediction alongside the truth derived from the rating.
    """
    feedback = "the support team was excellent and replied very quickly"
    submit_feedback(driver, wait, base_url, feedback, 5)

    # --- the submission is echoed back -------------------------------------
    assert feedback in text_of(wait, "submitted_text")
    assert "5" in text_of(wait, "submitted_rating")

    # --- the model's prediction is displayed -------------------------------
    predicted = text_of(wait, "predicted_sentiment")
    assert predicted in SENTIMENTS, \
        "unexpected predicted sentiment: {!r}".format(predicted)

    # --- the true label derived from the rating is displayed ---------------
    actual = text_of(wait, "actual_sentiment")
    assert actual == "positive", \
        "a 5-star rating must map to positive, got {!r}".format(actual)

    # --- whether the two agree is displayed --------------------------------
    match = text_of(wait, "prediction_match")
    assert match, "the prediction-match status is empty"
    assert ("YES" in match.upper()) == (predicted == actual), \
        "match status {!r} disagrees with predicted={} actual={}".format(
            match, predicted, actual)

    # --- the model version that made the call ------------------------------
    version = text_of(wait, "result_model_version")
    assert re.match(r"model v\d+", version), \
        "unexpected model version text: {!r}".format(version)

    print("  predicted={}  actual={}  match={}  by {}".format(
        predicted, actual, match, version))
    save_screenshot(driver, "01_feedback_submission")


def test_feedback_with_low_rating_is_labelled_negative(driver, wait, base_url):
    """A 1-star rating must be recorded as negative, whatever the model says."""
    submit_feedback(driver, wait, base_url,
                    "the delivery was late and the item arrived damaged", 1)

    assert text_of(wait, "actual_sentiment") == "negative"
    assert text_of(wait, "predicted_sentiment") in SENTIMENTS


def test_feedback_increments_the_retraining_counter(driver, wait, base_url):
    """
    Submitting feedback moves the auto-retraining progress counter.

    This is the visible link between using the app and triggering the MLOps
    pipeline, so it is worth asserting in the UI rather than only in the API.
    """
    before = api_stats(base_url)["new_samples"]

    submit_feedback(driver, wait, base_url,
                    "the website is fine but the search could be better", 3)

    shown = int(text_of(wait, "pending_after_submit"))
    assert shown == before + 1, \
        "counter shows {} but expected {}".format(shown, before + 1)

    # The progress panel lower down the page must agree.
    assert int(text_of(wait, "pending_samples")) == before + 1
    assert int(text_of(wait, "retrain_threshold")) > 0


def test_empty_feedback_is_rejected(driver, wait, base_url):
    """
    Submitting an empty form shows an error instead of storing rubbish.

    Validation is already unit-tested; this confirms the user actually SEES
    the rejection.
    """
    before = api_stats(base_url)["total_feedback"]

    driver.get(base_url + "/")
    wait.until(EC.element_to_be_clickable((By.ID, "feedback_text")))
    Select(driver.find_element(By.ID, "rating")).select_by_value("4")
    driver.find_element(By.ID, "submit_feedback").click()

    error = wait.until(EC.visibility_of_element_located((By.ID, "form_error")))
    assert error.text.strip(), "an error message should be shown"

    assert api_stats(base_url)["total_feedback"] == before, \
        "an invalid submission must not be stored"
    print("  error shown: {}".format(error.text.strip()))


# ===========================================================================
# 3. PREVIEW PREDICTION  (exercises POST /predict from the browser)
# ===========================================================================

@pytest.mark.parametrize("text,expected", [
    ("excellent service and very helpful staff", "positive"),
    ("terrible service and very rude staff", "negative"),
    ("it is okay but nothing special", "neutral"),
])
def test_preview_prediction(driver, wait, base_url, text, expected):
    """
    The Preview button asks the model for a prediction WITHOUT saving.

    It calls POST /predict with fetch(), so this also proves the JSON API
    works from inside the browser.
    """
    before = api_stats(base_url)["total_feedback"]

    driver.get(base_url + "/")
    field = wait.until(EC.element_to_be_clickable((By.ID, "feedback_text")))
    field.clear()
    field.send_keys(text)

    driver.find_element(By.ID, "preview_prediction").click()

    # The box starts out showing "predicting...", so wait for a real answer
    # rather than for the element merely becoming visible.
    wait.until(lambda d: d.find_element(
        By.ID, "preview_sentiment").text.strip() in SENTIMENTS)

    predicted = text_of(wait, "preview_sentiment")
    assert predicted == expected, \
        "expected {!r} for {!r}, got {!r}".format(expected, text, predicted)

    # Preview must not write anything to the database.
    assert api_stats(base_url)["total_feedback"] == before, \
        "Preview Prediction must not store feedback"

    print("  {!r} -> {}".format(text, predicted))


def test_preview_screenshot(driver, wait, base_url):
    """Capture one preview result as evidence."""
    driver.get(base_url + "/")
    field = wait.until(EC.element_to_be_clickable((By.ID, "feedback_text")))
    field.send_keys("great quality and very fast delivery")
    driver.find_element(By.ID, "preview_prediction").click()
    wait.until(lambda d: d.find_element(
        By.ID, "preview_sentiment").text.strip() in SENTIMENTS)
    save_screenshot(driver, "02_preview_prediction")


# ===========================================================================
# 4. THE DASHBOARD
# ===========================================================================

def test_dashboard_displays_real_values(driver, wait, base_url):
    """
    Every headline number on the dashboard is present and matches what the
    application actually holds.

    Comparing against /api/stats is what makes this a real test rather than
    a check that "some digits appeared".
    """
    stats = api_stats(base_url)
    driver.get(base_url + "/dashboard")
    wait.until(EC.visibility_of_element_located((By.ID, "total_feedback")))

    # --- feedback counts ---------------------------------------------------
    assert int(text_of(wait, "total_feedback")) == stats["total_feedback"]
    assert int(text_of(wait, "positive_count")) == stats["positive_count"]
    assert int(text_of(wait, "neutral_count")) == stats["neutral_count"]
    assert int(text_of(wait, "negative_count")) == stats["negative_count"]
    assert float(text_of(wait, "average_rating")) == pytest.approx(
        stats["average_rating"], abs=0.01)

    # --- retraining status -------------------------------------------------
    assert int(text_of(wait, "pending_samples")) == stats["new_samples"]
    assert int(text_of(wait, "retrain_threshold")) == stats["retrain_threshold"]
    assert text_of(wait, "retrain_ready").upper() in ("ELIGIBLE", "WAITING")

    # --- production model --------------------------------------------------
    version = text_of(wait, "model_version")
    assert version == "v{}".format(stats["model_version"])
    assert 0.0 <= float(text_of(wait, "model_accuracy")) <= 1.0
    assert text_of(wait, "model_status").lower() == "production"
    assert text_of(wait, "last_training_time"), "training time must be shown"

    print("  total={} pending={}/{} model={} accuracy={}".format(
        stats["total_feedback"], stats["new_samples"],
        stats["retrain_threshold"], version, text_of(wait, "model_accuracy")))
    save_screenshot(driver, "03_dashboard")


def test_dashboard_lists_recent_feedback(driver, wait, base_url):
    """Submitted feedback appears in the recent-feedback table."""
    marker = "a uniquely worded comment for the recent table check"
    submit_feedback(driver, wait, base_url, marker, 4)

    driver.get(base_url + "/dashboard")
    table = wait.until(EC.visibility_of_element_located(
        (By.ID, "recent_feedback_table")))

    rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
    assert len(rows) >= 1
    assert marker in table.text, "the new submission is not listed"


# ===========================================================================
# 5. THE MODEL / MLOPS PAGE
# ===========================================================================

def test_model_page_displays_production_model(driver, wait, base_url):
    """The production model, its four metrics and its history are shown."""
    driver.get(base_url + "/model")
    wait.until(EC.visibility_of_element_located((By.ID, "production_version")))

    # --- the live model ----------------------------------------------------
    version = text_of(wait, "production_version")
    assert re.match(r"^v\d+$", version), "bad version text: {!r}".format(version)
    assert text_of(wait, "production_status").lower() == "production"
    assert text_of(wait, "production_loaded").lower() == "yes", \
        "the production model should be loaded and serving"

    # --- all four metrics, each a sensible number --------------------------
    for metric_id in ("production_accuracy", "production_precision",
                      "production_recall", "production_f1"):
        value = float(text_of(wait, metric_id))
        assert 0.0 <= value <= 1.0, \
            "{} out of range: {}".format(metric_id, value)

    assert int(text_of(wait, "production_samples")) > 0

    # --- the quality gate rules are on show --------------------------------
    assert 0.0 < float(text_of(wait, "min_accuracy")) <= 1.0
    assert 0.0 <= float(text_of(wait, "accuracy_tolerance")) < 1.0

    print("  {} accuracy={} f1={} samples={}".format(
        version, text_of(wait, "production_accuracy"),
        text_of(wait, "production_f1"), text_of(wait, "production_samples")))
    save_screenshot(driver, "04_model_page")


def test_model_page_shows_version_history(driver, wait, base_url):
    """The full version history table lists at least the current model."""
    driver.get(base_url + "/model")
    table = wait.until(EC.visibility_of_element_located(
        (By.ID, "model_history_table")))

    rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
    assert len(rows) >= 1, "the history table should list at least one model"

    # The first row is the newest version and must carry a status badge.
    cells = rows[0].find_elements(By.TAG_NAME, "td")
    assert re.match(r"^v\d+$", cells[0].text.strip())
    assert cells[1].text.strip().lower() in ("production", "archived", "rejected")

    print("  history rows: {} | newest: {} [{}]".format(
        len(rows), cells[0].text.strip(), cells[1].text.strip()))


def test_model_page_matches_the_api(driver, wait, base_url):
    """What the page renders agrees with what the application reports."""
    stats = api_stats(base_url)

    driver.get(base_url + "/model")
    wait.until(EC.visibility_of_element_located((By.ID, "production_version")))

    assert text_of(wait, "production_version") == "v{}".format(
        stats["model_version"])
    assert float(text_of(wait, "production_accuracy")) == pytest.approx(
        stats["model_accuracy"], abs=0.0001)

"""
Fixtures for the Selenium browser tests.

These tests drive a REAL Chrome browser against a REAL running application -
either the local Flask development server or the deployed Docker container.
They are deliberately kept out of the default pytest run (pytest.ini sets
testpaths = tests) because they need a server to be up, which unit tests
must never depend on.

    Run against whatever is on port 5000:
        python -m pytest selenium_tests -v

    Watch the browser work (for a demonstration):
        set HEADLESS=false
        python -m pytest selenium_tests -v

    Point at a different address:
        set BASE_URL=http://localhost:5000

PROTECTING THE DEVELOPMENT BASELINE
-----------------------------------
These tests submit genuine feedback through the genuine form - that is the
whole point of an end-to-end test. But that writes to the shared database at
C:\\mlops-data\\feedback.db, which would leave the development baseline
dirty and could even push the pending-sample count over the retraining
threshold.

So the session fixture records the highest feedback id before any test runs,
and deletes everything above it afterwards. The database ends the session
exactly as it started.
"""

import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Make the project modules importable from this subdirectory.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import database                                   # noqa: E402

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "screenshots")

# How long an explicit wait will keep polling before giving up.
DEFAULT_TIMEOUT = 15


# ---------------------------------------------------------------------------
# CONFIGURATION FROM THE ENVIRONMENT
# ---------------------------------------------------------------------------

def _env_flag(name, default):
    """
    Read a boolean from the environment, accepting the usual spellings.

    Jenkins sets HEADLESS=1 while a person is more likely to type
    HEADLESS=false, so both are understood.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_base_url():
    """The address under test. Works for local Flask and for the container."""
    return os.environ.get("BASE_URL", "http://localhost:5000").rstrip("/")


def is_headless():
    """
    Headless by default.

    That is the safe choice: Jenkins runs as a service and cannot open a
    visible window at all (Windows Session 0 isolation), and an unattended
    run that popped up a browser would hang. Set HEADLESS=false to watch the
    tests drive Chrome during a demonstration.
    """
    return _env_flag("HEADLESS", True)


# ---------------------------------------------------------------------------
# SESSION SETUP
# ---------------------------------------------------------------------------

def _wait_for_app(url, attempts=15, delay=2):
    """Return True once the application answers /health."""
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def application_is_running():
    """
    Stop the whole run immediately if nothing is serving.

    Without this, every test would fail with a confusing browser error
    instead of the real reason.
    """
    base_url = get_base_url()
    if not _wait_for_app(base_url):
        pytest.exit(
            "\n\nThe application is not responding at {}\n"
            "Start it first, either:\n"
            "    python app.py\n"
            "or:\n"
            "    docker run -d -p 5000:5000 -v \"C:/mlops-data:/data\" "
            "--name smart_feedback_app smart-feedback:latest\n".format(base_url),
            returncode=2)
    print("\nApplication is up at {}".format(base_url))
    print("Browser mode: {}".format("HEADLESS" if is_headless() else "VISIBLE"))


@pytest.fixture(scope="session", autouse=True)
def preserve_database(application_is_running):
    """
    Record the database state before the session and restore it afterwards.

    Anything the browser submits during the tests is removed at the end, so
    the development baseline is left untouched.
    """
    marker = database.get_max_feedback_id()
    pending_before = database.count_new_samples()
    print("Baseline marker: highest feedback id = {}, pending = {}".format(
        marker, pending_before))

    yield

    removed = database.delete_feedback_after_id(marker)
    pending_after = database.count_new_samples()
    print("\n" + "-" * 58)
    print(" Selenium cleanup")
    print("-" * 58)
    print("  rows created by the browser and removed : {}".format(removed))
    print("  pending samples before / after          : {} / {}".format(
        pending_before, pending_after))
    if pending_after == pending_before:
        print("  development baseline RESTORED")
    else:
        print("  WARNING: pending count did not return to its original value")
    print("-" * 58)


# ---------------------------------------------------------------------------
# THE BROWSER
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def driver():
    """
    One Chrome instance for the whole session.

    Selenium 4's built-in Selenium Manager downloads and matches chromedriver
    automatically, so webdriver.Chrome() just works - exactly as in the
    Selenium practical, with no manual driver setup.
    """
    options = Options()
    if is_headless():
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # Keep the console readable: suppress Chrome's own noisy logging.
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    browser = webdriver.Chrome(options=options)

    # NOTE ON WAITS
    # The practical demonstrates both implicit and explicit waits. This suite
    # uses EXPLICIT waits only. Mixing the two is a documented Selenium
    # pitfall: the implicit wait interferes with WebDriverWait's polling and
    # produces unpredictable timeouts. Explicit waits also express exactly
    # what is being waited FOR, which makes a failure easier to read.
    browser.set_page_load_timeout(30)

    yield browser

    browser.quit()


@pytest.fixture
def wait(driver):
    """A ready-made explicit wait for the tests to use."""
    return WebDriverWait(driver, DEFAULT_TIMEOUT)


@pytest.fixture
def base_url():
    """The address under test."""
    return get_base_url()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def screenshot_dir():
    """Make sure the evidence folder exists before anything runs."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    return SCREENSHOT_DIR


def save_screenshot(driver, name):
    """
    Capture the current page as proof the workflow completed.

    The practical uses screenshots as lab evidence; here they are also
    archived by Jenkins after every build.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, "{}_{}.png".format(name, stamp))
    driver.save_screenshot(path)
    print("  screenshot saved: {}".format(os.path.basename(path)))
    return path


def api_stats(base_url):
    """
    Read /api/stats directly.

    Lets a test assert that the number RENDERED in the page matches the
    number the application actually holds, rather than just checking that
    some digits appeared.
    """
    import json
    with urllib.request.urlopen(base_url + "/api/stats", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def text_of(wait, element_id):
    """Wait for an element to be visible and return its trimmed text."""
    element = wait.until(EC.visibility_of_element_located((By.ID, element_id)))
    return element.text.strip()

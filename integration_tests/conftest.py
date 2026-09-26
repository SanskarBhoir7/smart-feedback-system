"""
Fixtures for the deployment integration tests.

These tests talk to the DEPLOYED application - normally the Docker container
Jenkins has just started - over real HTTP, exactly as a user's browser would.
They are the post-deployment check in the pipeline, replacing the browser
tests the project previously used.

Unlike tests/, nothing here is redirected to a temporary folder: the point is
to check the real deployment, including that feedback really lands in the
real mounted database. So they must clean up after themselves, and they do -
the session records the highest feedback id before any test runs and deletes
everything above it afterwards.

    Run against whatever is on port 5000:
        python -m pytest integration_tests -v

    Point somewhere else:
        set BASE_URL=http://localhost:5000
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import database                                  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "http://localhost:5000").rstrip("/")


class Response:
    """A minimal HTTP response: status, text and parsed JSON."""

    def __init__(self, status, body, content_type):
        self.status = status
        self.text = body
        self.content_type = content_type

    def json(self):
        return json.loads(self.text)


def request(method, path, json_body=None, form=None, timeout=30):
    """Send one real HTTP request to the deployed application."""
    data, headers = None, {}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Response(resp.status, resp.read().decode("utf-8"),
                            resp.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as error:
        return Response(error.code, error.read().decode("utf-8"),
                        error.headers.get("Content-Type", ""))


@pytest.fixture(scope="session", autouse=True)
def deployment_is_up():
    """Stop immediately, with the reason, if nothing is serving."""
    for _ in range(30):
        try:
            if request("GET", "/health", timeout=5).status == 200:
                print("\nDeployed application is up at {}".format(BASE_URL))
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    pytest.exit("The deployed application is not responding at {}. Start the "
                "container first.".format(BASE_URL), returncode=2)


@pytest.fixture(scope="session", autouse=True)
def restore_database(deployment_is_up):
    """Leave the shared database exactly as it was found."""
    marker = database.get_max_feedback_id()
    pending = database.count_new_samples()
    yield
    removed = database.delete_feedback_after_id(marker)
    print("\nIntegration cleanup: removed {} test row(s); pending {} -> {}".format(
        removed, pending, database.count_new_samples()))


@pytest.fixture
def api():
    """The request helper, for tests to use."""
    return request

"""
Shared pytest fixtures.

ISOLATION IS THE WHOLE POINT OF THIS FILE
-----------------------------------------
These tests train models, promote versions and reject candidates. If they ran
against the real database at C:\\mlops-data\\feedback.db and the real model
store, a test run would destroy the development baseline and the demo state.

So before ANY project module is imported, this file redirects every writable
path into a temporary directory:

    MLOPS_DATA_DIR      -> <temp>/            the shared data root
    FEEDBACK_DB_PATH    -> <temp>/test.db     the database
    MODEL_STORE_DIR     -> <temp>/models      the persistent model store
    config.BUILD_DIR    -> <temp>/build       candidate scratch space
    config.STAGED_MODEL_DIR -> <temp>/staged  the Docker staging folder

The order matters. app.py calls database.init_db() at import time, so if the
environment were not already redirected, importing it would create the real
database. The environment variables are therefore set at the very top of this
module, before the project imports below them.

Read-only inputs (the seed CSV, the training pool and the frozen holdout set)
deliberately keep pointing at the real committed files. Tests need the genuine
holdout set to verify it is never trained on, and nothing writes to them.
"""

import os
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------
# 1. REDIRECT EVERY WRITABLE PATH  (must happen before the project imports)
# ---------------------------------------------------------------------------

TEST_ROOT = tempfile.mkdtemp(prefix="sfs_tests_")

os.environ["MLOPS_DATA_DIR"] = TEST_ROOT
os.environ["FEEDBACK_DB_PATH"] = os.path.join(TEST_ROOT, "test_feedback.db")
os.environ["MODEL_STORE_DIR"] = os.path.join(TEST_ROOT, "models")

# Make the project modules importable from the tests/ subdirectory.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pytest                                    # noqa: E402

import config                                    # noqa: E402

# BUILD_DIR and STAGED_MODEL_DIR are derived from the project folder rather
# than from an environment variable, so they are redirected here instead.
config.BUILD_DIR = os.path.join(TEST_ROOT, "build")
config.STAGED_MODEL_DIR = os.path.join(TEST_ROOT, "staged")
config.MODEL_SEARCH_PATH = [config.STAGED_MODEL_DIR, config.MODEL_STORE]
config.CANDIDATE_MODEL_FILE = os.path.join(config.BUILD_DIR, "candidate_model.pkl")
config.CANDIDATE_METRICS_FILE = os.path.join(config.BUILD_DIR, "candidate_metrics.json")
config.RETRAIN_DECISION_FILE = os.path.join(config.BUILD_DIR, "retrain_decision.json")
config.ensure_directories()

import database                                  # noqa: E402
import dataset                                   # noqa: E402
import evaluate                                  # noqa: E402
import model_loader                              # noqa: E402
import retrain                                   # noqa: E402


# ---------------------------------------------------------------------------
# 2. SAFETY NET
# ---------------------------------------------------------------------------

# pytest requires its hook arguments to be named exactly 'config', which
# collides with this project's own config module. This alias lets the hooks
# below reach the project settings without shadowing pytest's parameter.
project_config = config


def pytest_configure(config):
    """
    Refuse to run if isolation failed.

    A typo in the redirection above could point the tests at the real data.
    Rather than risk destroying the development baseline, stop immediately.
    """
    for label, path in (("database", project_config.DB_PATH),
                        ("model store", project_config.MODEL_STORE),
                        ("build dir", project_config.BUILD_DIR),
                        ("staging dir", project_config.STAGED_MODEL_DIR)):
        if not os.path.abspath(path).startswith(os.path.abspath(TEST_ROOT)):
            raise RuntimeError(
                "Test isolation failed: {} points at {}, which is outside the "
                "temporary directory {}. Refusing to run.".format(
                    label, path, TEST_ROOT))


def pytest_unconfigure(config):
    """Delete the temporary directory once the whole session has finished."""
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def baseline_pipeline():
    """
    Train the model once for the whole test session.

    Training is the slowest thing these tests do, and most of them only need
    *a* working model rather than a freshly trained one. Training once and
    re-saving the same fitted pipeline keeps the suite fast.
    """
    X_train, y_train, _ = dataset.load_training_data(include_feedback=False)
    pipeline = dataset.build_pipeline()
    pipeline.fit(X_train, y_train)

    X_holdout, y_holdout = dataset.load_holdout_data()
    metrics, _ = evaluate.evaluate_model(pipeline, X_holdout, y_holdout)
    return pipeline, metrics, len(X_train)


@pytest.fixture(autouse=True)
def clean_database():
    """
    Give every test an empty database and an empty model store.

    autouse=True means no test can accidentally inherit state from the one
    before it, which is what keeps the retraining tests trustworthy.
    """
    for folder in (config.MODEL_STORE, config.STAGED_MODEL_DIR, config.BUILD_DIR):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    if os.path.exists(config.DB_PATH):
        os.remove(config.DB_PATH)
    database.init_db()

    # Drop the cached model so it cannot leak across tests.
    model_loader._cached_model = None
    model_loader._cached_version = None
    model_loader._last_error = None

    yield

    model_loader._cached_model = None
    model_loader._cached_version = None
    model_loader._last_error = None


@pytest.fixture
def production_model(baseline_pipeline):
    """
    An empty database with one trained model registered as production.

    This is the normal starting state for most tests: v1 is live.
    """
    import joblib

    pipeline, metrics, training_samples = baseline_pipeline
    version = database.get_next_version()
    path = os.path.join(config.MODEL_STORE, config.model_filename(version))
    joblib.dump(pipeline, path)

    database.insert_model_version(
        version=version,
        model_path=path,
        metrics=metrics,
        training_samples=training_samples,
        status=config.STATUS_PRODUCTION,
        notes="Baseline model created by the test fixture",
    )
    database.promote_to_production(version)

    model_loader._cached_model = None
    model_loader._cached_version = None
    return database.get_production_model()


@pytest.fixture
def client():
    """A Flask test client. No model is deployed unless a test asks for one."""
    import app as flask_app

    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def client_with_model(production_model, client):
    """A Flask test client with model v1 live and serving."""
    return client


# ---------------------------------------------------------------------------
# 4. HELPERS SHARED BY SEVERAL TEST MODULES
# ---------------------------------------------------------------------------

def add_feedback(count, text="the service was good and quite fast", rating=4,
                 source="user"):
    """Insert `count` ordinary feedback rows and return how many now pend."""
    for _ in range(count):
        database.insert_feedback(
            feedback_text=text, rating=rating,
            predicted_sentiment=None, source=source)
    return database.count_new_samples()


def add_mislabelled_feedback(count):
    """
    Insert deliberately corrupt training data, the same way
    seed_bad_feedback.py does: positive text rated 1, negative text rated 5.
    """
    positive_text = [
        "excellent service and very helpful staff",
        "great quality and fast delivery",
        "the app is fast and reliable",
        "wonderful support, very friendly",
        "perfect quality, exactly as described",
        "I love this app, easy to use",
        "brilliant design and smooth performance",
        "outstanding support and great service",
        "fantastic value and a great price",
        "highly recommend, excellent quality",
    ]
    negative_text = [
        "terrible service and very rude staff",
        "awful quality and slow delivery",
        "the app is slow and crashes constantly",
        "horrible support, very unhelpful",
        "poor quality, completely broken",
        "useless app, buggy and frustrating",
        "bad design and terrible performance",
        "worst support and worst service",
        "a waste of money at a high price",
        "would not recommend, awful quality",
    ]

    half = count // 2
    for index in range(half):
        database.insert_feedback(
            feedback_text=positive_text[index % len(positive_text)],
            rating=1, predicted_sentiment=None, source="bad_seed")
    for index in range(count - half):
        database.insert_feedback(
            feedback_text=negative_text[index % len(negative_text)],
            rating=5, predicted_sentiment=None, source="bad_seed")
    return database.count_new_samples()


def run_full_retrain_cycle():
    """
    Run the three pipeline stages exactly as Jenkins does.

    Returns (check_code, train_code, gate_code) so a test can assert on the
    same exit codes the Jenkinsfile relies on.
    """
    check_code = retrain.check()
    train_code = retrain.train_candidate()

    metrics, _, _ = evaluate.evaluate_model_file(config.CANDIDATE_MODEL_FILE)
    evaluate.save_metrics(metrics, config.CANDIDATE_METRICS_FILE)

    gate_code = retrain.quality_gate()
    return check_code, train_code, gate_code

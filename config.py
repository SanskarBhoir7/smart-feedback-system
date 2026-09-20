"""
Central configuration for the Smart Feedback System.

Every tunable value in this project lives in this one file. Nothing is hidden
inside the application code, so the whole MLOps lifecycle can be explained and
re-configured from a single place.

Anything here can be overridden with an environment variable, which is how the
Jenkins pipeline and the "failed quality gate" demonstration change behaviour
without editing source code.
"""

import os
import sys

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

# The project folder (where this file lives).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_data_root():
    """
    Pick a sensible shared-data location for the current machine.

    On Windows we use C:\\mlops-data. Inside the Docker container the
    Dockerfile sets MLOPS_DATA_DIR=/data, so this fallback is only used
    when the variable is missing.
    """
    if sys.platform == "win32":
        return r"C:\mlops-data"
    return "/data"


# DATA_ROOT is the shared folder that lives OUTSIDE both the Git repository
# and the Docker image. It holds runtime state that must survive a container
# being deleted and recreated.
DATA_ROOT = os.environ.get("MLOPS_DATA_DIR", _default_data_root())

# The SQLite database: collected feedback + the model version history.
DB_PATH = os.environ.get("FEEDBACK_DB_PATH", os.path.join(DATA_ROOT, "feedback.db"))

# MODEL_STORE is the PERSISTENT model artifact store. train.py and retrain.py
# always write new .pkl files here. It is never wiped by a Jenkins workspace
# clean, so model_v1.pkl ... model_vN.pkl accumulate safely across builds.
MODEL_STORE = os.environ.get("MODEL_STORE_DIR", os.path.join(DATA_ROOT, "models"))

# STAGED_MODEL_DIR is the workspace copy of the models folder. Jenkins copies
# the accepted production model into here right before `docker build`, so the
# Docker image bakes the model in - the deployment approach from Experiment 9.
STAGED_MODEL_DIR = os.path.join(BASE_DIR, "models")

# When loading a model we look in the workspace/baked folder first (this is
# what the running container sees at /app/models), then fall back to the
# persistent store (this is what a local development run sees).
MODEL_SEARCH_PATH = [STAGED_MODEL_DIR, MODEL_STORE]

# Scratch space used to pass the candidate model and its metrics between the
# separate Jenkins stages. Safe to delete at any time; it is gitignored.
BUILD_DIR = os.path.join(BASE_DIR, "build")

# The raw bootstrap dataset that ships with the repository.
SEED_CSV = os.path.join(BASE_DIR, "data", "feedback_seed.csv")

# prepare_data.py splits SEED_CSV once into these two files.
#
# TRAIN_POOL_CSV  the pool every model trains on, grown by user feedback
# HOLDOUT_CSV     a FIXED test set that is never trained on and never changes
#
# Freezing the holdout set is what makes accuracy comparable between model
# versions. If the test set moved every time the dataset grew, the quality
# gate would be comparing two numbers measured on different data, and
# feedback already used for training could leak into the test set.
TRAIN_POOL_CSV = os.path.join(BASE_DIR, "data", "train_pool.csv")
HOLDOUT_CSV = os.path.join(BASE_DIR, "data", "holdout.csv")

# Files written by the pipeline stages, in BUILD_DIR.
CANDIDATE_MODEL_FILE = os.path.join(BUILD_DIR, "candidate_model.pkl")
CANDIDATE_METRICS_FILE = os.path.join(BUILD_DIR, "candidate_metrics.json")
RETRAIN_DECISION_FILE = os.path.join(BUILD_DIR, "retrain_decision.json")


def model_filename(version):
    """Standard file name for a model version, e.g. 3 -> 'model_v3.pkl'."""
    return "model_v{}.pkl".format(version)


def model_basename(path):
    """
    Extract just the file name from a stored model path, whichever operating
    system wrote it.

    This exists because the database is written on the Windows host and read
    inside a Linux container. A path saved as

        C:\\mlops-data\\models\\model_v1.pkl

    is not understood by os.path.basename() under Linux, where a backslash is
    an ordinary character rather than a separator - it would return the whole
    string instead of 'model_v1.pkl', and the container would fail to find
    its own model.

    Normalising the separators first makes the lookup work on both systems.
    """
    return os.path.basename(str(path).replace("\\", "/"))


def resolve_model_path(filename):
    """
    Find an existing model file by name, searching the staged (baked) folder
    first and then the persistent store. Returns None if it is nowhere.
    """
    for folder in MODEL_SEARCH_PATH:
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def ensure_directories():
    """Create every folder the application needs, if it does not exist yet."""
    for folder in (DATA_ROOT, MODEL_STORE, STAGED_MODEL_DIR, BUILD_DIR):
        os.makedirs(folder, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. SENTIMENT LABELS
# ---------------------------------------------------------------------------

NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"

# The fixed order used for every metrics report, so columns never shuffle.
SENTIMENT_CLASSES = [NEGATIVE, NEUTRAL, POSITIVE]


def rating_to_sentiment(rating):
    """
    Convert a 1-5 star rating into the training label.

        1-2 -> negative
        3   -> neutral
        4-5 -> positive

    This is what turns ordinary user feedback into LABELLED TRAINING DATA,
    which is the idea the whole project is built on.
    """
    rating = int(rating)
    if rating <= 2:
        return NEGATIVE
    if rating == 3:
        return NEUTRAL
    return POSITIVE


# ---------------------------------------------------------------------------
# 3. MACHINE LEARNING SETTINGS
# ---------------------------------------------------------------------------

# Fixed so that every training run is reproducible. An examiner can re-run the
# training and get exactly the same numbers.
RANDOM_STATE = 42

# Fraction of the data held back for evaluation.
TEST_SIZE = 0.2


def _env_float(name, default):
    """Read a float from the environment, falling back to the default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name, default):
    """Read an integer from the environment, falling back to the default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# ---------------------------------------------------------------------------
# 4. AUTO-RETRAINING AND QUALITY GATE
# ---------------------------------------------------------------------------

# How many NEW feedback samples must be collected before the system becomes
# eligible for retraining.
RETRAIN_THRESHOLD = _env_int("RETRAIN_THRESHOLD", 10)

# QUALITY GATE CONDITION A - the absolute accuracy floor.
#
# FROZEN AT 0.85, chosen from measured results rather than invented:
#
#   * The real baseline model (v1, trained on the 360-row seed dataset)
#     scored 0.9167 accuracy on 72 held-out samples. 0.85 leaves 6.7
#     percentage points of headroom, so an honest retrain will not trip the
#     gate because of ordinary variation.
#
#   * A deliberately poisoned model measurably falls below it: 40 bad rows
#     scored 0.8194 and 50 bad rows scored 0.7778. The gate therefore
#     discriminates between a good model and a bad one in practice.
#
#   * 85% is also the example threshold used by the PS4 CI/CD Pipeline
#     practical ("if the new model's accuracy drops below... less than 85%").
MIN_ACCURACY = _env_float("MIN_ACCURACY", 0.85)

# QUALITY GATE CONDITION B - the non-regression allowance.
#
# A candidate may be at most this much worse than the current production model
# and still be accepted. This stops a working model being silently replaced by
# a worse one.
ACCURACY_TOLERANCE = _env_float("ACCURACY_TOLERANCE", 0.02)


# ---------------------------------------------------------------------------
# 5. FLASK APPLICATION
# ---------------------------------------------------------------------------

# 0.0.0.0 so the app is reachable from outside the Docker container,
# exactly as in Experiment 5 and Experiment 9.
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = _env_int("FLASK_PORT", 5000)
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

# Input validation limits for submitted feedback.
MIN_FEEDBACK_LENGTH = 3
MAX_FEEDBACK_LENGTH = 1000
MAX_NAME_LENGTH = 60
MIN_RATING = 1
MAX_RATING = 5

# How many recent submissions the dashboard table shows.
RECENT_FEEDBACK_LIMIT = 10


# ---------------------------------------------------------------------------
# 6. MODEL VERSION STATUS VALUES
# ---------------------------------------------------------------------------

STATUS_PRODUCTION = "production"   # currently serving predictions (only one)
STATUS_ARCHIVED = "archived"       # was production, replaced by a newer model
STATUS_REJECTED = "rejected"       # failed the quality gate, never deployed


def describe():
    """
    Print the active configuration. Every script calls this on startup so the
    Jenkins console log always records the settings a build actually used.
    """
    lines = [
        "----------------------------------------------------------",
        " Smart Feedback System - active configuration",
        "----------------------------------------------------------",
        "  Data root (shared)   : {}".format(DATA_ROOT),
        "  Database             : {}".format(DB_PATH),
        "  Model store (persist): {}".format(MODEL_STORE),
        "  Model staging (build): {}".format(STAGED_MODEL_DIR),
        "  Seed dataset         : {}".format(SEED_CSV),
        "  Retrain threshold    : {} new samples".format(RETRAIN_THRESHOLD),
        "  Min accuracy (gate)  : {}".format(
            "NOT FROZEN YET" if MIN_ACCURACY <= 0 else "{:.4f}".format(MIN_ACCURACY)
        ),
        "  Accuracy tolerance   : {:.4f}".format(ACCURACY_TOLERANCE),
        "----------------------------------------------------------",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    ensure_directories()
    describe()

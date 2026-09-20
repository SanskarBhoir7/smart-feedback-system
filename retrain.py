"""
Auto-retraining and the model quality gate.

This is the heart of the project. It is split into three subcommands so that
each one becomes its own visible stage in the Jenkins pipeline:

    python retrain.py --check     is retraining required?
    python retrain.py --train     train a candidate model
    python retrain.py --gate      apply the quality gate, accept or reject

State is passed between the stages through small files in build/, because
each Jenkins stage runs as a separate process.


EXIT CODE CONTRACT
------------------
Jenkins cannot read a Python variable, so the --check stage communicates its
decision through the process exit code, which `bat(returnStatus: true)`
captures reliably on Windows:

    --check   exit 0   retraining NOT required   -> env.RETRAIN_REQUIRED='false'
              exit 10  retraining IS required    -> env.RETRAIN_REQUIRED='true'
              exit 1   error

    --train   exit 0   candidate trained (or correctly skipped)
              exit 1   error

    --gate    exit 0   candidate ACCEPTED and promoted to production
              exit 1   candidate REJECTED - pipeline must stop, no deployment

Exit code 10 is used instead of 1 for "retraining required" so that a genuine
crash (which Python reports as 1) can never be mistaken for a decision.

`--check --quiet` prints nothing but the single word `true` or `false`, for
pipelines that prefer to capture stdout instead.


THE QUALITY GATE
----------------
A candidate must satisfy BOTH conditions to be accepted:

    A. ABSOLUTE FLOOR     accuracy >= MIN_ACCURACY (0.85)
    B. NON-REGRESSION     accuracy >= production accuracy - TOLERANCE (0.02)

Condition B is what stops a working model being silently replaced by a worse
one. Both models are scored on the SAME frozen holdout set, so the comparison
is valid.
"""

import argparse
import json
import os
import shutil
import sys

import joblib

import config
import database
import dataset
import evaluate


# Exit codes - see the contract in the module docstring.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_RETRAIN_REQUIRED = 10


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def write_json(path, payload):
    """Write a small state file for the next pipeline stage to read."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path):
    """Read a state file written by an earlier pipeline stage."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def banner(title):
    """Print a clearly delimited section header for the Jenkins console log."""
    print("")
    print("=" * 58)
    print(" {}".format(title))
    print("=" * 58)


# ---------------------------------------------------------------------------
# STAGE 1 - CHECK
# ---------------------------------------------------------------------------

def check(quiet=False):
    """
    Decide whether retraining is required.

    Compares the number of feedback samples collected since the last ACCEPTED
    training run against RETRAIN_THRESHOLD.
    """
    database.init_db()
    new_samples = database.count_new_samples()
    required = new_samples >= config.RETRAIN_THRESHOLD

    decision = {
        "new_samples": new_samples,
        "threshold": config.RETRAIN_THRESHOLD,
        "retrain_required": required,
    }
    write_json(config.RETRAIN_DECISION_FILE, decision)

    if quiet:
        # Machine-readable single token for `bat(returnStdout: true)` users.
        print("true" if required else "false")
        return EXIT_RETRAIN_REQUIRED if required else EXIT_OK

    banner("CHECK RETRAINING REQUIREMENT")
    print("  New samples          : {}".format(new_samples))
    print("  Retraining threshold : {}".format(config.RETRAIN_THRESHOLD))
    print("")
    if required:
        print("  RETRAINING REQUIRED")
        print("")
        print("  Threshold reached. The pipeline will train a candidate model,")
        print("  evaluate it, and apply the quality gate.")
        print("=" * 58)
        return EXIT_RETRAIN_REQUIRED

    print("  RETRAINING NOT REQUIRED")
    print("")
    print("  {} more sample(s) needed before retraining is triggered.".format(
        config.RETRAIN_THRESHOLD - new_samples))
    print("  The pipeline will skip training, evaluation, image build and")
    print("  deployment, and finish successfully.")
    print("=" * 58)
    return EXIT_OK


# ---------------------------------------------------------------------------
# STAGE 2 - TRAIN CANDIDATE
# ---------------------------------------------------------------------------

def train_candidate(force=False):
    """
    Train a candidate model on the current training pool.

    The candidate is written to build/ and is NOT yet a version - it has no
    version number and is not in the database. It only becomes a real model
    version if it passes the quality gate.
    """
    database.init_db()

    new_samples = database.count_new_samples()
    if new_samples < config.RETRAIN_THRESHOLD and not force:
        banner("TRAIN CANDIDATE - SKIPPED")
        print("  New samples {} < threshold {}.".format(
            new_samples, config.RETRAIN_THRESHOLD))
        print("  No candidate trained. This stage should not have run;")
        print("  check the Jenkins 'when' condition.")
        print("=" * 58)
        return EXIT_OK

    banner("TRAIN CANDIDATE MODEL")

    # Training data = committed pool + every piece of collected feedback.
    X_train, y_train, summary = dataset.load_training_data()
    print("  Pool samples         : {}".format(summary["pool_samples"]))
    print("  Feedback samples     : {}".format(summary["feedback_samples"]))
    print("  Total training data  : {}".format(summary["total_samples"]))
    print("  New since last train : {}".format(new_samples))

    model = dataset.build_pipeline()
    model.fit(X_train, y_train)
    print("  Vocabulary size      : {} features".format(
        len(model.named_steps["tfidf"].vocabulary_)))

    os.makedirs(config.BUILD_DIR, exist_ok=True)
    joblib.dump(model, config.CANDIDATE_MODEL_FILE)
    print("  Candidate saved to   : {}".format(config.CANDIDATE_MODEL_FILE))

    # Record how much data went in, so the gate can store it with the version.
    write_json(config.CANDIDATE_METRICS_FILE.replace("metrics", "training"), {
        "training_samples": summary["total_samples"],
        "pool_samples": summary["pool_samples"],
        "feedback_samples": summary["feedback_samples"],
        "new_samples": new_samples,
    })

    print("")
    print("  Candidate trained. It is NOT yet in production - it must pass")
    print("  the quality gate first.")
    print("=" * 58)
    return EXIT_OK


# ---------------------------------------------------------------------------
# STAGE 3 - QUALITY GATE
# ---------------------------------------------------------------------------

def _load_candidate_metrics():
    """Read the metrics produced by the Evaluate Candidate stage."""
    if not os.path.exists(config.CANDIDATE_METRICS_FILE):
        print("ERROR: candidate metrics not found at {}".format(
            config.CANDIDATE_METRICS_FILE))
        print("Run 'python evaluate.py --candidate' before the quality gate.")
        return None
    return read_json(config.CANDIDATE_METRICS_FILE)


def _load_training_info():
    """Read the training summary written by the Train Candidate stage."""
    path = config.CANDIDATE_METRICS_FILE.replace("metrics", "training")
    if os.path.exists(path):
        return read_json(path)
    return {"training_samples": 0}


def _score_production_on_holdout():
    """
    Re-score the current production model on the frozen holdout set.

    The stored accuracy in the database is already measured on this same
    fixed set, but re-scoring here proves it rather than trusting a number
    that was written by an earlier build.
    """
    record = database.get_production_model()
    if not record:
        return None, None

    model_path = config.resolve_model_path(config.model_basename(record["model_path"]))
    if not model_path:
        print("WARNING: production model file missing; using stored metrics.")
        return record, None

    metrics, _, _ = evaluate.evaluate_model_file(model_path)
    return record, metrics


def quality_gate():
    """
    Apply the quality gate and either promote or reject the candidate.

    Returns EXIT_OK when accepted, EXIT_ERROR when rejected so that the
    Jenkins stage turns red and deployment never happens.
    """
    database.init_db()

    if not os.path.exists(config.CANDIDATE_MODEL_FILE):
        banner("QUALITY GATE - SKIPPED")
        print("  No candidate model exists. Nothing to gate.")
        print("=" * 58)
        return EXIT_OK

    candidate_metrics = _load_candidate_metrics()
    if candidate_metrics is None:
        return EXIT_ERROR

    training_info = _load_training_info()
    production, production_now = _score_production_on_holdout()

    candidate_accuracy = candidate_metrics["accuracy"]

    banner("MODEL QUALITY GATE")
    print("  Candidate Accuracy   : {:.4f}".format(candidate_accuracy))
    print("  Required Accuracy    : {:.4f}   (MIN_ACCURACY)".format(
        config.MIN_ACCURACY))

    if production:
        stored = production["accuracy"]
        current = production_now["accuracy"] if production_now else stored
        print("  Production Accuracy  : {:.4f}   (v{}, re-scored on the "
              "same holdout)".format(current, production["version"]))
        if production_now and abs(current - stored) > 1e-9:
            print("     note: value stored at training time was {:.4f}".format(stored))
        print("  Regression Allowance : {:.4f}".format(config.ACCURACY_TOLERANCE))
    else:
        current = None
        print("  Production Accuracy  : none (no model in production yet)")
    print("")

    # ---- CONDITION A: absolute floor ------------------------------------
    passes_floor = candidate_accuracy >= config.MIN_ACCURACY
    print("  Condition A - absolute floor")
    print("     {:.4f} >= {:.4f}  ->  {}".format(
        candidate_accuracy, config.MIN_ACCURACY,
        "PASS" if passes_floor else "FAIL"))

    # ---- CONDITION B: no regression --------------------------------------
    if current is None:
        passes_regression = True
        print("  Condition B - no regression")
        print("     skipped, there is no production model to compare against")
    else:
        floor = current - config.ACCURACY_TOLERANCE
        passes_regression = candidate_accuracy >= floor
        print("  Condition B - no regression")
        print("     {:.4f} >= {:.4f}  ->  {}".format(
            candidate_accuracy, floor,
            "PASS" if passes_regression else "FAIL"))

    accepted = passes_floor and passes_regression
    print("")

    # ---- Decision --------------------------------------------------------
    if not accepted:
        reasons = []
        if not passes_floor:
            reasons.append("accuracy {:.4f} is below the required {:.4f}".format(
                candidate_accuracy, config.MIN_ACCURACY))
        if not passes_regression:
            reasons.append("accuracy {:.4f} regresses from production {:.4f}".format(
                candidate_accuracy, current))
        reason_text = "; ".join(reasons)

        print("  QUALITY GATE: FAILED")
        print("")
        print("  Reason: {}".format(reason_text))

        # The rejected candidate is still recorded, so the Model page can SHOW
        # that a rejection happened instead of it vanishing into a log file.
        version = database.get_next_version()
        rejected_path = os.path.join(config.MODEL_STORE,
                                     config.model_filename(version))
        shutil.copyfile(config.CANDIDATE_MODEL_FILE, rejected_path)
        database.insert_model_version(
            version=version,
            model_path=rejected_path,
            metrics=candidate_metrics,
            training_samples=training_info.get("training_samples", 0),
            status=config.STATUS_REJECTED,
            notes="Rejected by quality gate: {}".format(reason_text),
        )

        print("  Candidate recorded as v{} with status 'rejected'.".format(version))
        if production:
            print("  Production model v{} REMAINS ACTIVE.".format(production["version"]))
        print("")
        print("  Feedback samples were NOT marked as used, so the next")
        print("  pipeline run will try again as more data arrives.")
        print("  Deployment is blocked.")
        print("=" * 58)
        return EXIT_ERROR

    # ---- Accepted --------------------------------------------------------
    print("  QUALITY GATE: PASSED")
    print("")

    version = database.get_next_version()
    accepted_path = os.path.join(config.MODEL_STORE, config.model_filename(version))
    shutil.copyfile(config.CANDIDATE_MODEL_FILE, accepted_path)

    database.insert_model_version(
        version=version,
        model_path=accepted_path,
        metrics=candidate_metrics,
        training_samples=training_info.get("training_samples", 0),
        status=config.STATUS_PRODUCTION,
        notes="Accepted by quality gate (accuracy {:.4f})".format(candidate_accuracy),
    )
    database.promote_to_production(version)

    # Only now are the samples consumed. A rejection never eats training data.
    consumed = database.mark_all_samples_used()

    print("  Candidate model accepted.")
    print("  Saved to             : {}".format(accepted_path))
    print("  New model version    : v{}".format(version))
    if production:
        print("  Previous version     : v{} (now archived)".format(
            production["version"]))
    print("  Feedback consumed    : {} samples marked as used".format(consumed))
    print("")
    print("  v{} IS NOW THE PRODUCTION MODEL.".format(version))
    print("=" * 58)
    return EXIT_OK


# ---------------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-retraining pipeline: check, train candidate, quality gate.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="is retraining required? exit 10 = yes, 0 = no")
    group.add_argument("--train", action="store_true",
                       help="train a candidate model into build/")
    group.add_argument("--gate", action="store_true",
                       help="apply the quality gate; exit 1 = rejected")
    parser.add_argument("--quiet", action="store_true",
                        help="with --check, print only 'true' or 'false'")
    parser.add_argument("--force", action="store_true",
                        help="with --train, train even if below the threshold")
    args = parser.parse_args()

    if args.check:
        return check(quiet=args.quiet)
    if args.train:
        return train_candidate(force=args.force)
    return quality_gate()


if __name__ == "__main__":
    sys.exit(main())

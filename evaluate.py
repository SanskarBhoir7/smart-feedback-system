"""
Model evaluation.

This is the only place in the project where accuracy, precision, recall and
F1-score are calculated. train.py and retrain.py both import it, so the
production model and every candidate model are always measured by identical
code on an identical test set.

Can also be run directly, which is how the Jenkins "Evaluate Candidate" stage
works:

    python evaluate.py --candidate     evaluate the freshly trained candidate
    python evaluate.py --production    evaluate the live production model
    python evaluate.py --model PATH    evaluate any model file
"""

import argparse
import json
import os
import sys

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
import database
import dataset


# ---------------------------------------------------------------------------
# METRIC CALCULATION
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_predicted):
    """
    Calculate the four headline metrics.

    Averaging is 'macro': each of the three sentiment classes contributes
    equally, regardless of how many samples it has. That is the honest choice
    here, because it stops a model looking good simply by handling the two
    easy classes and ignoring 'neutral'.

    zero_division=0 keeps the run from crashing if the model never predicts
    one of the classes at all - it scores 0 for that class instead.
    """
    accuracy = accuracy_score(y_true, y_predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_predicted,
        average="macro",
        labels=config.SENTIMENT_CLASSES,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }


def evaluate_model(model, X_test, y_test):
    """Run a trained model over the test set and return its metrics."""
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)
    metrics["test_samples"] = len(y_test)
    return metrics, predictions


def evaluate_model_file(model_path):
    """
    Load a model from disk and score it on the FIXED holdout set.

    Every model version goes through this same function against the same 72
    frozen rows, which is what makes candidate-versus-production comparisons
    valid. The holdout set never grows and is never trained on, so no model
    is ever tested on data it has already seen.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))

    model = joblib.load(model_path)
    X_holdout, y_holdout = dataset.load_holdout_data()

    metrics, predictions = evaluate_model(model, X_holdout, y_holdout)
    return metrics, y_holdout, predictions


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def print_metrics_report(metrics, title="MODEL EVALUATION"):
    """Print the metrics block that appears in the Jenkins console log."""
    print("")
    print("=" * 58)
    print(" {}".format(title))
    print("=" * 58)
    print("  Accuracy   : {:.4f}".format(metrics["accuracy"]))
    print("  Precision  : {:.4f}   (macro average)".format(metrics["precision"]))
    print("  Recall     : {:.4f}   (macro average)".format(metrics["recall"]))
    print("  F1-Score   : {:.4f}   (macro average)".format(metrics["f1_score"]))
    if "test_samples" in metrics:
        print("  Holdout    : {} samples (fixed set)".format(metrics["test_samples"]))
    print("=" * 58)


def print_detailed_report(y_true, y_predicted):
    """Per-class breakdown and confusion matrix, useful during a viva."""
    print("\nPer-class report:")
    print(classification_report(
        y_true, y_predicted,
        labels=config.SENTIMENT_CLASSES,
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted)")
    print("               " + "".join("{:>10s}".format(c) for c in config.SENTIMENT_CLASSES))
    matrix = confusion_matrix(y_true, y_predicted, labels=config.SENTIMENT_CLASSES)
    for name, row in zip(config.SENTIMENT_CLASSES, matrix):
        print("  {:12s} ".format(name) + "".join("{:>10d}".format(v) for v in row))


def save_metrics(metrics, path):
    """Write metrics to JSON so the next Jenkins stage can read them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("\nMetrics written to: {}".format(path))


# ---------------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a sentiment model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", action="store_true",
                       help="evaluate the candidate model produced by retrain.py")
    group.add_argument("--production", action="store_true",
                       help="evaluate the current production model")
    group.add_argument("--model", metavar="PATH",
                       help="evaluate a specific model file")
    parser.add_argument("--detailed", action="store_true",
                        help="also print the per-class report and confusion matrix")
    args = parser.parse_args()

    database.init_db()

    if args.candidate:
        model_path = config.CANDIDATE_MODEL_FILE
        title = "CANDIDATE MODEL EVALUATION"

        # If the retrain step decided no retraining was needed, there is no
        # candidate to evaluate. That is a normal outcome, not a failure.
        if not os.path.exists(model_path):
            print("No candidate model found at {}".format(model_path))
            print("Nothing to evaluate - retraining was not required this run.")
            return 0

    elif args.production:
        record = database.get_production_model()
        if not record:
            print("ERROR: no production model exists yet. Run train.py first.")
            return 1
        model_path = config.resolve_model_path(
            config.model_basename(record["model_path"]))
        if not model_path:
            print("ERROR: production model file is missing from disk.")
            return 1
        title = "PRODUCTION MODEL EVALUATION (v{})".format(record["version"])

    else:
        model_path = args.model
        title = "MODEL EVALUATION - {}".format(config.model_basename(model_path))

    print("Evaluating: {}".format(model_path))
    metrics, y_test, predictions = evaluate_model_file(model_path)
    print_metrics_report(metrics, title)

    if args.detailed:
        print_detailed_report(y_test, predictions)

    # The candidate's metrics are handed to the Quality Gate stage via this file.
    if args.candidate:
        save_metrics(metrics, config.CANDIDATE_METRICS_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())

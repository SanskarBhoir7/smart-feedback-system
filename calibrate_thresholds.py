"""
Reproduce the two thresholds in config.py from validation data.

    LOW_CONFIDENCE_THRESHOLD   when a prediction is shown as "low confidence"
    DISAGREEMENT_CONFIDENCE    when a text/rating pair is flagged inconsistent

Neither was chosen by hand. Both come from OUT-OF-FOLD predictions on the
288-row training pool: the pool is split into 5 folds, and every row is
scored by a model trained on the OTHER four. So every score is what the model
does on text it has not seen - a genuine validation distribution.

The frozen holdout is deliberately NOT used. It is reserved for evaluating
models and for the quality gate; tuning anything on it would contaminate it.

    python calibrate_thresholds.py

Prints the derived values. It does not edit config.py: a threshold change is
a decision to make and document, not something a script should do silently.
"""

import csv
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
import dataset
import embeddings

warnings.filterwarnings("ignore")

POLARITY = {"negative": -1, "neutral": 0, "positive": 1}
LOW_CONFIDENCE_PERCENTILE = 10      # flag roughly the least-confident 10%


def main():
    X, y, _ = dataset.load_training_data(include_feedback=False)
    X, y = np.array(X, dtype=object), np.array(y)

    probabilities = np.zeros((len(X), 3))
    nearest = np.zeros(len(X))
    classes = None
    folds = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    embedder = embeddings.SentenceEmbedder().fit(X)
    E = embedder.transform(X)

    for train_idx, test_idx in folds.split(X, y):
        model = dataset.build_pipeline().fit(list(X[train_idx]), list(y[train_idx]))
        probabilities[test_idx] = model.predict_proba(list(X[test_idx]))
        nearest[test_idx] = (E[test_idx] @ E[train_idx].T).max(1)
        classes = list(model.classes_)

    predicted = np.array([classes[i] for i in probabilities.argmax(1)])
    top = probabilities.max(1)
    wrong = predicted != y

    print("=" * 64)
    print(" THRESHOLD CALIBRATION - out-of-fold, training pool only")
    print("=" * 64)
    print("  rows {} | out-of-fold accuracy {:.4f} | errors {}".format(
        len(X), 1 - wrong.mean(), wrong.sum()))

    # ---- 1. which signal best identifies an unreliable prediction? --------
    auc_top = roc_auc_score(wrong, -top)
    auc_nn = roc_auc_score(wrong, -nearest)
    print("\n  1. LOW CONFIDENCE")
    print("     How well each signal picks out wrong predictions (AUROC):")
    print("       top class probability         {:.3f}".format(auc_top))
    print("       similarity to training data   {:.3f}".format(auc_nn))
    print("     -> top class probability is used ({})".format(
        "better" if auc_top >= auc_nn else "NOTE: similarity scored higher"))

    threshold = np.percentile(top, LOW_CONFIDENCE_PERCENTILE)
    print("     {}th percentile of top probability: {:.3f}  (config uses {:.2f})".format(
        LOW_CONFIDENCE_PERCENTILE, threshold, config.LOW_CONFIDENCE_THRESHOLD))
    print("     errors below it: {} of {}".format((top[wrong] < threshold).sum(), wrong.sum()))

    challenge = os.path.join(config.BASE_DIR, "data", "challenge_set.csv")
    with open(challenge, newline="", encoding="utf-8") as handle:
        probes = [r["text"] for r in csv.DictReader(handle) if r["category"] == "low_information"]
    full = dataset.build_pipeline().fit(list(X), list(y))
    probe_top = full.predict_proba(probes).max(1)
    print("     low-information probes flagged: {} of {}".format(
        (probe_top < config.LOW_CONFIDENCE_THRESHOLD).sum(), len(probes)))

    # ---- 2. how confident is the model when it is WRONG on clean labels? --
    flips = np.array([POLARITY[p] * POLARITY[t] == -1 for p, t in zip(predicted, y)])
    print("\n  2. TEXT / RATING INCONSISTENCY")
    print("     opposite-polarity errors on CLEAN labels: {} of {}".format(flips.sum(), len(X)))
    if flips.any():
        print("     their confidences: {}".format(np.round(np.sort(top[flips]), 3).tolist()))
        print("     most confident: {:.3f}  (config uses {:.2f}, just above it)".format(
            top[flips].max(), config.DISAGREEMENT_CONFIDENCE))
    print("     A flip more confident than any clean-label error is more likely")
    print("     a bad rating than a model mistake.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())

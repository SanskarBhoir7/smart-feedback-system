"""
Formal comparison of the old and new model families.

    A  TF-IDF + Logistic Regression                (the original model)
    H  TF-IDF + sentence embeddings + Logistic Regression   (the current model)

Both are trained on the IDENTICAL data - the committed 288-row training pool,
no feedback - and scored on the IDENTICAL frozen 72-row holdout. That makes
the comparison about the REPRESENTATION alone.

Two evaluation sets, kept strictly apart:

    data/holdout.csv        the FORMAL frozen holdout. The quality gate uses
                            it. It measures in-distribution performance.

    data/challenge_set.csv  a small hand-written CHALLENGE set probing
                            negation, wording absent from the training
                            vocabulary, and near-meaningless input. Its
                            expected labels are a human reading of the text
                            alone. It is NOT the holdout, nothing is trained
                            or tuned on it, and it is never used by the gate.

Nothing is saved. Run it to reproduce the figures in the report:

    python compare_models.py
"""

import csv
import io
import os
import sys
import time
import warnings

import joblib

import config
import dataset
import evaluate

warnings.filterwarnings("ignore")

CHALLENGE_CSV = os.path.join(config.BASE_DIR, "data", "challenge_set.csv")


def build_tfidf_only():
    """The original model family, rebuilt exactly as it was."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", dataset.build_tfidf()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          C=5.0, random_state=config.RANDOM_STATE)),
    ])


def load_challenge():
    with open(CHALLENGE_CSV, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def timed_fit(builder, X, y):
    start = time.perf_counter()
    model = builder().fit(X, y)
    return model, time.perf_counter() - start


def latency_ms(model, runs=50):
    """
    Time one prediction on a sentence the model has NOT seen before.

    Every request uses a different sentence. Repeating one sentence would
    measure the embedding cache, not the model - the encoder only does real
    work for a sentence it has not already encoded, which is exactly the
    case for a new piece of user feedback.
    """
    for i in range(3):
        model.predict(["warm-up sentence number {}".format(i)])
    start = time.perf_counter()
    for i in range(runs):
        model.predict(["the delivery was late and the packaging was damaged, "
                       "order reference {}".format(i)])
    return (time.perf_counter() - start) / runs * 1000


def pickle_bytes(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell()


def score_challenge(model, rows, category):
    items = [r for r in rows if r["category"] == category]
    predicted = model.predict([r["text"] for r in items])
    return sum(p == r["expected"] for p, r in zip(predicted, items)), len(items)


def main():
    X_train, y_train, _ = dataset.load_training_data(include_feedback=False)
    X_hold, y_hold = dataset.load_holdout_data()
    challenge = load_challenge()

    # Warm the encoder first, so H's training time is the model, not the
    # one-off library import.
    dataset.build_pipeline().fit(X_train[:6], y_train[:6])

    models = {}
    for key, builder in (("A", build_tfidf_only), ("H", dataset.build_pipeline)):
        model, seconds = timed_fit(builder, X_train, y_train)
        metrics, predictions = evaluate.evaluate_model(model, X_hold, y_hold)
        models[key] = {"model": model, "fit_s": seconds, "metrics": metrics,
                       "pred": predictions}

    print("=" * 70)
    print(" MODEL COMPARISON - identical training data, identical frozen holdout")
    print("=" * 70)
    print("  training rows {} | holdout rows {}\n".format(len(X_train), len(X_hold)))
    print("  {:<28} {:>18} {:>18}".format("", "A  TF-IDF", "H  hybrid"))
    for name in ("accuracy", "precision", "recall", "f1_score"):
        print("  {:<28} {:>18.4f} {:>18.4f}".format(
            "holdout " + name, models["A"]["metrics"][name], models["H"]["metrics"][name]))
    for category in ("negation", "unseen"):
        a = score_challenge(models["A"]["model"], challenge, category)
        h = score_challenge(models["H"]["model"], challenge, category)
        print("  {:<28} {:>15}/{:<2} {:>15}/{:<2}".format(
            "challenge: " + category, a[0], a[1], h[0], h[1]))
    print("  {:<28} {:>18.3f} {:>18.3f}".format(
        "training time (s)", models["A"]["fit_s"], models["H"]["fit_s"]))
    print("  {:<28} {:>18.3f} {:>18.3f}".format(
        "latency (ms / request)", latency_ms(models["A"]["model"]), latency_ms(models["H"]["model"])))
    print("  {:<28} {:>18,} {:>18,}".format(
        "model_vN.pkl (bytes)", pickle_bytes(models["A"]["model"]), pickle_bytes(models["H"]["model"])))
    manifest_dir = config.resolve_encoder_dir()
    if manifest_dir:
        import embeddings
        mb = embeddings.read_manifest(manifest_dir)["total_bytes"] / 1e6
        print("  {:<28} {:>18} {:>15.1f} MB".format("shared frozen encoder", "-", mb))

    for key in ("A", "H"):
        print("\n  {} - per-class and confusion matrix:".format(key))
        evaluate.print_detailed_report(y_hold, models[key]["pred"])

    print("\n  CHALLENGE SET, item by item (expected = human reading of the text alone)")
    print("  {:<34} {:<9} {:<10} {:<10}".format("text", "expected", "A", "H"))
    for row in challenge:
        if row["category"] == "low_information":
            continue
        a = models["A"]["model"].predict([row["text"]])[0]
        h = models["H"]["model"].predict([row["text"]])[0]
        print("  {:<34} {:<9} {:<10} {:<10}".format(
            row["text"][:34], row["expected"],
            a + ("" if a == row["expected"] else " x"),
            h + ("" if h == row["expected"] else " x")))

    print("\n  LOW-INFORMATION INPUT - top probability (H flags below {:.2f})".format(
        config.LOW_CONFIDENCE_THRESHOLD))
    for row in challenge:
        if row["category"] != "low_information":
            continue
        a = models["A"]["model"].predict_proba([row["text"]])[0].max()
        h = models["H"]["model"].predict_proba([row["text"]])[0].max()
        print("  {:<34} A {:.2f}   H {:.2f}{}".format(
            row["text"], a, h, "  (flagged)" if h < config.LOW_CONFIDENCE_THRESHOLD else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

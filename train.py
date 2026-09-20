"""
Initial model training - creates version 1.

Responsibilities (kept deliberately separate from retrain.py):

    1. load the dataset          (seed CSV + any collected feedback)
    2. preprocess the text       (handled inside the pipeline)
    3. train the model
    4. evaluate the model
    5. save the model artifact   (to the PERSISTENT store, not the workspace)
    6. save the metrics
    7. create the first model version record, marked 'production'

Run it directly:

    python train.py            train v1 (does nothing if a model already exists)
    python train.py --force    train a new version even if one exists
    python train.py --detailed print the per-class report too

This script is idempotent by default, which is what lets the Jenkins
"Bootstrap Model" stage run on every build without creating duplicate models.
"""

import argparse
import os
import sys

import joblib

import config
import database
import dataset
import evaluate


def save_model(model, version):
    """
    Write the trained pipeline to the PERSISTENT model store.

    This is C:\\mlops-data\\models by default - deliberately outside the Git
    repository and outside the Jenkins workspace, so that model_v1.pkl,
    model_v2.pkl ... survive a workspace clean or a container being deleted.
    """
    config.ensure_directories()
    filename = config.model_filename(version)
    destination = os.path.join(config.MODEL_STORE, filename)
    joblib.dump(model, destination)
    return destination


def train(force=False, detailed=False):
    """Train version 1 of the model and register it as production."""
    config.describe()
    database.init_db()

    existing = database.get_production_model()
    if existing and not force:
        print("A production model already exists: v{} (accuracy {:.4f})".format(
            existing["version"], existing["accuracy"]))
        print("Nothing to do. Use --force to train a new version anyway,")
        print("or run retrain.py to go through the auto-retraining pipeline.")
        return 0

    # ---- 1. Load the training pool ---------------------------------------
    print("\n[1/6] Loading training data...")
    X_train, y_train, summary = dataset.load_training_data()
    print("      Pool samples     : {}".format(summary["pool_samples"]))
    print("      Feedback samples : {}".format(summary["feedback_samples"]))
    print("      Total training   : {}".format(summary["total_samples"]))

    # ---- 2. Load the fixed holdout set ------------------------------------
    print("\n[2/6] Loading the fixed holdout set...")
    X_test, y_test = dataset.load_holdout_data()
    print("      Holdout samples  : {} (frozen, never trained on)".format(len(X_test)))

    # ---- 3. Train --------------------------------------------------------
    print("\n[3/6] Training TF-IDF + Logistic Regression pipeline...")
    model = dataset.build_pipeline()
    model.fit(X_train, y_train)
    vocabulary_size = len(model.named_steps["tfidf"].vocabulary_)
    print("      Vocabulary size  : {} features".format(vocabulary_size))

    # ---- 4. Evaluate -----------------------------------------------------
    print("\n[4/6] Evaluating on the fixed holdout set...")
    metrics, predictions = evaluate.evaluate_model(model, X_test, y_test)
    evaluate.print_metrics_report(metrics, "BASELINE MODEL EVALUATION (fixed holdout)")
    if detailed:
        evaluate.print_detailed_report(y_test, predictions)

    # ---- 5. Save the artifact -------------------------------------------
    version = database.get_next_version()
    print("\n[5/6] Saving model artifact...")
    model_path = save_model(model, version)
    print("      Version          : v{}".format(version))
    print("      Saved to         : {}".format(model_path))

    # ---- 6. Register the version ----------------------------------------
    print("\n[6/6] Registering model version in the database...")
    database.insert_model_version(
        version=version,
        model_path=model_path,
        metrics=metrics,
        training_samples=len(X_train),
        status=config.STATUS_PRODUCTION,
        notes="Initial baseline model trained by train.py",
    )
    database.promote_to_production(version)

    # Any feedback used for training is now marked as consumed, so the
    # "new samples since last training" counter starts from zero.
    consumed = database.mark_all_samples_used()
    if consumed:
        print("      Marked {} feedback samples as used in training".format(consumed))

    print("\n" + "-" * 58)
    print(" MODEL v{} IS NOW THE PRODUCTION MODEL".format(version))
    print("-" * 58)

    # Printed so the real baseline can be used to choose MIN_ACCURACY
    # honestly, instead of inventing a number.
    if config.MIN_ACCURACY <= 0:
        print("\nNOTE: MIN_ACCURACY is not frozen yet in config.py.")
        print("      Observed baseline accuracy is {:.4f}.".format(metrics["accuracy"]))
        print("      Choose the quality gate threshold from this real figure.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Train the initial sentiment model.")
    parser.add_argument("--force", action="store_true",
                        help="train a new version even if a production model exists")
    parser.add_argument("--detailed", action="store_true",
                        help="print the per-class report and confusion matrix")
    args = parser.parse_args()
    return train(force=args.force, detailed=args.detailed)


if __name__ == "__main__":
    sys.exit(main())

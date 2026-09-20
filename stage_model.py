"""
Stage the production model into the Jenkins workspace for the Docker build.

WHY THIS EXISTS
---------------
Trained models live in the PERSISTENT store, outside the Git repository and
outside the Jenkins workspace:

    C:\\mlops-data\\models\\model_v1.pkl
                          model_v2.pkl
                          ...

That is deliberate - a Jenkins workspace clean or a deleted container must
never destroy the model history.

But Experiment 9's deployment approach bakes the model INTO the Docker image,
so the image is a self-contained deployment artifact. Docker can only copy
files from the build context, which is the workspace.

So immediately before `docker build`, this script copies the one model that
is currently in production out of the persistent store and into the workspace
models/ folder, where the Dockerfile's `COPY . .` will pick it up.

    PERSISTENT STORE                 WORKSPACE                DOCKER IMAGE
    C:\\mlops-data\\models\\      ->   <workspace>\\models\\  ->  /app/models/
    (authoritative, all versions)   (just the live one)      (baked in)

Usage:
    python stage_model.py            stage the current production model
    python stage_model.py --clean    empty the workspace models folder first
"""

import argparse
import os
import shutil
import sys

import config
import database


def clean_staging_dir():
    """Remove any previously staged models from the workspace."""
    if not os.path.isdir(config.STAGED_MODEL_DIR):
        return 0
    removed = 0
    for name in os.listdir(config.STAGED_MODEL_DIR):
        if name.endswith(".pkl"):
            os.remove(os.path.join(config.STAGED_MODEL_DIR, name))
            removed += 1
    return removed


def stage(clean=False):
    """Copy the production model from the persistent store into the workspace."""
    config.ensure_directories()
    database.init_db()

    print("=" * 58)
    print(" STAGE PRODUCTION MODEL FOR DOCKER BUILD")
    print("=" * 58)

    record = database.get_production_model()
    if not record:
        print("  ERROR: no production model exists.")
        print("  Run 'python train.py' to create the baseline model first.")
        return 1

    filename = config.model_basename(record["model_path"])
    source = os.path.join(config.MODEL_STORE, filename)

    if not os.path.exists(source):
        # Fall back to the search path in case it was staged but not stored.
        found = config.resolve_model_path(filename)
        if not found:
            print("  ERROR: production model file not found: {}".format(source))
            return 1
        source = found

    if clean:
        removed = clean_staging_dir()
        if removed:
            print("  Cleaned {} previously staged model file(s)".format(removed))

    destination = os.path.join(config.STAGED_MODEL_DIR, filename)
    os.makedirs(config.STAGED_MODEL_DIR, exist_ok=True)
    shutil.copyfile(source, destination)

    size_kb = os.path.getsize(destination) / 1024.0

    print("  Production version   : v{}".format(record["version"]))
    print("  Accuracy             : {:.4f}".format(record["accuracy"]))
    print("  Source (persistent)  : {}".format(source))
    print("  Staged to (workspace): {}".format(destination))
    print("  Size                 : {:.1f} KB".format(size_kb))
    print("")
    print("  The Docker image will bake in model v{}.".format(record["version"]))
    print("=" * 58)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Copy the production model into the workspace for Docker.")
    parser.add_argument("--clean", action="store_true",
                        help="remove previously staged .pkl files first")
    args = parser.parse_args()
    return stage(clean=args.clean)


if __name__ == "__main__":
    sys.exit(main())

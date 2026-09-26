"""
One-time preparation of the pretrained sentence encoder.

The encoder (all-MiniLM-L6-v2) is a frozen, versioned artifact - just like a
model_vN.pkl, except that it never changes between model versions. This script
puts it into the PERSISTENT store once:

    C:\\mlops-data\\encoders\\all-MiniLM-L6-v2\\
        model.safetensors, tokenizer.json, config.json, ...   the files
        encoder_manifest.json                                  what they are

From there stage_model.py copies it into the workspace before `docker build`,
so the image contains the byte-identical encoder the classifier was trained
against. Nothing is downloaded at prediction time - ever.

This is the ONLY step in the project that may need the internet, and only if
the encoder is not already in the local Hugging Face cache. The exact revision
is pinned in config.py, so every machine prepares identical files.

Usage:
    python prepare_encoder.py            prepare (does nothing if already done)
    python prepare_encoder.py --verify   re-hash the files and check the manifest
    python prepare_encoder.py --force    rebuild the prepared copy
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import config
import embeddings


def _snapshot_path():
    """
    Get the pinned revision from the Hugging Face cache, downloading it only
    if it is not there already.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(config.ENCODER_NAME,
                                 revision=config.ENCODER_REVISION,
                                 local_files_only=True)
    except Exception:                                   # noqa: BLE001
        print("  not in the local cache - downloading revision {} once".format(
            config.ENCODER_REVISION[:10]))
        return snapshot_download(config.ENCODER_NAME,
                                 revision=config.ENCODER_REVISION)


def prepare(force=False):
    target = os.path.join(config.ENCODER_STORE, config.ENCODER_DIRNAME)
    manifest_path = os.path.join(target, config.ENCODER_MANIFEST)

    print("=" * 58)
    print(" PREPARE SENTENCE ENCODER")
    print("=" * 58)
    print("  Encoder   : {}".format(config.ENCODER_NAME))
    print("  Revision  : {}".format(config.ENCODER_REVISION))
    print("  Store     : {}".format(target))

    if os.path.exists(manifest_path) and not force:
        with open(manifest_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("revision") == config.ENCODER_REVISION:
            print("")
            print("  Already prepared (weights {}...). Nothing to do.".format(
                existing["weights_sha256"][:12]))
            print("=" * 58)
            return 0
        print("  Prepared copy is a different revision - rebuilding.")

    try:
        source = _snapshot_path()
    except Exception as error:                          # noqa: BLE001
        # Most often: no internet. Say so plainly in the build log instead of
        # ending on an HTTP library's traceback.
        print("")
        print("  FAILED - the encoder is not prepared in this store, and it")
        print("  could not be downloaded:")
        print("    {}: {}".format(type(error).__name__, str(error)[:200]))
        print("  Run 'python prepare_encoder.py' once with internet access.")
        print("  After that, no step needs the network.")
        print("=" * 58)
        return 1

    if os.path.exists(target):
        shutil.rmtree(target)
    # symlinks=False copies the real files. The Hugging Face cache stores its
    # files as links into a shared blob folder, and a link would not survive
    # being copied into a Docker image.
    shutil.copytree(source, target, symlinks=False)

    weights = os.path.join(target, "model.safetensors")
    weights_sha256 = embeddings.sha256_of(weights)

    # Load it once from the new folder to prove it works, and read its size.
    embeddings.clear_caches()
    encoder = embeddings.load_encoder(target)
    dimension = int(encoder.get_embedding_dimension())

    files = sorted(
        os.path.relpath(os.path.join(root, name), target).replace("\\", "/")
        for root, _, names in os.walk(target) for name in names)
    total_bytes = sum(os.path.getsize(os.path.join(target, f)) for f in files)

    manifest = {
        "name": config.ENCODER_NAME,
        "revision": config.ENCODER_REVISION,
        "weights_file": "model.safetensors",
        "weights_sha256": weights_sha256,
        "dimension": dimension,
        "total_bytes": total_bytes,
        "files": files,
        "prepared_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    embeddings.clear_caches()

    print("")
    print("  Files     : {} ({:.1f} MB)".format(len(files), total_bytes / 1e6))
    print("  Dimension : {}".format(dimension))
    print("  SHA-256   : {}".format(weights_sha256))
    print("")
    print("  Encoder prepared.")
    print("=" * 58)
    return 0


def verify():
    """Re-hash the weights and confirm they still match the manifest."""
    target = config.resolve_encoder_dir()
    if not target:
        print("ERROR: no prepared encoder found. Run 'python prepare_encoder.py'.")
        return 1
    with open(os.path.join(target, config.ENCODER_MANIFEST), encoding="utf-8") as handle:
        manifest = json.load(handle)

    actual = embeddings.sha256_of(os.path.join(target, manifest["weights_file"]))
    missing = [f for f in manifest["files"] if not os.path.exists(os.path.join(target, f))]

    print("Encoder folder : {}".format(target))
    print("Revision       : {}".format(manifest["revision"]))
    print("Expected SHA   : {}".format(manifest["weights_sha256"]))
    print("Actual SHA     : {}".format(actual))
    print("Missing files  : {}".format(missing or "none"))

    if actual != manifest["weights_sha256"] or missing:
        print("\nVERIFICATION FAILED - the encoder files have changed or are incomplete.")
        return 1
    print("\nVERIFICATION PASSED - encoder intact.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Prepare the frozen sentence encoder.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the prepared copy even if it exists")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash the encoder and check it against its manifest")
    args = parser.parse_args()
    if args.verify:
        return verify()
    return prepare(force=args.force)


if __name__ == "__main__":
    sys.exit(main())

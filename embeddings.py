"""
The frozen pretrained sentence encoder, packaged as a scikit-learn transformer.

WHY IT IS A TRANSFORMER
-----------------------
The whole MLOps pipeline - training, evaluation, the quality gate, promotion,
staging and the Docker image - only ever talks to the model through
scikit-learn's .fit() / .predict() and joblib. Wrapping the encoder as a
transformer lets it slot into the existing Pipeline, so none of that machinery
had to change.

WHAT IT DOES
------------
It turns a sentence into a 384-number vector (an "embedding") using the
pretrained all-MiniLM-L6-v2 network. Sentences used in similar contexts get
similar vectors, so a classifier trained on these vectors can respond to
wording it never saw during training, provided the encoder places it near
familiar examples. It is a pretrained statistical representation, not
understanding, and it has known weak spots - negation in particular is only
weakly represented.

WHY IT NEVER GETS PICKLED
-------------------------
The encoder is FROZEN: it is never trained, and it is identical for every
model version. Pickling its 91 MB of weights into every model_vN.pkl would
waste space and hide which encoder a model depends on. Instead:

  * the weights live once, in the encoder artifact folder (config.py 1b)
  * this transformer loads them on first use and shares them process-wide
  * at .fit() it records the encoder's SHA-256 fingerprint as fingerprint_
  * at .transform() it refuses to run if the encoder it finds has a
    different fingerprint

That last check is a train/serve skew guard. A model trained against one
encoder can never silently be served with another.
"""

import hashlib
import json
import os
import threading

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import config


class EncoderNotFoundError(RuntimeError):
    """The prepared encoder folder is missing."""


class EncoderMismatchError(RuntimeError):
    """The encoder found differs from the one the model was trained with."""


# Process-wide state. The lock keeps it safe when Flask serves requests on
# several threads at once.
_lock = threading.Lock()
_encoders = {}            # encoder folder -> loaded SentenceTransformer
_manifests = {}           # encoder folder -> manifest dict
_vector_cache = {}        # (fingerprint, text) -> embedding
_VECTOR_CACHE_LIMIT = 50000


def _offline():
    """
    Forbid network access for the Hugging Face libraries.

    The encoder is always loaded from a local folder, so a network call could
    only mean something is wrong. Failing loudly beats silently downloading a
    different model at prediction time.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def sha256_of(path, chunk_size=1 << 20):
    """SHA-256 of a file, read in chunks so large weight files are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(encoder_dir):
    """The manifest prepare_encoder.py wrote beside the encoder files."""
    with _lock:
        if encoder_dir not in _manifests:
            with open(os.path.join(encoder_dir, config.ENCODER_MANIFEST),
                      encoding="utf-8") as handle:
                _manifests[encoder_dir] = json.load(handle)
        return _manifests[encoder_dir]


def locate_encoder():
    """Find the prepared encoder, or raise with instructions."""
    encoder_dir = config.resolve_encoder_dir()
    if encoder_dir is None:
        raise EncoderNotFoundError(
            "The sentence encoder '{}' is not prepared. Looked in: {}\n"
            "Run 'python prepare_encoder.py' once.".format(
                config.ENCODER_DIRNAME, config.ENCODER_SEARCH_PATH))
    return encoder_dir


def load_encoder(encoder_dir):
    """Load the encoder once per process and share it."""
    with _lock:
        if encoder_dir not in _encoders:
            _offline()
            from sentence_transformers import SentenceTransformer
            _encoders[encoder_dir] = SentenceTransformer(encoder_dir, device="cpu")
        return _encoders[encoder_dir]


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer: list of texts -> matrix of sentence embeddings.

    The embeddings are L2-normalised, so every vector has length 1 and sits
    on the same scale as a TF-IDF row. That is what lets the two feature sets
    be placed side by side for one classifier.
    """

    def __init__(self, encoder_name=config.ENCODER_NAME, batch_size=64):
        self.encoder_name = encoder_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """
        Nothing is learned - the encoder is frozen.

        What IS recorded is which encoder was used, so the saved model can
        later prove it is being served with the same one.
        """
        encoder_dir = locate_encoder()
        manifest = read_manifest(encoder_dir)
        if manifest.get("name") != self.encoder_name:
            raise EncoderMismatchError(
                "Configured encoder '{}' but the prepared one is '{}'.".format(
                    self.encoder_name, manifest.get("name")))
        self.fingerprint_ = manifest["weights_sha256"]
        self.dimension_ = int(manifest["dimension"])
        self.revision_ = manifest["revision"]
        return self

    def transform(self, X):
        encoder_dir = locate_encoder()
        manifest = read_manifest(encoder_dir)

        expected = getattr(self, "fingerprint_", None)
        if expected is not None and manifest["weights_sha256"] != expected:
            raise EncoderMismatchError(
                "This model was trained with encoder {}... but the encoder at "
                "{} is {}.... Refusing to predict with a different "
                "encoder.".format(expected[:12], encoder_dir,
                                  manifest["weights_sha256"][:12]))

        texts = ["" if text is None else str(text) for text in X]
        dimension = int(manifest["dimension"])
        if not texts:
            return np.zeros((0, dimension), dtype=np.float32)

        key = manifest["weights_sha256"]

        # The encoder is deterministic, so a sentence's vector never changes.
        # Caching means retraining does not re-encode the whole training pool.
        # The lookup is copied out under the lock, so another thread clearing
        # the cache cannot pull a vector away half-way through this request.
        with _lock:
            found = {t: _vector_cache.get((key, t)) for t in dict.fromkeys(texts)}
        missing = [t for t, vector in found.items() if vector is None]

        if missing:
            encoder = load_encoder(encoder_dir)
            vectors = encoder.encode(missing, batch_size=self.batch_size,
                                     normalize_embeddings=True,
                                     convert_to_numpy=True,
                                     show_progress_bar=False)
            with _lock:
                if len(_vector_cache) + len(missing) > _VECTOR_CACHE_LIMIT:
                    _vector_cache.clear()
                for text, vector in zip(missing, vectors):
                    vector = vector.astype(np.float32)
                    found[text] = vector
                    _vector_cache[(key, text)] = vector

        return np.vstack([found[t] for t in texts])


def warm_up():
    """
    Load the encoder now rather than on the first user request.

    Loading takes about 1.5 seconds. Doing it at startup keeps the first
    prediction as fast as every later one.
    """
    encoder_dir = locate_encoder()
    load_encoder(encoder_dir)
    return encoder_dir


def clear_caches():
    """Forget every loaded encoder and cached vector. Used by the tests."""
    with _lock:
        _encoders.clear()
        _manifests.clear()
        _vector_cache.clear()

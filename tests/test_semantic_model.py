"""
Tests for the semantic (sentence-embedding) half of the hybrid model.

These check BEHAVIOUR that the embeddings were introduced for, and the
engineering that keeps them safe to deploy:

    * the encoder loads from the prepared artifact, never the network
    * the classifier artifact stays small - the 91 MB encoder is not pickled
    * a model refuses to run with a different encoder than it was trained on
    * the model responds to wording that TF-IDF has never seen
    * negation is handled better than TF-IDF, but NOT solved - asserted
      honestly, not forced

The expected labels below come from reading the text alone. They are
challenge examples, not training data and not the frozen holdout, and nothing
in the model was tuned against them.
"""

import io
import json
import os
import shutil

import joblib
import numpy as np
import pytest

import config
import dataset
import embeddings
import model_loader


# ---------------------------------------------------------------------------
# THE ENCODER ARTIFACT
# ---------------------------------------------------------------------------

def test_encoder_is_prepared_with_a_manifest():
    """The prepared encoder carries a manifest pinning exactly what it is."""
    encoder_dir = config.resolve_encoder_dir()
    assert encoder_dir is not None

    with open(os.path.join(encoder_dir, config.ENCODER_MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["name"] == config.ENCODER_NAME
    assert manifest["revision"] == config.ENCODER_REVISION
    assert manifest["dimension"] == 384
    assert len(manifest["weights_sha256"]) == 64


def test_encoder_weights_match_their_fingerprint():
    """The weights on disk are byte-identical to the ones recorded."""
    encoder_dir = config.resolve_encoder_dir()
    manifest = embeddings.read_manifest(encoder_dir)
    actual = embeddings.sha256_of(os.path.join(encoder_dir, manifest["weights_file"]))
    assert actual == manifest["weights_sha256"]


def test_encoder_runs_without_network(monkeypatch):
    """
    Loading never contacts the internet. Offline mode is forced, so any
    attempted download would raise rather than silently fetch a model.
    """
    embeddings._offline()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    vectors = embeddings.SentenceEmbedder().fit(["x"]).transform(["an offline sentence"])
    assert vectors.shape == (1, 384)


def test_embeddings_are_normalised_and_deterministic():
    """Unit-length vectors, and the same sentence always gives the same one."""
    embedder = embeddings.SentenceEmbedder().fit(["x"])
    first = embedder.transform(["the delivery was quick"])
    embeddings.clear_caches()
    second = embeddings.SentenceEmbedder().fit(["x"]).transform(["the delivery was quick"])

    assert first.shape == (1, 384)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-4
    assert np.allclose(first, second, atol=1e-5)


def test_similar_sentences_have_similar_embeddings():
    """
    The property the whole change relies on: sentences used in similar
    contexts sit close together, even with no words in common.
    """
    embedder = embeddings.SentenceEmbedder().fit(["x"])
    v = embedder.transform([
        "the staff were phenomenal",        # unseen positive wording
        "wonderful and helpful staff",      # familiar positive wording
        "the product arrived broken",       # unrelated negative
    ])
    close = float(v[0] @ v[1])
    far = float(v[0] @ v[2])
    assert close > far


# ---------------------------------------------------------------------------
# SERIALISATION
# ---------------------------------------------------------------------------

def test_model_artifact_stays_small(baseline_pipeline):
    """
    A model version is the classifier only. The 91 MB encoder weights must
    NOT be pickled into every model_vN.pkl.
    """
    pipeline, _, _ = baseline_pipeline
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    assert buffer.tell() < 200_000, "the pickle has swallowed the encoder weights"


def test_model_records_its_encoder_fingerprint(baseline_pipeline):
    """At training time the model notes exactly which encoder it used."""
    pipeline, _, _ = baseline_pipeline
    embedder = dict(pipeline.named_steps["features"].transformer_list)["embedding"]
    manifest = embeddings.read_manifest(config.resolve_encoder_dir())
    assert embedder.fingerprint_ == manifest["weights_sha256"]
    assert embedder.revision_ == config.ENCODER_REVISION
    assert embedder.dimension_ == 384


def test_serialised_model_round_trips(baseline_pipeline):
    """Saved and reloaded, the model gives identical probabilities."""
    pipeline, _, _ = baseline_pipeline
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    reloaded = joblib.load(buffer)

    texts = ["the staff were phenomenal", "a complete waste of money"]
    assert np.allclose(pipeline.predict_proba(texts), reloaded.predict_proba(texts))


def test_model_refuses_a_different_encoder(baseline_pipeline):
    """
    Train/serve skew guard: a model paired with an encoder whose fingerprint
    differs must refuse to predict rather than serve wrong features.
    """
    pipeline, _, _ = baseline_pipeline
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    tampered = joblib.load(buffer)
    dict(tampered.named_steps["features"].transformer_list)["embedding"].fingerprint_ = "0" * 64

    with pytest.raises(embeddings.EncoderMismatchError):
        tampered.predict(["any text at all"])


def test_missing_encoder_is_reported_clearly(monkeypatch, tmp_path):
    """With no prepared encoder, the error says exactly what to run."""
    monkeypatch.setattr(config, "ENCODER_SEARCH_PATH", [str(tmp_path)])
    with pytest.raises(embeddings.EncoderNotFoundError, match="prepare_encoder.py"):
        embeddings.locate_encoder()


def test_loader_reports_a_missing_encoder_without_crashing(production_model, monkeypatch, tmp_path):
    """
    If the encoder vanishes, the app degrades gracefully: no model is served
    and /health can still report why.
    """
    monkeypatch.setattr(config, "ENCODER_SEARCH_PATH", [str(tmp_path)])
    embeddings.clear_caches()
    model_loader._cached_model = None
    model_loader._cached_version = None

    assert model_loader.is_ready() is False
    assert "prepare_encoder.py" in model_loader.get_last_error()


def test_staging_copies_the_encoder_once(production_model):
    """
    stage_model.py puts the encoder beside the model for the Docker build,
    verifies it, and skips the 91 MB copy when it is already staged.
    """
    import stage_model

    assert stage_model.stage_encoder() == 0
    staged = os.path.join(config.STAGED_ENCODER_DIR, config.ENCODER_DIRNAME)
    manifest = embeddings.read_manifest(config.resolve_encoder_dir())
    assert embeddings.sha256_of(os.path.join(staged, "model.safetensors")) == \
        manifest["weights_sha256"]

    stamp = os.path.getmtime(os.path.join(staged, "model.safetensors"))
    assert stage_model.stage_encoder() == 0
    assert os.path.getmtime(os.path.join(staged, "model.safetensors")) == stamp
    shutil.rmtree(config.STAGED_ENCODER_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# THE MODEL ITSELF
# ---------------------------------------------------------------------------

def test_production_pipeline_is_the_hybrid(baseline_pipeline):
    """Both feature sets are present, feeding one classifier."""
    pipeline, _, _ = baseline_pipeline
    names = [name for name, _ in pipeline.named_steps["features"].transformer_list]
    assert names == ["tfidf", "embedding"]
    assert "hybrid" in dataset.describe_model(pipeline)


def test_hybrid_meets_the_measured_holdout_accuracy(baseline_pipeline):
    """
    On the frozen holdout the hybrid scored 0.9583, against 0.9167 for the
    TF-IDF-only model it replaces. This guards that measured result.
    """
    _, metrics, _ = baseline_pipeline
    assert metrics["accuracy"] >= 0.95


@pytest.mark.parametrize("text,expected", [
    ("the staff were phenomenal", "positive"),
    ("an absolutely stellar experience", "positive"),
    ("a delightful and seamless purchase", "positive"),
    ("utterly infuriating ordeal", "negative"),
    ("the product is garbage", "negative"),
    ("this was a nightmare", "negative"),
])
def test_unseen_vocabulary(baseline_pipeline, text, expected):
    """
    Wording absent from the TF-IDF vocabulary. TF-IDF alone has nothing to go
    on for these; the embeddings still place them near familiar examples.
    """
    pipeline, _, _ = baseline_pipeline
    tfidf = dict(pipeline.named_steps["features"].transformer_list)["tfidf"]
    unseen = [w for w in tfidf.build_analyzer()(text) if w not in tfidf.vocabulary_]
    assert unseen, "this example should contain at least one unseen word"
    assert pipeline.predict([text])[0] == expected


def test_all_unseen_words_no_longer_collapse_to_one_answer(baseline_pipeline):
    """
    With TF-IDF alone, any fully-unseen text became the zero vector and got
    the SAME probabilities every time. The embeddings give each its own.
    """
    pipeline, _, _ = baseline_pipeline
    probabilities = pipeline.predict_proba([
        "utterly infuriating ordeal",
        "phenomenal stellar magnificent",
    ])
    assert not np.allclose(probabilities[0], probabilities[1], atol=0.05)
    assert pipeline.classes_[probabilities[0].argmax()] == "negative"
    assert pipeline.classes_[probabilities[1].argmax()] == "positive"


def test_negation_moves_away_from_the_wrong_polarity(baseline_pipeline):
    """
    The measured, honest claim: "not good" is no longer read as POSITIVE
    (the TF-IDF-only production model said positive, 0.53). It is NOT claimed
    that it becomes negative - sentence embeddings represent negation only
    weakly, and the hybrid reads it as neutral.
    """
    pipeline, _, _ = baseline_pipeline
    for text in ("not good", "not excellent"):
        assert pipeline.predict([text])[0] != "positive", text


def test_explicit_negation_with_clear_wording(baseline_pipeline):
    """A full negated sentence is read correctly."""
    pipeline, _, _ = baseline_pipeline
    assert pipeline.predict(["I did not like it"])[0] == "negative"


# ---------------------------------------------------------------------------
# CONFIDENCE
# ---------------------------------------------------------------------------

def test_predict_detailed_reports_probabilities(production_model):
    """The loader returns a full, consistent probability breakdown."""
    result = model_loader.predict_detailed("excellent service and very helpful staff")
    assert result["sentiment"] == "positive"
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
    assert result["confidence"] == max(result["probabilities"].values())
    assert result["low_confidence"] is False


@pytest.mark.parametrize("text", [
    "something something", "asdf qwerty", "lorem ipsum dolor", "blah blah blah",
])
def test_low_information_input_is_flagged(production_model, text):
    """
    Inputs with little to go on fall under the validation-derived threshold.
    The prediction is still made - the flag is advisory.
    """
    result = model_loader.predict_detailed(text)
    assert result["sentiment"] in config.SENTIMENT_CLASSES
    assert result["low_confidence"] is True
    assert result["confidence"] < config.LOW_CONFIDENCE_THRESHOLD


@pytest.mark.parametrize("text", [
    "excellent service and very helpful staff",
    "terrible service and very rude staff",
    "the staff were phenomenal",
])
def test_clear_input_is_not_flagged(production_model, text):
    """Clear feedback, familiar or not, is not flagged."""
    assert model_loader.predict_detailed(text)["low_confidence"] is False


def test_threshold_is_the_documented_value():
    """0.60 is the rounded 10th percentile of out-of-fold confidence (0.596)."""
    assert config.LOW_CONFIDENCE_THRESHOLD == pytest.approx(0.60)

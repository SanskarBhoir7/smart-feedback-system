# Smart Feedback System with Auto-Retraining Pipeline

**Mini Project Report — DevOps / CI-CD / MLOps**

---

## Abstract

This project implements a feedback-driven MLOps lifecycle in which a machine
learning model improves itself from ordinary application usage, but is never
allowed to get worse.

Users submit written feedback together with a 1–5 star rating. A sentiment
model predicts the sentiment of the text, while the rating independently
supplies the training label — so every submission automatically becomes a
labelled training sample. Once a configurable number of new samples has been
collected, an automated pipeline checks the new feedback for ratings that
contradict their own text, trains a candidate model on the rest, scores it
against a permanently frozen holdout set, and subjects it to a two-condition
quality gate. Only a candidate that clears the gate is versioned, packaged into
a Docker image, deployed, and checked over HTTP by automated tests; one that
fails is recorded as rejected and the existing production model continues to
serve.

The system was built with Flask, SQLite, scikit-learn, sentence-transformers,
Docker, Jenkins and pytest. The original TF-IDF model scored **0.9167** on the
fixed 72-sample holdout. After an evaluation exposed its dependence on exact
vocabulary, it was replaced by a **hybrid** of TF-IDF and a small frozen
sentence encoder. On identical training data it scores **0.9583** (F1 0.9582)
against 0.9167, and handles more negation and unseen-wording probes (8/12 vs
6/12, 9/11 vs 6/11), at the cost of a 2.14 GB image and ~12 ms predictions.
The production model trained on the live data (v4) scores **0.9306**. A new
data-quality layer excluded five contradictory feedback rows from its training
and catches 46 of 50 deliberately poisoned rows; the quality gate still
rejects a poisoned candidate (0.7778) when that layer is switched off.
**221 isolated tests** and **15 deployment integration tests** pass.

Jenkins runs the pipeline every five minutes. In both snapshots of its
retained builds that were inspected, every build that got past Checkout
succeeded (the only two failures could not resolve github.com just after the
machine woke), and two earlier builds carried out full retraining cycles. The revised
pipeline described here has been executed command by command on the
development machine but has not yet run inside Jenkins, because Jenkins builds
from GitHub and the revision has not been pushed.

---

## Problem Statement

A machine learning model deployed once begins to decay. Language drifts,
products change, and the data the model was trained on slowly stops
resembling the data it receives. Nobody notices until the predictions are
visibly poor. Retraining by hand is slow, easy to postpone and easy to forget.

Automating retraining is the obvious answer, and it introduces a worse
problem. A pipeline that deploys whatever it has just trained will eventually
replace a good model with a bad one — particularly when the incoming data is
noisy, unrepresentative, or deliberately poisoned. In a feedback system the
noise is built in: the label comes from the star rating, and nothing stops a
user rating *"Good"* with one star.

This project therefore addresses three requirements at once:

1. retraining must happen automatically, driven by real usage;
2. feedback whose rating contradicts its own text must not silently become a
   wrong training label; and
3. no model may reach production without passing an explicit, measurable
   quality check against a stable yardstick.

---

## Motivation

The supplied CI/CD practical draws the distinction that motivated this work:

> **Traditional CI/CD:** Code change → Build software → Run unit tests → Deploy
> **ML CI/CD (MLOps):** Code/Data change → Retrain model → Evaluate metrics →
> Package artifact → Deploy inference API

and states the safeguard directly:

> "If the new model's accuracy drops below a defined threshold (e.g., less
> than 85%), the pipeline automatically fails and stops, preventing a broken
> or inferior model from reaching production."

That sentence describes a quality gate but stops short of building one. This
project takes it as the starting point and constructs the full loop around
it: the data source that triggers retraining, the frozen yardstick that makes
the accuracy comparison meaningful, the versioning that records what
happened, and the deployment that only occurs when the gate is satisfied.

---

## Objectives

1. Build a Flask application that collects feedback and converts it into
   labelled training data automatically.
2. Train and serve a sentiment classifier through a web API, reporting the
   confidence of every prediction.
3. Detect feedback whose rating contradicts its text — without hand-written
   sentiment rules — and keep it out of training without deleting it.
4. Trigger retraining automatically once a configurable threshold of new
   samples is reached.
5. Evaluate every model version on an identical, permanently frozen holdout
   set so that versions are genuinely comparable.
6. Enforce a quality gate capable of **rejecting** a candidate and protecting
   the production model.
7. Version every model — and the encoder it depends on — and maintain a
   complete, auditable history including rejections and exclusions.
8. Package the approved model into a Docker image as a self-contained,
   offline deployment artifact.
9. Automate the lifecycle in a Jenkins declarative pipeline that ends by
   testing the deployed container over HTTP.

---

## Scope

**In scope.** Feedback collection; three-class sentiment classification with
a small, frozen, pretrained sentence encoder (`all-MiniLM-L6-v2`, 22.7 M
parameters, CPU only); text/rating consistency checks; advisory confidence
flags; automatic retraining on a sample-count threshold; a two-condition
quality gate; model and encoder versioning with a persistent artifact store;
containerised deployment with a mounted database; a Jenkins pipeline with
conditional stages; automated unit, integration and deployment testing.

**Out of scope, deliberately.** Fine-tuning the encoder, large language
models, GPUs, Kubernetes, cloud infrastructure, message queues, distributed
training, authentication, and multi-node orchestration. The project targets a
scale that can be run locally and explained completely in a viva.

---

## Technology Stack

| Layer | Technology | Version | Why |
|---|---|---|---|
| Language | Python | 3.12.6 | matches the practicals |
| Web | Flask | 3.1.3 | used in the Docker and ML deployment practicals |
| Templating | Jinja2 | bundled | ships with Flask |
| Frontend | plain CSS + vanilla JS | — | no framework needed |
| Database | SQLite (`sqlite3`) | stdlib | zero-configuration, file-based |
| ML | scikit-learn | 1.5.1 | used in the CI/CD practical |
| Sentence encoder | sentence-transformers (`all-MiniLM-L6-v2`) | 5.7.0 | small, CPU-friendly, widely used |
| Encoder runtime | PyTorch, CPU-only build | 2.13.0+cpu | the GPU build adds gigabytes this project never uses |
| Encoder support | transformers / tokenizers / huggingface-hub / safetensors | 5.15.0 / 0.22.2 / 1.27.0 / 0.8.0 | pinned sentence-transformers dependencies |
| Persistence | joblib | 1.4.2 | the practicals' `model.pkl` approach |
| Numerics | numpy / scipy | 1.26.4 / 1.14.1 | scikit-learn dependencies |
| Testing | pytest | 8.3.4 | used in the CI/CD practical |
| Containers | Docker | 29.6.2 | Docker practicals |
| CI/CD | Jenkins | 2.568.3 | Jenkins practicals |
| SCM | Git / GitHub | 2.55.0 | Git practical |

Every version is pinned in `requirements.txt` so that a model artifact
trained on the host loads correctly inside the container.

---

## System Architecture

Full diagrams: **[docs/architecture.md](docs/architecture.md)**

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        v                                                          │
   USER submits feedback (text + 1-5 rating)                       │
        │                                                          │
        ├──> model PREDICTS sentiment + confidence ─┐              │
        └──> rating gives the LABEL ────────────────┤              │
                                                    v              │
                                   LABELLED TRAINING DATA          │
                                            │                      │
                              enough new samples? (>= 10)          │
                                       yes  v                      │
                 LAYER 1   DATA-QUALITY CHECK: rating contradicts  │
                           text? -> excluded (kept, not deleted)   │
                                            v                      │
                                   TRAIN CANDIDATE                 │
                                            v                      │
                         EVALUATE on the FIXED 72-row holdout      │
                                            v                      │
                 LAYER 2             QUALITY GATE                  │
                                     /              \              │
                                 FAIL                PASS          │
                                  │                   │            │
                      keep production model   promote to production│
                      block deployment                │            │
                                                      v            │
                                   bake model + encoder into image │
                                                      v            │
                                              deploy container ────┘
                                                      v
                                 integration tests over real HTTP
```

### Layered view

```
   PRESENTATION   templates/ + static/        3 pages, stable element IDs
   APPLICATION    app.py                      routes, validation, rendering
   SERVICE        model_loader.py             loads the ONE production model
                  data_quality.py             text / rating consistency
   MODEL          dataset.py + embeddings.py  the hybrid pipeline
   DATA           database.py                 all SQL, Flask-free, migrations
   PIPELINE       train / evaluate / retrain  offline, run by Jenkins
   CONFIG         config.py                   every threshold and path
```

### Design decisions

**The web application never trains a model.** It only consumes whichever
version is marked `production` in the database. Training and promotion belong
to `train.py` and `retrain.py`, executed by Jenkins. This prevents the running
service from quietly changing the model serving live users.

**Models are baked into the image; the database is mounted.** The image is an
immutable artifact containing the exact approved model and encoder. The
database is runtime state and must survive the container being destroyed and
recreated — which Jenkins does on every deployment.

**The encoder is an artifact, not a download.** It is fetched once, pinned to a
revision, fingerprinted, and moved through the same route as the models:
persistent store → workspace → image. Nothing downloads anything at runtime.

**The `model_versions` table is the single source of truth** for which model
is live. There is no parallel registry file that could drift out of sync.

---

## Data Flow

```
  POST /feedback
      │
      ├─ validate text (3-1000 chars) and rating (1-5)   -> 400 on failure
      │
      ├─ model_loader.predict_detailed(text)  -> "positive", confidence 0.98
      ├─ config.rating_to_sentiment(1)        -> "negative"
      │
      ├─ database.insert_feedback(...)
      │     text, rating, prediction, label, whether they matched,
      │     confidence, low-confidence flag, model version,
      │     used_in_training = 0
      │
      ├─ data_quality: opposite polarity at >= 0.87?  same text, opposite rating?
      │
      └─ render: predicted / confidence / actual / match / version /
                 low-confidence note / inconsistency note / pending counter
```

The rating is the label. When the model disagrees with it, the model is
usually the one recorded as wrong — which is how the system measures its own
live accuracy. But when the model disagrees *strongly and in the opposite
direction*, the rating is the more likely error, and the row is flagged
instead of being trusted.

---

## Database Design

### Table `feedback`

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | |
| `name` | TEXT | optional |
| `feedback_text` | TEXT NOT NULL | the submitted text |
| `rating` | INTEGER NOT NULL | 1–5 |
| `predicted_sentiment` | TEXT | what the model said |
| `actual_sentiment` | TEXT NOT NULL | derived from the rating |
| `prediction_correct` | INTEGER | 0/1, NULL if no model |
| `prediction_confidence` | REAL | top class probability |
| `low_confidence` | INTEGER | 1 if below the low-confidence threshold |
| `quality_flag` | TEXT | why an accepted run excluded the row, else NULL |
| `model_version` | INTEGER | which model predicted |
| `source` | TEXT | `user` / `bad_seed` |
| `used_in_training` | INTEGER | 0 until a candidate using it is accepted |
| `created_at` | TEXT | timestamp |

### Table `model_versions`

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | |
| `version` | INTEGER UNIQUE | 1, 2, 3 … never reused |
| `model_path` | TEXT | location of the `.pkl` |
| `accuracy`, `precision`, `recall`, `f1_score` | REAL | macro-averaged |
| `training_samples` | INTEGER | how much data went in |
| `status` | TEXT | `production` / `archived` / `rejected` |
| `notes` | TEXT | the quality gate's reason, and any exclusions |
| `created_at` | TEXT | timestamp |

### Three decisions worth defending

**`used_in_training` is a flag, not a derived count.** It makes "new samples
since last training" an exact `COUNT(*) WHERE used_in_training = 0` rather
than an inference from a possibly stale number.

**Samples are marked used only when a candidate is ACCEPTED.** If the gate
rejects a candidate the samples stay pending, so the next pipeline run retries
with them. A rejection must never silently consume training data. This is
asserted by `test_rejection_does_not_consume_the_samples`.

**An exclusion is recorded only when a candidate is ACCEPTED, and then it
sticks.** The disagreement check depends on the current model. Without a
record, a later model that happened to be less sure about a text could re-admit
a label an earlier run had judged wrong — which is exactly what would have
happened to *"Good"* rated 1 star (v3 read it as positive at 0.92, v4 only at
0.64). A rejected run records nothing, and a person can clear a flag.

**Migration.** The three new columns are added to an existing database by
`ALTER TABLE ... ADD COLUMN` when `init_db()` finds them missing. No row is
rewritten. The live database was backed up before the migration was applied,
and the old pipeline kept running successfully against the migrated database
(Jenkins builds #320–#323).

---

## ML Methodology

### Problem formulation

Three-class text classification. The rating is mapped to the label:

| Rating | Label |
|---|---|
| 1–2 | negative |
| 3 | neutral |
| 4–5 | positive |

### The original model (v1–v3)

A single scikit-learn `Pipeline`:

```
TfidfVectorizer(ngram_range=(1,1), min_df=2, sublinear_tf=True)
        │
        v
LogisticRegression(C=5.0, class_weight='balanced', max_iter=1000)
```

Wrapping both steps in one pipeline means `joblib` saves the vectorizer and
the classifier as a **single artifact**, matching the practicals' `model.pkl`
and removing any risk of the saved vocabulary drifting out of step with the
model.

Its settings were chosen by 5-fold cross-validation **on the training set
only**, never the holdout set:

| Configuration | CV accuracy |
|---|---|
| unigram, `min_df=2`, `C=5` | **0.9202** ← chosen |
| unigram, `min_df=1`, `C=1` | 0.9168 |
| 1–2 gram, `min_df=1`, `C=10` | 0.9063 |
| 1–2 gram, `min_df=2`, `C=10` | 0.8924 |
| 1–2 gram, `min_df=1`, `C=1` | 0.8855 |

Word pairs such as "not bad" were expected to help but measurably hurt:
bigrams generate far more features than 288 training samples can support, so
the model overfits.

---

## Model Improvement: Adding a Semantic Representation

### What the evaluation exposed

An evaluation of the deployed TF-IDF model (v3) tested five inputs. The
results, re-measured for this report:

| Input | Rating | v3 prediction |
|---|---|---|
| *"not good"* | 1 | **positive** 0.53 |
| *"Good"* | 1 | positive 0.92 |
| *"not good"* | 5 | positive 0.53 |
| *"the experience could be better."* | 4 | neutral 0.46 |
| *"something something"* | — | negative 0.38 |

Two separate weaknesses show here. First, **TF-IDF only knows words it has
seen**: it scores *"not good"* by the weight of *good*, and gives meaningless
text a confident-looking label because nothing in it carries any weight.
Second, **the label pipeline trusted every rating** — *"Good"* at 1 star went
into training as a negative example. The first is a model problem; the second
is a data problem, addressed separately (see *Data Quality*).

### Options considered

| Option | Decision |
|---|---|
| Add hand-written rules for negation words | **Rejected** — keyword rules are brittle, and the brief excluded them |
| Word bigrams (`"not good"` as a feature) | **Rejected** — already measured to hurt (0.9202 → 0.9063 CV) |
| Fine-tune a transformer | **Rejected** — 288 rows are far too few, training would take minutes per run and need much more memory |
| Frozen sentence embeddings + logistic regression | **Measured** |
| TF-IDF **and** embeddings side by side | **Measured, chosen** |

### Selection, by cross-validation on the training pool only

| Candidate | CV accuracy | Holdout accuracy | Negation | Unseen wording |
|---|---|---|---|---|
| TF-IDF only | 0.9202 | 0.9167 | 6 / 12 | 6 / 11 |
| embeddings only | 0.9304 | 0.8472 | 8 / 12 | 8 / 11 |
| **TF-IDF + embeddings** | **0.9411** | **0.9583** | **8 / 12** | **9 / 11** |

Embeddings alone looked best on paper for unfamiliar wording but **lost
7 points on the holdout** — this project's own quality gate would have
rejected them. The hybrid keeps TF-IDF's precision on familiar wording and
adds the encoder's reach to unfamiliar wording.

```
text --+--> TF-IDF (which words appear)            -> sparse vector --+
       |                                                              +--> LogisticRegression
       +--> all-MiniLM-L6-v2 sentence embedding    -> 384 numbers    --+    (C=5, balanced)
            (frozen, pretrained, never fine-tuned)
```

The encoder maps a sentence to 384 numbers such that sentences used in
similar contexts get similar vectors. That is a learned statistical
representation of text, not comprehension: it makes the model less dependent
on exact words, but it does not *understand* language as a person does, and
the limitations below show where it fails.

### The challenge set

`data/challenge_set.csv` holds 31 hand-written probes: 12 negation pairs
(*good / not good, excellent / not excellent, I liked it / I did not like
it …*), 11 sentences written in vocabulary absent from training (*"the staff
were phenomenal"*, *"I feel cheated"*), and 8 near-meaningless inputs. Their
expected labels are a human reading of the text alone. It is **not** the
holdout: nothing is trained or tuned on it and the gate never uses it. It is
far too small to be a benchmark; it exists to show *where* the models differ.

### Formal comparison (identical data, identical metrics)

Both trained on the 288-row training pool and scored on the frozen 72-row
holdout (`python compare_models.py`):

| | TF-IDF + LR (original) | TF-IDF + MiniLM + LR (current) |
|---|---|---|
| Representation | 1-gram TF-IDF | 1-gram TF-IDF + 384-dim sentence embedding |
| Classifier | Logistic Regression | Logistic Regression |
| Accuracy | 0.9167 | **0.9583** |
| Precision (macro) | 0.9196 | **0.9630** |
| Recall (macro) | 0.9167 | **0.9583** |
| F1 (macro) | 0.9145 | **0.9582** |
| Negation probes | 6 / 12 | **8 / 12** |
| Unseen-wording probes | 6 / 11 | **9 / 11** |
| Challenge set overall (23 labelled) | 12 / 23 | **17 / 23** |
| Low-information probes flagged | — | 8 / 8 |
| Training time | **0.011 s** | 0.443 s |
| Inference time (new sentence) | **0.47 ms** | 12.3 ms |
| Model artifact (`model_vN.pkl`) | **10,289 B** | 19,921 B |
| Shared encoder | — | 91.6 MB |
| Docker image | **645 MB** | 2.14 GB |

On the **live** training data (pool + 23 clean feedback rows, the data v4 was
trained on) the ordering holds: TF-IDF 0.9028 accuracy / 0.9136 precision /
0.9028 recall / 0.9001 F1; hybrid 0.9306 / 0.9425 / 0.9306 / 0.9298; negation
5/12 → 8/12; unseen wording 8/11 → 10/11.

Per class on the formal comparison, the hybrid's remaining errors are all
neutral sentences read as positive (neutral recall 0.88; negative and
positive recall 1.00). TF-IDF's were spread across neutral → negative (3),
neutral → positive (2) and positive → neutral (1).

### Where it improved, and where it did not

Item by item, for the two models of the formal comparison above:

| Probe | TF-IDF | Hybrid | Expected |
|---|---|---|---|
| *the staff were phenomenal* | negative ✗ | positive ✓ | positive |
| *I feel cheated* | neutral ✗ | negative ✓ | negative |
| *utterly infuriating ordeal* | neutral ✗ | negative ✓ | negative |
| *I did not like it* | neutral ✗ | negative ✓ | negative |
| *not bad* | negative ✗ | neutral ✓ | neutral |
| *not excellent* | positive ✗ | neutral ✓ | neutral |
| *not good* | neutral ✗ | neutral ✗ | negative |
| *not very good* | positive ✗ | neutral ✗ | negative |
| *I liked it* | neutral ✗ | neutral ✗ | positive |
| *good* | positive ✓ | neutral ✗ | positive |
| *I feel robbed* | neutral ✗ | neutral ✗ | negative |

The hybrid never moved a negated phrase to the *opposite* polarity — its
negation errors are all "neutral" — but it did not learn that *not good* is
negative, and it lost the bare word *good*. No rule was added to correct
either; they are reported as measured.

### Engineering choices that came with it

* **CPU-only PyTorch** from PyTorch's CPU wheel index, instead of the default
  build that bundles GPU libraries.
* **The encoder is never pickled.** `model_vN.pkl` stores the TF-IDF
  vocabulary, the classifier weights and the encoder's SHA-256 fingerprint.
  At load time the model checks the encoder it finds and refuses to run with
  a different one (`EncoderMismatchError`).
* **An embedding cache** means a sentence is encoded once per process; the
  retraining and data-quality passes re-read the same texts repeatedly.
* **The model warms up when it loads**, so the first user request is not the
  one that pays for loading the encoder.

---

## Dataset Preparation

No suitable dataset was supplied, so a **synthetic seed dataset of 360 rows**
was written by hand, balanced at 120 rows per class.

**A first attempt failed and the failure was instructive.** An initial
180-row dataset produced only **0.5833 accuracy**. Analysis showed the cause:
370 of 543 unique words appeared exactly once, so test sentences shared
almost no vocabulary with training sentences. The dataset was rebuilt at 360
rows with deliberate vocabulary recurrence, and accuracy rose to **0.9167**.

In hindsight this was the first sign of the weakness the evaluation later
exposed: the TF-IDF model could only succeed if the test vocabulary had
already been seen. The sentence encoder reduces that dependence.

---

## Fixed Holdout Strategy

This is the methodological core of the project.

```
  data/feedback_seed.csv              360 rows
            │
            │  prepare_data.py — ONE stratified split, random_state=42
      ┌─────┴──────┐
      v            v
 train_pool.csv   holdout.csv
  288 rows         72 rows       <- FROZEN. Never trained on. Never grows.
      │                │
      +  user feedback │
      v                v
  TRAINING DATA    EVALUATION DATA
  (grows)          (fixed forever)
```

### Why it is necessary

If the test set were regenerated as the dataset grew, then v1's accuracy and
v2's accuracy would be measured on **different rows**, and the quality gate
would be comparing two numbers that are not comparable. Worse, feedback
already used for *training* could later appear in the *test* set and inflate
the score — textbook test-set contamination.

New user feedback therefore always becomes **training** data. The yardstick
never moves. The model change did not move it either: the hybrid was chosen
by cross-validation on the training pool, its thresholds were derived from
out-of-fold predictions on the training pool, and the challenge set is a
separate file.

### How it is enforced

* `prepare_data.py` refuses to regenerate the split without `--force`.
* `prepare_data.py --verify` proves the split is a clean partition:
  **0 overlap, 0 rows lost**, stratified 96/96/96 and 24/24/24.
* `tests/test_model_safety.py` asserts automatically that no holdout row ever
  appears in the training data, that the holdout does not grow after
  retraining, and that candidate and production are scored on identical rows
  in identical order.

---

## Model Training

`train.py` creates version 1:

1. load the training pool plus any collected feedback, minus rows the
   data-quality check flags
2. load the fixed holdout set
3. fit the hybrid pipeline
4. evaluate on the holdout
5. save the artifact to the **persistent store**, `C:\mlops-data\models\`
6. register the version in the database as `production`

The script is **idempotent** — if a production model exists it exits 0 without
doing anything, which makes the Jenkins bootstrap stage safe to run on every
build.

### Live model history

| Version | Model | Training rows | Accuracy | Precision | Recall | F1 | How |
|---|---|---|---|---|---|---|---|
| v1 | TF-IDF | 288 | 0.9167 | 0.9196 | 0.9167 | 0.9145 | `train.py` baseline |
| v2 | TF-IDF | 298 | 0.9167 | | | | accepted by the gate, Jenkins build #3 |
| v3 | TF-IDF | 308 | 0.9167 | | | | accepted by the gate, Jenkins build #75 |
| **v4** | **hybrid** | 311 | **0.9306** | **0.9425** | **0.9306** | **0.9298** | accepted by the gate; 5 rows excluded |

v4 was produced by `retrain.py --train --force` (which bypasses only the
10-sample threshold) and `retrain.py --gate`, which re-scored v3 on the same
holdout and accepted v4 on accuracy alone. Neutral remains the hardest class:
v4's neutral recall is 0.79, the five errors all neutral sentences read as
positive.

---

## Model Evaluation

`evaluate.py` is the only place the four metrics are computed, so production
and candidate models are always measured by identical code on identical data.

Averaging is **macro**: each class contributes equally regardless of support.
This prevents a model from looking good by handling the two easy classes and
ignoring neutral. `zero_division=0` prevents a crash if a class is never
predicted.

`random_state=42` is fixed throughout, and the encoder is deterministic, so an
examiner can re-run training and reproduce the figures exactly.
`test_training_is_reproducible` and `test_evaluation_is_reproducible` assert
this.

---

## Data Quality: Text / Rating Inconsistency

### The problem

The label is the rating. A rating that contradicts its text is a wrong label,
and the evaluation found such rows in the live data: *"Good"* rated 1 star
(id 48), and *"not good"* submitted once at 1 star and once at 5 (ids 51, 52).

### Two checks, no keywords

| Check | Flags a row when | Needs a model? |
|---|---|---|
| **Model disagreement** | the production model predicts the opposite polarity to the rating with confidence ≥ 0.87 | yes |
| **Label conflict** | the same normalised text also appears with an opposite-polarity rating | no |

`data_quality.py` contains no sentiment words; a test parses its source and
fails if one appears. Neutral never counts as a contradiction.

### What happens to a flagged row

It is **kept**, shown on the dashboard with its reason, and — while
`EXCLUDE_INCONSISTENT_FEEDBACK` is on (the default) — **left out of
retraining**. An accepted run records the reason in `quality_flag`, so the
exclusion survives later model changes.

### Measured effect

* **Live data:** five rows excluded before v4 was trained — ids 31, 32, 51, 52
  (label conflict) and 48 (the then-production v3 read *"Good"* as positive
  at 0.92). Trained on the same data with those five rows **included**, the
  hybrid scored 0.9167 instead of 0.9306 — one holdout row, so a small
  difference, but in the expected direction.
* **Poisoned data** (50 rows: positive text rated 1, negative text rated 5,
  on a copy of the live database): 46 caught. The four missed were two texts
  submitted twice — *"perfect quality, exactly as described"* (read as
  positive at 0.84, below the threshold) and *"would not recommend, awful
  quality"* (misread as positive at 0.54).

### Limitation, stated plainly

The disagreement check can only vouch for a rating when the model is sure
what the text means. v4 reads a bare *"Good"* as positive at only 0.64, so a
*new* *"Good"* rated 1 star would not be flagged.

---

## Prediction Confidence

Every prediction reports its top class probability. Below 0.60 it is marked
**low confidence**: shown to the user, stored with the feedback, and
**advisory only** — the prediction is still returned, and nothing is blocked
or excluded because of it.

### How both thresholds were derived

`calibrate_thresholds.py` uses **out-of-fold** predictions on the training
pool (5-fold: each prediction comes from a model that never trained on that
row). The holdout is untouched.

```
rows 288 | out-of-fold accuracy 0.9410 | errors 17

LOW CONFIDENCE — how well each signal picks out wrong predictions (AUROC)
    top class probability         0.906   <- used
    similarity to training data   0.694
  10th percentile of top probability: 0.596   (config uses 0.60)
  errors below it: 8 of 17;  low-information probes flagged: 8 of 8

INCONSISTENCY — opposite-polarity errors on CLEAN labels: 6 of 288
  their confidences: 0.556 0.600 0.715 0.733 0.754 0.861
  most confident: 0.861                        (config uses 0.87)
```

**The trade-off.** At 0.60, about one prediction in ten is flagged and about
half the model's errors are caught. Catching most errors would mean flagging
a large share of correct predictions — so the flag stays advisory rather than
blocking. 0.87 sits just above every opposite-polarity mistake the model made
on clean labels: a more confident flip is more likely a bad rating than a
model error. Both are configurable by environment variable.

---

## Auto-Retraining Pipeline

```
  new_samples = COUNT(feedback WHERE used_in_training = 0)

  if new_samples < RETRAIN_THRESHOLD (10):
      exit 0   -> pipeline skips training, build and deployment
  else:
      assemble training data: pool + feedback, minus flagged rows
      train candidate
      evaluate on the FIXED holdout
      apply the quality gate
```

Retraining is split into three subcommands so that each becomes its own
visible Jenkins stage, with state passed between the separate processes
through small files in `build/`:

| Command | Produces |
|---|---|
| `retrain.py --check` | `build/retrain_decision.json` + exit code |
| `retrain.py --train` | `build/candidate_model.pkl`, `build/candidate_training.json` (what was used and excluded) |
| `evaluate.py --candidate` | `build/candidate_metrics.json` |
| `retrain.py --gate` | promotion or rejection + exit code |

### Exit-code contract

A Jenkins stage is a separate process, so Python cannot set a Jenkins
variable. The decision travels by exit code:

| Command | Exit | Meaning |
|---|---|---|
| `--check` | 0 | retraining NOT required |
| `--check` | 10 | retraining IS required |
| `--check` | 1 | error |
| `--gate` | 0 | accepted |
| `--gate` | 1 | rejected — stop the pipeline |

`10` rather than `1` means a genuine Python crash can never be mistaken for a
decision. Verified through `cmd.exe`, the interpreter Jenkins `bat` uses.

---

## Quality Gate

A candidate must satisfy **both** conditions:

| | Condition | Default |
|---|---|---|
| **A** — absolute floor | `accuracy >= MIN_ACCURACY` | 0.85 |
| **B** — no regression | `accuracy >= production − TOLERANCE` | 0.02 |

Condition A alone would permit a slow decline; Condition B prevents a working
model being silently replaced by a worse one. Both models are re-scored on
the same frozen holdout before comparison, so the numbers are valid — even
across model families. The gate was not changed for the new model; it judged
hybrid v4 against TF-IDF v3 on the same terms as every earlier candidate.

### How the threshold was chosen

`MIN_ACCURACY = 0.85` was derived from measurement with the original model:

| Evidence | Value |
|---|---|
| Real baseline accuracy | 0.9167 — leaves 6.7 points of headroom |
| Poisoned model, 40 bad rows | 0.8194 — fails |
| Poisoned model, 50 bad rows | 0.7778 — fails clearly |
| CI/CD practical's own example threshold | 85% |

### Both outcomes, as measured with the current model

Accepted (the live promotion of v4):

```
Candidate Accuracy   : 0.9306
Required Accuracy    : 0.8500
Production Accuracy  : 0.9167   (v3, re-scored on the same holdout)
QUALITY GATE: PASSED
New model version    : v4      Previous version : v3 (archived)
```

Rejected (50 poisoned rows, data-quality filter off, copy of the live
database):

```
Candidate Accuracy   : 0.7778
Required Accuracy    : 0.8500   (MIN_ACCURACY)
Production Accuracy  : 0.9306   (v4, re-scored on the same holdout)
QUALITY GATE: FAILED
Reason: accuracy 0.7778 is below the required 0.8500; accuracy 0.7778
        regresses from production 0.9306
Candidate recorded as v5 with status 'rejected'.
Production model v4 REMAINS ACTIVE.
Deployment is blocked.
```

A rejected candidate is still saved and registered, so the Model page can
**show** the rejection rather than the event disappearing into a log file.

### Two layers against poisoned data

Candidate accuracy with N poisoned rows added to the live data; the gate
needs max(0.85, 0.9306 − 0.02) = 0.9106:

| Bad rows | TF-IDF, no filter (original) | Hybrid, filter off | Hybrid, filter on (default) |
|---|---|---|---|
| 0 | 0.9167 | 0.9167 | 0.9306 |
| 20 | 0.8750 | 0.8333 **FAIL** | 0.9167 (18 caught) |
| 30 | 0.8611 | 0.8194 **FAIL** | 0.9167 (28 caught) |
| 40 | 0.8194 **FAIL** | 0.7917 **FAIL** | 0.9167 (37 caught) |
| 50 | 0.7778 **FAIL** | 0.7778 **FAIL** | 0.9167 (46 caught) |

(The TF-IDF column was measured earlier against a production model at
0.9167, so its gate verdicts used the 0.85 floor.)

Two findings. The hybrid is **more sensitive** to poisoned labels than
TF-IDF was — 20 bad rows already fail the gate. And with the filter on, the
candidate stays within the gate's tolerance at every level tested: the filter
removes most of the poison before it can do damage, and the gate remains the
backstop for what slips through.

---

## Model Versioning

```
  C:\mlops-data\models\          PERSISTENT authoritative store
      model_v1.pkl ... v4.pkl        every version ever produced,
                                     including rejected candidates
  C:\mlops-data\encoders\
      all-MiniLM-L6-v2\              the encoder, once, with
        encoder_manifest.json        name, revision, SHA-256, dimension

  <workspace>\models\            only the LIVE model  } copied by
  <workspace>\encoders\          the encoder          } stage_model.py

  Docker image                   the approved model + encoder, baked in
```

**The encoder is versioned together with the classifier.** Each
`model_vN.pkl` records the SHA-256 of the encoder weights it was trained
with; the manifest records the pinned Hugging Face revision
(`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) and the same SHA-256. A model
cannot silently run on a different encoder. Paths are resolved from
configuration at run time, never taken from the pickle, so the same artifact
works on Windows and in the Linux container.

Version numbers are sequential and never reused. Exactly one row may hold
`status = 'production'`, enforced by `promote_to_production()`, which demotes
the incumbent to `archived` in the same transaction. Older TF-IDF versions
still load and can still be evaluated, so a manual rollback remains possible.

---

## Flask Application

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | feedback form |
| `/feedback` | POST | submit feedback |
| `/dashboard` | GET | statistics, data quality and model status |
| `/model` | GET | model versions and metrics |
| `/predict` | POST | predict without storing (JSON) |
| `/api/stats` | GET | dashboard data as JSON |
| `/api/model` | GET | model data as JSON |
| `/health` | GET | liveness probe |

`app.py` contains **no SQL and no scikit-learn**. Everything is delegated to
`database.py`, `model_loader.py` and `data_quality.py`.

`/predict` now returns the prediction, the model version, the confidence, the
low-confidence flag and all three class probabilities. The result page shows
the confidence and, where they apply, a low-confidence note and an
inconsistency note that says the row will be left out of retraining. The
dashboard adds Data Quality counts and per-row confidence and quality columns.

**Validation.** Text must be 3–1000 characters; rating must be an integer
1–5. Invalid form submissions return 400 with the error displayed and the
typed text preserved. Invalid API requests return 400 with a JSON error.

**Graceful degradation.** `/health` returns **200 even with no model loaded**,
reporting the model state separately — the service being up and a model being
deployed are different facts. `/predict` returns 503. Feedback is still
stored, because its value as training data does not depend on a model
existing.

---

## Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLOPS_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
RUN mkdir -p /data/models
EXPOSE 5000
CMD ["python", "app.py"]
```

### Differences from the Docker practical, and why

| Change | Reason |
|---|---|
| `python:3.12-slim`, not `3.9-slim` | the model is trained on Python 3.12 with scikit-learn 1.5.1; loading that pickle under a different version raises `InconsistentVersionWarning` and can fail |
| `ENV MLOPS_DATA_DIR=/data` | redirects SQLite out of the container's writable layer into the mounted volume |
| `ENV PYTHONUNBUFFERED=1` | otherwise Flask's logs do not reach `docker logs` promptly |
| `ENV HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | the encoder is baked in; any attempt to download fails loudly instead of happening silently |
| volume mount | so collected feedback survives the `rm`/`run` cycle Jenkins performs on every deploy |

### Verified behaviour (current image)

| Check | Result |
|---|---|
| Image size | **2.14 GB** (was 645 MB) |
| of which PyTorch CPU | 750 MB |
| Encoder in image | `/app/encoders/all-MiniLM-L6-v2`, 88 MB, 12 files |
| Model in image | `/app/models/model_v4.pkl` |
| Database baked into image | **None** |
| Offline mode | `HF_HUB_OFFLINE=1` confirmed inside the container |
| Rebuild with cached dependency layers | 11 s |
| Container healthy after `docker run` | 8–14 s |
| Integration tests against the container | **15 passed** |
| Database after tests and redeploys | unchanged — 28 rows, 5 quality flags |

### An architectural finding

A container started **without** the volume reports `model_loaded: false`. The
`.pkl` is in the image, but the *database* holds the record of which version
is production. The image carries the artifact, the database carries the
deployment decision, and the volume is therefore mandatory.

---

## Jenkins CI/CD

Declarative pipeline, twelve stages, Windows `bat` throughout.

```
   cron('H/5 * * * *')
          v
   Checkout -> Install Dependencies -> Bootstrap Encoder and Model -> Run Tests
          │                                                             │
          │                                                tests fail ──┴--> STOP
          v
   Check Retraining Requirement
    ┌─────┴──────────────────────────┐
    │ exit 0                          │ exit 10
    v                                 v
  SKIP everything, finish GREEN   Train Candidate (data-quality filter)
                                      v
                                  Evaluate Candidate
                                      v
                                  QUALITY GATE --FAIL--> STOP (no deploy)
                                      │ PASS
                                      v
                                  Stage Production Model (model + encoder)
                                      v
                                  Build Docker Image (:v${BUILD_NUMBER})
                                      v
                                  Deploy (rm -f, run -d, poll /health)
                                      v
                                  Deployment Integration Tests
```

Changes for the new model: the bootstrap stage runs `prepare_encoder.py`
before `train.py` (idempotent: it only checks an already prepared encoder),
the health check allows 60 attempts because the container takes longer to
start, the pipeline timeout rose from 30 to 45 minutes to cover a first-time
PyTorch install, and the final stage runs `pytest integration_tests`.

### Why a cron trigger rather than Poll SCM

Feedback lives in SQLite, not in Git. Poll SCM watches for **code** changes
and would never notice new **data**. The cron trigger wakes the pipeline to
ask the database whether enough feedback has arrived. This is the concrete
difference between traditional CI/CD and MLOps: the pipeline is triggered by
data as well as by code.

### Why most runs do almost nothing

Below the threshold the pipeline trains nothing, builds nothing and redeploys
nothing — it prints `RETRAINING NOT REQUIRED` and finishes green.

### Evidence from the live Jenkins

Job `Smart-Feedback-CI-CD`, Jenkins 2.568.3, running as a Windows service
under LocalSystem:

| Evidence | What it shows |
|---|---|
| Builds #307–#326 all SUCCESS (34–83 s each); of #513–#532, 18 SUCCESS | the cron trigger, `checkout scm`, the test stage and the exit-code check work in Jenkins |
| Builds #513 and #531 failed at Checkout: `Could not resolve host: github.com`; the next build succeeded each time | the machine's network was not up yet after waking — not a project fault |
| Each logs `Stage "…" skipped due to when conditional` for all seven conditional stages | the `when` logic is evaluated by Jenkins itself |
| Model v2 promoted 20 Sep 23:10; image `smart-feedback:v3` created within a minute | a full cycle — train, gate, stage, build, deploy — ran in build #3 |
| Model v3 promoted 21 Sep 19:46; image `smart-feedback:v75` created within a minute | a second full cycle, build #75 |
| Builds #320–#323 ran after the schema migration | the migration is backward compatible |

Builds #3 and #75 have rotated out of the retained 20, so their console logs
are gone; the evidence for them is the database rows and the image
timestamps.

**Pending:** the revised pipeline (encoder bootstrap, hybrid model, integration
tests) has not yet run inside Jenkins. Every command in it has been run on the
development machine in pipeline order, including a full promotion, staging,
image build, deployment, health check and integration tests.

### Windows adaptations

| Purpose | Linux (practical) | Windows (this project) |
|---|---|---|
| run a step | `sh 'python train.py'` | `bat 'python train.py'` |
| variable | `${BUILD_NUMBER}` | `%BUILD_NUMBER%` |
| tolerate missing container | `docker rm -f x \|\| true` | `bat(returnStatus: true, script: 'docker rm -f x')` |

### Runtime prerequisite

`jenkins_env_check.bat` checks the Jenkins runtime before the pipeline is
debugged: python, git, Docker CLI and daemon access, read/write access to
`C:\mlops-data`, the encoder (`prepare_encoder.py --verify`), the exit-code
contract, the test suite and model staging. All seven checks pass. An earlier
version of this report stated that LocalSystem cannot reach Docker; that was
an assumption, and the Jenkins jobs on this machine disproved it — LocalSystem
finds Python and ran `docker build` successfully.

---

## Automated Verification

### Why Selenium was removed

The project previously ended its pipeline with 15 Selenium browser tests. They
were removed and replaced with pytest suites, for four reasons:

1. **They duplicated coverage.** Every page, field and value the browser
   tests checked can be checked from the rendered HTML, which Flask's test
   client produces without a browser.
2. **They added infrastructure to the Jenkins machine** — Chrome, a matching
   driver, and headless mode because a Windows service cannot open a window —
   for a stage whose purpose is to confirm that the container serves correct
   responses.
3. **They were the slowest and least deterministic part of the pipeline.**
4. **The application has almost no client-side logic** — a character counter
   and a preview button that calls `/predict`.

What was given up is stated honestly: the JavaScript itself (the counter,
and the code that writes the preview response into the page) is no longer
exercised.

### What replaced it

| Suite | Tests | Runs against | Covers |
|---|---|---|---|
| `tests/test_pages.py` | 24 | Flask test client, temp DB | the pages as rendered: form, navigation, result panel, counter, dashboard and model page values matched against the API |
| `integration_tests/test_deployment.py` | 15 | the deployed container, real HTTP | `/health` (hybrid model loaded, version matches the DB), `/predict` (full response, unseen wording, low confidence, invalid input), pages, static assets, 404, and a submission that must land in the real mounted database |

The integration tests restore the database afterwards (highest id recorded
before, everything above it deleted after) — verified: 28 rows before and
after.

### The full test suite

| Suite | Tests | Result |
|---|---|---|
| `tests/test_model.py` | 16 | passed |
| `tests/test_semantic_model.py` | 32 | passed |
| `tests/test_data_quality.py` | 34 | passed |
| `tests/test_api.py` | 19 | passed |
| `tests/test_pages.py` | 24 | passed |
| `tests/test_validation.py` | 28 | passed |
| `tests/test_database.py` | 30 | passed |
| `tests/test_retraining.py` | 24 | passed |
| `tests/test_model_safety.py` | 14 | passed |
| **Isolated total** | **221** | **passed** (~53 s) |
| `integration_tests/test_deployment.py` vs Docker | 15 | passed |

The negation and unseen-vocabulary tests assert the behaviour that was
measured (for example, that *"not good"* moves away from positive), not a
desired answer, and none of them force a result.

---

## Results

All figures below were measured during development. Nothing is estimated.

### Primary results table

| Component | Result |
|---|---|
| Hybrid vs TF-IDF, identical data | 0.9583 vs 0.9167 accuracy, 0.9582 vs 0.9145 F1 |
| Production model v4 (live data) | **0.9306 accuracy**, 0.9298 F1 |
| Candidate accepted | **Verified** — v4 over v3 |
| Candidate rejected | **Verified** — 0.7778, production retained |
| Data-quality filter | **Verified** — 5 live rows, 46 / 50 poisoned rows |
| Pytest | **221 passed** |
| Deployment integration tests | **15 passed** |
| Docker | **Verified** — 2.14 GB, offline, healthy in 8–14 s |
| Container persistence | **Verified** |
| Jenkins below-threshold runs | **Verified in Jenkins** — #307–#326, #513–#532 (2 network failures at Checkout) |
| Jenkins full cycles | **Verified by evidence** — builds #3, #75 |
| **Revised pipeline in Jenkins** | **Pending push** |

### The evaluator's five cases, re-tested

Old = v3 (TF-IDF, production at the time of the evaluation). New = v4
(hybrid, production now). Probabilities are negative / neutral / positive.

| Input | Rating → label | Old prediction | New prediction | Flagged? |
|---|---|---|---|---|
| *"not good"* | 1 → negative | positive 0.53 (0.13 / 0.33 / 0.53) | neutral 0.68 (0.25 / 0.68 / 0.07) | **yes** — label conflict with the 5-star copy; excluded from v4 |
| *"Good"* | 1 → negative | positive 0.92 (0.02 / 0.06 / 0.92) | positive 0.64 (0.07 / 0.29 / 0.64) | **yes** — model disagreement (judged by v3 at 0.92); excluded from v4, recorded |
| *"not good"* | 5 → positive | positive 0.53 — scored "correct" only because a wrong prediction met a wrong rating | neutral 0.68 | **yes** — label conflict; excluded from v4 |
| *"the experience could be better."* | 4 → positive | neutral 0.46 (0.33 / 0.46 / 0.21) | positive 0.70 (0.07 / 0.22 / 0.70) | no |
| *"something something"* | — | negative 0.38 (0.38 / 0.36 / 0.26) | neutral 0.46 (0.38 / 0.46 / 0.16) | **low confidence** |

Read honestly: the new model no longer calls *"not good"* positive, but calls
it neutral, not negative. It still reads *"Good"* as positive — correctly;
the rating was the error, and it no longer reaches training. *"the experience
could be better."* at 4 stars was a genuine submission (id 50) and part of
v4's training data, so v4's positive answer there partly reflects having seen
that exact text; the same model family trained without it answers neutral.
Meaningless input is now flagged instead of receiving a confident-looking
label. The model was not tuned toward any of these five inputs.

### Challenge set, full

| Category | TF-IDF | Hybrid |
|---|---|---|
| Negation (12) | 6 | **8** |
| Unseen wording (11) | 6 | **9** |
| Low information flagged (8) | — | **8** |

---

## Limitations

* **Negation is still weak.** *"not good"* and *"not very good"* are read as
  neutral, *"I liked it"* as neutral; 8 of 12 negation probes right. Sentence
  embeddings represent negation only weakly, and no rule was added to hide it.
* **Bare single words are read less confidently** (*"Good"* 0.64), so the
  disagreement check cannot catch a new *"Good"* rated 1 star.
* **One live feedback text is effectively memorised** (id 50, above).
* **The disagreement check is only as reliable as the model**; 4 of 50
  poisoned rows passed it.
* **Small, synthetic data.** 360 hand-written rows plus 28 real submissions;
  one holdout row is 1.4 accuracy points, and the challenge set is 31
  sentences. Small differences should not be over-read.
* **A heavier deployment.** Image 645 MB → 2.14 GB, prediction ~0.5 → ~12 ms,
  retraining 0.01 → 0.45 s, and importing the encoder libraries adds 15–20 s
  to a cold start on Windows.
* **Single-node deployment**, no orchestration or automated rollback.
* **No authentication.**
* **Threshold-based triggering only** — not drift or live-accuracy based.
* **Shared global Python** across the Jenkins jobs on this machine; another
  job pinning a different Flask version could change this project's
  environment between builds.

---

## Practical Mapping

| Practical | Project component | What is demonstrated |
|---|---|---|
| **Experiment 1** — Explore Git, GitHub Commands and Source Code Management | repository, `.gitignore`, branch strategy | `init`, `add`, `status`, `commit`, `log`, `branch`, `checkout`, `remote add origin`, `push`, `pull` |
| **Experiment 2** — Jenkins installation and setup | Jenkins on `localhost:8080`, `jenkins_env_check.bat` | installing and exploring the CI/CD environment |
| **Experiment 3** — Demonstrate CI/CD using Jenkins | `jenkins_env_check.bat` as a Freestyle job | Freestyle projects, *Execute Windows batch command*, Console Output, non-zero exit codes — and the basis for choosing `bat` over `sh` |
| **Experiment 4** — Explore Docker commands for content management | container lifecycle commands; the pipeline's teardown step | `docker ps`, `stop`, `rm`, `images`, `logs`, image/container distinction |
| **Experiment 5** — Develop a simple containerized application | `Dockerfile`, Flask in a container, `-p 5000:5000` | Dockerfile authoring, building an image, port mapping, running a Flask app in Docker |
| **PS4 — CI/CD Pipeline / ML CI-CD Practical** | `Jenkinsfile`, `train.py`, `tests/`, the **quality gate** | Freestyle vs Pipeline-as-Code, Groovy stages, `Code + Data + Model`, retrain → evaluate → package → deploy, and the quality-gate concept this project is built around |
| **Experiment 9** — Deploying Applications/Models using Docker through Jenkins | `app.py` `/predict`, Deploy stage, `healthcheck.py`, `integration_tests/` | training vs inference, baking the model artifact into the image, `docker rm -f` before `docker run`, testing the live API |

**The Selenium practicals are no longer demonstrated by this project** — see
*Automated Verification* for the reasons and the replacement.

### Note on PS4's experiment number

PS4 is referred to throughout as **"PS4 — CI/CD Pipeline / ML CI-CD
Practical"**. Its experiment number cannot be established from the supplied
material, so none is asserted. Experiment 9 opens by referencing an
*"Exp 8: Training & Testing"*, and PS4 covers exactly that ground — it even
names its Jenkins job `MLOps-LabExp8` — but that is circumstantial and not
sufficient to claim a number.

---

## Learning Outcomes

1. **MLOps is not CI/CD with a model attached.** Traditional pipelines version
   code; an ML pipeline must version code, data *and* model — and here the
   encoder too — and must be triggered by data changes as well as code
   changes.

2. **An automated pipeline without a quality gate is a liability.** The most
   valuable stage in this project is the one that *refuses* to deploy.

3. **Evaluation methodology determines whether metrics mean anything.** The
   fixed holdout set made it possible to compare a hybrid model against a
   TF-IDF model on equal terms, and to let the unchanged gate decide.

4. **A better representation is not free.** The hybrid gained 4 accuracy
   points and generalises further, and cost a 3.3× larger image, 25× slower
   predictions and a new artifact to version. Whether that is worth it is a
   judgement the measurements make possible, not one they make for you.

5. **An improvement on paper can fail the gate.** Embeddings alone looked
   better on unfamiliar wording and would have been rejected for losing
   7 points on the holdout.

6. **The label can be the bug.** Part of what looked like model error was
   bad ratings. Improving the model would not have fixed that; checking the
   data did.

7. **Thresholds should come from validation data.** Out-of-fold predictions
   gave both thresholds a stated derivation — and showed that the obvious
   signal (similarity to training data) was the weaker one.

8. **Decisions made by a model need a memory.** A less confident successor
   model would have re-admitted a label its predecessor rejected. Recording
   accepted exclusions fixed that.

9. **Test assumptions by measuring them.** Bigrams were expected to help and
   measurably hurt; the poisoning calibration showed the new model is *more*
   sensitive to bad labels, not less.

10. **Containers separate the immutable from the mutable.** The image carries
    the model and encoder; the database carries the deployment decision.

11. **Correct your own documentation.** This report once stated that
    LocalSystem could not reach Docker. The Jenkins jobs on this machine
    showed otherwise, and the claim was withdrawn.

---

## Conclusion

The project delivers a working, feedback-driven MLOps lifecycle. Feedback
submitted through a Flask application becomes labelled training data
automatically; contradictory ratings are detected and kept out of training;
crossing a configurable threshold triggers retraining; every candidate is
scored on a permanently frozen holdout set; and a two-condition quality gate
decides whether it is fit to deploy. Approved models are versioned together
with their encoder, baked into an offline Docker image, deployed, and checked
over HTTP.

Replacing the TF-IDF model with a TF-IDF + sentence-embedding hybrid was
justified by measurement on identical data: 0.9583 against 0.9167 on the
frozen holdout, and better results on negation and unseen wording. It is not
a solved problem — negation remains weak and bare words are read with less
confidence — and it made the deployment substantially heavier. Those costs
are stated alongside the gains.

The central lesson is unchanged and reinforced: automated retraining is only
safe when it is guarded. This project now guards it twice — once on the data,
once on the model — and measured both layers doing their job.

---

## Future Scope

1. **Better negation handling** without rules — more negated examples in the
   training pool, or fine-tuning once far more data exists — judged, as
   always, by the unchanged quality gate.
2. **A review queue for flagged rows**, so a person can confirm or clear each
   exclusion from the dashboard.
3. **Drift-based triggering.** Retrain when live prediction accuracy drops or
   the input distribution shifts, rather than on a fixed sample count.
4. **Automated rollback.** Archived artifacts are retained, so redeploying a
   previous version could be a single pipeline stage.
5. **A smaller image**, for example by exporting the encoder to ONNX and
   dropping PyTorch from the runtime.
6. **Human approval gate.** Jenkins `input` could require sign-off before a
   new model goes live.
7. **Real dataset.** Replace the synthetic seed data with genuinely collected
   feedback.
8. **Authentication and rate limiting** before any real deployment.
9. **Run the revised pipeline in Jenkins** — the immediate next step, once
   the revision is pushed.

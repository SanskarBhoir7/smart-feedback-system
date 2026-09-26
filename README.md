# Smart Feedback System with Auto-Retraining Pipeline

A mini project demonstrating a complete, automated MLOps lifecycle: user
feedback becomes labelled training data, inconsistent labels are filtered out,
a threshold triggers retraining, a quality gate decides whether the new model
is good enough to deploy, and only then is it packaged into Docker, deployed,
and checked over HTTP by an automated test suite.

> **Verification status.** Everything in this README has been executed on the
> development machine. Jenkins runs as a Windows service and its builds are
> the evidence for the pipeline (see [§18](#18-jenkins-pipeline-explained)).
> One item is still pending: the pipeline **as revised here** (hybrid model,
> encoder bootstrap, integration tests) has not yet run *inside Jenkins*,
> because Jenkins builds from GitHub and the revision has not been pushed.
> Every command it runs has been executed locally with the same exit codes.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Problem statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Architecture](#4-architecture)
5. [Technology stack](#5-technology-stack)
6. [Folder structure](#6-folder-structure)
7. [Installation](#7-installation)
8. [Database setup](#8-database-setup)
9. [The sentiment model](#9-the-sentiment-model)
10. [Model training](#10-model-training)
11. [Running locally](#11-running-locally)
12. [Feedback workflow](#12-feedback-workflow)
13. [Data quality and confidence](#13-data-quality-and-confidence)
14. [Auto-retraining](#14-auto-retraining)
15. [Quality gate](#15-quality-gate)
16. [Docker](#16-docker)
17. [Jenkins setup](#17-jenkins-setup)
18. [Jenkins pipeline explained](#18-jenkins-pipeline-explained)
19. [Tests](#19-tests)
20. [Deployment integration tests](#20-deployment-integration-tests)
21. [Limitations](#21-limitations)
22. [Troubleshooting](#22-troubleshooting)
23. [Demonstration procedure](#23-demonstration-procedure)
24. [Practical mapping](#24-practical-mapping)

---

## 1. Project overview

Users submit written feedback with a 1–5 star rating. The current production
model predicts the sentiment of the text; the star rating independently gives
the *actual* label. Both are stored, so **every submission becomes a labelled
training sample**.

Once enough new samples accumulate, the pipeline trains a candidate model,
scores it on a frozen holdout set, and puts it through a quality gate. Only a
candidate that clears the gate is packaged into a Docker image and deployed.
A candidate that fails is recorded and discarded, and the existing production
model keeps serving.

The defining feature is not sentiment analysis. It is this loop:

```
FEEDBACK -> DATA-QUALITY CHECK -> TRAINING DATA -> RETRAINING -> EVALUATION
         -> QUALITY GATE -> DEPLOYMENT -> DEPLOYMENT INTEGRATION TESTS
```

## 2. Problem statement

A machine learning model deployed once begins to decay. Real language drifts
away from the data the model was trained on, and nobody notices until the
predictions are visibly poor. Retraining by hand is slow and easy to forget.

But automated retraining introduces a worse danger: a pipeline that deploys
whatever it just trained will eventually replace a good model with a bad one —
especially if the incoming data is noisy or deliberately poisoned. In this
project that is a real risk, not a hypothetical one: the training label comes
from the star rating, so a rating that contradicts its own text (*"Good"*,
1 star) becomes a wrong label.

This project addresses both halves: retraining is automatic, **and** it is
guarded twice — contradictory feedback is filtered out before training, and no
model reaches production without passing an explicit, measurable quality check
against a fixed yardstick.

## 3. Objectives

1. Collect feedback through a web application and turn it into labelled
   training data automatically.
2. Train and serve a sentiment classifier through a Flask API, reporting how
   confident each prediction is.
3. Detect feedback whose rating contradicts its text, and keep it out of
   training without deleting it.
4. Trigger retraining automatically once a configurable number of new
   samples has been collected.
5. Evaluate every model version on an identical, frozen holdout set.
6. Enforce a quality gate that can **reject** a candidate and protect
   production.
7. Version every model — and the text encoder it depends on — and keep a
   full, auditable history.
8. Package the approved model into a Docker image as a self-contained,
   offline deployment artifact.
9. Automate the whole lifecycle in a Jenkins pipeline, ending with automated
   tests against the deployed container.

## 4. Architecture

Full diagrams and design rationale: **[docs/architecture.md](docs/architecture.md)**

```
   USER ──> Flask app ──┬──> model predicts sentiment (+ confidence)
                        └──> rating gives the actual label
                                    │
                                    v
                        SQLite: labelled training data
                                    │
                          >= 10 new samples?
                                    │ yes
                                    v
            LAYER 1  data-quality check: rating contradicts text?
                                    │  flagged rows left out, not deleted
                                    v
                      Jenkins: train candidate
                                    v
                  evaluate on the FIXED 72-row holdout
                                    v
            LAYER 2         QUALITY GATE
                          /              \
                      FAIL                PASS
                        │                   │
               keep production        promote, bake model + encoder
               block deployment       into Docker image, deploy
                                            │
                                            v
                               integration tests over real HTTP
```

Four separations make the design work:

* **The web app never trains.** It only consumes whichever model is marked
  `production`. Training belongs to `train.py` and `retrain.py`, run by Jenkins.
* **Models are baked into the image; the database is a mounted volume.** The
  image is immutable and self-contained; runtime state survives container
  recreation.
* **The holdout set is frozen.** Every model version is scored on the same 72
  rows, so accuracy numbers are actually comparable.
* **The encoder is an artifact, not a download.** The 91.6 MB sentence encoder
  is fetched once, fingerprinted, and copied alongside the models. Nothing
  downloads anything at prediction time, and the container runs offline.

## 5. Technology stack

| Layer | Choice | Version |
|---|---|---|
| Language | Python | 3.12.6 |
| Web framework | Flask | 3.1.3 |
| Frontend | Jinja2 + plain CSS/JS | — |
| Database | SQLite (`sqlite3`, stdlib) | — |
| ML (classifier, TF-IDF) | scikit-learn | 1.5.1 |
| Sentence encoder | sentence-transformers, `all-MiniLM-L6-v2` (frozen) | 5.7.0 |
| Encoder runtime | PyTorch, **CPU-only build** | 2.13.0+cpu |
| Encoder dependencies | transformers / tokenizers / huggingface-hub / safetensors | 5.15.0 / 0.22.2 / 1.27.0 / 0.8.0 |
| Model persistence | joblib | 1.4.2 |
| Numerics | numpy / scipy | 1.26.4 / 1.14.1 |
| Tests | pytest | 8.3.4 |
| Containers | Docker | 29.6.2 |
| CI/CD | Jenkins (Declarative Pipeline) | 2.568.3 |
| SCM | Git | 2.55.0 |

Versions are pinned in `requirements.txt` so the model artifact trained on the
host loads correctly inside the container. PyTorch is the CPU-only build on
both platforms. On Linux (the Docker image) it comes from PyTorch's CPU wheel
index as `2.13.0+cpu`, because PyPI's Linux wheel bundles GPU libraries that
would add gigabytes to the image. On Windows (the Jenkins host) PyPI's wheel is
already CPU-only and is published as plain `2.13.0`, so that is what is
pinned there — pinning `+cpu` on Windows made pip re-download torch.
`httpcore` and `h11` are pinned to the versions a clean install resolves, so
pip does not downgrade `h11` in the shared Jenkins Python.

## 6. Folder structure

```
smart-feedback-system/
├── app.py                   Flask routes, validation, rendering. No SQL, no ML.
├── config.py                Every threshold and path, in one place.
├── database.py              All SQLite access, including schema migration.
├── dataset.py               Data assembly + the model pipeline definition.
├── embeddings.py            The frozen sentence encoder as a scikit-learn step.
├── data_quality.py          Text / rating consistency checks (no keywords).
├── model_loader.py          Loads and caches the production model.
├── prepare_data.py          One-time split into training pool + fixed holdout.
├── prepare_encoder.py       One-time encoder download, fingerprint, --verify.
├── train.py                 Initial training (creates v1).
├── evaluate.py              Accuracy / precision / recall / F1.
├── retrain.py               --check | --train | --gate
├── stage_model.py           Copies the live model + encoder in for Docker.
├── compare_models.py        TF-IDF vs hybrid on identical data (reproducible).
├── calibrate_thresholds.py  Derives the two confidence thresholds.
├── seed_bad_feedback.py     Injects mislabelled data for the failure demo.
├── healthcheck.py           Polls /health after deployment.
├── jenkins_env_check.bat    Verifies the Jenkins runtime.
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── Jenkinsfile
├── pytest.ini
├── .gitignore
├── README.md
├── PROJECT_REPORT.md
│
├── data/
│   ├── feedback_seed.csv    360 rows, 120 per class
│   ├── train_pool.csv       288 rows  (grows with user feedback)
│   ├── holdout.csv          72 rows   (FROZEN, never trained on)
│   └── challenge_set.csv    31 probes (negation, unseen wording, noise) - NOT a test set for the gate
│
├── docs/
│   └── architecture.md
│
├── models/                  Staged production model (gitignored)
├── encoders/                Staged sentence encoder (gitignored)
├── build/                   Candidate model between stages (gitignored)
│
├── templates/               base, index, dashboard, model, error
├── static/css/style.css
├── static/js/main.js
│
├── tests/                   221 isolated tests (never touch real data)
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_semantic_model.py
│   ├── test_data_quality.py
│   ├── test_api.py
│   ├── test_pages.py
│   ├── test_validation.py
│   ├── test_database.py
│   ├── test_retraining.py
│   └── test_model_safety.py
│
└── integration_tests/       15 tests against the DEPLOYED application
    ├── conftest.py
    └── test_deployment.py
```

Everything that must survive a workspace clean lives **outside** the repository:

```
C:\mlops-data\
├── feedback.db                    Runtime database
├── models\
│   └── model_v1.pkl ... v4.pkl    Every model version ever produced (~10-20 KB each)
└── encoders\
    └── all-MiniLM-L6-v2\          The frozen encoder (91.6 MB) + encoder_manifest.json
```

## 7. Installation

Prerequisites: Python 3.12, Git, Docker Desktop. Java 17 or 21 is required
only for Jenkins.

```bash
git clone <your-repository-url>
```

```bash
cd smart-feedback-system
```

```bash
python -m pip install -r requirements.txt
```

This downloads roughly 1 GB the first time (the CPU PyTorch wheel is most of
it).

Create the persistent data directory (PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path C:\mlops-data\models
```

Fetch the sentence encoder **once** (needs internet access):

```bash
python prepare_encoder.py
```

It downloads the pinned revision of `sentence-transformers/all-MiniLM-L6-v2`
into `C:\mlops-data\encoders\all-MiniLM-L6-v2`, records the SHA-256 of the
weights in `encoder_manifest.json`, and is idempotent — run again, it checks
the existing copy and downloads nothing. From then on no step needs the
network. Re-check the copy at any time:

```bash
python prepare_encoder.py --verify
```

All paths and thresholds can be overridden by environment variable — see
`config.py`. The defaults assume `C:\mlops-data`.

## 8. Database setup

The database is created automatically on first use. To create it explicitly:

```bash
python database.py
```

**Schema.**

`feedback` — one row per submission:

| Column | Purpose |
|---|---|
| `id` | primary key |
| `name` | optional |
| `feedback_text` | the text |
| `rating` | 1–5 |
| `predicted_sentiment` | what the model said |
| `actual_sentiment` | derived from the rating (the training label) |
| `prediction_correct` | 0/1, or NULL if no model was available |
| `prediction_confidence` | the model's top class probability |
| `low_confidence` | 1 if that was below the low-confidence threshold |
| `quality_flag` | why an accepted training run excluded this row, else NULL |
| `model_version` | which model made the prediction |
| `source` | `user` / `bad_seed` |
| `used_in_training` | 0 until a candidate using it is **accepted** |
| `created_at` | timestamp |

`model_versions` — one row per trained model:

| Column | Purpose |
|---|---|
| `version` | 1, 2, 3 … never reused |
| `model_path` | where the `.pkl` lives |
| `accuracy`, `precision`, `recall`, `f1_score` | macro-averaged |
| `training_samples` | how much data went in |
| `status` | `production` / `archived` / `rejected` |
| `notes` | the quality gate's reason, and any exclusions |

This table is the **single source of truth** for which model is live. There is
no separate registry file that could drift out of sync.

**Migration.** `prediction_confidence`, `low_confidence` and `quality_flag`
were added with the hybrid model. `init_db()` adds any missing column to an
existing database with `ALTER TABLE ... ADD COLUMN`; it never drops or
rewrites a row. The live database was backed up before the migration was
first applied.

## 9. The sentiment model

```
text --+--> TF-IDF (which words appear)            -> sparse vector --+
       |                                                              +--> Logistic Regression
       +--> MiniLM sentence embedding (frozen,    -> 384 numbers    --+    (the only trained part)
            pretrained, never fine-tuned)
```

The original model was TF-IDF + Logistic Regression. It was replaced by this
**hybrid**, because TF-IDF can only use words it saw in training: a sentence
made entirely of new words carries no information for it at all.

**The encoder.** `sentence-transformers/all-MiniLM-L6-v2` turns a sentence
into 384 numbers such that sentences used in similar contexts get similar
vectors. It is small (22.7 M parameters, 91.6 MB) and runs on the CPU. It is
**frozen** — only the logistic regression is trained — so retraining stays
fast (~0.45 s) and cannot damage the encoder. It is a statistical text
representation; it does not *understand* language the way a person does, and
§21 lists exactly where it fails.

**Why both halves, not just embeddings.** Measured on identical data — the
288-row training pool, no feedback — and the identical frozen holdout, each
candidate chosen by 5-fold cross-validation on the training pool only:

| | CV accuracy | Holdout accuracy | Negation probes | Unseen-wording probes |
|---|---|---|---|---|
| TF-IDF only (original) | 0.9202 | 0.9167 | 6 / 12 | 6 / 11 |
| embeddings only | 0.9304 | 0.8472 | 8 / 12 | 8 / 11 |
| **TF-IDF + embeddings (chosen)** | **0.9411** | **0.9583** | **8 / 12** | **9 / 11** |

Embeddings alone generalise better to new wording but **lost 7 points on the
holdout** and would have been rejected by this project's own quality gate.
Combined, TF-IDF keeps precision on familiar wording and the embeddings add
reach to unfamiliar wording.

**The full comparison** (`python compare_models.py` reproduces it):

| | TF-IDF + LR (original) | TF-IDF + MiniLM + LR (current) |
|---|---|---|
| Representation | word counts, 1-grams | word counts + 384-dim sentence embedding |
| Classifier | Logistic Regression, C=5, balanced | same |
| Holdout accuracy | 0.9167 | **0.9583** |
| Holdout precision (macro) | 0.9196 | **0.9630** |
| Holdout recall (macro) | 0.9167 | **0.9583** |
| Holdout F1 (macro) | 0.9145 | **0.9582** |
| Negation probes | 6 / 12 | **8 / 12** |
| Unseen-wording probes | 6 / 11 | **9 / 11** |
| Low-information probes flagged | — (no flag existed) | 8 / 8 |
| Training time | **0.011 s** | 0.443 s |
| Prediction latency (new sentence) | **0.47 ms** | 12.3 ms |
| `model_vN.pkl` | **10,289 bytes** | 19,921 bytes |
| Extra shared artifact | none | 91.6 MB encoder |
| Docker image | **645 MB** | 2.14 GB |

On the **real** training data (the pool plus the 23 clean feedback rows that
model v4 used) the ordering is the same: TF-IDF 0.9028 accuracy / 0.9001 F1,
hybrid 0.9306 / 0.9298; negation 5/12 → 8/12; unseen wording 8/11 → 10/11.

The probes come from `data/challenge_set.csv`, a small hand-written set kept
strictly apart from the holdout: nothing is trained or tuned on it and the
quality gate never sees it. It is 31 sentences — enough to show *where* the
models differ, far too few to be a statistical benchmark.

**Why it is less keyword-dependent — and where it is not.** TF-IDF reacts only
to exact words it has seen; *"the staff were phenomenal"* came out
**negative** because none of those words carried positive weight in its
vocabulary. The embedding places that sentence near other praise, and the
hybrid answers positive (0.82). But the embedding represents negation only
weakly: *"not good"* moves from positive (TF-IDF) to **neutral**, not to
negative, and a bare *"good"* is read as neutral. No rule was added to force
these cases; they are reported as measured.

**The artifact.** joblib saves the whole pipeline as one small
`model_vN.pkl`: the TF-IDF vocabulary, the classifier weights, and the
encoder's **fingerprint** (SHA-256 of its weights). The 91.6 MB encoder weights
are *not* pickled — they live once in the encoder folder. On loading, the
model checks the encoder it finds against its fingerprint and refuses to
predict with a different one (`EncoderMismatchError`) rather than silently
producing wrong numbers. Older TF-IDF-only versions (v1–v3) still load and can
still be evaluated.

## 10. Model training

The dataset split is created once and then frozen:

```bash
python prepare_data.py
```

```bash
python prepare_data.py --verify
```

`--verify` confirms the split is a clean partition — 0 overlap between the
training pool and the holdout set, 0 rows lost.

Train the baseline model:

```bash
python train.py
```

```bash
python train.py --detailed
```

`train.py` is **idempotent**: if a production model already exists it prints a
message and exits 0, which is what makes the Jenkins bootstrap stage safe to
run on every build. Use `--force` to train anyway.

Evaluate an existing model:

```bash
python evaluate.py --production --detailed
```

**Model history on the live database:**

| Version | Model | Training rows | Holdout accuracy | How it got there |
|---|---|---|---|---|
| v1 | TF-IDF | 288 | 0.9167 | `train.py` baseline |
| v2 | TF-IDF | 298 | 0.9167 | accepted by the gate in Jenkins build #3 |
| v3 | TF-IDF | 308 | 0.9167 | accepted by the gate in Jenkins build #75 |
| **v4** | **hybrid** | 311 | **0.9306** | accepted by the gate; 5 inconsistent rows excluded |

v4 metrics: accuracy 0.9306, precision 0.9425, recall 0.9306, F1 0.9298. It was
produced by `retrain.py --train --force` (the `--force` bypasses only the
10-sample threshold, never the gate) followed by `retrain.py --gate`, which
re-scored v3 on the same holdout (0.9167) and accepted v4.

## 11. Running locally

```bash
python app.py
```

Then open <http://localhost:5000>. The first start takes longer than it used
to: importing the encoder libraries costs roughly 15–20 s in a cold Windows
process (inside the Linux container the whole start-up takes 8–14 s).

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

Test the API from PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body '{"text": "excellent service and very helpful staff"}'
```

Returns:

```json
{"prediction": "positive", "model_version": 4, "confidence": 0.9932,
 "low_confidence": false,
 "probabilities": {"negative": 0.0017, "neutral": 0.005, "positive": 0.9932}}
```

`/health` deliberately returns **200 even when no model is loaded** — the
service being up and a model being deployed are two different things, and the
Jenkins deploy stage needs to distinguish them. It also reports
`model_family` (`hybrid` or `tfidf`).

## 12. Feedback workflow

1. Enter feedback text and choose a rating (name is optional).
2. The production model predicts the sentiment and how confident it is.
3. The rating is converted to the actual label: **1–2 → negative, 3 → neutral,
   4–5 → positive**.
4. Both are stored, with the confidence and whether they agreed.
5. The result panel shows the prediction, its confidence, the actual
   sentiment, whether they matched, and the model version that answered —
   plus a warning if the prediction is low-confidence or the rating
   contradicts the text (§13).
6. The pending-sample counter advances towards the retraining threshold.

**Preview Prediction** asks the model for an answer via `POST /predict`
without storing anything, and shows the confidence too.

## 13. Data quality and confidence

### Text / rating inconsistency

The training label comes from the rating, so *"It is a great experience."*
rated 1 star becomes a **negative** training example — a wrong label. The
evaluation of the previous version found such rows in the live data:
*"Good"* rated 1 star, and *"not good"* rated 5.

`data_quality.py` runs two checks. **Neither uses sentiment keywords** — a
test parses the module's source and fails if one appears.

| Check | Flags a row when | Needs a model? |
|---|---|---|
| **Model disagreement** | the production model predicts the *opposite* polarity to the rating (positive vs negative) with confidence ≥ **0.87** | yes |
| **Label conflict** | the same text (case, spacing and punctuation normalised) was also submitted with an opposite-polarity rating | no |

Neutral never counts as a contradiction — *"neutral text, 4 stars"* is a
difference of opinion, not a wrong label.

What happens to a flagged row:

* it is **kept** in the database and shown on the dashboard with its reason;
* while `EXCLUDE_INCONSISTENT_FEEDBACK=1` (the default) it is **left out of
  retraining**;
* when a training run using that exclusion is **accepted**, the reason is
  written to the row's `quality_flag`, so the exclusion **sticks** — a later
  model that happens to be less sure about the text cannot quietly re-admit
  a label already judged wrong. A rejected run records nothing.

On the live data the check excluded five rows before v4 was trained: ids 31,
32, 51 and 52 (the same text submitted with opposite ratings) and id 48
(*"Good"*, 1 star, which the then-production model read as positive at 0.92).

Switch the exclusion off (the rows are still flagged, just not excluded):

```powershell
$env:EXCLUDE_INCONSISTENT_FEEDBACK = "0"
```

### Low confidence

Every prediction reports its top class probability. Below **0.60** it is
marked **low confidence** — shown to the user, stored, but **advisory only**:
the prediction is still returned and nothing is blocked.

### Where the two thresholds come from

Neither number was invented. `python calibrate_thresholds.py` derives both
from **out-of-fold** predictions on the training pool (5-fold, so every
prediction is made by a model that did not train on that row). The holdout is
not used, so it stays an unbiased yardstick.

| Threshold | Value | Derivation |
|---|---|---|
| `LOW_CONFIDENCE_THRESHOLD` | 0.60 | 10th percentile of the top probability (0.596). Top probability separates right from wrong predictions far better (AUROC 0.906) than similarity to the training data (0.694). Catches 8 of the 17 out-of-fold errors and 8 of 8 meaningless probes. |
| `DISAGREEMENT_CONFIDENCE` | 0.87 | The most confident opposite-polarity mistake the model made on **clean** labels was 0.861. A flip more confident than any clean-label error is more likely a bad rating than a model mistake. |

The trade-off is stated honestly: at 0.60, half the model's errors are still
*not* flagged, and a threshold high enough to catch most of them would flag a
large share of correct predictions too.

## 14. Auto-retraining

Configured in `config.py`:

```python
RETRAIN_THRESHOLD = 10       # new samples before retraining is eligible
```

Check whether retraining is needed:

```bash
python retrain.py --check
```

The exit code carries the decision, because each Jenkins stage is a separate
process:

| Exit | Meaning |
|---|---|
| `0` | retraining NOT required |
| `10` | retraining IS required |
| `1` | error |

`10` rather than `1` means a genuine Python crash can never be misread as a
decision. `--quiet` prints only `true` or `false` for pipelines that prefer
to capture stdout.

Run the three stages by hand:

```bash
python retrain.py --train
```

```bash
python evaluate.py --candidate --detailed
```

```bash
python retrain.py --gate
```

`--train` prints what went in and what was excluded, and writes it to
`build/candidate_training.json`, which Jenkins archives:

```
Pool samples     : 288
Feedback samples : 23 used of 28 collected
Excluded as inconsistent: 5 row(s) [31, 32, 48, 51, 52]
   model disagreement : 1
   label conflict     : 4
Total training   : 311
```

**Training data grows; the holdout set never does.** New feedback always
becomes training data, so the yardstick stays fixed and model versions remain
comparable.

## 15. Quality gate

A candidate must satisfy **both** conditions:

| | Condition | Default |
|---|---|---|
| **A** | `accuracy >= MIN_ACCURACY` | 0.85 |
| **B** | `accuracy >= production_accuracy − TOLERANCE` | 0.02 |

Condition B is what stops a working model being silently replaced by a worse
one. Both models are re-scored on the same frozen holdout, so the comparison
is valid — including across model families: the gate compared hybrid v4
against TF-IDF v3 on the same 72 rows and approved the change on measured
accuracy alone.

`MIN_ACCURACY = 0.85` was chosen from measurement, not invented:

* the real baseline scored **0.9167** — 6.7 points of headroom, so an honest
  retrain will not trip the gate by accident;
* a deliberately poisoned model scored **0.7778** — the gate discriminates;
* 85% is also the example threshold used in the CI/CD practical.

Accepted (the live v4 promotion):

```
QUALITY GATE: PASSED
Candidate model accepted.
New model version    : v4
Previous version     : v3 (now archived)
```

Rejected (Scenario C, filter off, on a copy of the live database):

```
Candidate Accuracy   : 0.7778
Required Accuracy    : 0.8500   (MIN_ACCURACY)
Production Accuracy  : 0.9306   (v4, re-scored on the same holdout)
QUALITY GATE: FAILED
Candidate recorded as v5 with status 'rejected'.
Production model v4 REMAINS ACTIVE.
Deployment is blocked.
```

A rejection **does not consume the training samples** — they stay pending so
the next run tries again as more feedback arrives.

**Two layers, measured.** With 50 poisoned rows (positive text rated 1 star,
negative text rated 5), on a copy of the live database:

| | Poisoned rows trained on | Candidate accuracy | Gate |
|---|---|---|---|
| Layer 2 only (filter off) | 50 | 0.7778 | **FAIL** — production kept |
| Layer 1 + 2 (default) | 4 (46 caught) | 0.9167 | pass |

The filter does most of the work; the gate remains the backstop for whatever
slips through. The four rows it missed were two texts — *"perfect quality,
exactly as described"* (read as positive, but only at 0.84) and *"would not
recommend, awful quality"* (misread as positive at 0.54).

## 16. Docker

Stage the approved model **and the encoder** into the build context, then
build:

```powershell
python stage_model.py --clean
```

```powershell
docker build -t smart-feedback:v4 -t smart-feedback:latest .
```

Run with the port mapping and the persistent volume:

```powershell
docker run -d -p 5000:5000 -v "C:/mlops-data:/data" --name smart_feedback_app smart-feedback:latest
```

Container lifecycle:

```powershell
docker ps
```

```powershell
docker logs smart_feedback_app
```

```powershell
docker stop smart_feedback_app
```

```powershell
docker rm smart_feedback_app
```

```powershell
docker images smart-feedback
```

**The volume is mandatory.** Without it the container starts but reports
`model_loaded: false` — the `.pkl` is baked into the image, but the *database*
holds the record of which version is production. That is correct design: the
image carries the artifact, the database carries the deployment decision.

**The encoder is baked in, and the container is offline.** `stage_model.py`
copies the encoder folder into `encoders/` (skipping the copy if the
fingerprint already matches, verifying the SHA-256 after copying), and the
image carries it at `/app/encoders`. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` make any attempt to reach the internet fail loudly
instead of silently downloading. Nothing is downloaded per prediction or per
container start.

**Measured cost.** The image grew from 645 MB to **2.14 GB**: the CPU PyTorch
runtime is about 750 MB, the other encoder libraries most of the rest, the
encoder itself 88 MB. A rebuild with cached dependency layers takes about
11 s; the first build downloads the wheels and takes minutes. The container is
healthy 8–14 s after `docker run`.

Differences from the Docker practical, and why:

| Change | Reason |
|---|---|
| `python:3.12-slim` instead of `3.9-slim` | the model is trained on Python 3.12 with scikit-learn 1.5.1; loading that pickle under a different version can fail |
| `ENV MLOPS_DATA_DIR=/data` | redirects SQLite out of the container's writable layer into the volume |
| `ENV PYTHONUNBUFFERED=1` | without it Flask's logs do not reach `docker logs` promptly, making a failed deploy hard to diagnose |
| `ENV HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | the encoder is baked in; the container must never try to download it |
| volume mount | so feedback survives the `rm`/`run` cycle Jenkins performs on every deployment |

## 17. Jenkins setup

**Before anything else, verify the runtime.** Create a Freestyle job with one
*Execute Windows batch command* step:

```bat
call jenkins_env_check.bat
```

It runs seven checks — `python`, `git`, `docker --version` + `docker ps`,
read/write access to `C:\mlops-data`, `prepare_encoder.py --verify`,
`retrain.py --check`, and `pytest` + `stage_model.py` — then reports a
pass/fail verdict. All seven pass on the development machine.

**Which account runs Jenkins.** On this machine the Jenkins service runs as
**LocalSystem**, and that works: it finds the same Python as the logged-in
user, and the `ML-Deployment-Exp9` job ran `docker build` successfully under
it. If `docker ps` fails for your Jenkins account, check that the account can
open the Docker named pipe (Administrators and SYSTEM normally can; other
accounts must be in `docker-users`), then restart the service — group
membership is read at logon. Do not loosen Docker's own permissions.

Then create the pipeline job:

1. **New Item** → name it `Smart-Feedback-CI-CD` → **Pipeline** → OK
2. **Pipeline** → *Pipeline script from SCM* → Git → your repository URL
3. Script Path: `Jenkinsfile`
4. Save → **Build Now**

The `Jenkinsfile` declares `triggers { cron('H/5 * * * *') }`, so after the
first manual build Jenkins runs every five minutes on its own.

**First run after this revision.** On this machine *Install Dependencies*
installs nothing (every pinned package is already in the Python Jenkins
uses), and *Bootstrap Encoder and Model* only confirms the prepared encoder in
`C:\mlops-data\encoders`. So a normal build needs no internet beyond Jenkins'
own `git fetch` — verified by the replay in §18. On a fresh machine, pip and
`prepare_encoder.py` need internet access once; if the encoder cannot be
downloaded, `prepare_encoder.py` says so plainly and exits 1.

## 18. Jenkins pipeline explained

Twelve stages. Everything after the check is conditional.

| # | Stage | Runs | On failure |
|---|---|---|---|
| 1 | Checkout | always | stop |
| 2 | Install Dependencies | always | stop |
| 3 | Bootstrap Encoder and Model | always (both idempotent) | stop |
| 4 | Run Tests | always | **stop — nothing is built** |
| 5 | Check Retraining Requirement | always | stop |
| 6 | Train Candidate (data-quality filter included) | only if required | stop |
| 7 | Evaluate Candidate | only if required | stop |
| 8 | **Quality Gate** | only if required | **stop — no deployment** |
| 9 | Stage Production Model (model + encoder) | only if required | stop |
| 10 | Build Docker Image | only if required | stop |
| 11 | Deploy | only if required | stop |
| 12 | Deployment Integration Tests | only if required | stop |

The pipeline timeout is 45 minutes (it was 30): the first build after a
dependency change installs PyTorch.

**Why a cron trigger and not Poll SCM.** Feedback is stored in SQLite, not in
Git. A Poll SCM trigger watches for *code* changes and would never notice new
*data*. The cron trigger wakes the pipeline up to ask the database whether
enough feedback has arrived.

**Why most runs do almost nothing.** If the threshold has not been reached,
the pipeline trains nothing, builds nothing and redeploys nothing. It prints
`RETRAINING NOT REQUIRED` and finishes green. Rebuilding an unchanged model
every five minutes would be pointless churn and would restart the live
service for no reason.

**Evidence from the live Jenkins** (job `Smart-Feedback-CI-CD`, Jenkins
2.568.3, service account LocalSystem):

* of the 20 builds Jenkins retains (**#513–#532** at the time of writing),
  **18 succeeded**, each logging `RETRAINING NOT REQUIRED` and
  `Stage "..." skipped due to when conditional` for all seven conditional
  stages, so the `when` logic is verified in Jenkins itself. The other two
  failed at Checkout with `Could not resolve host: github.com` — the network
  was not up yet after the machine woke — and the next build succeeded each
  time. (An earlier snapshot, #307–#326, was 20 / 20 SUCCESS.);
* two **full retraining cycles** ran in Jenkins: model v2 and v3 were promoted
  by the gate at 23:10 on 20 Sep and 19:46 on 21 Sep, and the Docker images
  `smart-feedback:v3` and `smart-feedback:v75` — tagged with the Jenkins
  build number — were created within a minute of each promotion. Those two
  builds' console logs have since rotated out, so the evidence is the database
  rows and image timestamps rather than the logs;
* builds #320–#323 ran **after** the database migration and still passed —
  the old pipeline reads the migrated database without problems.

**Not yet verified inside Jenkins:** the revised pipeline, because Jenkins
builds from GitHub. Before pushing, it was replayed as closely as possible
without Jenkins: the stages, `bat` commands, `when` conditions and exit-code
handling were **parsed from the Jenkinsfile itself** and run through
`cmd.exe` in a fresh clone, with the SYSTEM account's machine PATH, an empty
user profile, no user site-packages, and every non-local network connection
made by Python blocked and logged.

| Replay | Result |
|---|---|
| Below threshold, live data | 5 stages pass, 7 skipped by `when`; **0** network attempts; live database byte-identical before and after |
| Retraining path, copy of the live data + 10 new rows | **all 12 stages pass** — exit 10 → 5 inconsistent rows excluded → v5 accepted (0.9306) → model + encoder staged → image built → container healthy serving v5 → 15 / 15 integration tests; **0** non-local network attempts |

The replay caught one real problem, now fixed: `torch==2.13.0+cpu` did not
match the Windows host's torch (`2.13.0`), so the first Jenkins build would
have re-downloaded torch (see §5). What a replay cannot show is Jenkins itself
running as SYSTEM; the first build after the push will.

**Windows.** Every step uses `bat`, not `sh`. The practicals show `sh` because
they assume a Linux agent; this Jenkins runs on Windows, where `sh` fails.
Experiment 3 uses *Execute Windows batch command* for the same reason.

| Purpose | Linux (practical) | Windows (this project) |
|---|---|---|
| run a step | `sh 'python train.py'` | `bat 'python train.py'` |
| variable | `${BUILD_NUMBER}` | `%BUILD_NUMBER%` |
| tolerate missing container | `docker rm -f x \|\| true` | `bat(returnStatus: true, script: 'docker rm -f x')` |

## 19. Tests

```bash
python -m pytest -q
```

**Result: 221 passed** (about 53 s).

| Module | Tests | Covers |
|---|---|---|
| `test_model.py` | 16 | artifact exists, loads, all three classes |
| `test_semantic_model.py` | 32 | encoder manifest and fingerprint, offline loading, normalised deterministic embeddings, serialisation round trip, wrong-encoder refusal, staging, hybrid accuracy, unseen vocabulary, negation, probabilities, low confidence |
| `test_data_quality.py` | 34 | both checks, no-keyword source guard, replay of the real evaluation data, sticky exclusion, migration |
| `test_api.py` | 19 | all eight routes |
| `test_pages.py` | 24 | rendered pages: form, navigation, result panel, counter, dashboard, model page, the script the preview button uses |
| `test_validation.py` | 28 | empty, short, invalid, malformed input |
| `test_database.py` | 30 | insert, retrieve, versioning, promotion |
| `test_retraining.py` | 24 | threshold, candidate, gate accept/reject, filter vs poisoned data |
| `test_model_safety.py` | 14 | holdout isolation, shared test set |

The negation and unseen-wording tests assert what was **measured** — that
*"not good"* moves away from positive, for example — not what one might wish;
none of them force an answer.

The suite is fully isolated: `tests/conftest.py` redirects the database, the
model store, the build directory and the staging directories into a temporary
folder **before any project module is imported**, and aborts the run if that
redirection ever fails. The encoder is the one exception — it is read-only and
read from `C:\mlops-data\encoders` — so the tests need `prepare_encoder.py` to
have run once. Running the tests cannot write to `C:\mlops-data`.

`pytest.ini` sets `testpaths = tests`, so the deployment integration tests
(§20) are not part of the default run — they need a running server.

## 20. Deployment integration tests

These tests check the **deployed** application — normally the container
Jenkins has just started — over real HTTP, as a user's browser would.

```bash
python -m pytest integration_tests -v
```

**Result: 15 passed** against the Docker container.

They cover `/health` (model loaded, hybrid family, served version matches the
database), `/predict` (full response, unseen wording, low-confidence flag,
invalid and form-encoded input), every page and static asset, a 404, and the
full feedback path: a submission through the real form lands in the real
mounted database and appears in `/api/stats`, and an invalid one is rejected
and not stored. `BASE_URL` (default `http://localhost:5000`) points them
elsewhere.

**Baseline protection.** A session fixture records the highest feedback id
before the run and deletes everything above it afterwards, so the database
ends exactly as it started (verified: 28 rows before and after).

**Why no browser automation.** The project previously drove Chrome with
Selenium for this stage. It was removed: the tests were slow, needed Chrome
and a matching driver on the Jenkins machine, and every behaviour they
checked is covered more directly by `tests/test_pages.py` (rendered HTML
through Flask's test client) and by these HTTP tests. The pages have only two
pieces of client-side behaviour, a character counter and the preview button.
The preview button calls `/predict`, which both suites test. What is no longer
tested is the JavaScript itself: the counter, and the code that writes the
preview response into the page.

## 21. Limitations

* **Negation is still weak.** *"not good"* is read as neutral, *"not very
  good"* as neutral, *"I liked it"* as neutral (challenge set: 8 of 12 negation
  probes right). Sentence embeddings represent negation only weakly, and
  nothing was hard-coded to hide it.
* **Bare single words are read less confidently.** *"Good"* scores positive
  at only 0.64 with v4 (0.92 with TF-IDF v3). So a **new** *"Good"* rated
  1 star would not be flagged as inconsistent — 0.64 is below the 0.87
  disagreement threshold. The existing row (id 48) stays excluded because
  that decision was recorded.
* **One feedback text is memorised.** *"the experience could be better."*
  rated 4 stars (id 50) was a clean row and is part of v4's training data, so
  v4 answers positive (0.70) for that exact text — not neutral, as the same
  model family trained without it does.
* **The disagreement check is only as good as the model.** If the model is
  unsure what a text means, it cannot vouch that the rating is wrong; 4 of 50
  poisoned rows got past it.
* **The dataset is small and synthetic.** 360 hand-written rows plus 28 real
  submissions; the challenge set is 31 sentences. Differences of one or two
  holdout rows (1.4 points each) should not be over-read.
* **Heavier deployment.** Image 645 MB → 2.14 GB; prediction ~0.5 ms → ~12 ms;
  retraining 0.01 s → 0.45 s; importing the encoder libraries adds ~15–20 s to
  a cold start on Windows.
* **Single-node deployment.** One container on one host; no orchestration,
  scaling or rollback automation. Archived `.pkl` files are retained, so a
  manual rollback is possible.
* **No authentication.** Anyone who can reach the port can submit feedback.
* **Shared global Python.** Jenkins jobs on this machine share one Python
  installation; another job pinning a different Flask version could change
  this project's environment between builds.

## 22. Troubleshooting

**`EncoderNotFoundError: ... Run 'python prepare_encoder.py' once.`**
The encoder has not been fetched on this machine (or not staged for Docker).
Run `python prepare_encoder.py`, then `python stage_model.py --clean` before
building the image.

**`EncoderMismatchError`**
The model was trained with a different encoder than the one found. Do not
work around it — the model's numbers would be meaningless. Check the encoder
with `python prepare_encoder.py --verify`; if it was replaced, restore the
original or retrain.

**The first request or `python app.py` is slow**
Importing PyTorch and sentence-transformers takes 15–20 s in a cold Windows
process. Later predictions take ~12 ms. `model_loader` warms the model up
when it loads, so the first *user* request is not the slow one.

**`docker ps` fails from Jenkins but works in your terminal**
The Jenkins service account cannot open the Docker named pipe. See §17 and
run `jenkins_env_check.bat` to confirm.

**Container starts but `model_loaded` is false**
The volume is missing. The `.pkl` is baked into the image but the *database*
records which version is production. Re-run with
`-v "C:/mlops-data:/data"`.

**`InconsistentVersionWarning` when loading a model**
The scikit-learn that created the pickle differs from the one loading it.
Reinstall from the pinned `requirements.txt`; the Dockerfile's
`python:3.12-slim` matches the host deliberately.

**Port 5000 already in use**

```powershell
Get-NetTCPConnection -LocalPort 5000 -State Listen
```

Stop the process, or `docker rm -f smart_feedback_app`.

**Integration tests fail immediately with "application is not responding"**
Nothing is serving on `BASE_URL`. Start `python app.py` or the container
first.

**`retrain.py --check` seems to do nothing**
That is correct below the threshold. It prints `RETRAINING NOT REQUIRED` and
exits 0. Check the pending count on the dashboard.

**The quality gate keeps failing**
Check the Model page for the rejected version and its recorded reason. If you
injected bad data for a demonstration, undo it:

```bash
python seed_bad_feedback.py --undo
```

**A row was excluded by mistake**
Exclusions are recorded, not deleted. `database.clear_quality_flag(row_id)`
clears one; the next training run judges the row afresh.

**Tests pass individually but fail together**
Should not happen — every test gets a fresh temporary database. If it does,
check that `tests/conftest.py` still redirects the paths before importing any
project module.

## 23. Demonstration procedure

### Setup

```powershell
python prepare_data.py
```

```powershell
python prepare_encoder.py
```

```powershell
python train.py
```

```powershell
python stage_model.py --clean
```

```powershell
docker build -t smart-feedback:v4 -t smart-feedback:latest .
```

```powershell
docker run -d -p 5000:5000 -v "C:/mlops-data:/data" --name smart_feedback_app smart-feedback:latest
```

```powershell
python healthcheck.py --require-model --retries 60
```

### Scenario A — normal operation

1. Open <http://localhost:5000> and submit one piece of feedback.
2. Show the result: predicted sentiment and its confidence, actual sentiment,
   whether they matched, and the model version.
3. Submit *"something something"* — it is marked **low confidence** (0.46).
4. Open the Dashboard — the counts, the pending-sample counter and the Data
   Quality cards.
5. Open Model / MLOps — v4 in production with its four metrics.
6. Run `python retrain.py --check` → **exit 0**, `RETRAINING NOT REQUIRED`.

*Point to make:* below the threshold the pipeline builds and deploys nothing.

### Scenario B — auto-retraining and promotion

1. Submit feedback until the counter reaches **10 / 10**. Include
   *"It is a great experience."* rated **1 star**: the result panel warns that
   the rating contradicts the text (the model reads it as positive, 0.98) and
   that it will be left out of retraining.
2. `python retrain.py --check` → **exit 10**, `RETRAINING REQUIRED`.
3. `python retrain.py --train` — the log lists the excluded row.
4. `python evaluate.py --candidate --detailed`
5. `python retrain.py --gate` → **QUALITY GATE: PASSED**, v5 promoted, v4 archived.
6. Refresh Model / MLOps — v5 is production, and its note records the
   exclusion.

*Point to make:* using the app produced the data that improved the model, and
the contradictory rating did not become a training label.

### Scenario C — poisoned data meets both layers

```powershell
python seed_bad_feedback.py
```

Injects 50 deliberately mislabelled rows — positive text rated 1 star,
negative text rated 5.

**Layer 1 — the filter:**

```powershell
python retrain.py --train
```

Expect about 46 of the 50 poisoned rows in the `Excluded as inconsistent`
list, alongside the rows already on record (measured on a copy of the live
database: 51 rows = 46 poisoned + 5 on record). Do **not** run the gate here:
the filtered candidate scored 0.9167 in that measurement and would pass and be
promoted.

**Layer 2 — the gate**, with the filter switched off:

```powershell
$env:EXCLUDE_INCONSISTENT_FEEDBACK = "0"
```

```powershell
python retrain.py --train
```

```powershell
python evaluate.py --candidate
```

```powershell
python retrain.py --gate
```

Expect **QUALITY GATE: FAILED** (0.7778 in the measurement above), exit code
**1**, the candidate recorded as `rejected`, and production unchanged. Refresh
Model / MLOps to show the rejected version and its reason.

Then undo:

```powershell
Remove-Item Env:EXCLUDE_INCONSISTENT_FEEDBACK
```

```powershell
python seed_bad_feedback.py --undo
```

*Point to make:* the filter stops most bad data before training; the gate
catches a bad model if the data gets through anyway.

### Deployment check

```powershell
python -m pytest integration_tests -v
```

### Tests

```powershell
python -m pytest -q
```

## 24. Practical mapping

| Practical | Project component | What is demonstrated |
|---|---|---|
| **Experiment 1** — Git / GitHub | repository, `.gitignore`, branches | `init`, `add`, `commit`, `branch`, `checkout`, `remote add`, `push`, `pull` |
| **Experiment 2** — Jenkins installation & setup | Jenkins on `localhost:8080`, `jenkins_env_check.bat` | installing and configuring the CI/CD environment; the `.war` method from this practical was the one used |
| **Experiment 3** — Jenkins CI (Freestyle) | `jenkins_env_check.bat` as a Freestyle job | *Execute Windows batch command*, Console Output, exit codes — and the reason this project uses `bat` rather than `sh` |
| **Experiment 4** — Docker CLI | container lifecycle in §16, the pipeline's teardown step | `docker ps`, `stop`, `rm`, `images`, `logs` |
| **Experiment 5** — Dockerised Flask application | `Dockerfile`, port mapping `5000:5000` | Dockerfile structure, building an image, port mapping, running a Flask app in a container |
| **PS4 — CI/CD Pipeline / ML CI-CD Practical** | `Jenkinsfile`, `train.py`, `tests/`, the **quality gate** | pipeline-as-code, graphical stages, `Code + Data + Model`, and the quality-gate concept this whole project is built around |
| **Experiment 9** — Deploying models via Docker through Jenkins | `app.py` `/predict`, the Deploy stage, `healthcheck.py`, `integration_tests/` | training vs inference, baking the model into the image, `docker rm -f` before `docker run`, testing the live API |

**The Selenium practicals are no longer demonstrated by this project.** The
browser tests were replaced by pytest suites (§19, §20) for the reasons given
in §20.

**PS4 is not given an experiment number.** Its true number cannot be
established from the supplied material, so it is referred to throughout as
*PS4 — CI/CD Pipeline / ML CI-CD Practical*. Experiment 9 references an
"Exp 8: Training & Testing"; PS4 covers exactly that ground and even names its
Jenkins job `MLOps-LabExp8`, but that is not firm enough to assert a number.

---

## Verification summary

| Component | Result |
|---|---|
| Hybrid vs TF-IDF, identical data | **Measured** — holdout 0.9583 vs 0.9167; negation 8/12 vs 6/12; unseen 9/11 vs 6/11 |
| Model v4 (hybrid, live data) | **0.9306** accuracy, 0.9298 F1 (72-sample fixed holdout) |
| Candidate accepted | **Verified** — v4 promoted over v3 (0.9306 vs 0.9167) |
| Candidate rejected | **Verified** — 0.7778, production retained (filter off) |
| Data-quality filter | **Verified** — 5 live rows excluded; 46 / 50 poisoned rows caught |
| Sticky exclusion | **Verified** — recorded on the live database after v4 was accepted |
| Pytest | **221 passed** |
| Deployment integration tests vs Docker | **15 passed** |
| Docker build (encoder baked in, offline) | **Verified** — 2.14 GB, healthy in 8–14 s |
| Container persistence | **Verified** — database unchanged across redeploys (28 rows) |
| Holdout isolation | **Verified** — 0 overlap; holdout unchanged |
| `jenkins_env_check.bat` | **Verified** — 7 / 7 checks pass |
| Jenkins below-threshold runs | **Verified in Jenkins** — #307–#326 all SUCCESS; #513–#532 18 SUCCESS, 2 Checkout failures (no DNS after wake) |
| Jenkins full retraining cycle | **Verified by evidence** — v2 / v3 promotions match images v3 / v75 (logs rotated out) |
| Revised pipeline, Jenkins-like replay | **Verified** — both paths pass offline under SYSTEM-like conditions (§18) |
| **Revised pipeline inside Jenkins** | **Pending** — runs after the revision is pushed |

# Smart Feedback System with Auto-Retraining Pipeline

**Mini Project Report — DevOps / CI-CD / MLOps**

---

## Abstract

This project implements a feedback-driven MLOps lifecycle in which a machine
learning model improves itself from ordinary application usage, but is never
allowed to get worse.

Users submit written feedback together with a 1–5 star rating. A sentiment
model predicts the sentiment of the text, while the rating independently
supplies the true label — so every submission automatically becomes a
labelled training sample. Once a configurable number of new samples has been
collected, an automated pipeline trains a candidate model, scores it against
a permanently frozen holdout set, and subjects it to a two-condition quality
gate. Only a candidate that clears the gate is versioned, packaged into a
Docker image and deployed; one that fails is recorded as rejected and the
existing production model continues to serve.

The system was built with Flask, SQLite, scikit-learn, Docker, Jenkins and
Selenium. The baseline model achieved **0.9167 accuracy** on a fixed
72-sample holdout set. Both quality-gate outcomes were verified end to end:
a genuine improvement was promoted at 0.9306, and a deliberately poisoned
candidate scoring 0.7778 was rejected with production left untouched.
**130 unit and integration tests** and **15 Selenium browser tests** pass.

One component is not fully verified: Jenkins could not be started on the
development machine because of a system-level Java networking fault. The
pipeline is syntax-validated and its logic was simulated end to end, but an
actual Jenkins build remains pending. This is stated explicitly rather than
glossed over — see *Limitations / Pending Verification*.

---

## Problem Statement

A machine learning model deployed once begins to decay. Language drifts,
products change, and the data the model was trained on slowly stops
resembling the data it receives. Nobody notices until the predictions are
visibly poor. Retraining by hand is slow, easy to postpone and easy to forget.

Automating retraining is the obvious answer, and it introduces a worse
problem. A pipeline that deploys whatever it has just trained will eventually
replace a good model with a bad one — particularly when the incoming data is
noisy, unrepresentative, or deliberately poisoned.

This project therefore addresses two requirements at once:

1. retraining must happen automatically, driven by real usage; and
2. no model may reach production without passing an explicit, measurable
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
2. Train and serve a sentiment classifier through a web API.
3. Trigger retraining automatically once a configurable threshold of new
   samples is reached.
4. Evaluate every model version on an identical, permanently frozen holdout
   set so that versions are genuinely comparable.
5. Enforce a quality gate capable of **rejecting** a candidate and protecting
   the production model.
6. Version every model and maintain a complete, auditable history including
   rejections.
7. Package the approved model into a Docker image as a self-contained
   deployment artifact.
8. Automate the lifecycle in a Jenkins declarative pipeline.
9. Verify the deployed application through a real browser using Selenium.

---

## Scope

**In scope.** Feedback collection; three-class sentiment classification;
automatic retraining on a sample-count threshold; a two-condition quality
gate; model versioning with a persistent artifact store; containerised
deployment with a mounted database; a Jenkins pipeline with conditional
stages; automated unit, integration and browser testing.

**Out of scope, deliberately.** Kubernetes, cloud infrastructure, message
queues, distributed training, large pretrained language models,
authentication, and multi-node orchestration. The project targets a scale
that can be run locally and explained completely in a viva.

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
| Persistence | joblib | 1.4.2 | the practicals' `model.pkl` approach |
| Numerics | numpy / scipy | 1.26.4 / 1.14.1 | scikit-learn dependencies |
| Unit testing | pytest | 8.3.4 | used in the CI/CD practical |
| Browser testing | Selenium (Python) | 4.27.1 | the Selenium practical |
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
        ├──> model PREDICTS sentiment ──────┐                      │
        └──> rating gives the TRUE label ───┤                      │
                                            v                      │
                                   LABELLED TRAINING DATA          │
                                            │                      │
                              enough new samples? (>= 10)          │
                                       yes  v                      │
                                   TRAIN CANDIDATE                 │
                                            v                      │
                         EVALUATE on the FIXED 72-row holdout      │
                                            v                      │
                                     QUALITY GATE                  │
                                     /              \              │
                                 FAIL                PASS          │
                                  │                   │            │
                      keep production model   promote to production│
                      block deployment                │            │
                                                      v            │
                                            bake into Docker image │
                                                      v            │
                                              deploy container ────┘
                                                      v
                                            Selenium verifies the UI
```

### Layered view

```
   PRESENTATION   templates/ + static/        3 pages, stable element IDs
   APPLICATION    app.py                      routes, validation, rendering
   SERVICE        model_loader.py             loads the ONE production model
   DATA           database.py                 all SQL, Flask-free
   PIPELINE       train / evaluate / retrain  offline, run by Jenkins
   CONFIG         config.py                   every threshold and path
```

### Design decisions

**The web application never trains a model.** It only consumes whichever
version is marked `production` in the database. Training and promotion belong
to `train.py` and `retrain.py`, executed by Jenkins. This prevents the running
service from quietly changing the model serving live users.

**Models are baked into the image; the database is mounted.** The image is an
immutable artifact containing the exact approved model. The database is
runtime state and must survive the container being destroyed and recreated —
which Jenkins does on every deployment.

**The `model_versions` table is the single source of truth** for which model
is live. There is no parallel registry file that could drift out of sync.

---

## Data Flow

```
  POST /feedback
      │
      ├─ validate text (3-1000 chars) and rating (1-5)   -> 400 on failure
      │
      ├─ model_loader.predict(text)          -> "positive"
      ├─ config.rating_to_sentiment(5)       -> "positive"
      │
      ├─ database.insert_feedback(...)
      │     text, rating, prediction, true label, whether they matched,
      │     model version, used_in_training = 0
      │
      └─ render: predicted / actual / match / version / pending counter
```

The rating is the source of truth. When the model disagrees with it, the
model is recorded as wrong — which is how the system measures its own live
accuracy and accumulates the data needed to correct itself.

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
| `notes` | TEXT | the quality gate's reason |
| `created_at` | TEXT | timestamp |

### Two decisions worth defending

**`used_in_training` is a flag, not a derived count.** It makes "new samples
since last training" an exact `COUNT(*) WHERE used_in_training = 0` rather
than an inference from a possibly stale number.

**Samples are marked used only when a candidate is ACCEPTED.** If the gate
rejects a candidate the samples stay pending, so the next pipeline run retries
with them. A rejection must never silently consume training data. This is
asserted by `test_rejection_does_not_consume_the_samples`.

---

## ML Methodology

### Problem formulation

Three-class text classification. The rating is mapped to the label:

| Rating | Label |
|---|---|
| 1–2 | negative |
| 3 | neutral |
| 4–5 | positive |

### Model

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

### Hyperparameters chosen by experiment, not assumption

Eight configurations were compared using 5-fold cross-validation **on the
training set only**, never the holdout set:

| Configuration | CV accuracy |
|---|---|
| unigram, `min_df=2`, `C=5` | **0.9202** ← chosen |
| unigram, `min_df=1`, `C=1` | 0.9168 |
| 1–2 gram, `min_df=1`, `C=10` | 0.9063 |
| 1–2 gram, `min_df=2`, `C=10` | 0.8924 |
| 1–2 gram, `min_df=1`, `C=1` | 0.8855 |

Word pairs such as "not bad" were expected to help but measurably hurt:
bigrams generate far more features than 288 training samples can support, so
the model overfits. Single words won. `min_df=2` discards words appearing
only once, which cannot generalise anyway.

---

## Dataset Preparation

No suitable dataset was supplied, so a **synthetic seed dataset of 360 rows**
was written by hand, balanced at 120 rows per class.

**A first attempt failed and the failure was instructive.** An initial
180-row dataset produced only **0.5833 accuracy**. Analysis showed the cause:
370 of 543 unique words appeared exactly once, so test sentences shared
almost no vocabulary with training sentences. The sentences had been written
with too much creative variety — real feedback reuses sentiment words
("slow", "excellent", "poor") constantly. The dataset was rebuilt at 360 rows
with deliberate vocabulary recurrence, and accuracy rose to **0.9167**.

This is recorded because it is a genuine result: for a small text dataset,
vocabulary recurrence matters more than sentence count.

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
never moves.

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

1. load the training pool plus any collected feedback
2. load the fixed holdout set
3. fit the TF-IDF + Logistic Regression pipeline
4. evaluate on the holdout
5. save the artifact to the **persistent store**, `C:\mlops-data\models\`
6. register the version in the database as `production`

The script is **idempotent** — if a production model exists it exits 0 without
doing anything, which makes the Jenkins bootstrap stage safe to run on every
build.

### Measured baseline (model v1)

```
Accuracy   0.9167        Precision  0.9196
Recall     0.9167        F1-Score   0.9145
Training samples 288     Holdout 72     Vocabulary 201 features
```

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.89 | 1.00 | 0.94 | 24 |
| neutral | 0.95 | 0.79 | 0.86 | 24 |
| positive | 0.92 | 0.96 | 0.94 | 24 |

Neutral is the hardest class, which is expected — it sits between the other
two and lacks strongly polarised vocabulary.

---

## Model Evaluation

`evaluate.py` is the only place the four metrics are computed, so production
and candidate models are always measured by identical code on identical data.

Averaging is **macro**: each class contributes equally regardless of support.
This is the honest choice here, because it prevents a model from looking good
by handling the two easy classes and ignoring neutral. `zero_division=0`
prevents a crash if a class is never predicted.

`random_state=42` is fixed throughout, so an examiner can re-run training and
reproduce the figures exactly. `test_training_is_reproducible` and
`test_evaluation_is_reproducible` assert this.

---

## Auto-Retraining Pipeline

```
  new_samples = COUNT(feedback WHERE used_in_training = 0)

  if new_samples < RETRAIN_THRESHOLD (10):
      exit 0   -> pipeline skips training, build and deployment
  else:
      train candidate on (training pool + ALL feedback)
      evaluate on the FIXED holdout
      apply the quality gate
```

Retraining is split into three subcommands so that each becomes its own
visible Jenkins stage, with state passed between the separate processes
through small files in `build/`:

| Command | Produces |
|---|---|
| `retrain.py --check` | `build/retrain_decision.json` + exit code |
| `retrain.py --train` | `build/candidate_model.pkl` |
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
the same frozen holdout before comparison, so the numbers are valid.

### How the threshold was chosen

`MIN_ACCURACY = 0.85` was derived from measurement, not invented:

| Evidence | Value |
|---|---|
| Real baseline accuracy | 0.9167 — leaves 6.7 points of headroom |
| Poisoned model, 40 bad rows | 0.8194 — fails |
| Poisoned model, 50 bad rows | 0.7778 — fails clearly |
| CI/CD practical's own example threshold | 85% |

So the gate does not trip on ordinary variation, but does discriminate
against a genuinely degraded model.

### Both outcomes, as measured

Accepted:

```
Candidate Accuracy   : 0.9306
Required Accuracy    : 0.8500
Production Accuracy  : 0.9167
Condition A  0.9306 >= 0.8500  -> PASS
Condition B  0.9306 >= 0.8967  -> PASS
QUALITY GATE: PASSED
New model version    : v2      Previous version : v1 (archived)
```

Rejected:

```
Candidate Accuracy   : 0.7778
Required Accuracy    : 0.8500
Production Accuracy  : 0.9306
Condition A  0.7778 >= 0.8500  -> FAIL
Condition B  0.7778 >= 0.9106  -> FAIL
QUALITY GATE: FAILED
Candidate recorded as v3 with status 'rejected'.
Production model v2 REMAINS ACTIVE.  Deployment is blocked.
```

A rejected candidate is still saved and registered, so the Model page can
**show** the rejection rather than the event disappearing into a log file.

---

## Model Versioning

```
  C:\mlops-data\models\          PERSISTENT authoritative store
      model_v1.pkl                   every version ever produced,
      model_v2.pkl                   including rejected candidates
      model_v3.pkl

  <workspace>\models\            only the LIVE model, copied by
      model_v2.pkl               stage_model.py before docker build

  Docker image                   the one approved model, baked in
```

Version numbers are sequential and never reused. Exactly one row may hold
`status = 'production'`, enforced by `promote_to_production()`, which demotes
the incumbent to `archived` in the same transaction. Rejected versions keep
their status permanently — a rejection is a historical record, not a former
production model.

Archived artifacts are retained, so a manual rollback is possible.

---

## Flask Application

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | feedback form |
| `/feedback` | POST | submit feedback |
| `/dashboard` | GET | statistics and model status |
| `/model` | GET | model versions and metrics |
| `/predict` | POST | predict without storing (JSON) |
| `/api/stats` | GET | dashboard data as JSON |
| `/api/model` | GET | model data as JSON |
| `/health` | GET | liveness probe |

`app.py` contains **no SQL and no scikit-learn**. Everything is delegated to
`database.py` and `model_loader.py`.

**Validation.** Text must be 3–1000 characters; rating must be an integer
1–5. Invalid form submissions return 400 with the error displayed and the
typed text preserved. Invalid API requests return 400 with a JSON error.
Verified by 28 tests, each asserting both the status code and that nothing
reached the database.

**Graceful degradation.** `/health` returns **200 even with no model loaded**,
reporting the model state separately — the service being up and a model being
deployed are different facts, and the deploy stage needs both. `/predict`
returns 503. Feedback is still stored, because its value as training data
does not depend on a model existing.

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
RUN mkdir -p /data/models
EXPOSE 5000
CMD ["python", "app.py"]
```

### Differences from the Docker practical, and why

| Change | Reason |
|---|---|
| `python:3.12-slim`, not `3.9-slim` | the model is trained on Python 3.12 with scikit-learn 1.5.1; loading that pickle under a different version raises `InconsistentVersionWarning` and can fail |
| `ENV MLOPS_DATA_DIR=/data` | redirects SQLite out of the container's writable layer into the mounted volume |
| `ENV PYTHONUNBUFFERED=1` | otherwise Flask's logs do not reach `docker logs` promptly, making a failed deployment hard to diagnose |
| volume mount | so collected feedback survives the `rm`/`run` cycle Jenkins performs on every deploy |

### Verified behaviour

* Build context reduced to **244 kB** by `.dockerignore`; image 645 MB.
* **No `.db` file anywhere in the image** — confirmed by searching the
  filesystem inside a running container.
* The staged `model_v1.pkl` **is** present at `/app/models/`.
* Feedback **survived** container `stop` + `rm` + recreate: 3 rows before, 3
  rows after.
* The full test suite passes **inside** the built image.

### An architectural finding

A container started **without** the volume reports `model_loaded: false`. The
`.pkl` is in the image, but the *database* holds the record of which version
is production. This is correct design — the image carries the artifact, the
database carries the deployment decision — and it means the volume is
mandatory, not optional.

---

## Jenkins CI/CD

> **An actual Jenkins build has not been executed.** See *Limitations /
> Pending Verification*.

Declarative pipeline, twelve stages, Windows `bat` throughout.

```
   cron('H/5 * * * *')
          v
   Checkout -> Install Dependencies -> Bootstrap Model -> Run Tests
          │                                                  │
          │                                     tests fail ──┴--> STOP
          v
   Check Retraining Requirement
    ┌─────┴──────────────────────────┐
    │ exit 0                          │ exit 10
    v                                 v
  SKIP everything, finish GREEN   Train Candidate
                                      v
                                  Evaluate Candidate
                                      v
                                  QUALITY GATE --FAIL--> STOP (no deploy)
                                      │ PASS
                                      v
                                  Stage Production Model
                                      v
                                  Build Docker Image (:v${BUILD_NUMBER})
                                      v
                                  Deploy (rm -f, run -d, poll /health)
                                      v
                                  Selenium UI Verification
```

### Why a cron trigger rather than Poll SCM

Feedback lives in SQLite, not in Git. Poll SCM watches for **code** changes
and would never notice new **data**. The cron trigger wakes the pipeline to
ask the database whether enough feedback has arrived. This is the concrete
difference between traditional CI/CD and MLOps: the pipeline is triggered by
data as well as by code.

### Why most runs do almost nothing

Below the threshold the pipeline trains nothing, builds nothing and redeploys
nothing — it prints `RETRAINING NOT REQUIRED` and finishes green. Rebuilding
an unchanged model every five minutes would be pointless churn and would
restart the live service for no reason.

### Windows adaptations

| Purpose | Linux (practical) | Windows (this project) |
|---|---|---|
| run a step | `sh 'python train.py'` | `bat 'python train.py'` |
| variable | `${BUILD_NUMBER}` | `%BUILD_NUMBER%` |
| tolerate missing container | `docker rm -f x \|\| true` | `bat(returnStatus: true, script: 'docker rm -f x')` |

### Runtime prerequisite

Jenkins must run as an account in the local `docker-users` group. The MSI
installer's default `LocalSystem` is not a member and cannot reach the Docker
named pipe; this project's Python also lives under the user profile, which
`LocalSystem` does not have on its PATH. `jenkins_env_check.bat` verifies all
of this before the pipeline is debugged.

---

## Selenium Testing

Python Selenium driven by pytest — one language across the whole project, no
`npm` dependency.

| Group | Tests |
|---|---|
| Application loads / navigation | 2 |
| Feedback workflow | 4 |
| Preview prediction (3 classes + screenshot) | 4 |
| Dashboard | 2 |
| Model / MLOps page | 3 |
| **Total** | **15** |

Practical concepts used: **locators** (`By.ID`, `By.CSS_SELECTOR`,
`By.TAG_NAME`), **explicit waits** (`WebDriverWait` with
`expected_conditions`), the **`Select`** class for the rating dropdown,
**assertions** in every test, and **screenshots** as evidence.

Explicit waits are used exclusively — mixing them with `implicitly_wait()` is
a documented Selenium pitfall that produces unpredictable timeouts.

Stable element IDs were designed into the templates from the start
(`#feedback_text`, `#rating`, `#submit_feedback`, `#predicted_sentiment`,
`#actual_sentiment`, `#prediction_match`, `#total_feedback`, `#model_version`,
`#pending_samples`, `#production_version`), so no fragile XPath is needed.

**Assertions compare against the API.** Rather than merely checking that
digits appeared, the dashboard and model tests fetch `/api/stats` and assert
the rendered values match — proving the UI shows real data.

**Baseline protection.** The tests submit genuine feedback through the
genuine form. A session fixture records the highest feedback id beforehand
and deletes everything above it afterwards, leaving the database exactly as
it started. Verified after every run.

Environment: `BASE_URL` (default `http://localhost:5000`) and `HEADLESS`
(default true, since a Jenkins service cannot open a window under Windows
Session 0 isolation).

---

## Results

All figures below were measured during development. Nothing is estimated.

### Primary results table

| Component | Result |
|---|---|
| Model v1 | **0.9167 accuracy** |
| Candidate accepted | **Verified** |
| Candidate rejected | **Verified** |
| Pytest | **130 passed** |
| Selenium | **15 passed** |
| Docker | **Verified** |
| Container persistence | **Verified** |
| Jenkinsfile syntax | **Verified** |
| **Actual Jenkins build** | **NOT YET VERIFIED** |

### Model performance

| Version | Accuracy | Precision | Recall | F1 | Samples | Outcome |
|---|---|---|---|---|---|---|
| v1 baseline | 0.9167 | 0.9196 | 0.9167 | 0.9145 | 288 | production |
| v2 candidate (good data) | 0.9306 | 0.9315 | 0.9306 | 0.9302 | 298 | **accepted** |
| v3 candidate (poisoned) | 0.7778 | — | — | — | 350 | **rejected** |

All scored on the same fixed 72-sample holdout set.

### Quality gate calibration

| Injected bad rows | Candidate accuracy | Gate at 0.85 |
|---|---|---|
| 0 | 0.9167 | pass |
| 20 | 0.8750 | pass |
| 30 | 0.8611 | pass |
| 40 | 0.8194 | **FAIL** |
| 50 | 0.7778 | **FAIL** |

### Testing

| Suite | Tests | Result |
|---|---|---|
| `tests/test_model.py` | 16 | passed |
| `tests/test_api.py` | 19 | passed |
| `tests/test_validation.py` | 28 | passed |
| `tests/test_database.py` | 30 | passed |
| `tests/test_retraining.py` | 23 | passed |
| `tests/test_model_safety.py` | 14 | passed |
| **Unit / integration total** | **130** | **passed** |
| `selenium_tests/test_feedback_ui.py` | 15 | passed |
| Selenium vs local Flask | 15 | passed |
| Selenium vs Docker container | 15 | passed |
| Full suite inside the Docker image | 130 | passed |

Running the suite inside the built image confirms the artifact itself is
sound on Linux with the pinned dependencies, not merely on the development
host.

### Docker

| Check | Result |
|---|---|
| Build context size | 244 kB |
| Image size | 645 MB |
| Database baked into image | **None** — verified by filesystem search |
| Staged model in image | `/app/models/model_v1.pkl`, 10289 bytes |
| Feedback before teardown | 3 rows |
| Feedback after `stop` + `rm` + recreate | **3 rows** |

### Jenkins pipeline simulation

Every stage command executed in order with real exit codes. **This is a
simulation, not a Jenkins build.**

| Stage | Below threshold | Good data | Poisoned data |
|---|---|---|---|
| Run Tests | passed | — | — |
| Check Retraining | exit 0 → false | exit 10 → true | exit 10 → true |
| Train Candidate | **skipped** | exit 0 | exit 0 |
| Evaluate Candidate | **skipped** | 0.9306 | 0.7917 |
| Quality Gate | **skipped** | **PASSED**, v2 promoted | **FAILED, exit 1** |
| Stage Production Model | **skipped** | exit 0 | **not reached** |
| Build Docker Image | **skipped** | `smart-feedback:v42` | **not reached** |
| Deploy | **skipped** | healthy | **not reached** |
| Selenium | **skipped** | skipped | **not reached** |

---

## Limitations / Pending Verification

### Actual Jenkins execution is NOT verified

**Jenkins 2.568.3 could not be started on the development machine.** This is
not a project defect and not a Jenkins misconfiguration.

**Symptom** — Jetty fails during startup:

```
java.io.IOException: Unable to establish loopback connection
java.net.SocketException: Invalid argument: connect
  at sun.nio.ch.PipeImpl$Initializer$LoopbackConnector.run
```

**Diagnosis** — isolated with a minimal Java program:

| Operation | Result |
|---|---|
| Plain TCP loopback connect | OK |
| `Pipe.open()` | OK |
| `Selector.open()` | **FAILS** |

`Selector.open()` fails identically on **JDK 21.0.11, 24 and 25**, under the
default selector, `WindowsSelectorProvider` and `WEPollSelectorProvider`, and
with `-Djava.net.preferIPv4Stack=true`. It also fails outside any sandbox.
Any Java NIO server — Jenkins, Tomcat, Jetty, Spring Boot — would fail
identically on this machine.

**Likely cause.** Three antivirus products are installed simultaneously
(Quick Heal AntiVirus Pro, Windows Defender, McAfee). One of their network
drivers appears to reject the loopback socket pair that NIO selectors create.
Windows Firewall was checked: every `java.exe` rule is *Allow*, with no block
rules.

**Remedies, all requiring administrator rights** (unavailable on the
development account): `netsh winsock reset` plus a reboot; an antivirus
exclusion for `java.exe`; or removing the redundant antivirus products.

### Verification status, stated precisely

| Claim | Status |
|---|---|
| Application verified locally | **Verified** |
| Application verified inside Docker | **Verified** |
| `Jenkinsfile` Groovy syntax | **Verified** — parsed with `groovy-all-2.4.21.jar` extracted from `jenkins.war` |
| Jenkins pipeline logic | **Simulated** — all stages, all three scenarios |
| Jenkins runtime prerequisites | **Verified** — all six checks pass as the target user |
| **Actual Jenkins build** | **NOT YET VERIFIED** |

What the simulation covered: every stage command, in order, with real exit
codes, across all three scenarios. What remains unproven: Jenkins' own
orchestration — `when` expression evaluation, `checkout scm`, artifact
archiving, the cron trigger firing, and the stage view.

### Other limitations

* **The dataset is synthetic** — 360 hand-written rows, artificially balanced.
  Real feedback would be noisier and skewed towards positive ratings.
* **The model is intentionally simple.** TF-IDF + Logistic Regression was
  chosen for explainability over peak accuracy. It cannot handle sarcasm,
  negation scope or unseen vocabulary.
* **Single-node deployment.** One container on one host, no orchestration or
  automated rollback. Archived artifacts are retained so manual rollback is
  possible.
* **No authentication.** Anyone who can reach the port can submit feedback.
* **Threshold-based triggering only.** Retraining is triggered by sample
  count, not by detected drift or by a drop in live accuracy.

---

## Practical Mapping

| Practical | Project component | What is demonstrated |
|---|---|---|
| **Experiment 1** — Explore Git, GitHub Commands and Source Code Management | repository, `.gitignore`, branch strategy | `init`, `add`, `status`, `commit`, `log`, `branch`, `checkout`, `remote add origin`, `push`, `pull` |
| **Experiment 2** — Jenkins installation and setup | Jenkins on `localhost:8080`, `jenkins_env_check.bat` | installing and exploring the CI/CD environment; the practical's `.war` method was the one attempted here |
| **Experiment 3** — Demonstrate CI/CD using Jenkins | `jenkins_env_check.bat` as a Freestyle job | Freestyle projects, *Execute Windows batch command*, Console Output, non-zero exit codes — and the basis for choosing `bat` over `sh` |
| **Experiment 4** — Explore Docker commands for content management | container lifecycle commands; the pipeline's teardown step | `docker ps`, `stop`, `rm`, `images`, `logs`, image/container distinction |
| **Experiment 5** — Develop a simple containerized application | `Dockerfile`, Flask in a container, `-p 5000:5000` | Dockerfile authoring, building an image, port mapping, running a Flask app in Docker |
| **PS4 — CI/CD Pipeline / ML CI-CD Practical** | `Jenkinsfile`, `train.py`, `tests/test_model.py`, the **quality gate** | Freestyle vs Pipeline-as-Code, Groovy stages, `Code + Data + Model`, retrain → evaluate → package → deploy, and the quality-gate concept this project is built around |
| **Experiment 9** — Deploying Applications/Models using Docker through Jenkins | `app.py` `/predict`, Deploy stage, `healthcheck.py` | training vs inference, baking the model artifact into the image, `docker rm -f` before `docker run`, testing the live API with `Invoke-RestMethod` |
| **Selenium (Python practical)** | `selenium_tests/test_feedback_ui.py` | `By` locators, implicit vs explicit waits, the `Select` class, assertions, `save_screenshot` |
| **Selenium (Node.js practical)** | the same suite, in Python | concepts adapted rather than the language — see below |

### Note on the two Selenium practicals

The first Selenium practical is Python; the second is Node.js with
`selenium-webdriver`. This project uses **Python Selenium throughout**,
keeping the entire codebase in one language, reusing the existing pytest
infrastructure, and avoiding an `npm` dependency for a single test suite.

The Node.js practical's concepts are nonetheless all present, translated:

| Node.js practical concept | Python equivalent used here |
|---|---|
| `async/await` to force sequential execution | not required — the Selenium Python binding is synchronous by design, so the ordering problem `async/await` solves does not arise |
| `driver.wait(until.elementLocated(...), 10000)` | `WebDriverWait(driver, 15).until(EC.presence_of_element_located(...))` |
| Node's built-in `assert` module | pytest's plain `assert` statement |
| `fs.writeFileSync('shot.png', image, 'base64')` | `driver.save_screenshot(path)` |

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
   code; an ML pipeline must version code, data *and* model, and must be
   triggered by data changes as well as code changes. The cron trigger exists
   precisely because feedback lives in SQLite rather than Git.

2. **An automated pipeline without a quality gate is a liability.** The most
   valuable stage in this project is the one that *refuses* to deploy.
   Demonstrating rejection matters more than demonstrating success.

3. **Evaluation methodology determines whether metrics mean anything.** The
   fixed holdout set was the single most important design decision. Without
   it, "candidate 0.93 vs production 0.92" compares two numbers measured on
   different data.

4. **Data quality dominates model choice at small scale.** Rebuilding the
   dataset for vocabulary recurrence moved accuracy from 0.5833 to 0.9167.
   No amount of hyperparameter tuning would have achieved that.

5. **Test assumptions by measuring them.** Bigrams were expected to help and
   measurably hurt. Cross-validation, not intuition, chose the final
   configuration.

6. **Containers separate the immutable from the mutable.** Discovering that a
   container without its volume reports `model_loaded: false` clarified the
   architecture: the image carries the artifact, the database carries the
   deployment decision.

7. **Cross-platform bugs hide until you actually run the thing.** A Windows
   path stored in the database could not be parsed by `os.path.basename()`
   inside a Linux container. No unit test would have caught it — only running
   the container did.

8. **Tests find real defects, not just regressions.** Writing a test for
   duplicate model versions exposed a connection leak in every database
   function: a failed statement skipped `close()`, and on Windows the leaked
   handle locked the database file.

9. **Honest reporting is part of engineering.** Jenkins could not be started
   on this machine. Recording that precisely — with the diagnosis and the
   distinction between simulated and verified — is more useful than a vague
   claim of success.

---

## Conclusion

The project delivers a working, feedback-driven MLOps lifecycle. Feedback
submitted through a Flask application becomes labelled training data
automatically; crossing a configurable threshold triggers retraining; every
candidate is scored on a permanently frozen holdout set; and a two-condition
quality gate decides whether it is fit to deploy. Approved models are
versioned, baked into a Docker image and deployed, then verified in a real
browser.

Both quality-gate outcomes were demonstrated with measured results: a genuine
improvement was promoted from 0.9167 to 0.9306, and a deliberately poisoned
candidate scoring 0.7778 was rejected with the production model left
untouched and its training samples preserved for the next attempt. 130 unit
and integration tests and 15 Selenium browser tests pass, against both a
local server and the deployed container.

The central lesson is that automated retraining is only safe when it is
guarded. A pipeline that deploys whatever it just trained is worse than no
pipeline at all, because it automates degradation. The quality gate — and the
fixed holdout set that makes its comparison meaningful — is what turns
automation into something trustworthy.

One objective is incompletely met. Jenkins could not be started on the
development machine because of a system-level Java networking fault
unrelated to this project. The pipeline is written, syntax-validated and
logically simulated across all three scenarios, and every command it invokes
is independently verified — but an actual Jenkins build has not run, and this
report does not claim otherwise.

---

## Future Scope

1. **Drift-based triggering.** Retrain when live prediction accuracy drops or
   the input distribution shifts, rather than on a fixed sample count. The
   `prediction_correct` column already records what is needed.
2. **Automated rollback.** Archived artifacts are retained, so redeploying a
   previous version could be a single pipeline stage.
3. **Human approval gate.** Jenkins `input` could require sign-off before a
   new model goes live, useful in a regulated setting.
4. **Richer models.** Word embeddings or a small transformer, with the
   existing quality gate deciding objectively whether the added complexity
   actually helps.
5. **Real dataset.** Replace the synthetic seed data with genuinely collected
   feedback.
6. **Experiment tracking.** MLflow or similar for richer comparison than the
   `model_versions` table provides.
7. **Authentication and rate limiting** before any real deployment.
8. **Complete the Jenkins verification** on a machine with a working Java NIO
   stack — the immediate next step.

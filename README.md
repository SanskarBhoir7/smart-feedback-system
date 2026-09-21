# Smart Feedback System with Auto-Retraining Pipeline

A mini project demonstrating a complete, automated MLOps lifecycle: user
feedback becomes labelled training data, a threshold triggers retraining, a
quality gate decides whether the new model is good enough to deploy, and only
then is it packaged into Docker, deployed, and verified in a real browser.

> **Verification status.** Everything in this README has been executed except
> a live Jenkins build. Jenkins could not start on the development machine
> (see [Limitations](#19-limitations--pending-verification)). The `Jenkinsfile`
> is syntax-validated and its full stage logic was simulated, but an actual
> Jenkins run is **still pending**.

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
9. [Model training](#9-model-training)
10. [Running locally](#10-running-locally)
11. [Feedback workflow](#11-feedback-workflow)
12. [Auto-retraining](#12-auto-retraining)
13. [Quality gate](#13-quality-gate)
14. [Docker](#14-docker)
15. [Jenkins setup](#15-jenkins-setup)
16. [Jenkins pipeline explained](#16-jenkins-pipeline-explained)
17. [Tests](#17-tests)
18. [Selenium](#18-selenium)
19. [Limitations / pending verification](#19-limitations--pending-verification)
20. [Troubleshooting](#20-troubleshooting)
21. [Demonstration procedure](#21-demonstration-procedure)
22. [Practical mapping](#22-practical-mapping)

---

## 1. Project overview

Users submit written feedback with a 1–5 star rating. The current production
model predicts the sentiment of the text; the star rating independently gives
the *true* label. Both are stored, so **every submission becomes a labelled
training sample**.

Once enough new samples accumulate, the pipeline trains a candidate model,
scores it on a frozen holdout set, and puts it through a quality gate. Only a
candidate that clears the gate is packaged into a Docker image and deployed.
A candidate that fails is recorded and discarded, and the existing production
model keeps serving.

The defining feature is not sentiment analysis. It is this loop:

```
FEEDBACK -> TRAINING DATA -> RETRAINING -> EVALUATION
         -> QUALITY GATE -> DEPLOYMENT -> BROWSER VERIFICATION
```

## 2. Problem statement

A machine learning model deployed once begins to decay. Real language drifts
away from the data the model was trained on, and nobody notices until the
predictions are visibly poor. Retraining by hand is slow and easy to forget.

But automated retraining introduces a worse danger: a pipeline that deploys
whatever it just trained will eventually replace a good model with a bad one —
especially if the incoming data is noisy or deliberately poisoned.

This project addresses both halves: retraining is automatic, **and** it is
guarded. No model reaches production without passing an explicit, measurable
quality check against a fixed yardstick.

## 3. Objectives

1. Collect feedback through a web application and turn it into labelled
   training data automatically.
2. Train and serve a sentiment classifier through a Flask API.
3. Trigger retraining automatically once a configurable number of new
   samples has been collected.
4. Evaluate every model version on an identical, frozen holdout set.
5. Enforce a quality gate that can **reject** a candidate and protect
   production.
6. Version every model and keep a full, auditable history.
7. Package the approved model into a Docker image as a self-contained
   deployment artifact.
8. Automate the whole lifecycle in a Jenkins pipeline.
9. Verify the deployed application in a real browser with Selenium.

## 4. Architecture

Full diagrams and design rationale: **[docs/architecture.md](docs/architecture.md)**

```
   USER ──> Flask app ──┬──> model predicts sentiment
                        └──> rating gives the true label
                                    │
                                    v
                        SQLite: labelled training data
                                    │
                          >= 10 new samples?
                                    │ yes
                                    v
                      Jenkins: train candidate
                                    v
                  evaluate on the FIXED 72-row holdout
                                    v
                            QUALITY GATE
                          /              \
                      FAIL                PASS
                        │                   │
               keep production        promote, bake into
               block deployment       Docker image, deploy
                                            │
                                            v
                                   Selenium verifies the UI
```

Three separations make the design work:

* **The web app never trains.** It only consumes whichever model is marked
  `production`. Training belongs to `train.py` and `retrain.py`, run by Jenkins.
* **Models are baked into the image; the database is a mounted volume.** The
  image is immutable and self-contained; runtime state survives container
  recreation.
* **The holdout set is frozen.** Every model version is scored on the same 72
  rows, so accuracy numbers are actually comparable.

## 5. Technology stack

| Layer | Choice | Version |
|---|---|---|
| Language | Python | 3.12.6 |
| Web framework | Flask | 3.1.3 |
| Frontend | Jinja2 + plain CSS/JS | — |
| Database | SQLite (`sqlite3`, stdlib) | — |
| ML | scikit-learn (TF-IDF + Logistic Regression) | 1.5.1 |
| Model persistence | joblib | 1.4.2 |
| Numerics | numpy / scipy | 1.26.4 / 1.14.1 |
| Unit tests | pytest | 8.3.4 |
| Browser tests | Selenium (Python) | 4.27.1 |
| Containers | Docker | 29.6.2 |
| CI/CD | Jenkins (Declarative Pipeline) | 2.568.3 |
| SCM | Git | 2.55.0 |

Versions are pinned in `requirements.txt` so the model artifact trained on the
host loads correctly inside the container.

## 6. Folder structure

```
smart-feedback-system/
├── app.py                   Flask routes, validation, rendering. No SQL, no ML.
├── config.py                Every threshold and path, in one place.
├── database.py              All SQLite access. Flask-free.
├── dataset.py               Data assembly + the model pipeline definition.
├── model_loader.py          Loads and caches the production model.
├── prepare_data.py          One-time split into training pool + fixed holdout.
├── train.py                 Initial training (creates v1).
├── evaluate.py              Accuracy / precision / recall / F1.
├── retrain.py               --check | --train | --gate
├── stage_model.py           Copies the live model into the workspace for Docker.
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
│   └── holdout.csv          72 rows   (FROZEN, never trained on)
│
├── docs/
│   └── architecture.md
│
├── models/                  Staged production model (gitignored)
├── build/                   Candidate model between stages (gitignored)
│
├── templates/               base, index, dashboard, model, error
├── static/css/style.css
├── static/js/main.js
│
├── tests/                   130 unit + integration tests
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_validation.py
│   ├── test_database.py
│   ├── test_retraining.py
│   └── test_model_safety.py
│
└── selenium_tests/          15 browser tests
    ├── conftest.py
    ├── test_feedback_ui.py
    └── screenshots/
```

Two directories live **outside** the repository:

```
C:\mlops-data\
├── feedback.db              Runtime database
└── models\
    └── model_v1.pkl         Authoritative store: every version ever produced
```

## 7. Installation

Prerequisites: Python 3.12, Git, Docker Desktop, Google Chrome. Java 17 or 21
is required only for Jenkins.

```bash
git clone <your-repository-url>
cd smart-feedback-system
```

```bash
python -m pip install -r requirements.txt
```

Create the persistent data directory (PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path C:\mlops-data\models
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
| `actual_sentiment` | derived from the rating (the truth) |
| `prediction_correct` | 0/1, or NULL if no model was available |
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
| `notes` | the quality gate's reason |

This table is the **single source of truth** for which model is live. There is
no separate registry file that could drift out of sync.

## 9. Model training

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

**Measured baseline (v1):**

```
Accuracy   0.9167      Precision  0.9196
Recall     0.9167      F1-Score   0.9145      (72-sample fixed holdout)
```

## 10. Running locally

```bash
python app.py
```

Then open <http://localhost:5000>.

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

Test the API from PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body '{"text": "excellent service and very helpful staff"}'
```

Returns `{"prediction": "positive", "model_version": 1}`.

`/health` deliberately returns **200 even when no model is loaded** — the
service being up and a model being deployed are two different things, and the
Jenkins deploy stage needs to distinguish them.

## 11. Feedback workflow

1. Enter feedback text and choose a rating (name is optional).
2. The production model predicts the sentiment.
3. The rating is converted to the true label: **1–2 → negative, 3 → neutral,
   4–5 → positive**.
4. Both are stored, along with whether they agreed.
5. The result panel shows the prediction, the actual sentiment, whether they
   matched, and the model version that answered.
6. The pending-sample counter advances towards the retraining threshold.

**Preview Prediction** asks the model for an answer via `POST /predict`
without storing anything — useful for demonstrating the API from the UI.

## 12. Auto-retraining

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

**Training data grows; the holdout set never does.** New feedback always
becomes training data, so the yardstick stays fixed and model versions remain
comparable.

## 13. Quality gate

A candidate must satisfy **both** conditions:

| | Condition | Default |
|---|---|---|
| **A** | `accuracy >= MIN_ACCURACY` | 0.85 |
| **B** | `accuracy >= production_accuracy − TOLERANCE` | 0.02 |

Condition B is what stops a working model being silently replaced by a worse
one. Both models are re-scored on the same frozen holdout, so the comparison
is valid.

`MIN_ACCURACY = 0.85` was chosen from measurement, not invented:

* the real baseline scored **0.9167** — 6.7 points of headroom, so an honest
  retrain will not trip the gate by accident;
* a deliberately poisoned model scored **0.7778** — the gate discriminates;
* 85% is also the example threshold used in the CI/CD practical.

Accepted:

```
QUALITY GATE: PASSED
Candidate model accepted.
New model version    : v2
Previous version     : v1 (now archived)
```

Rejected:

```
Candidate Accuracy   : 0.7778
Required Accuracy    : 0.8500
Production Accuracy  : 0.9306
QUALITY GATE: FAILED
Candidate recorded as v3 with status 'rejected'.
Production model v2 REMAINS ACTIVE.
Deployment is blocked.
```

A rejection **does not consume the training samples** — they stay pending so
the next run tries again as more feedback arrives.

## 14. Docker

Stage the approved model into the build context, then build:

```powershell
python stage_model.py --clean
```

```powershell
docker build -t smart-feedback:v1 -t smart-feedback:latest .
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

Differences from the Docker practical, and why:

| Change | Reason |
|---|---|
| `python:3.12-slim` instead of `3.9-slim` | the model is trained on Python 3.12 with scikit-learn 1.5.1; loading that pickle under a different version can fail |
| `ENV MLOPS_DATA_DIR=/data` | redirects SQLite out of the container's writable layer into the volume |
| `ENV PYTHONUNBUFFERED=1` | without it Flask's logs do not reach `docker logs` promptly, making a failed deploy hard to diagnose |
| volume mount | so feedback survives the `rm`/`run` cycle Jenkins performs on every deployment |

## 15. Jenkins setup

> Jenkins execution is **not yet verified** on the development machine — see
> [Limitations](#19-limitations--pending-verification).

**Before anything else, verify the runtime.** Create a Freestyle job with one
*Execute Windows batch command* step:

```bat
call jenkins_env_check.bat
```

It checks `python`, `git`, `docker --version`, `docker ps`, read/write access
to `C:\mlops-data`, `retrain.py --check`, `pytest`, and `stage_model.py`, then
reports a pass/fail verdict.

**Critical: which account runs Jenkins.** The MSI installer defaults to
`LocalSystem`, which is **not** a member of the local `docker-users` group and
therefore cannot reach the Docker named pipe — `docker ps` will fail. This
project's Python also lives under the user profile, which `LocalSystem` does
not have on its PATH.

> Run Jenkins as your own account. Do not loosen Docker's permissions to work
> around it.

To change it: `services.msc` → Jenkins → Properties → *Log On* → *This
account* → your user → restart the service. Group membership is only read at
logon, so the restart is required.

Then create the pipeline job:

1. **New Item** → name it `smart-feedback-pipeline` → **Pipeline** → OK
2. **Pipeline** → *Pipeline script from SCM* → Git → your repository URL
3. Script Path: `Jenkinsfile`
4. Save → **Build Now**

The `Jenkinsfile` declares `triggers { cron('H/5 * * * *') }`, so after the
first manual build Jenkins polls every five minutes on its own.

## 16. Jenkins pipeline explained

Twelve stages. Everything after the check is conditional.

| # | Stage | Runs | On failure |
|---|---|---|---|
| 1 | Checkout | always | stop |
| 2 | Install Dependencies | always | stop |
| 3 | Bootstrap Model | always (idempotent) | stop |
| 4 | Run Tests | always | **stop — nothing is built** |
| 5 | Check Retraining Requirement | always | stop |
| 6 | Train Candidate | only if required | stop |
| 7 | Evaluate Candidate | only if required | stop |
| 8 | **Quality Gate** | only if required | **stop — no deployment** |
| 9 | Stage Production Model | only if required | stop |
| 10 | Build Docker Image | only if required | stop |
| 11 | Deploy | only if required | stop |
| 12 | Selenium UI Verification | only if required | stop |

**Why a cron trigger and not Poll SCM.** Feedback is stored in SQLite, not in
Git. A Poll SCM trigger watches for *code* changes and would never notice new
*data*. The cron trigger wakes the pipeline up to ask the database whether
enough feedback has arrived.

**Why most runs do almost nothing.** If the threshold has not been reached,
the pipeline trains nothing, builds nothing and redeploys nothing. It prints
`RETRAINING NOT REQUIRED` and finishes green. Rebuilding an unchanged model
every five minutes would be pointless churn and would restart the live
service for no reason.

**Windows.** Every step uses `bat`, not `sh`. The practicals show `sh` because
they assume a Linux agent; this Jenkins runs on Windows, where `sh` fails.
Experiment 3 uses *Execute Windows batch command* for the same reason.

| Purpose | Linux (practical) | Windows (this project) |
|---|---|---|
| run a step | `sh 'python train.py'` | `bat 'python train.py'` |
| variable | `${BUILD_NUMBER}` | `%BUILD_NUMBER%` |
| tolerate missing container | `docker rm -f x \|\| true` | `bat(returnStatus: true, script: 'docker rm -f x')` |

## 17. Tests

```bash
python -m pytest -q
```

**Result: 130 passed.**

| Module | Tests | Covers |
|---|---|---|
| `test_model.py` | 16 | artifact exists, loads, all three classes |
| `test_api.py` | 19 | all eight routes |
| `test_validation.py` | 28 | empty, short, invalid, malformed input |
| `test_database.py` | 30 | insert, retrieve, versioning, promotion |
| `test_retraining.py` | 23 | threshold, candidate, gate accept/reject |
| `test_model_safety.py` | 14 | holdout isolation, shared test set |

`pytest.ini` sets `testpaths = tests`, so the Selenium suite is excluded from
the default run — browser tests need a running server, which unit tests must
never depend on.

The suite is fully isolated: `tests/conftest.py` redirects the database, the
model store, the build directory and the staging directory into a temporary
folder **before any project module is imported**, and aborts the run if that
redirection ever fails. Running the tests cannot touch `C:\mlops-data`.

## 18. Selenium

Python Selenium, driven by pytest, against a running application.

```bash
python -m pytest selenium_tests -v
```

**Result: 15 passed.**

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `BASE_URL` | `http://localhost:5000` | the target |
| `HEADLESS` | `true` | `false` shows the browser |

```powershell
$env:HEADLESS = "false"; python -m pytest selenium_tests -v
```

Headless is the default because a Jenkins service cannot open a visible window
on Windows (Session 0 isolation). Set `HEADLESS=false` for a live
demonstration. `1`, `true`, `yes`, `on` all mean headless; `0`, `false`, `no`
mean visible.

The same tests run unchanged against the local Flask server and the Docker
container — both verified.

Concepts from the Selenium practical, all used: **locators** (`By.ID`,
`By.CSS_SELECTOR`, `By.TAG_NAME`), **explicit waits** (`WebDriverWait` with
`expected_conditions`), the **`Select`** class for the rating dropdown,
**assertions** in every test, and **screenshots** as evidence.

> The suite uses explicit waits only, not `implicitly_wait()`. Mixing the two
> is a documented Selenium pitfall — the implicit wait interferes with
> `WebDriverWait`'s polling and produces unpredictable timeouts.

Screenshots are written to `selenium_tests/screenshots/` and archived by
Jenkins:

```
01_feedback_submission_<timestamp>.png
02_preview_prediction_<timestamp>.png
03_dashboard_<timestamp>.png
04_model_page_<timestamp>.png
```

**Baseline protection.** These tests submit genuine feedback through the
genuine form. A session fixture records the highest feedback id before the run
and deletes everything above it afterwards, so the development database ends
exactly as it started.

## 19. Limitations / pending verification

### Actual Jenkins execution is NOT verified

Jenkins 2.568.3 **could not be started** on the development machine. It is not
a Jenkins configuration problem and not a project defect.

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

Any Java NIO server — Jenkins, Tomcat, Jetty, Spring Boot — would fail the
same way on this machine. The likely cause is one of the **three antivirus
products installed simultaneously** (Quick Heal, Windows Defender, McAfee)
intercepting the loopback socket pair that NIO selectors create. Windows
Firewall rules were checked and all `java.exe` entries are *Allow*.

**Remedies, all requiring administrator rights:**

1. `netsh winsock reset` followed by a reboot.
2. Add an antivirus exclusion for `java.exe`.
3. Remove the redundant antivirus products.

### What this means for the project

| Claim | Status |
|---|---|
| Application verified locally | **Verified** |
| Application verified in Docker | **Verified** |
| `Jenkinsfile` Groovy syntax | **Verified** — parsed with `groovy-all-2.4.21.jar` from `jenkins.war` |
| Jenkins pipeline logic | **Simulated** — every stage command run in order with real exit codes |
| Jenkins runtime prerequisites | **Verified** — all six checks pass as the target user |
| **Actual Jenkins build** | **NOT YET VERIFIED** |

The simulation covered all three scenarios (below threshold, successful
retraining, failed quality gate) and exercised every command, exit code and
conditional the pipeline relies on. What remains unproven is Jenkins' own
orchestration: `when` evaluation, `checkout scm`, artifact archiving and the
stage view.

### Other limitations

* **The dataset is synthetic.** 360 hand-written rows, balanced 120 per class.
  Real feedback would be noisier and less balanced.
* **The model is intentionally simple.** TF-IDF + Logistic Regression was
  chosen for explainability, not peak accuracy.
* **Single-node deployment.** One container on one host; no orchestration,
  scaling or rollback automation. Archived `.pkl` files are retained, so a
  manual rollback is possible.
* **No authentication.** Anyone who can reach the port can submit feedback.

## 20. Troubleshooting

**`docker ps` fails from Jenkins but works in your terminal**
The Jenkins service account is not in `docker-users`. Run Jenkins as your own
user and restart the service — group membership is only read at logon. Do not
loosen Docker's permissions. Run `jenkins_env_check.bat` to confirm.

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

**Selenium cannot find Chrome**
Selenium 4.27's Selenium Manager resolves chromedriver automatically; ensure
Chrome is installed. In Jenkins, always run headless — a service cannot open
a window.

**Selenium tests fail immediately with "application is not responding"**
Nothing is serving on `BASE_URL`. Start `python app.py` or the container
first; the suite exits early with this message rather than producing a wall
of confusing browser errors.

**`retrain.py --check` seems to do nothing**
That is correct below the threshold. It prints `RETRAINING NOT REQUIRED` and
exits 0. Check the pending count on the dashboard.

**The quality gate keeps failing**
Check the Model page for the rejected version and its recorded reason. If you
injected bad data for a demonstration, undo it:

```bash
python seed_bad_feedback.py --undo
```

**Jenkins will not start (`Unable to establish loopback connection`)**
See [Limitations](#19-limitations--pending-verification). Requires
administrator rights to resolve.

**Tests pass individually but fail together**
Should not happen — every test gets a fresh temporary database. If it does,
check that `tests/conftest.py` still redirects the paths before importing any
project module.

## 21. Demonstration procedure

### Setup

```powershell
python prepare_data.py
```

```powershell
python train.py
```

```powershell
python stage_model.py --clean
```

```powershell
docker build -t smart-feedback:v1 -t smart-feedback:latest .
```

```powershell
docker run -d -p 5000:5000 -v "C:/mlops-data:/data" --name smart_feedback_app smart-feedback:latest
```

```powershell
python healthcheck.py --require-model
```

### Scenario A — normal operation

1. Open <http://localhost:5000> and submit one piece of feedback.
2. Show the result: predicted sentiment, actual sentiment, whether they
   matched, and the model version.
3. Open the Dashboard — the counts and the pending-sample counter have moved.
4. Open Model / MLOps — v1 in production with its four metrics.
5. Run `python retrain.py --check` → **exit 0**, `RETRAINING NOT REQUIRED`.

*Point to make:* below the threshold the pipeline builds and deploys nothing.

### Scenario B — auto-retraining and promotion

1. Submit feedback until the counter reaches **10 / 10**.
2. `python retrain.py --check` → **exit 10**, `RETRAINING REQUIRED`.
3. `python retrain.py --train`
4. `python evaluate.py --candidate --detailed`
5. `python retrain.py --gate` → **QUALITY GATE: PASSED**, v2 promoted, v1 archived.
6. Refresh Model / MLOps — v2 is production, v1 is the previous version.

*Point to make:* using the app produced the data that improved the model.

### Scenario C — failed quality gate (the important one)

```powershell
python seed_bad_feedback.py
```

Injects 50 deliberately mislabelled rows — positive text rated 1 star,
negative text rated 5.

```powershell
python retrain.py --check
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

Expect **QUALITY GATE: FAILED**, exit code **1**, the candidate recorded as
`rejected`, and production unchanged. Refresh Model / MLOps to show the
rejected version and its reason.

Then undo:

```powershell
python seed_bad_feedback.py --undo
```

*Point to make:* the pipeline protected production from bad data. This is the
whole reason the gate exists.

### Browser verification

```powershell
$env:HEADLESS = "false"; python -m pytest selenium_tests -v
```

Chrome opens and drives the application; screenshots land in
`selenium_tests/screenshots/`.

### Tests

```powershell
python -m pytest -q
```

## 22. Practical mapping

| Practical | Project component | What is demonstrated |
|---|---|---|
| **Experiment 1** — Git / GitHub | repository, `.gitignore`, branches | `init`, `add`, `commit`, `branch`, `checkout`, `remote add`, `push`, `pull` |
| **Experiment 2** — Jenkins installation & setup | Jenkins on `localhost:8080`, `jenkins_env_check.bat` | installing and configuring the CI/CD environment; the `.war` method from this practical was the one used |
| **Experiment 3** — Jenkins CI (Freestyle) | `jenkins_env_check.bat` as a Freestyle job | *Execute Windows batch command*, Console Output, exit codes — and the reason this project uses `bat` rather than `sh` |
| **Experiment 4** — Docker CLI | container lifecycle in §14, the pipeline's teardown step | `docker ps`, `stop`, `rm`, `images`, `logs` |
| **Experiment 5** — Dockerised Flask application | `Dockerfile`, port mapping `5000:5000` | Dockerfile structure, building an image, port mapping, running a Flask app in a container |
| **PS4 — CI/CD Pipeline / ML CI-CD Practical** | `Jenkinsfile`, `train.py`, `tests/test_model.py`, the **quality gate** | pipeline-as-code, graphical stages, `Code + Data + Model`, and the quality-gate concept this whole project is built around |
| **Experiment 9** — Deploying models via Docker through Jenkins | `app.py` `/predict`, the Deploy stage, `healthcheck.py` | training vs inference, baking `model.pkl` into the image, `docker rm -f` before `docker run`, testing the live API with `Invoke-RestMethod` |
| **Selenium (Python practical)** | `selenium_tests/test_feedback_ui.py` | locators, explicit waits, the `Select` class, assertions, screenshots |
| **Selenium (Node.js practical)** | same suite, implemented in Python | concepts adapted, not the language — see the note below |

**On the two Selenium practicals.** The first is Python, the second is
Node.js. This project uses **Python Selenium throughout**, keeping the whole
codebase in one language and reusing the existing pytest infrastructure with
no `npm` dependency. The Node.js practical's *concepts* are nonetheless all
present, translated to their Python equivalents:

| Node.js practical | Python equivalent used here |
|---|---|
| `async/await` for sequencing | synchronous Python — the Selenium Python binding blocks by design, so the ordering problem `async/await` solves does not arise |
| `driver.wait(until.elementLocated(...))` | `WebDriverWait(...).until(EC.presence_of_element_located(...))` |
| Node's `assert` module | pytest's plain `assert` |
| `fs.writeFileSync` for screenshots | `driver.save_screenshot(...)` |

**PS4 is not given an experiment number.** Its true number cannot be
established from the supplied material, so it is referred to throughout as
*PS4 — CI/CD Pipeline / ML CI-CD Practical*. Experiment 9 references an
"Exp 8: Training & Testing"; PS4 covers exactly that ground and even names its
Jenkins job `MLOps-LabExp8`, but that is not firm enough to assert a number.

---

## Verification summary

| Component | Result |
|---|---|
| Model v1 | **0.9167** accuracy (72-sample fixed holdout) |
| Candidate accepted | **Verified** — v2 promoted at 0.9306 |
| Candidate rejected | **Verified** — 0.7778, production retained |
| Pytest | **130 passed** |
| Selenium | **15 passed** |
| Selenium vs local Flask | **Verified** |
| Selenium vs Docker container | **Verified** |
| Docker build | **Verified** |
| Container persistence | **Verified** — data survived `stop` + `rm` + recreate |
| Holdout isolation | **Verified** — 0 overlap |
| `Jenkinsfile` syntax | **Verified** |
| Jenkins pipeline logic | **Simulated** (3 scenarios) |
| **Actual Jenkins build** | **NOT YET VERIFIED** |

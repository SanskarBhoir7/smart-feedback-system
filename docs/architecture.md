# System Architecture

Smart Feedback System with Auto-Retraining Pipeline

---

## 1. The idea in one picture

The project is not "a feedback website with a model attached". The point is
the closed loop: using the application produces the data that improves the
model, and a quality gate decides whether that improvement is real.

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        v                                                          │
   USER submits feedback                                           │
   (text + 1-5 rating)                                             │
        │                                                          │
        ├──> model PREDICTS sentiment ──────┐                      │
        │                                   │                      │
        └──> rating gives the TRUE label ───┤                      │
                                            v                      │
                                   stored together as              │
                                   LABELLED TRAINING DATA          │
                                            │                      │
                                            v                      │
                              enough new samples? (>= 10)          │
                                            │                      │
                                       yes  v                      │
                                   TRAIN CANDIDATE MODEL           │
                                            │                      │
                                            v                      │
                         EVALUATE on the FIXED 72-row holdout      │
                                            │                      │
                                            v                      │
                                     QUALITY GATE                  │
                                     /              \              │
                                 FAIL                PASS          │
                                  │                   │            │
                        keep production      promote to production │
                        model, block                  │            │
                        deployment                    v            │
                                            bake into Docker image │
                                                      │            │
                                                      v            │
                                              deploy container ────┘
                                                      │
                                                      v
                                            Selenium verifies the UI
```

---

## 2. Component architecture

```
                         ┌──────────────────────────────┐
   Chrome / user ───────>│   Flask app  (port 5000)     │
                         │                              │
                         │  Pages:  /  /dashboard       │
                         │          /model              │
                         │  API:    /predict            │
                         │          /api/stats          │
                         │          /api/model          │
                         │          /health             │
                         └───┬──────────────────┬───────┘
                             │                  │
                  app.py ────┤                  ├──── model_loader.py
              (routes only)  │                  │  (loads the ONE model
                             │                  │   marked 'production')
                             v                  v
                  ┌────────────────────┐  ┌──────────────────┐
                  │  database.py       │  │  models/         │
                  │  (all SQL)         │  │  model_vN.pkl    │
                  └─────────┬──────────┘  └────────▲─────────┘
                            │                      │
                            v                      │
              ┌───────────────────────────┐        │
              │  SQLite: feedback.db      │        │
              │   - feedback              │        │
              │   - model_versions        │        │
              │     (which model is LIVE) │        │
              └─────────┬─────────────────┘        │
                        │                          │
      ┌─────────────────┴──────────────────────────┴──────────────┐
      │                  OFFLINE / PIPELINE SIDE                  │
      │                                                           │
      │   dataset.py    assembles training data + fixed holdout   │
      │   train.py      creates v1                                │
      │   evaluate.py   accuracy / precision / recall / F1        │
      │   retrain.py    --check | --train | --gate                │
      │   stage_model.py copies the live model into the workspace │
      └───────────────────────────────────────────────────────────┘
```

**Key separation:** the web application is a *consumer* of models. It can
never train or promote one. Only `train.py` and `retrain.py` do that, and
Jenkins runs them. This is what stops the running service from quietly
changing the model that is serving users.

---

## 3. Where state lives

This is the single most important design decision in the project, and the
one most worth explaining in a viva.

```
  C:\mlops-data\                <- PERSISTENT, outside Git and outside Docker
      feedback.db               <- runtime state: every submission
      models\
          model_v1.pkl          <- authoritative artifact store
          model_v2.pkl             (every version ever produced,
          model_v3.pkl              including rejected ones)

  <workspace>\                  <- the Git repository / Jenkins workspace
      models\
          model_v2.pkl          <- ONLY the live model, copied here by
                                   stage_model.py right before docker build
      build\                    <- scratch: candidate model between stages
                                   (gitignored, never deployed)

  Docker image                  <- IMMUTABLE deployment artifact
      /app/...                     code + the one approved model baked in
      /data                        mount point for C:\mlops-data
```

Three rules follow from this:

1. **Models are baked into the image.** The image is self-contained: it holds
   the exact model version that was approved. This is the deployment approach
   from the ML deployment practical (Experiment 9).
2. **The database is a mounted volume.** It is runtime state. Recreating the
   container must never destroy collected feedback, and Jenkins recreates the
   container on every deployment.
3. **The persistent store is the authority.** A Jenkins workspace clean or a
   deleted container cannot destroy the model history.

A consequence worth knowing: the image carries the model *file*, but the
database carries the *decision* about which version is live. A container
started without the volume reports `model_loaded: false` — correct
behaviour, not a bug. The volume is mandatory.

---

## 4. The fixed holdout set

Model versions can only be compared if they are measured on the same data.

```
  data/feedback_seed.csv          360 hand-written rows (120 per class)
            │
            │  prepare_data.py — ONE stratified split, random_state=42
            │
      ┌─────┴──────┐
      v            v
 train_pool.csv   holdout.csv
  288 rows         72 rows        <- FROZEN. Never trained on. Never grows.
      │                │
      +  user feedback │
      │                │
      v                v
  TRAINING DATA    EVALUATION DATA
  (grows forever)  (fixed forever)
```

Why it matters:

* If the test set were regenerated as the dataset grew, v1's accuracy and
  v2's accuracy would be measured on different rows, and the quality gate
  would be comparing two numbers that are not comparable.
* Feedback already used for *training* could later land in the *test* set and
  inflate the score. That is test-set contamination.

New user feedback therefore always becomes **training** data, never test
data. `prepare_data.py --verify` proves the split is a clean partition
(0 overlap, 0 rows lost), and `tests/test_model_safety.py` asserts it
automatically.

---

## 5. The quality gate

A candidate must satisfy **both** conditions to be deployed.

```
   Candidate trained
          │
          v
   Scored on the fixed 72-row holdout
          │
          v
   ┌──────────────────────────────────────────────┐
   │ A. ABSOLUTE FLOOR                            │
   │    accuracy >= MIN_ACCURACY (0.85)           │
   ├──────────────────────────────────────────────┤
   │ B. NO REGRESSION                             │
   │    accuracy >= production − TOLERANCE (0.02) │
   │    (production is re-scored on the SAME      │
   │     holdout, so the comparison is valid)     │
   └──────────────────────────────────────────────┘
          │                          │
       BOTH PASS                  EITHER FAILS
          │                          │
          v                          v
   exit 0                       exit 1
   save model_vN.pkl            save model_vN.pkl
   status = production          status = REJECTED
   archive the previous         production UNCHANGED
   mark samples used            samples NOT consumed
          │                          │
          v                          v
   pipeline continues           pipeline STOPS
   to build + deploy            no build, no deploy
```

Two details that matter:

* **Condition B** is what prevents a working model being silently replaced by
  a worse one. Condition A alone would allow a slow decline.
* **A rejection does not consume the training samples.** They stay pending so
  the next run tries again as more feedback arrives. A rejection must never
  eat data.

`MIN_ACCURACY = 0.85` was chosen from measurement, not invented: the real
baseline scored 0.9167 (6.7 points of headroom), a deliberately poisoned
model scored 0.7778, and 85% is the figure the CI/CD practical itself uses
as its example threshold.

---

## 6. Exit-code contract

Each Jenkins stage is a separate process, so Python cannot set a Jenkins
variable. The decision travels by exit code.

| Command | Exit | Meaning |
|---|---|---|
| `retrain.py --check` | `0` | retraining NOT required |
| `retrain.py --check` | `10` | retraining IS required |
| `retrain.py --check` | `1` | error |
| `retrain.py --gate` | `0` | candidate accepted |
| `retrain.py --gate` | `1` | candidate rejected — stop |

`10` rather than `1` for "required" means a genuine Python crash (which
Python reports as `1`) can never be mistaken for a decision.

Jenkins captures it with:

```groovy
def code = bat(returnStatus: true, script: 'python retrain.py --check')
if (code == 10)      { env.RETRAIN_REQUIRED = 'true'  }
else if (code == 0)  { env.RETRAIN_REQUIRED = 'false' }
else { error("retrain.py --check failed with exit code ${code}") }
```

Every later stage is guarded by
`when { expression { env.RETRAIN_REQUIRED == 'true' } }`.

---

## 7. Pipeline flow

```
   cron H/5 * * * *  (data-driven: feedback lives in SQLite, not in Git,
                      so Poll SCM would never notice it)
          │
          v
   Checkout ─> Install Dependencies ─> Bootstrap Model ─> Run Tests
          │                                                  │
          │                                     tests fail ──┴──> STOP
          v
   Check Retraining Requirement
          │
    ┌─────┴──────────────────────────┐
    │ exit 0                          │ exit 10
    v                                 v
  SKIP everything below           Train Candidate
  finish GREEN                        │
  (no build, no redeploy)             v
                                  Evaluate Candidate
                                      │
                                      v
                                  QUALITY GATE ──FAIL──> STOP (no deploy)
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

The below-threshold path is deliberate: rebuilding and redeploying an
unchanged model every five minutes would be pointless churn and would
restart the live service for no reason.

---

## 8. Request flow for one submission

```
  POST /feedback
      │
      ├─ app.py validates text and rating          (rejects -> 400 + error)
      │
      ├─ model_loader.predict(text)  ──> production model  ──> "positive"
      │
      ├─ config.rating_to_sentiment(5)             ──> "positive"
      │
      ├─ database.insert_feedback(...)
      │      stores text, rating, prediction, true label,
      │      whether they matched, which model version,
      │      used_in_training = 0
      │
      └─ render the result panel:
             predicted sentiment, actual sentiment,
             match status, and the pending-sample counter
```

The rating is the source of truth. If the model disagrees with it, the model
is recorded as wrong — which is how the system measures its own live
accuracy and accumulates the data to correct itself.

---

## 9. Module responsibilities

| Module | Owns | Never does |
|---|---|---|
| `config.py` | every threshold and path | logic |
| `database.py` | all SQL | import Flask |
| `dataset.py` | assembling data, the model definition | I/O of models |
| `train.py` | creating v1 | retraining |
| `evaluate.py` | the four metrics | deciding anything |
| `retrain.py` | threshold, candidate, quality gate | serving |
| `stage_model.py` | copying the live model for Docker | choosing it |
| `model_loader.py` | loading and caching the live model | training |
| `app.py` | routes, validation, rendering | SQL or ML |

`dataset.py` exists because `train.py`, `retrain.py` and `evaluate.py` must
build and split the data identically — if they differed, comparing a
candidate against production would be meaningless. Putting it in `train.py`
would have created a circular import with `evaluate.py`.

# System Architecture

Smart Feedback System with Auto-Retraining Pipeline

---

## 1. The idea in one picture

The project is not "a feedback website with a model attached". The point is
the closed loop: using the application produces the data that improves the
model, and two independent checks decide whether that data and that
improvement can be trusted.

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        v                                                          │
   USER submits feedback                                           │
   (text + 1-5 rating)                                             │
        │                                                          │
        ├──> model PREDICTS sentiment (+ confidence) ─┐            │
        │                                             │            │
        └──> rating gives the TRUE label ─────────────┤            │
                                                      v            │
                                   stored together as              │
                                   LABELLED TRAINING DATA          │
                                            │                      │
                                            v                      │
                              enough new samples? (>= 10)          │
                                            │                      │
                                       yes  v                      │
                          DATA-QUALITY FILTER          <- layer 1  │
                          drop rows whose rating                   │
                          contradicts their own text               │
                                            │                      │
                                            v                      │
                                   TRAIN CANDIDATE MODEL           │
                                            │                      │
                                            v                      │
                         EVALUATE on the FIXED 72-row holdout      │
                                            │                      │
                                            v                      │
                                     QUALITY GATE      <- layer 2  │
                                     /              \              │
                                 FAIL                PASS          │
                                  │                   │            │
                        keep production      promote to production │
                        model, block                  │            │
                        deployment                    v            │
                                            bake model + encoder   │
                                            into Docker image      │
                                                      │            │
                                                      v            │
                                              deploy container ────┘
                                                      │
                                                      v
                                     integration tests over real HTTP
```

**Two layers of defence.** The data-quality filter stops contradictory labels
from reaching training at all; the quality gate stops a degraded model from
reaching production. They are tested independently, and each catches things
the other cannot.

---

## 2. Component architecture

```
                         ┌──────────────────────────────┐
   browser / user ──────>│   Flask app  (port 5000)     │
                         │                              │
                         │  Pages:  /  /dashboard       │
                         │          /model              │
                         │  API:    /predict            │
                         │          /api/stats          │
                         │          /api/model          │
                         │          /health             │
                         └───┬──────────────┬───────────┘
                             │              │
                  app.py ────┤              ├──── model_loader.py
              (routes only)  │              │  loads the ONE model marked
                             │              │  'production', warms it up,
                             │              │  returns label + confidence
                             v              v
            ┌──────────────────────┐  ┌──────────────────────────────┐
            │  database.py         │  │  model_vN.pkl  (~20 KB)      │
            │  (all SQL)           │  │   TF-IDF vocabulary          │
            └─────────┬────────────┘  │   classifier weights         │
                      │               │   encoder FINGERPRINT        │
                      v               └──────────────┬───────────────┘
        ┌───────────────────────────┐                │ needs, and
        │  SQLite: feedback.db      │                │ verifies
        │   - feedback              │                v
        │     (+ confidence)        │  ┌──────────────────────────────┐
        │   - model_versions        │  │  embeddings.py               │
        │     (which model is LIVE) │  │  frozen all-MiniLM-L6-v2     │
        └─────────┬─────────────────┘  │  encoder (91.6 MB), loaded   │
                  │                    │  once and shared             │
                  │                    └──────────────────────────────┘
      ┌───────────┴──────────────────────────────────────────────────┐
      │                  OFFLINE / PIPELINE SIDE                     │
      │                                                              │
      │   prepare_encoder.py  puts the pinned encoder in the store   │
      │   dataset.py          training data, holdout, model layout   │
      │   data_quality.py     flags ratings that contradict the text │
      │   train.py            creates the first version              │
      │   evaluate.py         accuracy / precision / recall / F1     │
      │   retrain.py          --check | --train | --gate             │
      │   stage_model.py      copies live model + encoder to build   │
      └──────────────────────────────────────────────────────────────┘
```

**Key separation:** the web application is a *consumer* of models. It can
never train or promote one. Only `train.py` and `retrain.py` do that, and
Jenkins runs them. This is what stops the running service from quietly
changing the model that is serving users.

---

## 3. The model

```
                 ┌──> TF-IDF over single words ─────────> sparse vector ──┐
                 │    (which words appear)                                │
   feedback text ┤                                                        ├──> Logistic   ──> negative
                 │                                                        │    Regression      neutral
                 └──> all-MiniLM-L6-v2 sentence encoder ──> 384 numbers ──┘    (C=5)           positive
                      FROZEN, pretrained, never retrained                        + probabilities
```

One scikit-learn `Pipeline`: a `FeatureUnion` places both feature sets side
by side, and one classifier learns from both. Only the classifier and the
TF-IDF vocabulary are learned; the encoder is pretrained and frozen.

**Why both halves, not just embeddings.** Measured on the identical frozen
holdout, each candidate chosen by 5-fold cross-validation on the training
pool only:

| | CV acc | Holdout acc | Negation | Unseen wording |
|---|---|---|---|---|
| TF-IDF only | 0.9202 | 0.9167 | 6/12 | 6/11 |
| Embeddings only | 0.9304 | 0.8472 | 8/12 | 8/11 |
| **TF-IDF + embeddings** | **0.9411** | **0.9583** | 8/12 | **9/11** |

Embeddings alone generalised better to unfamiliar wording but lost seven
points on the holdout and would have been rejected by our own quality gate.
TF-IDF keeps precision on familiar wording; the embeddings add reach to
wording never seen in training. The encoder produces a *pretrained
statistical representation* — sentences used in similar contexts get similar
vectors. It is not understanding, and it represents negation only weakly.

---

## 4. Where state lives

This is the single most important design decision in the project, and the
one most worth explaining in a viva.

```
  C:\mlops-data\                <- PERSISTENT, outside Git and outside Docker
      feedback.db               <- runtime state: every submission
      models\
          model_v1.pkl ...      <- authoritative artifact store: every
          model_v4.pkl             version ever produced, rejected ones too
      encoders\
          all-MiniLM-L6-v2\     <- the frozen encoder, written ONCE by
              model.safetensors    prepare_encoder.py, pinned by revision,
              encoder_manifest.json   fingerprinted by SHA-256

  <workspace>\                  <- the Git repository / Jenkins workspace
      models\model_v4.pkl       <- ONLY the live model   } copied by
      encoders\all-MiniLM-L6-v2 <- and its encoder       } stage_model.py
      build\                    <- scratch: candidate model between stages
                                   (gitignored, never deployed)

  Docker image                  <- IMMUTABLE deployment artifact
      /app/models/              code + the one approved model
      /app/encoders/            + the encoder it was trained against
      /data                     mount point for C:\mlops-data
```

Four rules follow from this:

1. **Models and their encoder are baked into the image.** The image is
   self-contained: it holds the exact approved model AND the byte-identical
   encoder it was trained against. This is the deployment approach from the
   ML deployment practical (Experiment 9).
2. **The database is a mounted volume.** It is runtime state. Recreating the
   container must never destroy collected feedback, and Jenkins recreates the
   container on every deployment.
3. **The persistent store is the authority.** A Jenkins workspace clean or a
   deleted container cannot destroy the model history.
4. **Nothing is downloaded at prediction time.** The container runs with
   `HF_HUB_OFFLINE=1`; the encoder can only come from the image.

**The encoder is never pickled into a model.** Each `model_vN.pkl` records
the encoder's SHA-256 fingerprint instead. At prediction time the encoder
found on disk must match it, or the model refuses to predict. A model can
therefore never be silently served with a different encoder than the one it
was trained against — a train/serve skew guard.

A consequence worth knowing: the image carries the model *file*, but the
database carries the *decision* about which version is live. A container
started without the volume reports `model_loaded: false` — correct
behaviour, not a bug. The volume is mandatory.

---

## 5. The fixed holdout set

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
      │  (minus rows   │
      │   flagged as   │
      │   inconsistent)│
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

A separate, hand-written **challenge set** (`data/challenge_set.csv`) probes
negation and unfamiliar wording. It is *not* the holdout: nothing is trained,
tuned or gated on it. It exists only to report behaviour the holdout, drawn
from the same distribution as training, cannot reveal.

---

## 6. Data quality — layer 1

The training label comes from the **rating**, not the text. A rating that
contradicts its own text is a wrong label. Two checks, neither using
sentiment keywords:

```
   collected feedback
          │
          ├──> MODEL DISAGREEMENT   the production model reads the text as
          │                         the OPPOSITE polarity to the rating,
          │                         with confidence >= 0.87
          │
          └──> LABEL CONFLICT       the same text (normalised) was also
                                    submitted with an opposite-polarity
                                    rating - both cannot be right
          │
          v
   flagged rows are KEPT in the database, shown on the dashboard,
   and LEFT OUT of retraining  (EXCLUDE_INCONSISTENT_FEEDBACK=1)
```

Neutral never counts in either check — "neutral text, 4 stars" is a
difference of opinion, not a contradiction.

**0.87 is derived, not chosen.** On clean labels the model made 6
opposite-polarity errors in 288 out-of-fold predictions, the most confident
at 0.861. A flip more confident than any clean-label error is more likely a
bad rating than a model mistake. `calibrate_thresholds.py` reproduces it.

**Measured on the real evaluation data**, the first retraining run excluded
ids 31, 32, 48, 51 and 52: four by label conflict (`It was great.` 5★/1★,
`not good` 1★/5★) and one by model disagreement (`Good` 1★). The check is
only as good as the model reading the text: a text the model itself misreads
cannot be vouched against its rating.

---

## 7. The quality gate — layer 2

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

**The gate approved the change of model family.** The hybrid did not bypass
the gate: it was trained as an ordinary candidate and scored 0.9306 against
the TF-IDF production model's 0.9167 on the same holdout, so it passed both
conditions and became v4.

---

## 8. Low confidence — an advisory signal

Every prediction carries its top class probability. Below **0.60** the
prediction is marked *low confidence*. It is advisory: the prediction is
still made, returned and stored, the three classes are unchanged, and the
quality gate is unaffected.

| Signal | AUROC for picking out wrong predictions |
|---|---|
| **Top class probability** | **0.906** — used |
| Similarity to the nearest training sentence | 0.694 — rejected |

0.60 is the rounded 10th percentile (0.596) of top probability over genuine
out-of-fold predictions, so it flags about 10% of normal feedback. It flags
8 of 8 near-meaningless probes (`something something`, `asdf qwerty`...) —
**and** some genuine but unfamiliar feedback (`I feel cheated`). On this
dataset the model cannot tell "no information" from "unfamiliar wording", so
the flag means *limited evidence*, not *meaningless input*.

---

## 9. Exit-code contract

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

## 10. Pipeline flow

```
   cron H/5 * * * *  (data-driven: feedback lives in SQLite, not in Git,
                      so Poll SCM would never notice it)
          │
          v
   Checkout ─> Install Dependencies ─> Bootstrap Encoder and Model ─> Run Tests
          │                                                              │
          │                                                 tests fail ──┴──> STOP
          v
   Check Retraining Requirement
          │
    ┌─────┴──────────────────────────┐
    │ exit 0                          │ exit 10
    v                                 v
  SKIP everything below           Train Candidate  (inconsistent rows excluded)
  finish GREEN                        │
  (no build, no redeploy)             v
                                  Evaluate Candidate
                                      │
                                      v
                                  QUALITY GATE ──FAIL──> STOP (no deploy)
                                      │ PASS
                                      v
                                  Stage Production Model  (model + encoder)
                                      v
                                  Build Docker Image (:v${BUILD_NUMBER})
                                      v
                                  Deploy (rm -f, run -d, poll /health)
                                      v
                                  Deployment Integration Tests (real HTTP)
```

The below-threshold path is deliberate: rebuilding and redeploying an
unchanged model every five minutes would be pointless churn and would
restart the live service for no reason.

---

## 11. Request flow for one submission

```
  POST /feedback
      │
      ├─ app.py validates text and rating          (rejects -> 400 + error)
      │
      ├─ model_loader.predict_detailed(text)
      │      -> "positive", confidence 0.93, low_confidence False
      │
      ├─ config.rating_to_sentiment(5)             ──> "positive"
      │
      ├─ database.insert_feedback(...)
      │      stores text, rating, prediction, confidence, true label,
      │      whether they matched, which model version,
      │      used_in_training = 0
      │
      ├─ data_quality: does this rating contradict its own text?
      │
      └─ render the result panel:
             predicted sentiment + confidence, actual sentiment,
             match status, any low-confidence or mismatch warning,
             and the pending-sample counter
```

The rating is the source of truth for training. When the model disagrees
with it, the model is recorded as wrong — unless the disagreement is so
confident that the *rating* is the more likely mistake, in which case the row
is flagged and kept out of training.

---

## 12. Module responsibilities

| Module | Owns | Never does |
|---|---|---|
| `config.py` | every threshold and path | logic |
| `database.py` | all SQL, schema migration | import Flask |
| `dataset.py` | assembling data, the model definition | I/O of models |
| `embeddings.py` | the frozen encoder as a transformer, its fingerprint check | learn anything |
| `data_quality.py` | text/rating consistency checks | use sentiment keywords |
| `prepare_encoder.py` | putting the pinned encoder in the store, once | run at prediction time |
| `train.py` | creating the first version | retraining |
| `evaluate.py` | the four metrics | deciding anything |
| `retrain.py` | threshold, candidate, quality gate | serving |
| `stage_model.py` | copying the live model and encoder for Docker | choosing them |
| `model_loader.py` | loading, warming and caching the live model | training |
| `app.py` | routes, validation, rendering | SQL or ML |

`dataset.py` exists because `train.py`, `retrain.py` and `evaluate.py` must
build and split the data identically — if they differed, comparing a
candidate against production would be meaningless.

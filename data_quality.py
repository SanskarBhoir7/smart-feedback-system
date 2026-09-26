"""
Text / rating consistency checks for collected feedback.

THE PROBLEM
-----------
The training label comes from the STAR RATING, not the text:

    1-2 stars -> negative     3 -> neutral     4-5 -> positive

So a submission whose rating contradicts its own text becomes a WRONG
training label. The evaluation found exactly this:

    "It was great."  + 1 star  -> trained as NEGATIVE    (it entered v3)
    "not good"       + 5 stars -> scored "correct", because the model's wrong
                                   answer happened to agree with the wrong
                                   rating

WHAT THIS MODULE DOES NOT DO
----------------------------
It contains no sentiment keywords and no rules such as "if the text contains
'good'". Every judgement below comes either from the trained model or from
comparing submissions with each other.

THE TWO CHECKS
--------------
1. MODEL DISAGREEMENT
   The production model reads the text and predicts the OPPOSITE polarity to
   the rating - positive where the rating says negative, or the reverse - with
   confidence of at least config.DISAGREEMENT_CONFIDENCE (0.87, derived from
   validation data). Neutral never counts: "neutral text, 4 stars" is a mild
   difference of opinion, not a contradiction.

   Limitation: this is only as good as the model. If the model is itself
   unsure what a text means, it cannot vouch that the rating is wrong.

2. LABEL CONFLICT
   The same text (after normalising case, spacing and punctuation) has been
   submitted with both a positive and a negative rating. Both labels cannot
   be right, and nothing in the text says which one is, so every copy is
   flagged. This needs no model at all, so it still works when check 1 cannot.

WHAT HAPPENS TO A FLAGGED ROW
-----------------------------
Nothing is deleted. The row is kept, shown with a warning, and - while
config.EXCLUDE_INCONSISTENT_FEEDBACK is on - left out of retraining.
"""

import re
from collections import defaultdict

import numpy as np

import config

POLARITY = {config.NEGATIVE: -1, config.NEUTRAL: 0, config.POSITIVE: 1}

REASON_DISAGREEMENT = "model_disagreement"
REASON_CONFLICT = "label_conflict"

REASON_TEXT = {
    REASON_DISAGREEMENT: "the model reads the text as the opposite of the rating",
    REASON_CONFLICT: "the same text was also submitted with an opposite rating",
}


def normalise(text):
    """
    Reduce text to a comparison key: lower case, letters and digits only,
    single spaces. "It was great." and ' it  WAS great ' become the same key.

    This is plain string normalisation for exact matching - it carries no
    notion of sentiment.
    """
    text = re.sub(r"[^a-z0-9' ]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def is_opposite(predicted, label):
    """True when one is positive and the other negative."""
    return POLARITY.get(predicted, 0) * POLARITY.get(label, 0) == -1


def is_disagreement(predicted, confidence, label):
    """Check 1 for a single prediction that has already been made."""
    if predicted is None or confidence is None:
        return False
    return is_opposite(predicted, label) and confidence >= config.DISAGREEMENT_CONFIDENCE


def find_label_conflicts(records):
    """
    Check 2. Returns the set of record ids whose normalised text also occurs
    with an opposite-polarity label.
    """
    groups = defaultdict(list)
    for record in records:
        key = normalise(record["feedback_text"])
        if key:
            groups[key].append(record)

    conflicted = set()
    for members in groups.values():
        polarities = {POLARITY[m["actual_sentiment"]] for m in members}
        if 1 in polarities and -1 in polarities:
            conflicted.update(m["id"] for m in members)
    return conflicted


def find_model_disagreements(records, model):
    """
    Check 1 over many records at once. Returns {id: (predicted, confidence)}.

    The model re-reads every text now, so every row is judged by the same,
    current production model - including rows stored before this check
    existed.
    """
    if model is None or not records:
        return {}
    probabilities = model.predict_proba([r["feedback_text"] for r in records])
    classes = list(model.classes_)
    flagged = {}
    for record, row in zip(records, probabilities):
        index = int(np.argmax(row))
        predicted, confidence = classes[index], float(row[index])
        if is_disagreement(predicted, confidence, record["actual_sentiment"]):
            flagged[record["id"]] = (predicted, round(confidence, 4))
    return flagged


def assess(records, model=None):
    """
    Run both checks. Returns {record id: [reason, ...]} for flagged rows only.

    model may be None - for example before the first model exists - in which
    case only the label-conflict check runs.

    A row an earlier ACCEPTED training run already excluded carries its
    reason in record["quality_flag"], and stays flagged whatever today's
    checks say. Otherwise a later production model that happened to be less
    confident about a text could quietly re-admit a label that was already
    judged wrong. See database.set_quality_flags().
    """
    flags = defaultdict(list)

    def add(record_id, reason):
        if reason and reason not in flags[record_id]:
            flags[record_id].append(reason)

    for record in records:
        for reason in (record.get("quality_flag") or "").split(","):
            add(record["id"], reason.strip())
    for record_id in find_label_conflicts(records):
        add(record_id, REASON_CONFLICT)
    for record_id in find_model_disagreements(records, model):
        add(record_id, REASON_DISAGREEMENT)
    return {record_id: reasons for record_id, reasons in flags.items() if reasons}


def summarise(flags):
    """Counts per reason, for logs and the dashboard."""
    counts = {REASON_DISAGREEMENT: 0, REASON_CONFLICT: 0}
    for reasons in flags.values():
        for reason in reasons:
            counts[reason] += 1
    return {
        "flagged_rows": len(flags),
        "model_disagreement": counts[REASON_DISAGREEMENT],
        "label_conflict": counts[REASON_CONFLICT],
    }

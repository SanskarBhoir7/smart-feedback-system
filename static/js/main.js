/* ------------------------------------------------------------------
   Smart Feedback System - client-side script

   Plain JavaScript, no framework. Two small jobs:

     1. a live character counter on the feedback box
     2. a "Preview Prediction" button that calls POST /predict

   The preview button is what makes the JSON API visible in the UI: it
   asks the production model for a prediction WITHOUT storing anything,
   so you can show the API working without polluting the database.

   Nothing here is required for the app to function. Submitting the form
   is a normal HTML POST, so Selenium and a browser with JavaScript
   disabled both behave identically.
   ------------------------------------------------------------------ */

(function () {
    "use strict";

    // ---- 1. Character counter -------------------------------------
    var textarea = document.getElementById("feedback_text");
    var counter = document.getElementById("char_count");

    if (textarea && counter) {
        var updateCount = function () {
            counter.textContent = textarea.value.length;
        };
        textarea.addEventListener("input", updateCount);
        updateCount();
    }

    // ---- 2. Preview prediction via the JSON API --------------------
    var previewButton = document.getElementById("preview_prediction");
    var previewBox = document.getElementById("preview_result");
    var previewSentiment = document.getElementById("preview_sentiment");

    if (previewButton && previewBox && previewSentiment && textarea) {
        previewButton.addEventListener("click", function () {
            var text = textarea.value.trim();

            if (text.length < 3) {
                previewBox.classList.remove("hidden");
                previewSentiment.textContent = "enter more text first";
                previewSentiment.className = "pill";
                return;
            }

            previewButton.disabled = true;
            previewSentiment.textContent = "predicting...";
            previewSentiment.className = "pill";
            previewBox.classList.remove("hidden");

            fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            })
                .then(function (response) {
                    return response.json().then(function (data) {
                        return { ok: response.ok, data: data };
                    });
                })
                .then(function (result) {
                    if (result.ok && result.data.prediction) {
                        previewSentiment.textContent = result.data.prediction;
                        previewSentiment.className = "pill pill-" + result.data.prediction;
                    } else {
                        previewSentiment.textContent = result.data.error || "prediction failed";
                        previewSentiment.className = "pill pill-negative";
                    }
                })
                .catch(function () {
                    previewSentiment.textContent = "could not reach the server";
                    previewSentiment.className = "pill pill-negative";
                })
                .finally(function () {
                    previewButton.disabled = false;
                });
        });
    }
})();

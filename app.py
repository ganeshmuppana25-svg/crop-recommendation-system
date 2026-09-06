"""
app.py
------
Flask backend for the Crop Recommendation System (Phase 1).

Routes:
  GET  /               -> web page (templates/index.html)
  POST /api/predict    -> predict crop from 7 soil/weather inputs
  GET  /api/metrics    -> real model evaluation metrics + feature importance
  GET  /api/dataset    -> dataset information (stats, classes, ranges)
  GET  /api/health     -> service health check

The prediction ALWAYS comes from the trained DecisionTreeClassifier stored in
model/crop_model.pkl. There are no hardcoded predictions or if/else crop rules.

Run:  python app.py   (then open http://127.0.0.1:5000)
"""

import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

app = Flask(__name__)

# Loaded once at startup by load_artifacts()
STATE = {
    "model": None,        # the trained DecisionTreeClassifier
    "features": [],       # ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    "classes": [],        # crop class labels known to the model
    "meta": {},           # contents of model/metadata.json
    "model_type": "",
}

# Short GENERAL reference information per crop (for the UI info card).
# This is static educational text only - it is NOT used for prediction.
CROP_INFO = {
    "rice": "Rice is a Kharif cereal grown in flooded paddies. It thrives in high "
            "rainfall, high humidity and warm temperatures.",
    "maize": "Maize (corn) is a widely grown cereal. It prefers warm temperatures, "
             "moderate rainfall and well-drained soil.",
    "chickpea": "Chickpea is a Rabi pulse grown in cool, dry climates on "
                "well-drained soil with low to moderate water need.",
    "kidneybeans": "Kidney beans are a Rabi pulse preferring cool temperatures and "
                   "moderate rainfall.",
    "pigeonpeas": "Pigeon peas are a Kharif pulse suited to warm climates with "
                  "moderate rainfall.",
    "mothbeans": "Moth beans are a hardy pulse of arid regions; they need very "
                 "little water and tolerate high temperatures.",
    "mungbean": "Mung bean is a short-duration Kharif pulse that grows best in "
                "warm, humid conditions.",
    "blackgram": "Black gram is a Kharif pulse grown in warm, humid climates with "
                 "moderate rainfall.",
    "lentil": "Lentil is a Rabi pulse grown in cool climates with low rainfall and "
              "well-drained soil.",
    "pomegranate": "Pomegranate is a perennial fruit suited to dry, warm climates; "
                   "it is drought tolerant.",
    "banana": "Banana is a tropical fruit that needs hot, humid conditions and "
              "plenty of water year-round.",
    "mango": "Mango is a tropical fruit tree preferring hot summers with moderate "
             "rainfall.",
    "grapes": "Grapes are a vine fruit best grown in warm, dry climates with low "
              "humidity.",
    "watermelon": "Watermelon is a warm-season fruit needing hot weather and "
                  "moderate water.",
    "muskmelon": "Muskmelon is a warm-season fruit that prefers hot, relatively dry "
                 "conditions.",
    "apple": "Apple is a temperate fruit requiring cool climate and winter "
             "chilling hours.",
    "orange": "Orange is a citrus fruit of subtropical climates with moderate "
              "temperature and rainfall.",
    "papaya": "Papaya is a tropical fruit thriving in warm, humid climates.",
    "coconut": "Coconut is a tropical palm of hot, humid coastal regions with very "
               "high rainfall.",
    "cotton": "Cotton is a Kharif fibre crop grown in warm climates, often on "
              "black cotton soil.",
    "jute": "Jute is a fibre crop of hot, humid regions with alluvial soil.",
    "coffee": "Coffee is a plantation crop of cool, humid highlands with moderate "
              "rainfall and shade.",
}



def load_artifacts() -> None:
    """Load the trained model and metadata once at startup."""
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first."
        )
    bundle = joblib.load(MODEL_PATH)
    STATE["model"] = bundle["model"]
    STATE["features"] = list(bundle["features"])
    STATE["classes"] = list(bundle["classes"])
    STATE["model_type"] = bundle.get("model_type", "DecisionTreeClassifier")
    if METADATA_PATH.exists():
        STATE["meta"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"[startup] Model loaded: {STATE['model_type']} | "
          f"classes: {len(STATE['classes'])} | features: {STATE['features']}")


def validation_ranges() -> dict:
    """Permitted input ranges derived from the training dataset."""
    return STATE["meta"].get("validation_ranges", {})


def validate_payload(payload) -> tuple:
    """
    Validate the JSON payload and return (values, errors).
    values = list of 7 floats in STATE["features"] order (empty if invalid).
    """
    errors = []
    if not isinstance(payload, dict):
        return [], ["Request body must be a JSON object with the 7 input fields."]

    ranges = validation_ranges()
    values = []
    for feature in STATE["features"]:
        raw = payload.get(feature)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors.append(f"{feature}: value is required.")
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{feature}: '{raw}' is not a valid number.")
            continue
        if num != num:  # NaN check
            errors.append(f"{feature}: value must be a finite number.")
            continue
        allowed = ranges.get(feature)
        if allowed and not (allowed["min"] <= num <= allowed["max"]):
            errors.append(
                f"{feature}: {num} is outside the valid range "
                f"[{allowed['min']}, {allowed['max']}]."
            )
            continue
        values.append(num)

    if errors:
        return [], errors
    return values, []


def relative_to_average(feature: str, value: float) -> str:
    """Describe where an input sits vs the dataset average (honest and simple)."""
    stats = STATE["meta"].get("dataset", {}).get("feature_stats", {}).get(feature)
    if not stats:
        return f"{value}"
    mean = stats.get("mean", 0)
    diff_pct = (value - mean) / mean * 100 if mean else 0
    if abs(diff_pct) <= 15:
        return f"{value} (close to the dataset average of {mean})"
    if diff_pct > 0:
        return f"{value} (above the dataset average of {mean})"
    return f"{value} (below the dataset average of {mean})"


def build_explanation(crop: str, values: dict, prob: float) -> str:
    """
    Short, HONEST explanation: model size, top feature importances and where the
    inputs sit relative to dataset averages. The tree itself cannot produce
    human-readable reasons, so none are invented here.
    """
    imp = STATE["meta"].get("feature_importance", {})
    top3 = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_txt = ", ".join(f"{f} ({i * 100:.1f}%)" for f, i in top3)
    train_rows = STATE["meta"].get("model", {}).get("train_rows", "the full")
    rel = [f"{f} was {relative_to_average(f, v)}" for f, v in values.items()]
    return (
        f"This prediction comes from a Decision Tree trained on {train_rows} "
        f"soil/weather samples. The model's most influential features are: "
        f"{top_txt}. Your input: {'; '.join(rel)}. Based on these values the "
        f"tree matched your conditions to the '{crop}' pattern it learned "
        f"during training, with {prob * 100:.2f}% probability. Note: a "
        f"Decision Tree provides probabilities, not human-readable reasons - "
        f"treat this as statistical guidance, not agronomic advice."
    )


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    """Serve the web page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept JSON with the 7 input features, run the Decision Tree and return
    the predicted crop, confidence, top-3 probabilities and supporting info.
    """
    if STATE["model"] is None:
        return jsonify({"success": False, "error": "Model is not loaded."}), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON with the 7 input fields.",
        }), 400

    values, errors = validate_payload(payload)
    if errors:
        return jsonify({"success": False, "error": "Invalid input.", "details": errors}), 400

    features = STATE["features"]
    model = STATE["model"]
    X = np.array([values], dtype=float)

    try:
        crop = str(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    except Exception as exc:  # never leak a raw traceback to the client
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500

    classes = [str(c) for c in model.classes_]
    order = np.argsort(proba)[::-1]  # descending by probability
    top3 = [
        {"crop": classes[i], "confidence": round(float(proba[i]) * 100, 2)}
        for i in order[:3]
    ]

    values_dict = {f: v for f, v in zip(features, values)}
    explanation = build_explanation(crop, values_dict, float(proba[order[0]]))

    return jsonify({
        "success": True,
        "crop": crop,
        "confidence": round(float(proba[order[0]]) * 100, 2),
        "top3": top3,
        "input_summary": values_dict,
        "crop_info": CROP_INFO.get(
            crop, "General information for this crop is not available."
        ),
        "explanation": explanation,
        "feature_importance": STATE["meta"].get("feature_importance", {}),
        "model_accuracy": STATE["meta"].get("metrics", {}).get("accuracy"),
    })


@app.route("/api/metrics")
def api_metrics():
    """Real evaluation metrics of the trained model."""
    meta = STATE["meta"]
    return jsonify({
        "success": True,
        "model": meta.get("model", {}),
        "metrics": meta.get("metrics", {}),
        "feature_importance": meta.get("feature_importance", {}),
    })


@app.route("/api/dataset")
def api_dataset():
    """Dataset information: rows, classes, distribution, stats, valid ranges."""
    meta = STATE["meta"]
    return jsonify({
        "success": True,
        "dataset": meta.get("dataset", {}),
        "validation_ranges": meta.get("validation_ranges", {}),
        "class_means": meta.get("class_means", {}),
    })


@app.route("/api/health")
def api_health():
    """Simple health check for the demo."""
    return jsonify({
        "success": True,
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_type": STATE["model_type"],
        "n_classes": len(STATE["classes"]),
    })


# --------------------------------------------------------------------------- #
# Error handlers (always return JSON, never raw tracebacks)                   #
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(_e):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    load_artifacts()
    # debug=False keeps the model loaded once; port can be overridden with PORT
    import os

    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)

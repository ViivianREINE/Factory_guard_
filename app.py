
# FactoryGuard AI — Flask REST API
# File: app.py
# Purpose: Serve the trained XGBoost model as a production REST endpoint.
#          Accepts JSON sensor readings, returns failure probability + SHAP.
# To run:
#   python app.py
#   Default: http://localhost:5000


from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
import time
import os

# App setup 
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False   # preserve key order in responses

# ── Load artifacts at startup (not per request — this is the production way) ─
MODEL_DIR = "factoryguard_outputs"

print("Loading FactoryGuard AI artifacts...")
model     = joblib.load(f"{MODEL_DIR}/factoryguard_model.joblib")
explainer = joblib.load(f"{MODEL_DIR}/factoryguard_explainer.joblib")
features  = joblib.load(f"{MODEL_DIR}/factoryguard_features.joblib")
print(f"  ✅ Model loaded. Expecting {len(features)} features.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SHAP Explanation Generator (same logic as Week 3)
# ─────────────────────────────────────────────────────────────────────────────

def generate_shap_explanation(shap_vals, feature_names, feature_values, top_n=5):
    """Generate human-readable SHAP explanation for the API response."""
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals,
        "feature_value": feature_values
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_n)

    contributors = []
    for _, row in shap_df.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        magnitude = (
            "strongly" if row["abs_shap"] > 0.3
            else "moderately" if row["abs_shap"] > 0.1
            else "slightly"
        )
        contributors.append({
            "feature": row["feature"],
            "value": round(float(row["feature_value"]), 4),
            "shap_contribution": round(float(row["shap_value"]), 4),
            "effect": f"{magnitude} {direction} failure probability"
        })

    # Physics-based narrative
    top = shap_df.iloc[0]
    f, v = top["feature"], top["feature_value"]
    if "temperature" in f.lower() or "temp" in f.lower():
        narrative = f"High failure risk — elevated temperature ({v:.1f}K). Possible coolant failure."
    elif "torque" in f.lower():
        narrative = f"High failure risk — excessive torque ({v:.1f}N·m). Machine under mechanical stress."
    elif "wear" in f.lower():
        narrative = f"High failure risk — critical tool wear ({v:.1f}min). Immediate replacement needed."
    elif "speed" in f.lower() or "rpm" in f.lower():
        narrative = f"High failure risk — abnormal rotational speed ({v:.1f}RPM). Check belt/motor."
    elif "power" in f.lower():
        narrative = f"High failure risk — abnormal power ({v:.1f}W). Combined torque/speed anomaly."
    else:
        narrative = f"High failure risk driven by '{f}' = {v:.3f}. Inspect machine immediately."

    return {
        "top_contributors": contributors,
        "narrative_summary": narrative,
        "recommended_action": "Dispatch technician for inspection within 24 hours."
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Health Check
# GET /health → returns API status
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint used by load balancers and monitoring."""
    return jsonify({
        "status": "ok",
        "service": "FactoryGuard AI",
        "version": "1.0.0",
        "model": "XGBoost + SMOTE (Tuned)"
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Predict
# POST /predict → accepts JSON sensor data, returns prediction + explanation
#
# Expected input (JSON):
# {
#   "air_temperature_K": 298.5,
#   "process_temperature_K": 308.8,
#   "rotational_speed_rpm": 1400,
#   "torque_Nm": 55.0,
#   "tool_wear_min": 180,
#   "type_encoded": 0,
#   ... (all feature-engineered columns)
# }
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Accepts:  JSON with all required sensor features
    Returns:  failure_probability, prediction, SHAP explanation, latency_ms
    """
    t_start = time.perf_counter()

    # ── 1. Parse request ──────────────────────────────────────────────────
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload."}), 400

    if not payload:
        return jsonify({"error": "Empty request body."}), 400

    # ── 2. Validate features ──────────────────────────────────────────────
    missing = [f for f in features if f not in payload]
    if missing:
        return jsonify({
            "error": f"Missing {len(missing)} required features.",
            "missing_features": missing,
            "required_features": features
        }), 422

    # ── 3. Build input DataFrame ──────────────────────────────────────────
    input_df = pd.DataFrame([[payload[f] for f in features]], columns=features)

    # ── 4. Inference ──────────────────────────────────────────────────────
    try:
        failure_prob  = float(model.predict_proba(input_df)[0][1])
        prediction    = int(model.predict(input_df)[0])
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    # ── 5. SHAP explanation ───────────────────────────────────────────────
    try:
        # Transform through scaler first
        scaler      = model.named_steps["scaler"]
        xgb_step    = model.named_steps["clf"]
        scaled_vals = scaler.transform(input_df)
        scaled_df   = pd.DataFrame(scaled_vals, columns=features)

        xgb_explainer = shap.TreeExplainer(xgb_step)
        shap_vals      = xgb_explainer.shap_values(scaled_df)[0]

        explanation = generate_shap_explanation(
            shap_vals, features, scaled_vals[0]
        )
    except Exception as e:
        explanation = {"error": f"SHAP explanation failed: {str(e)}"}

    # ── 6. Build response ─────────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

    response = {
        "prediction": prediction,
        "prediction_label": "FAILURE" if prediction == 1 else "NORMAL",
        "failure_probability": round(failure_prob, 4),
        "confidence_pct": round(failure_prob * 100, 2),
        "alert": failure_prob >= 0.5,
        "explanation": explanation,
        "latency_ms": latency_ms,
        "latency_ok": latency_ms < 50,
        "model_version": "xgboost_smote_tuned_v1"
    }

    status_code = 200
    return jsonify(response), status_code


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Features Info
# GET /features → returns list of expected features and their descriptions
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/features", methods=["GET"])
def get_features():
    """Return list of all required input features."""
    return jsonify({
        "n_features": len(features),
        "features": features
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Starting FactoryGuard AI API server...")
    print("   Endpoints:")
    print("     GET  /health   → health check")
    print("     POST /predict  → failure prediction + SHAP explanation")
    print("     GET  /features → required feature list")
    print("   Running on http://0.0.0.0:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

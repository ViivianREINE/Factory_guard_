"""
FactoryGuard AI — Example API Client
=====================================
Run this script AFTER starting the Flask API server (python app.py).

This script shows how a factory monitoring system would call the API
and interpret the response in real time.
"""

import requests
import json

BASE_URL = "http://localhost:5000"

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Health Check
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Health Check")
print("=" * 60)
resp = requests.get(f"{BASE_URL}/health")
print(json.dumps(resp.json(), indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Normal Machine (Low Failure Risk)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Predict — NORMAL machine (expected: low failure probability)")
print("=" * 60)

# NOTE: In production this payload would be built automatically by the
# feature engineering pipeline from raw sensor readings.
# For demo purposes we send a simplified payload (you must include ALL features).
# Here we show the concept with base sensor values:

normal_payload = {
    "Type": 1,                               # M-type machine
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 10.0,
    # Lag features
    "Air temperature [K]_lag1": 298.0,
    "Air temperature [K]_lag2": 297.9,
    "Process temperature [K]_lag1": 308.5,
    "Process temperature [K]_lag2": 308.4,
    "Rotational speed [rpm]_lag1": 1498.0,
    "Rotational speed [rpm]_lag2": 1497.0,
    "Torque [Nm]_lag1": 39.8,
    "Torque [Nm]_lag2": 39.5,
    "Tool wear [min]_lag1": 9.0,
    "Tool wear [min]_lag2": 8.0,
    # Rolling means (1h)
    "Air temperature [K]_roll_mean_1h": 298.1,
    "Air temperature [K]_roll_std_1h": 0.05,
    "Process temperature [K]_roll_mean_1h": 308.6,
    "Process temperature [K]_roll_std_1h": 0.04,
    "Rotational speed [rpm]_roll_mean_1h": 1500.0,
    "Rotational speed [rpm]_roll_std_1h": 2.0,
    "Torque [Nm]_roll_mean_1h": 40.0,
    "Torque [Nm]_roll_std_1h": 0.5,
    "Tool wear [min]_roll_mean_1h": 9.5,
    "Tool wear [min]_roll_std_1h": 0.5,
    # Rolling means (4h, 8h) — simplified here
    "Air temperature [K]_roll_mean_4h": 298.0,
    "Air temperature [K]_roll_std_4h": 0.1,
    "Process temperature [K]_roll_mean_4h": 308.5,
    "Process temperature [K]_roll_std_4h": 0.08,
    "Rotational speed [rpm]_roll_mean_4h": 1499.0,
    "Rotational speed [rpm]_roll_std_4h": 3.0,
    "Torque [Nm]_roll_mean_4h": 40.1,
    "Torque [Nm]_roll_std_4h": 0.7,
    "Tool wear [min]_roll_mean_4h": 8.0,
    "Tool wear [min]_roll_std_4h": 1.0,
    "Air temperature [K]_roll_mean_8h": 297.9,
    "Air temperature [K]_roll_std_8h": 0.2,
    "Process temperature [K]_roll_mean_8h": 308.4,
    "Process temperature [K]_roll_std_8h": 0.1,
    "Rotational speed [rpm]_roll_mean_8h": 1498.0,
    "Rotational speed [rpm]_roll_std_8h": 4.0,
    "Torque [Nm]_roll_mean_8h": 40.0,
    "Torque [Nm]_roll_std_8h": 0.8,
    "Tool wear [min]_roll_mean_8h": 7.0,
    "Tool wear [min]_roll_std_8h": 1.5,
    # EMA features
    "Air temperature [K]_ema10": 298.0,
    "Air temperature [K]_ema40": 298.0,
    "Process temperature [K]_ema10": 308.5,
    "Process temperature [K]_ema40": 308.5,
    "Rotational speed [rpm]_ema10": 1499.0,
    "Rotational speed [rpm]_ema40": 1498.0,
    "Torque [Nm]_ema10": 40.0,
    "Torque [Nm]_ema40": 40.0,
    "Tool wear [min]_ema10": 9.0,
    "Tool wear [min]_ema40": 8.5,
    # Domain features
    "Power_W": 40.0 * (1500.0 * 2 * 3.14159 / 60),
    "Temp_diff_K": 308.6 - 298.1,
    "Torque_per_rpm": 40.0 / 1500.0,
    "Wear_x_Torque": 10.0 * 40.0,
}

resp = requests.post(f"{BASE_URL}/predict", json=normal_payload)
result = resp.json()
print(f"  Prediction    : {result.get('prediction_label')}")
print(f"  Failure Prob  : {result.get('failure_probability')}")
print(f"  Alert Triggered: {result.get('alert')}")
print(f"  Latency (ms)  : {result.get('latency_ms')}")
print(f"  Latency OK    : {result.get('latency_ok')}")

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: High-Risk Machine (Expected: FAILURE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Predict — HIGH RISK machine (expected: high failure probability)")
print("=" * 60)

# Same payload but with high temperature, high torque, heavy wear
high_risk_payload = normal_payload.copy()
high_risk_payload.update({
    "Air temperature [K]": 310.0,                 # very hot
    "Process temperature [K]": 320.0,             # extremely hot
    "Torque [Nm]": 75.0,                          # very high torque
    "Tool wear [min]": 240.0,                     # near end of tool life
    "Rotational speed [rpm]": 1200.0,             # low speed + high torque = overload
    "Wear_x_Torque": 240.0 * 75.0,
    "Temp_diff_K": 320.0 - 310.0,
    "Torque_per_rpm": 75.0 / 1200.0,
    "Power_W": 75.0 * (1200.0 * 2 * 3.14159 / 60),
})

resp = requests.post(f"{BASE_URL}/predict", json=high_risk_payload)
result = resp.json()
print(f"  Prediction    : {result.get('prediction_label')}")
print(f"  Failure Prob  : {result.get('failure_probability')}")
print(f"  Confidence %  : {result.get('confidence_pct')}%")
print(f"  Alert Triggered: {result.get('alert')}")
print(f"  Latency (ms)  : {result.get('latency_ms')}")
print(f"  Latency OK (<50ms): {result.get('latency_ok')}")

print("\n  SHAP Explanation:")
explanation = result.get("explanation", {})
if "narrative_summary" in explanation:
    print(f"  → {explanation['narrative_summary']}")
    print(f"  → {explanation['recommended_action']}")
    print("\n  Top Contributors:")
    for c in explanation.get("top_contributors", []):
        print(f"    • {c['feature']} = {c['value']:.3f} → {c['effect']} (SHAP={c['shap_contribution']:+.4f})")

print("\n" + "=" * 60)
print("✅ API test complete")

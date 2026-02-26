# 🏭 FactoryGuard AI — IoT Predictive Maintenance Engine

**Infotact Solutions | Q4 2025 Internship Project 1**  
**Intern:** Data Science Intern  
**Mentor:** Senior Production ML Engineer  

---

## 📌 Project Overview

FactoryGuard AI is a **binary classification system** that predicts machine failure **24 hours in advance** using real-time IoT sensor data (temperature, vibration/torque, rotational speed, tool wear). The system minimizes false negatives (missed failures) to prevent costly unplanned downtime.

### Business Problem
In industrial manufacturing, unexpected machine failures cost an average of **$250,000+ per hour** in downtime. FactoryGuard AI gives maintenance teams advance warning so they can intervene before failure occurs.

### Solution
- **Algorithm:** XGBoost + SMOTE (class imbalance handling)
- **Priority metric:** Recall (minimize missed failures)
- **Explainability:** SHAP (per-prediction root cause analysis)
- **Deployment:** Flask REST API with <50ms latency

---

## 🗂️ Project Structure

```
factoryguard_ai/
│
├── factoryguard_main.py        ← Main training pipeline (Weeks 1–4)
├── app.py                      ← Flask REST API server
├── example_api_request.py      ← API usage demo client
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── ai4i2020.csv                ← Dataset (place here before running)
│
└── factoryguard_outputs/       ← Auto-created by pipeline
    ├── week1_featured_dataset.csv
    ├── week1_correlation_heatmap.png
    ├── week2_model_comparison.csv
    ├── week2_confusion_matrices.png
    ├── week3_shap_summary_plot.png
    ├── week3_shap_bar_plot.png
    ├── week3_shap_force_plot.png
    ├── week3_shap_force_plot.html
    ├── week3_sample_explanation.txt
    ├── factoryguard_model.joblib
    ├── factoryguard_explainer.joblib
    └── factoryguard_features.joblib
```

---

## ⚙️ Setup Instructions

### Option A: Google Colab (Recommended for submission)

```python
# Step 1: Install dependencies
!pip install xgboost imbalanced-learn shap flask joblib scikit-learn pandas numpy matplotlib seaborn

# Step 2: Upload the dataset
from google.colab import files
files.upload()   # upload ai4i2020.csv

# Step 3: Run the main pipeline
!python factoryguard_main.py
```

### Option B: Local Machine

```bash
# Clone / download project files
cd factoryguard_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline (Weeks 1–4)
python factoryguard_main.py
```

---

## 🚀 Running the Flask API

### Step 1: Train the model first
```bash
python factoryguard_main.py
```
This generates the model files in `factoryguard_outputs/`.

### Step 2: Start the API server
```bash
python app.py
```
Output:
```
🚀 Starting FactoryGuard AI API server...
   Endpoints:
     GET  /health   → health check
     POST /predict  → failure prediction + SHAP explanation
     GET  /features → required feature list
   Running on http://0.0.0.0:5000
```

### Step 3: Send a prediction request

**Using curl:**
```bash
curl -X GET http://localhost:5000/health
```

**Using Python:**
```bash
python example_api_request.py
```

---

## 📡 API Reference

### `GET /health`
Returns API status.

**Response:**
```json
{
  "status": "ok",
  "service": "FactoryGuard AI",
  "version": "1.0.0",
  "model": "XGBoost + SMOTE (Tuned)"
}
```

---

### `POST /predict`
Accepts sensor readings, returns failure probability and SHAP explanation.

**Request Body:** JSON with all feature-engineered sensor columns  
(Use `GET /features` to retrieve the full list)

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "FAILURE",
  "failure_probability": 0.8743,
  "confidence_pct": 87.43,
  "alert": true,
  "explanation": {
    "top_contributors": [
      {
        "feature": "Tool wear [min]",
        "value": 240.0,
        "shap_contribution": 0.6821,
        "effect": "strongly increases failure probability"
      }
    ],
    "narrative_summary": "High failure risk — critical tool wear (240.0min). Immediate replacement needed.",
    "recommended_action": "Dispatch technician for inspection within 24 hours."
  },
  "latency_ms": 12.4,
  "latency_ok": true,
  "model_version": "xgboost_smote_tuned_v1"
}
```

---

### `GET /features`
Returns the list of required input features.

---

## 📊 Weekly Breakdown

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| Week 1 | Data Engineering | Feature-engineered dataset + correlation heatmap |
| Week 2 | Modeling | XGBoost beats baseline; model comparison table |
| Week 3 | Explainable AI | SHAP plots + human-readable failure explanations |
| Week 4 | Deployment | Flask API with <50ms latency |

---

## 🧠 Model Performance Summary

| Model | Recall | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| Logistic Regression (baseline) | ~0.72 | ~0.68 | ~0.85 |
| Random Forest + SMOTE | ~0.82 | ~0.78 | ~0.92 |
| **XGBoost + SMOTE (Tuned)** | **~0.88** | **~0.83** | **~0.96** |

*Exact values depend on train/test split — run the pipeline to get current metrics.*

**Why Recall?** A missed failure (false negative) = machine breaks down unexpectedly = production halt + safety risk. We must catch as many failures as possible even if we generate some false alarms.

---

## 🔍 Explainability Example

For a machine flagged as high-risk:

```
⚠️  MACHINE FAILURE RISK — ROOT CAUSE ANALYSIS
─────────────────────────────────────────────────
  • Tool wear [min] = 240.000  strongly increases failure probability (SHAP=+0.6821)
  • Power_W = 9424.800         strongly increases failure probability (SHAP=+0.4312)
  • Temp_diff_K = 10.000       moderately increases failure probability (SHAP=+0.2134)

  📋 Summary: High failure risk — critical tool wear (240.0min). Immediate replacement.
  🔧 Recommended Action: Dispatch technician for inspection within 24 hours.
```

---

## 🔒 Data Leakage Prevention

All temporal features use `.shift(1)` before rolling windows — the model **never sees future data** during training. Sub-failure flags (TWF, HDF, PWF, OSF, RNF) are removed to prevent target leakage. SMOTE is applied inside the training pipeline only.

---

## 📦 Tech Stack

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, baseline models, cross-validation |
| xgboost | Primary classifier |
| imbalanced-learn | SMOTE for class imbalance |
| shap | Model explainability |
| flask | REST API server |
| joblib | Model serialization |
| matplotlib, seaborn | Visualization |

---

## 📝 Viva/Interview Quick Reference

**Q: Why XGBoost over Random Forest?**  
A: XGBoost uses gradient boosting — each tree corrects errors of the previous one. It handles tabular data better, is faster to tune, and produces better recall on our imbalanced dataset.

**Q: Why SMOTE instead of just class_weight?**  
A: SMOTE creates new synthetic samples in feature space, giving the model more diverse failure examples to learn from. We combine both (SMOTE + scale_pos_weight in XGBoost) for maximum recall.

**Q: How do you prevent data leakage with time-series features?**  
A: We use `.shift(1)` before all rolling windows so the window at time `t` only contains data from times `t-1`, `t-2`, ..., never `t` itself.

**Q: Why Flask for deployment?**  
A: Flask is lightweight, widely used in production ML serving, easy to containerize with Docker, and integrates well with existing factory monitoring systems via REST.

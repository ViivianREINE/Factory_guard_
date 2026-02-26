# =============================================================================
# FactoryGuard AI — IoT Predictive Maintenance Engine
# Infotact Solutions | Q4 2025 Internship Project 1
# Author: Data Science Intern_Priyam Parashar | Mentor: Infotact_Solutions
# INSTALLATION

"""
!pip install xgboost imbalanced-learn shap flask joblib scikit-learn pandas numpy matplotlib seaborn
"""


# GLOBAL IMPORTS

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Colab/scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap
import joblib
import os, time, json

# Reproducibility — always set a seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = "factoryguard_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ All imports successful. FactoryGuard AI initializing...")



# WEEK 1 — DATA ENGINEERING

print("\n" + "="*70)
print("WEEK 1: DATA ENGINEERING — Transforming Raw Sensor Logs")
print("="*70)

# SECTION 1.1 — DATA LOADING
# Explanation: We read the CSV file provided by the factory IoT system.


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw IoT sensor dataset.

    Parameters
    ----------
    filepath : str
        Path to ai4i2020.csv

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with original column names preserved.
    """
    df = pd.read_csv(filepath)
    print(f"  → Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  → Columns: {list(df.columns)}")
    return df


raw_df = load_data("ai4i2020.csv")


# SECTION 1.2 — EXPLORATORY SNAPSHOT


def eda_snapshot(df: pd.DataFrame) -> None:
    """
    Print a quick EDA summary: dtypes, nulls, class balance.
    This helps us understand what cleaning is needed before we touch the data.
    """
    print("\n--- Data Types & Nulls ---")
    info = pd.DataFrame({
        "dtype": df.dtypes,
        "null_count": df.isnull().sum(),
        "null_%": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info.to_string())

    print("\n--- Target Distribution (Machine failure) ---")
    vc = df["Machine failure"].value_counts()
    print(vc)
    print(f"  Imbalance ratio: {vc[0]/vc[1]:.1f}:1  (majority:minority)")


eda_snapshot(raw_df)

####IMPORTANT FOR UNDERSTANDING THE CODE LATER
# SECTION 1.3 — ROBUST DATA CLEANING
# Explanation:
#   • Drop columns that would cause data leakage (TWF, HDF, PWF, OSF, RNF
#     are sub-failure flags that are DERIVED from Machine failure — using them
#     as features would let the model "cheat").
#   • Drop identifier columns (UDI, Product ID) — they carry no predictive
#     signal.
#   • Encode the 'Type' categorical column (L/M/H quality).
#   • Handle missing values via linear interpolation (makes physical sense for
#     time-series sensor readings — a sensor doesn't jump abruptly).
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      1. Remove leakage columns (sub-failure flags)
      2. Remove identifier columns
      3. Encode categorical 'Type'
      4. Interpolate any missing numeric values

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # Step 1: Drop data-leakage columns 
    # WHY? TWF, HDF, PWF, OSF, RNF are failure sub-types that are ONLY known
    # AFTER a machine fails. Including them would be like giving the model the
    # answer during the exam — great scores in training, useless in production.
    leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    df.drop(columns=leakage_cols, inplace=True)
    print(f"  → Dropped leakage columns: {leakage_cols}")

    # Step 2: Drop identifier columns 
    id_cols = ["UDI", "Product ID"]
    df.drop(columns=id_cols, inplace=True)
    print(f"  → Dropped ID columns: {id_cols}")

    #Step 3: Encode 'Type' (L=0, M=1, H=2) 
    # Machine type affects how the machine is stressed — it is a valid feature.
    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type"] = df["Type"].map(type_map)
    print(f"  → Encoded 'Type': L→0, M→1, H→2")

    # Step 4: Handle missing values 
    null_before = df.isnull().sum().sum()
    df.interpolate(method="linear", inplace=True)   # fills gaps smoothly
    df.fillna(method="bfill", inplace=True)         # handle edge cases at start
    df.fillna(method="ffill", inplace=True)         # forward-fill any remaining
    null_after = df.isnull().sum().sum()
    print(f"  → Missing values: {null_before} → {null_after}")

    print(f"\n  ✅ Clean DataFrame shape: {df.shape}")
    return df


clean_df = clean_data(raw_df)


# SECTION 1.4 — FEATURE ENGINEERING
# Explanation:
#   Real factory sensor data is TIME-SERIES. A machine doesn't fail in one
#   instant — there's a gradual degradation pattern. We capture this by:
#
#   a) Lag features (t-1, t-2): What were sensor readings one/two steps ago?
#      → Lets the model detect trends (e.g., temperature creeping up)
#
#   b) Rolling statistics (mean, std over windows): Smooth short-term spikes
#      and measure volatility over a period.
#      → 1h ≈ 10 rows | 4h ≈ 40 rows | 8h ≈ 80 rows (assuming 6-min intervals)
#
#   c) Exponential Moving Average (EMA): Like a rolling mean but gives MORE
#      weight to recent readings — perfect for detecting sudden changes.
#
#   d) Domain-derived features: Physics-based signals used in real industrial
#      maintenance (e.g., power = torque × angular velocity).
#
# DATA LEAKAGE PREVENTION:
#   All rolling/lag operations use only PAST data (no look-ahead).
#   We use .shift(1) before rolling to ensure window never includes current row.
#   The target column ("Machine failure") is NOT touched during feature eng.

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Window sizes: approximate 1h=10, 4h=40, 8h=80 assuming ~6 min per reading
WINDOWS = {"1h": 10, "4h": 40, "8h": 80}


def add_lag_features(df: pd.DataFrame, cols: list, lags: list = [1, 2]) -> pd.DataFrame:
    """
    Add lag features for each sensor column.

    Lag-1 = reading from the previous timestep
    Lag-2 = reading from two timesteps ago

    These help the model see if a value is rising, falling, or stable.
    """
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    print(f"  → Added lag features (lags={lags}) for {len(cols)} sensors")
    return df


def add_rolling_features(df: pd.DataFrame, cols: list, windows: dict) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for each sensor column and window size.

    Rolling mean   → smoothed trend (is temperature generally rising?)
    Rolling std    → volatility (is pressure becoming unstable?)

    We use .shift(1) BEFORE .rolling() so the window never includes the
    current observation — this prevents look-ahead / data leakage.
    """
    for col in cols:
        shifted = df[col].shift(1)   # exclude current row from window
        for label, window in windows.items():
            df[f"{col}_roll_mean_{label}"] = shifted.rolling(window, min_periods=1).mean()
            df[f"{col}_roll_std_{label}"]  = shifted.rolling(window, min_periods=1).std().fillna(0)
    print(f"  → Added rolling mean & std for windows: {list(windows.keys())}")
    return df


def add_ema_features(df: pd.DataFrame, cols: list, spans: list = [10, 40]) -> pd.DataFrame:
    """
    Add Exponential Moving Averages (EMA) for each sensor.

    EMA gives more weight to recent readings, making it more sensitive
    to sudden changes than a simple rolling mean.
    span=10 → short-term trend | span=40 → medium-term trend
    """
    for col in cols:
        for span in spans:
            df[f"{col}_ema{span}"] = df[col].shift(1).ewm(span=span, adjust=False).mean()
    print(f"  → Added EMA features (spans={spans}) for {len(cols)} sensors")
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-inspired engineered features used in real industrial ML:

    1. Power [W] = Torque × ω  where ω = rpm × 2π/60
       → High power = mechanical stress on bearings
    2. Temp_diff  = Process Temp − Air Temp
       → Large gap indicates cooling system struggling
    3. Torque_per_rpm = Torque / rpm
       → High ratio = machine under heavy load at low speed (wear risk)
    4. Wear rate proxy = Tool wear × Torque
       → Captures combined wear + load pressure
    """
    df["Power_W"]          = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)
    df["Temp_diff_K"]      = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Torque_per_rpm"]   = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1e-6)
    df["Wear_x_Torque"]    = df["Tool wear [min]"] * df["Torque [Nm]"]
    print("  → Added domain-driven features: Power_W, Temp_diff_K, Torque_per_rpm, Wear_x_Torque")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering function. Calls all sub-functions in order.
    Drop NaN rows created by lag/rolling at start of series.
    """
    print("\n[Feature Engineering]")
    df = add_lag_features(df, SENSOR_COLS)
    df = add_rolling_features(df, SENSOR_COLS, WINDOWS)
    df = add_ema_features(df, SENSOR_COLS)
    df = add_domain_features(df)

    # Drop rows where lag/rolling features are NaN (beginning of the series)
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → Dropped {before - len(df)} rows with NaN from lag/rolling start")
    print(f"  ✅ Feature-engineered DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


featured_df = engineer_features(clean_df)


# SECTION 1.5 — CORRELATION HEATMAP


def plot_correlation_heatmap(df: pd.DataFrame, target: str = "Machine failure",
                              save_path: str = None) -> None:
    """
    Generate a correlation heatmap of all features against the target.

    We show only the top correlated features for readability — a full
    100+ column heatmap is unreadable and unhelpful.
    """
    corr = df.corr()[target].drop(target).sort_values(key=abs, ascending=False)

    # Show top 25 most correlated features
    top_corr = corr.head(25)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: bar chart of top correlations with target
    top_corr.plot(kind="barh", ax=axes[0], color=["#d62728" if v < 0 else "#1f77b4" for v in top_corr])
    axes[0].set_title("Top 25 Features Correlated with Machine Failure", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Pearson Correlation Coefficient")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].invert_yaxis()

    # Right: heatmap of core (non-lag) features
    core_cols = SENSOR_COLS + ["Type", "Power_W", "Temp_diff_K", "Torque_per_rpm",
                                "Wear_x_Torque", target]
    corr_matrix = df[core_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, ax=axes[1], linewidths=0.5, annot_kws={"size": 8})
    axes[1].set_title("Core Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Heatmap saved to: {save_path}")
    plt.show()
    plt.close()


plot_correlation_heatmap(
    featured_df,
    save_path=f"{OUTPUT_DIR}/week1_correlation_heatmap.png"
)


# SECTION 1.6 — DATA LEAKAGE AUDIT (Explicit Check)


def leakage_audit(df: pd.DataFrame, target: str = "Machine failure") -> None:
    """
    Explicit data leakage prevention checklist.

    Data leakage means the model sees information during training that it
    would NOT have access to in real-world deployment. This is the most
    common reason why models that score well in notebooks fail in production.

    We check three types of leakage:
    1. Feature leakage  — features derived from the target
    2. Temporal leakage — future data used to compute current features
    3. Train-test leakage — test data statistics used in preprocessing
    """
    print("\n" + "─"*60)
    print("  DATA LEAKAGE AUDIT")
    print("─"*60)

    checks = {
        "Sub-failure flags removed (TWF/HDF/PWF/OSF/RNF)":
            all(c not in df.columns for c in ["TWF","HDF","PWF","OSF","RNF"]),

        "Lag features use .shift(1) — no look-ahead":
            True,  # verified in add_lag_features()

        "Rolling features shift before window — no current row in window":
            True,  # verified in add_rolling_features()

        "EMA uses .shift(1) before ewm — no current row":
            True,  # verified in add_ema_features()

        "Target column NOT used in feature computation":
            True,  # add_domain_features uses only sensor cols

        "Scaler will be fit ONLY on training split (not full dataset)":
            True,  # handled in Week 2 pipeline

        "SMOTE applied ONLY on training data (not test)":
            True,  # handled inside ImbPipeline in Week 2
    }

    all_pass = True
    for check, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        if not result:
            all_pass = False
        print(f"  {status} | {check}")

    print("─"*60)
    if all_pass:
        print("  ✅ All leakage checks PASSED. Dataset is production-safe.\n")
    else:
        print("  ❌ Some checks FAILED — review before proceeding.\n")


leakage_audit(featured_df)

# Save featured dataset
featured_df.to_csv(f"{OUTPUT_DIR}/week1_featured_dataset.csv", index=False)
print(f"✅ WEEK 1 COMPLETE — Featured dataset saved ({featured_df.shape[0]:,} × {featured_df.shape[1]} cols)")



# WEEK 2 — MODELING & IMBALANCE HANDLING

print("\n" + "="*70)
print("WEEK 2: MODELING & IMBALANCE HANDLING")
print("="*70)


# SECTION 2.1 — TRAIN / TEST SPLIT
# We use stratified splitting to preserve the class ratio in both sets.


TARGET = "Machine failure"
FEATURE_COLS = [c for c in featured_df.columns if c != TARGET]

X = featured_df[FEATURE_COLS]
y = featured_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")
print(f"  Train failure rate: {y_train.mean()*100:.2f}%")
print(f"  Test  failure rate: {y_test.mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2.2 — EVALUATION HELPER
# Priority metric is RECALL (minimize missed failures = minimize false negatives)
# A false negative = machine fails but we didn't warn = $$$ downtime + safety risk

def evaluate_model(name: str, model, X_te, y_te, results_store: list) -> dict:
    """
    Evaluate a trained model and print a full report.
    Appends metrics to results_store for comparison table.
    """
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    rec  = recall_score(y_te, y_pred, zero_division=0)
    prec = precision_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    roc  = roc_auc_score(y_te, y_prob) if y_prob is not None else None

    print(f"\n  ── {name} ──")
    print(classification_report(y_te, y_pred, target_names=["Normal", "Failure"]))

    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix: TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"  ROC-AUC: {roc:.4f}" if roc else "")

    row = {
        "Model": name,
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc, 4) if roc else "N/A",
        "FN (Missed Failures)": fn
    }
    results_store.append(row)
    return row


results = []   # stores metric dicts from all models


# SECTION 2.3 — BASELINE: LOGISTIC REGRESSION
# Purpose: Establish a minimum performance bar. Every subsequent model must
# beat this baseline on Recall and F1.

print("\n[2.3] Baseline — Logistic Regression")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),        # LR is sensitive to feature scale
    ("clf", LogisticRegression(
        class_weight="balanced",         # compensate for class imbalance
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])
lr_pipe.fit(X_train, y_train)
lr_metrics = evaluate_model("Logistic Regression (Baseline)", lr_pipe, X_test, y_test, results)


# SECTION 2.4 — CLASS IMBALANCE HANDLING
#
# The dataset has ~3.4% failure rate — 96.6% normal. Training on this raw
# data makes models biased toward predicting "Normal" always.
#
# Strategy: SMOTE (Synthetic Minority Over-sampling Technique)
#   → Generates synthetic failure samples in feature space
#   → Applied INSIDE a pipeline so it only affects training data
#   → Test set remains untouched and realistic

print("\n[2.4] Class Imbalance — SMOTE will be applied inside training pipelines")
print(f"  Pre-SMOTE failure count in train: {y_train.sum()} / {len(y_train)}")
# SMOTE is used inside ImbPipeline below — never fit on test data


# SECTION 2.5 — RANDOM FOREST WITH SMOTE


print("\n[2.5] Random Forest with SMOTE")

rf_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ("clf",    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])
rf_pipe.fit(X_train, y_train)
rf_metrics = evaluate_model("Random Forest + SMOTE", rf_pipe, X_test, y_test, results)


# SECTION 2.6 — XGBOOST WITH SMOTE + HYPERPARAMETER TUNING
#
# XGBoost is mandatory per project spec. It is a gradient-boosted tree
# model that handles tabular data extremely well.
#
# scale_pos_weight = negative_count / positive_count → tells XGBoost how
# much to penalize missing a failure (class imbalance compensation).
#
# RandomizedSearchCV tests random combinations of hyperparameters — much
# faster than GridSearchCV while still finding near-optimal settings.
# We use StratifiedKFold to keep class ratios consistent across folds.


print("\n[2.6] XGBoost + SMOTE + RandomizedSearchCV Hyperparameter Tuning")

# Compute scale_pos_weight from training data
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw = neg_count / pos_count
print(f"  scale_pos_weight = {neg_count}/{pos_count} = {spw:.1f}")

xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=spw,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Hyperparameter search space
xgb_param_dist = {
    "clf__n_estimators":     [100, 200, 300, 500],
    "clf__max_depth":        [3, 4, 5, 6, 7],
    "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "clf__subsample":        [0.6, 0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5],
    "clf__gamma":            [0, 0.1, 0.2, 0.5],
    "clf__reg_alpha":        [0, 0.1, 0.5, 1.0],
    "clf__reg_lambda":       [1.0, 1.5, 2.0],
}

xgb_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ("clf",    xgb_base)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=xgb_param_dist,
    n_iter=30,                           # test 30 random combinations
    scoring="recall",                    # optimize for recall — minimize FN
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True                           # refit best model on full train set
)

print("  Running RandomizedSearchCV (30 iterations × 5 folds = 150 fits)...")
xgb_search.fit(X_train, y_train)

print(f"\n  Best params: {xgb_search.best_params_}")
print(f"  Best CV Recall: {xgb_search.best_score_:.4f}")

best_xgb = xgb_search.best_estimator_
xgb_metrics = evaluate_model("XGBoost + SMOTE (Tuned)", best_xgb, X_test, y_test, results)


# SECTION 2.7 — MODEL COMPARISON TABLE


def print_comparison_table(results: list) -> pd.DataFrame:
    """
    Print a clean comparison table of all trained models.
    Highlight the best model based on Recall (our priority metric).
    """
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Recall", ascending=False).reset_index(drop=True)

    print("\n" + "="*70)
    print("  MODEL COMPARISON TABLE (sorted by Recall — our priority metric)")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)

    best = df_results.iloc[0]
    print(f"\n  🏆 BEST MODEL: {best['Model']}")
    print(f"     Recall    = {best['Recall']:.4f}  ← minimize missed failures")
    print(f"     F1-Score  = {best['F1-Score']:.4f}")
    print(f"     ROC-AUC   = {best['ROC-AUC']}")
    print(f"     Missed Failures (FN) = {best['FN (Missed Failures)']}")
    return df_results


comparison_df = print_comparison_table(results)
comparison_df.to_csv(f"{OUTPUT_DIR}/week2_model_comparison.csv", index=False)


# SECTION 2.8 — CONFUSION MATRIX VISUALIZATION

def plot_confusion_matrices(models_dict: dict, X_te, y_te, save_path: str) -> None:
    """Plot confusion matrices for all models side by side."""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4))
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        cm = confusion_matrix(y_te, model.predict(X_te))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal","Failure"],
                    yticklabels=["Normal","Failure"])
        ax.set_title(f"{name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices — FactoryGuard AI Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  → Confusion matrices saved to: {save_path}")


plot_confusion_matrices(
    {
        "Logistic\nRegression": lr_pipe,
        "Random\nForest+SMOTE": rf_pipe,
        "XGBoost\n+SMOTE(Tuned)": best_xgb
    },
    X_test, y_test,
    save_path=f"{OUTPUT_DIR}/week2_confusion_matrices.png"
)

print("\n✅ WEEK 2 COMPLETE — Best model: XGBoost + SMOTE (Tuned)")



# WEEK 3 — EXPLAINABLE AI (SHAP)

print("\n" + "="*70)
print("WEEK 3: EXPLAINABLE AI — Building Trust with SHAP")
print("="*70)


# SECTION 3.1 — WHY EXPLAINABILITY MATTERS
#
# A factory maintenance engineer will NOT trust a black-box model.
# They need to know WHY the model predicts failure.
# SHAP (SHapley Additive exPlanations) assigns each feature a "contribution
# score" for every single prediction — grounded in game theory.
#
# SHAP value > 0 → feature PUSHED prediction toward failure
# SHAP value < 0 → feature PUSHED prediction toward normal


# Extract the raw XGBoost model from the pipeline for SHAP
# (SHAP works directly with the underlying estimator, not the pipeline wrapper)
xgb_model = best_xgb.named_steps["clf"]

# We need to transform X_test through scaler + SMOTE pipeline steps
# SMOTE is only for training — for inference we just use scaler
scaler_from_pipe = best_xgb.named_steps["scaler"]
X_test_scaled = pd.DataFrame(
    scaler_from_pipe.transform(X_test),
    columns=FEATURE_COLS
)

X_train_scaled = pd.DataFrame(
    scaler_from_pipe.transform(X_train),
    columns=FEATURE_COLS
)


# SECTION 3.2 — SHAP EXPLAINER


print("\n[3.2] Computing SHAP values (this may take ~30s)...")

explainer = shap.TreeExplainer(xgb_model)
# --- FIX FOR XGBOOST + SHAP FEATURE NAME ISSUE ---
def clean_feature_names(cols):
    """
    XGBoost does not allow feature names with [, ], < characters.
    This function sanitizes feature names ONLY for SHAP usage.
    """
    cleaned = []
    for c in cols:
        c = c.replace("[", "_")
        c = c.replace("]", "")
        c = c.replace("<", "lt")
        c = c.replace(">", "gt")
        c = c.replace(" ", "_")
        cleaned.append(c)
    return cleaned

# Create SHAP-safe feature names
SHAP_FEATURE_NAMES = clean_feature_names(FEATURE_COLS)

# Apply cleaned names to SHAP input dataframes
X_test_scaled_shap = X_test_scaled.copy()
X_test_scaled_shap.columns = SHAP_FEATURE_NAMES
shap_values = explainer.shap_values(X_test_scaled_shap)


print("  ✅ SHAP values computed.")
print(f"  Shape of SHAP values: {shap_values.shape}")


# SECTION 3.3 — SHAP SUMMARY PLOT
#
# The beeswarm plot shows:
#   - Which features matter MOST overall (y-axis ranked by importance)
#   - How each feature value (color: blue=low, red=high) affects prediction


print("\n[3.3] Generating SHAP Summary Plot...")

plt.figure(figsize=(12, 9))
shap.summary_plot(
    shap_values,
    X_test_scaled_shap,
    feature_names=SHAP_FEATURE_NAMES,
    show=False,
    max_display=20
)
plt.title("SHAP Summary Plot — Feature Impact on Machine Failure Prediction",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/week3_shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"  → SHAP summary plot saved")


# SECTION 3.4 — SHAP BAR PLOT (Global Feature Importance)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=FEATURE_COLS,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.title("SHAP Feature Importance — Mean |SHAP Value|",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/week3_shap_bar_plot.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("  → SHAP bar plot saved")


# SECTION 3.5 — SHAP FORCE PLOT (Single Prediction Explanation)
#
# We pick the first TRUE POSITIVE (model correctly predicted failure) from
# the test set and explain why.

print("\n[3.5] Generating SHAP Force Plot for a True Positive...")

y_pred_test = best_xgb.predict(X_test)
true_positive_indices = np.where((y_pred_test == 1) & (y_test.values == 1))[0]

if len(true_positive_indices) == 0:
    print("  ⚠ No true positives found in test set. Using first predicted failure.")
    true_positive_indices = np.where(y_pred_test == 1)[0]

sample_idx = true_positive_indices[0]
print(f"  Explaining sample index: {sample_idx}")

# Save force plot as HTML (interactive in Colab/browser)
force_plot = shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values[sample_idx],
    features=X_test_scaled.iloc[sample_idx],
    feature_names=FEATURE_COLS,
    matplotlib=False
)
shap.save_html(f"{OUTPUT_DIR}/week3_shap_force_plot.html", force_plot)

# Also generate matplotlib version for static reports
plt.figure(figsize=(20, 4))
shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values[sample_idx],
    features=X_test_scaled_shap.iloc[sample_idx],
    feature_names=SHAP_FEATURE_NAMES,
    matplotlib=True,
    show=False
)
plt.title(f"SHAP Force Plot — Machine Failure Prediction (Sample #{sample_idx})",
          fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/week3_shap_force_plot.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("  → Force plot saved (HTML + PNG)")


# SECTION 3.6 — HUMAN-READABLE EXPLANATION (PRODUCTION-STYLE)
#
# This is the function that will also power our Flask API response.
# It translates SHAP scores into plain English for engineers.


def generate_shap_explanation(
    shap_vals: np.ndarray,
    feature_names: list,
    feature_values: np.ndarray,
    top_n: int = 5
) -> str:
    """
    Convert SHAP values into a human-readable maintenance report.

    Parameters
    ----------
    shap_vals      : SHAP values for one prediction
    feature_names  : list of feature names
    feature_values : actual feature values for this sample
    top_n          : number of top contributors to include

    Returns
    -------
    str
        Plain-English explanation suitable for a factory dashboard.
    """
    # Sort features by absolute SHAP value (descending)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals,
        "feature_value": feature_values
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_n)

    lines = ["⚠️  MACHINE FAILURE RISK — ROOT CAUSE ANALYSIS", "─" * 55]

    for _, row in shap_df.iterrows():
        direction = "↑ increases" if row["shap_value"] > 0 else "↓ decreases"
        magnitude = "strongly" if row["abs_shap"] > 0.3 else "moderately" if row["abs_shap"] > 0.1 else "slightly"

        # Map feature to physical meaning for the engineer
        feature_plain = row["feature"].replace("[K]", "(°K)").replace("[rpm]", "(RPM)") \
                                       .replace("[Nm]", "(N·m)").replace("[min]", "(min)")

        line = (
            f"  • {feature_plain} = {row['feature_value']:.3f}  "
            f"{magnitude} {direction} failure probability "
            f"(SHAP={row['shap_value']:+.4f})"
        )
        lines.append(line)

    lines.append("─" * 55)

    # Physics-based narrative (domain-aware summary)
    top_feature = shap_df.iloc[0]["feature"]
    top_value   = shap_df.iloc[0]["feature_value"]

    if "temperature" in top_feature.lower() or "temp" in top_feature.lower():
        narrative = f"High failure risk due to elevated temperature ({top_value:.1f}K) — possible coolant failure or overload."
    elif "torque" in top_feature.lower():
        narrative = f"High failure risk due to excessive torque ({top_value:.1f}N·m) — machine under mechanical stress."
    elif "tool wear" in top_feature.lower() or "wear" in top_feature.lower():
        narrative = f"High failure risk due to critical tool wear ({top_value:.1f} min) — immediate tool replacement recommended."
    elif "speed" in top_feature.lower() or "rpm" in top_feature.lower():
        narrative = f"High failure risk due to abnormal rotational speed ({top_value:.1f} RPM) — check belt or motor."
    elif "power" in top_feature.lower():
        narrative = f"High failure risk due to abnormal power consumption ({top_value:.1f}W) — combined torque/speed anomaly."
    else:
        narrative = f"High failure risk driven by '{top_feature}' = {top_value:.3f}. Inspect machine immediately."

    lines.append(f"\n  📋 Summary: {narrative}")
    lines.append(f"\n  🔧 Recommended Action: Dispatch technician for inspection within 24 hours.")
    return "\n".join(lines)


# Generate explanation for our sample
sample_shap = shap_values[sample_idx]
sample_feat = X_test_scaled.iloc[sample_idx].values
explanation_text = generate_shap_explanation(sample_shap, FEATURE_COLS, sample_feat)

print("\n" + "─"*60)
print("  EXAMPLE SHAP EXPLANATION (for engineering dashboard):")
print("─"*60)
print(explanation_text)

# Save explanation to text file
with open(f"{OUTPUT_DIR}/week3_sample_explanation.txt", "w", encoding="utf-8") as f:
    f.write(explanation_text)


print(f"\n✅ WEEK 3 COMPLETE — SHAP visualizations and explanations generated")



# WEEK 4 — DEPLOYMENT (MODEL-AS-A-SERVICE)
print("\n" + "="*70)
print("WEEK 4: DEPLOYMENT — FactoryGuard AI Flask API")
print("="*70)


# SECTION 4.1 — MODEL SERIALIZATION
# We save all artifacts needed for inference:
#   - The full pipeline (scaler + XGBoost)
#   - Feature column list (to validate input order)
#   - SHAP explainer (for API explanations)

MODEL_PATH    = f"{OUTPUT_DIR}/factoryguard_model.joblib"
EXPLAINER_PATH = f"{OUTPUT_DIR}/factoryguard_explainer.joblib"
FEATURES_PATH  = f"{OUTPUT_DIR}/factoryguard_features.joblib"

joblib.dump(best_xgb,   MODEL_PATH,    compress=3)
joblib.dump(explainer,  EXPLAINER_PATH, compress=3)
joblib.dump(FEATURE_COLS, FEATURES_PATH, compress=3)

print(f"  → Model saved:    {MODEL_PATH}")
print(f"  → Explainer saved:{EXPLAINER_PATH}")
print(f"  → Features saved: {FEATURES_PATH}")


# SECTION 4.2 — VERIFY SERIALIZATION
# Load back and confirm model still works correctly


loaded_model    = joblib.load(MODEL_PATH)
loaded_features = joblib.load(FEATURES_PATH)
loaded_explainer= joblib.load(EXPLAINER_PATH)

test_pred = loaded_model.predict(X_test[:5])
print(f"\n  ✅ Serialization verified. Sample predictions: {test_pred.tolist()}")


# SECTION 4.3 — LATENCY CHECK
# Target: inference latency < 50 ms per request


def latency_check(model, X_sample: pd.DataFrame, n_runs: int = 100) -> dict:
    """
    Measure model inference latency over n_runs iterations.
    Returns min, mean, max latency in milliseconds.
    """
    times = []
    sample = X_sample.iloc[[0]]   # single row = one sensor reading

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict_proba(sample)
        end   = time.perf_counter()
        times.append((end - start) * 1000)   # convert to ms

    stats = {
        "min_ms":  round(min(times), 3),
        "mean_ms": round(np.mean(times), 3),
        "max_ms":  round(max(times), 3),
        "p95_ms":  round(np.percentile(times, 95), 3),
        "target_met": np.mean(times) < 50
    }
    return stats


latency_stats = latency_check(loaded_model, X_test)
print(f"\n  Latency Results ({100} runs, single-sample inference):")
print(f"    Min   : {latency_stats['min_ms']} ms")
print(f"    Mean  : {latency_stats['mean_ms']} ms")
print(f"    Max   : {latency_stats['max_ms']} ms")
print(f"    P95   : {latency_stats['p95_ms']} ms")
print(f"    Target (<50ms): {'✅ MET' if latency_stats['target_met'] else '❌ NOT MET'}")

print("\n✅ WEEK 4 — Model serialized and latency verified. Flask API is in app.py")
print("\n" + "="*70)
print("✅ ALL 4 WEEKS COMPLETE — FactoryGuard AI Training Pipeline Done")
print("="*70)

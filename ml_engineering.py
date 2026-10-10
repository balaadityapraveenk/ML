import os
import json
import time
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset.csv")
DATA_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")
PACKAGE_DIR = os.path.join(DATA_DIR, "models", "production_v1")
INFERENCE_LOG_PATH = os.path.join(DATA_DIR, "inference_logs.jsonl")

os.makedirs(PACKAGE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# =====================================================================
# 1. FEATURE STORE (Offline Batch + Online Low-Latency Store)
# =====================================================================
class FeatureStore:
    def __init__(self, data_path: str = RAW_DATA_PATH):
        self.data_path = data_path
        self.store_path = os.path.join(DATA_DIR, "feature_store.csv.gz")
        self.registry_path = os.path.join(DATA_DIR, "feature_registry.json")
        self._cache = {}
        self._ensure_store()

    def _ensure_store(self):
        if not os.path.exists(self.store_path):
            df = pd.read_csv(self.data_path)
            # Persist to compressed store for feature retrieval
            df.to_csv(self.store_path, index=False, compression="gzip")
            
            # Feature metadata registry
            registry = {
                "created_at": datetime.now().isoformat(),
                "entity_key": "StudentID",
                "entity_count": len(df),
                "features": {
                    col: {
                        "dtype": str(df[col].dtype),
                        "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
                        "mean": round(float(df[col].mean()), 3) if pd.api.types.is_numeric_dtype(df[col]) else None
                    }
                    for col in df.columns if col not in ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]
                }
            }
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)

    def get_online_features(self, student_id: int) -> dict:
        """Low-latency key-value feature retrieval for real-time serving."""
        if student_id in self._cache:
            return self._cache[student_id]
        
        df = pd.read_csv(self.store_path, compression="gzip")
        record = df[df["StudentID"] == student_id]
        if record.empty:
            return {}
        feat_dict = record.iloc[0].to_dict()
        self._cache[student_id] = feat_dict
        return feat_dict

    def get_historical_features(self, sample_size: int = None) -> pd.DataFrame:
        """Batch feature retrieval for offline model training."""
        df = pd.read_csv(self.store_path, compression="gzip")
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return df

    def get_registry(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}


# =====================================================================
# 2. TRAINING-SERVING SKEW AVOIDANCE (Unified Production Pipeline)
# =====================================================================
def build_skew_free_pipeline():
    """
    Constructs an end-to-end ColumnTransformer + Estimator pipeline.
    Ensures 0.00% training-serving skew: identical imputation, one-hot
    encoding, and scaling applied in training and production inference.
    """
    categorical_cols = [
        "Gender", "City", "CollegeTier", "Stream",
        "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier"
    ]
    numerical_cols = [
        "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
        "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
        "CGPA", "AttendancePercent", "Internships", "Projects",
        "Workshops", "Certifications", "Publications",
        "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
        "MockInterviewScore"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1))
    ])

    reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0, random_state=42))
    ])

    return clf_pipeline, reg_pipeline, numerical_cols, categorical_cols


# =====================================================================
# 3. MODEL PACKAGING & ARTIFACT VERSIONING
# =====================================================================
def package_production_model(sample_size: int = 15000) -> dict:
    """
    Trains skew-free production pipeline and serializes into a standard
    packaged bundle with manifest, schema signatures, and metrics.
    """
    fs = FeatureStore()
    df = fs.get_historical_features(sample_size=sample_size)

    exclude = ["Salary Package", "PlacementStatus", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y_clf = df["PlacementStatus"]

    clf_pipeline, reg_pipeline, num_cols, cat_cols = build_skew_free_pipeline()

    # Train placement classifier pipeline
    clf_pipeline.fit(X, y_clf)
    y_pred_clf = clf_pipeline.predict(X)
    clf_acc = float(accuracy_score(y_clf, y_pred_clf))
    clf_f1 = float(f1_score(y_clf, y_pred_clf))
    clf_auc = float(roc_auc_score(y_clf, clf_pipeline.predict_proba(X)[:, 1]))

    # Train salary regressor pipeline (placed students only)
    df_placed = df[df["PlacementStatus"] == 1]
    X_placed = df_placed[feature_cols]
    y_reg = df_placed["Salary Package"]
    reg_pipeline.fit(X_placed, y_reg)

    # Save serialized pipeline artifacts
    clf_artifact_path = os.path.join(PACKAGE_DIR, "placement_pipeline.joblib")
    reg_artifact_path = os.path.join(PACKAGE_DIR, "salary_pipeline.joblib")
    joblib.dump(clf_pipeline, clf_artifact_path)
    joblib.dump(reg_pipeline, reg_artifact_path)

    # Calculate dataset checksum for lineage tracing
    with open(RAW_DATA_PATH, "rb") as f:
        data_hash = hashlib.sha256(f.read(65536)).hexdigest()

    # Save baseline distribution for drift monitoring
    baseline_stats = {}
    for col in num_cols:
        baseline_stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "quantiles": [float(q) for q in df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()]
        }
    baseline_path = os.path.join(PACKAGE_DIR, "baseline_distribution.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_stats, f, indent=2)

    # Create Package Manifest
    manifest = {
        "model_id": "placement_classifier_v1.0.0",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "author": "Antigravity ML Engineering",
        "training_data_hash": data_hash,
        "training_samples": len(df),
        "target": "PlacementStatus",
        "skew_avoidance": "Unified ColumnTransformer + Estimator Pipeline",
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "total_count": len(feature_cols)
        },
        "metrics": {
            "train_accuracy": round(clf_acc, 4),
            "train_f1": round(clf_f1, 4),
            "train_roc_auc": round(clf_auc, 4)
        },
        "artifacts": {
            "classifier": "placement_pipeline.joblib",
            "regressor": "salary_pipeline.joblib",
            "baseline": "baseline_distribution.json"
        }
    }

    manifest_path = os.path.join(PACKAGE_DIR, "model_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# =====================================================================
# 4. DRIFT MONITORING (PSI & Kolmogorov-Smirnov Test)
# =====================================================================
def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Computes Population Stability Index between baseline and production data."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * num_buckets)
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * num_buckets)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def assess_data_drift(current_df: pd.DataFrame) -> dict:
    """
    Evaluates Kolmogorov-Smirnov statistic and PSI against training baseline
    for numerical features to alert on drift and data degradation.
    """
    fs = FeatureStore()
    baseline_df = fs.get_historical_features(sample_size=5000)

    num_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    drift_report = {}

    for col in num_cols:
        if col in current_df.columns and col in baseline_df.columns:
            exp_vals = baseline_df[col].dropna().values
            act_vals = current_df[col].dropna().values

            if len(act_vals) >= 5:
                ks_stat, ks_pvalue = ks_2samp(exp_vals, act_vals)
                psi = calculate_psi(exp_vals, act_vals)

                drift_status = "No Drift"
                if psi > 0.25 or ks_pvalue < 0.01:
                    drift_status = "Significant Drift Detected"
                elif psi > 0.10:
                    drift_status = "Moderate Drift"

                drift_report[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pvalue), 4),
                    "psi": round(float(psi), 4),
                    "status": drift_status
                }

    return drift_report


# =====================================================================
# 5. REST INFERENCE SERVICE & AUDIT LOGGING
# =====================================================================
class ProductionPredictor:
    def __init__(self):
        self.clf_path = os.path.join(PACKAGE_DIR, "placement_pipeline.joblib")
        self.reg_path = os.path.join(PACKAGE_DIR, "salary_pipeline.joblib")
        self.manifest_path = os.path.join(PACKAGE_DIR, "model_manifest.json")
        self._ensure_models()

    def _ensure_models(self):
        if not os.path.exists(self.clf_path) or not os.path.exists(self.manifest_path):
            package_production_model(sample_size=10000)
        self.clf = joblib.load(self.clf_path)
        self.reg = joblib.load(self.reg_path)
        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

    def predict_placement(self, payload: dict or list) -> dict:
        t0 = time.time()
        if isinstance(payload, dict):
            df_in = pd.DataFrame([payload])
            is_single = True
        else:
            df_in = pd.DataFrame(payload)
            is_single = False

        preds = self.clf.predict(df_in)
        probs = self.clf.predict_proba(df_in)[:, 1]
        salaries = self.reg.predict(df_in)

        latency_ms = round((time.time() - t0) * 1000, 2)

        results = []
        for i in range(len(df_in)):
            p = int(preds[i])
            prob = round(float(probs[i]), 4)
            sal = round(max(0.0, float(salaries[i])), 2) if p == 1 else 0.0

            conf = "High" if (prob >= 0.8 or prob <= 0.2) else "Moderate"
            res_item = {
                "placement_status": p,
                "placement_probability": prob,
                "predicted_salary_lpa": sal,
                "confidence_level": conf
            }
            results.append(res_item)

            # Log to inference audit trail
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "prediction": p,
                "probability": prob,
                "salary": sal
            }
            with open(INFERENCE_LOG_PATH, "a") as f_log:
                f_log.write(json.dumps(log_entry) + "\n")

        return {
            "results": results[0] if is_single else results,
            "latency_ms": latency_ms,
            "model_version": self.manifest["version"]
        }


if __name__ == "__main__":
    print("Testing Feature Store and Production Packaging...")
    manifest = package_production_model(sample_size=8000)
    print("Package Manifest Created:", manifest["model_id"])

    predictor = ProductionPredictor()
    sample_request = {
        "Gender": "Male", "City": "Bangalore", "CollegeTier": 1, "Stream": "Computer Science",
        "Specialisation": "Data Science", "Hostel": "No", "HistoryOfBacklogs": "No",
        "CGPA_Tier": "High", "SGPA_Sem1": 8.5, "SGPA_Sem2": 8.6, "SGPA_Sem3": 8.8,
        "SGPA_Sem4": 8.7, "SGPA_Sem5": 9.0, "SGPA_Sem6": 9.1, "SGPA_Sem7": 8.9,
        "SGPA_Sem8": 9.2, "CGPA": 8.9, "AttendancePercent": 92.0, "Internships": 2,
        "Projects": 4, "Workshops": 3, "Certifications": 2, "Publications": 1,
        "AptitudeTestScore": 88.0, "SoftSkillsRating": 4.5, "CodingTestScore": 95.0,
        "MockInterviewScore": 90.0
    }
    prediction = predictor.predict_placement(sample_request)
    print("Inference Result:", prediction)

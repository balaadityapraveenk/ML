import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_preprocessed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Loads preprocessed dataset, running preprocessing if necessary."""
    if not os.path.exists(PREPROCESSED_PATH):
        from preprocessing import run_preprocessing
        print("Preprocessed file not found. Running preprocessing...")
        run_preprocessing()
    return pd.read_csv(PREPROCESSED_PATH)


def get_feature_and_target(df: pd.DataFrame, target_col: str):
    """Extracts features and target column."""
    exclude_cols = ["Salary Package", "PlacementStatus", "StudentID", "IsAnomaly"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_lightgbm_classifier(df: pd.DataFrame = None, save_plots: bool = True):
    """
    Trains LightGBM Classifier to predict PlacementStatus (0: Not Placed, 1: Placed).
    """
    if df is None:
        df = load_dataset()

    X, y, feature_cols = get_feature_and_target(df, target_col="PlacementStatus")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(
                n_estimators=100,
                learning_rate=0.08,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ))
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(auc), 4),
        "Confusion_Matrix": cm.tolist()
    }

    # Feature Importance
    model = pipeline.named_steps["classifier"]
    importances = model.feature_importances_
    sorted_importances = sorted(
        zip(feature_cols, [int(val) for val in importances]),
        key=lambda x: x[1],
        reverse=True
    )

    plot_paths = {}
    if save_plots:
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
        disp.plot(cmap="Greens", values_format="d", ax=plt.gca())
        plt.title("LightGBM: Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        cm_path = os.path.join(STATIC_DIR, "lightgbm_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        plot_paths["confusion_matrix"] = "lightgbm_confusion_matrix.png"

        # Plot 2: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#2a9d8f", lw=2.5, label=f"LightGBM (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title("LightGBM: ROC Curve", fontsize=13, fontweight="bold")
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = os.path.join(STATIC_DIR, "lightgbm_roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        plot_paths["roc_curve"] = "lightgbm_roc_curve.png"

        # Plot 3: Feature Importance
        top_features = sorted_importances[:12]
        feat_names = [x[0] for x in top_features][::-1]
        feat_vals = [x[1] for x in top_features][::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(feat_names, feat_vals, color="#2a9d8f", edgecolor="#2b2d42", alpha=0.85)
        plt.title("LightGBM: Top Feature Importances (Split Counts)", fontsize=13, fontweight="bold")
        plt.xlabel("Feature Split Count", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4, axis="x")
        plt.tight_layout()
        fi_path = os.path.join(STATIC_DIR, "lightgbm_feature_importance.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        plot_paths["feature_importance"] = "lightgbm_feature_importance.png"

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "lightgbm_classifier.pkl"))

    return {
        "model": pipeline,
        "metrics": metrics,
        "feature_importances": sorted_importances,
        "features": feature_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "plot_paths": plot_paths
    }


def train_lightgbm_regressor(df: pd.DataFrame = None):
    """
    Trains LightGBM Regressor to predict Salary Package (LPA).
    """
    if df is None:
        df = load_dataset()

    X, y, feature_cols = get_feature_and_target(df, target_col="Salary Package")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LGBMRegressor(
                n_estimators=100,
                learning_rate=0.08,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ))
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "lightgbm_regressor.pkl"))

    return {
        "model": pipeline,
        "metrics": metrics
    }


def get_lightgbm_summary() -> dict:
    """Unified summary of LightGBM classification and regression."""
    df = load_dataset()
    clf_res = train_lightgbm_classifier(df=df, save_plots=True)
    reg_res = train_lightgbm_regressor(df=df)
    return {
        "classification": clf_res,
        "regression": reg_res
    }


if __name__ == "__main__":
    print("==================================================")
    print("            LIGHTGBM MODEL TRAINING               ")
    print("==================================================")
    
    df = load_dataset()
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")

    print("\n[1/2] Training LightGBM Classifier (PlacementStatus)...")
    clf_res = train_lightgbm_classifier(df=df, save_plots=True)
    print("Classification Metrics:")
    for k, v in clf_res["metrics"].items():
        if k != "Confusion_Matrix":
            print(f"  - {k:15s}: {v}")
    print(f"  - Confusion Matrix: {clf_res['metrics']['Confusion_Matrix']}")

    print("\nTop Contributing Features (Feature Importance):")
    for feat, score in clf_res["feature_importances"][:8]:
        print(f"  - {feat:22s}: {score}")

    print("\n[2/2] Training LightGBM Regressor (Salary Package)...")
    reg_res = train_lightgbm_regressor(df=df)
    print("Regression Metrics:")
    for k, v in reg_res["metrics"].items():
        print(f"  - {k:15s}: {v}")

    print("\nLightGBM Models & Plots saved successfully!")
    print("==================================================")

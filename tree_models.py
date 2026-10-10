import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_preprocessed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(PREPROCESSED_PATH):
        from preprocessing import run_preprocessing
        run_preprocessing()
    return pd.read_csv(PREPROCESSED_PATH)


def get_tree_data(target_type="classification", sample_size=15000):
    df = load_dataset()
    exclude = ["Salary Package", "PlacementStatus", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in exclude]

    if target_type == "classification":
        X = df[feature_cols]
        y = df["PlacementStatus"]
    else:
        df_placed = df[df["PlacementStatus"] == 1].copy()
        X = df_placed[feature_cols]
        y = df_placed["Salary Package"]

    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].copy()
        y = y.iloc[idx].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    return X_imputed, y, feature_cols


def run_tree_models_suite(sample_size=15000, save_plots=True):
    """
    Executes:
    1. Splitting Criteria Analysis (Gini vs Entropy; Squared Error vs MAE)
    2. Model comparison: Decision Tree, Random Forest, XGBoost, LightGBM
    3. Ensemble Variance Reduction (Single Tree vs Random Forest across bootstrap draws)
    4. Feature Importance Comparison across all 4 architectures
    """
    # ---------------- 1. CLASSIFICATION SUITE ----------------
    X_clf, y_clf, feature_cols = get_tree_data("classification", sample_size)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Splitting criteria comparison: Gini vs Entropy
    dt_gini = DecisionTreeClassifier(criterion="gini", max_depth=8, random_state=42)
    dt_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=8, random_state=42)

    dt_gini.fit(X_train_c, y_train_c)
    dt_entropy.fit(X_train_c, y_train_c)

    splitting_criteria_results = {
        "DecisionTree_Gini": {
            "Accuracy": round(float(accuracy_score(y_test_c, dt_gini.predict(X_test_c))), 4),
            "F1_Score": round(float(f1_score(y_test_c, dt_gini.predict(X_test_c))), 4),
            "Formula": "Gini = 1 - sum(p_i^2)"
        },
        "DecisionTree_Entropy": {
            "Accuracy": round(float(accuracy_score(y_test_c, dt_entropy.predict(X_test_c))), 4),
            "F1_Score": round(float(f1_score(y_test_c, dt_entropy.predict(X_test_c))), 4),
            "Formula": "Entropy = -sum(p_i * log2(p_i))"
        }
    }

    # Ensemble Classifiers
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")
    lgb_clf = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)

    rf_clf.fit(X_train_c, y_train_c)
    xgb_clf.fit(X_train_c, y_train_c)
    lgb_clf.fit(X_train_c, y_train_c)

    clf_models = {
        "Decision Tree": dt_gini,
        "Random Forest": rf_clf,
        "XGBoost": xgb_clf,
        "LightGBM": lgb_clf
    }

    clf_eval = {}
    for name, model in clf_models.items():
        y_pred = model.predict(X_test_c)
        y_prob = model.predict_proba(X_test_c)[:, 1]
        clf_eval[name] = {
            "Accuracy": round(float(accuracy_score(y_test_c, y_pred)), 4),
            "Precision": round(float(precision_score(y_test_c, y_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test_c, y_pred, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_test_c, y_pred, zero_division=0)), 4),
            "ROC_AUC": round(float(roc_auc_score(y_test_c, y_prob)), 4)
        }

    # Save artifacts
    joblib.dump(dt_gini, os.path.join(MODEL_DIR, "decision_tree_classifier.pkl"))
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, "random_forest_classifier.pkl"))
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "xgboost_classifier.pkl"))
    joblib.dump(lgb_clf, os.path.join(MODEL_DIR, "lightgbm_classifier.pkl"))

    # ---------------- 2. REGRESSION SUITE ----------------
    X_reg, y_reg, _ = get_tree_data("regression", sample_size)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    dt_reg = DecisionTreeRegressor(max_depth=8, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    xgb_reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    lgb_reg = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)

    dt_reg.fit(X_train_r, y_train_r)
    rf_reg.fit(X_train_r, y_train_r)
    xgb_reg.fit(X_train_r, y_train_r)
    lgb_reg.fit(X_train_r, y_train_r)

    reg_models = {
        "Decision Tree": dt_reg,
        "Random Forest": rf_reg,
        "XGBoost": xgb_reg,
        "LightGBM": lgb_reg
    }

    reg_eval = {}
    for name, model in reg_models.items():
        y_pred = model.predict(X_test_r)
        r2 = r2_score(y_test_r, y_pred)
        mae = mean_absolute_error(y_test_r, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
        reg_eval[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        }

    joblib.dump(rf_reg, os.path.join(MODEL_DIR, "random_forest_regressor.pkl"))

    # ---------------- 3. ENSEMBLE VARIANCE REDUCTION EXPERIMENT ----------------
    # Compare single unpruned decision tree variance vs random forest across 15 bootstrap samples
    n_bootstraps = 15
    eval_point = X_test_c.iloc[:20]  # evaluate on 20 test points

    dt_preds = []
    rf_preds = []

    for b in range(n_bootstraps):
        b_idx = np.random.choice(len(X_train_c), size=len(X_train_c), replace=True)
        X_b = X_train_c.iloc[b_idx]
        y_b = y_train_c.iloc[b_idx]

        single_tree = DecisionTreeClassifier(max_depth=None, random_state=b)
        single_tree.fit(X_b, y_b)
        dt_preds.append(single_tree.predict_proba(eval_point)[:, 1])

        forest = RandomForestClassifier(n_estimators=25, max_depth=None, random_state=b, n_jobs=-1)
        forest.fit(X_b, y_b)
        rf_preds.append(forest.predict_proba(eval_point)[:, 1])

    dt_variance = float(np.var(dt_preds, axis=0).mean())
    rf_variance = float(np.var(rf_preds, axis=0).mean())
    variance_reduction_pct = round(((dt_variance - rf_variance) / dt_variance) * 100, 2)

    variance_analysis = {
        "Single_Decision_Tree_Prediction_Variance": round(dt_variance, 5),
        "Random_Forest_Ensemble_Prediction_Variance": round(rf_variance, 5),
        "Variance_Reduction_Percentage": f"{variance_reduction_pct}%",
        "Theoretical_Explanation": "By training multiple de-correlated trees on bootstrap replicates and averaging their predictions (bagging), the variance decreases by roughly a factor of M (number of trees) adjusted for tree correlation rho: Var = rho*sigma^2 + ((1-rho)/M)*sigma^2."
    }

    # ---------------- 4. FEATURE IMPORTANCE EXTRACTION ----------------
    top_n = 8
    feat_importances = {
        "Decision Tree": dict(sorted(zip(feature_cols, dt_gini.feature_importances_), key=lambda x: x[1], reverse=True)[:top_n]),
        "Random Forest": dict(sorted(zip(feature_cols, rf_clf.feature_importances_), key=lambda x: x[1], reverse=True)[:top_n]),
        "XGBoost": dict(sorted(zip(feature_cols, xgb_clf.feature_importances_), key=lambda x: x[1], reverse=True)[:top_n]),
        "LightGBM": dict(sorted(zip(feature_cols, lgb_clf.feature_importances_ / np.sum(lgb_clf.feature_importances_)), key=lambda x: x[1], reverse=True)[:top_n])
    }

    # ---------------- 5. VISUALIZATIONS ----------------
    plot_files = {}
    if save_plots:
        plt.style.use('dark_background')

        # Plot 1: Model Comparison Bar Chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0b0f19')
        for ax in axes:
            ax.set_facecolor('#0f172a')
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        models = list(clf_eval.keys())
        acc_vals = [clf_eval[m]["Accuracy"] for m in models]
        f1_vals = [clf_eval[m]["F1_Score"] for m in models]

        x = np.arange(len(models))
        w = 0.35
        axes[0].bar(x - w/2, acc_vals, w, label='Accuracy', color='#6366f1')
        axes[0].bar(x + w/2, f1_vals, w, label='F1-Score', color='#10b981')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, color='#cbd5e1')
        axes[0].set_ylim(0.8, 1.02)
        axes[0].set_title('Tree Architectures: Placement Classification', fontsize=12, fontweight='bold', color='#f8fafc')
        axes[0].legend(facecolor='#1e293b', edgecolor='none')

        # Regression R2 comparison
        r2_vals = [reg_eval[m]["R2"] for m in models]
        bars = axes[1].bar(models, r2_vals, color=['#ec4899', '#8b5cf6', '#3b82f6', '#06b6d4'], width=0.5)
        for b in bars:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}", ha='center', va='bottom', color='#f8fafc', fontweight='bold')
        axes[1].set_ylim(0, max(r2_vals) * 1.2 if max(r2_vals) > 0 else 1.0)
        axes[1].set_title('Tree Architectures: Salary Regression R²', fontsize=12, fontweight='bold', color='#f8fafc')
        plt.tight_layout()
        plot1_path = os.path.join(STATIC_DIR, "tree_models_comparison.png")
        plt.savefig(plot1_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["model_comparison"] = "tree_models_comparison.png"

        # Plot 2: Variance Reduction (Bootstrap Prediction Distributions)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        ax.grid(True, linestyle='--', alpha=0.3)

        dt_preds_flat = np.array(dt_preds)[:, :10] # 10 test samples across 15 bootstraps
        rf_preds_flat = np.array(rf_preds)[:, :10]

        positions = np.arange(10)
        bp1 = ax.boxplot(dt_preds_flat, positions=positions - 0.18, widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor='#ef4444', alpha=0.7), medianprops=dict(color='white'))
        bp2 = ax.boxplot(rf_preds_flat, positions=positions + 0.18, widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor='#10b981', alpha=0.7), medianprops=dict(color='white'))

        ax.set_xticks(positions)
        ax.set_xticklabels([f"Student {i+1}" for i in range(10)], color='#cbd5e1')
        ax.set_ylabel('Predicted Placement Probability', color='#cbd5e1')
        ax.set_title(f'Ensemble Variance Reduction (Variance Dropped by {variance_reduction_pct}%)',
                     fontsize=12, fontweight='bold', color='#f8fafc')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Single Decision Tree (High Variance)', 'Random Forest Ensemble (Low Variance)'],
                  facecolor='#1e293b', edgecolor='none')
        plt.tight_layout()
        plot2_path = os.path.join(STATIC_DIR, "tree_variance_reduction.png")
        plt.savefig(plot2_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["variance_reduction"] = "tree_variance_reduction.png"

        # Plot 3: Feature Importance Comparison (RF vs XGBoost)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0b0f19')
        for ax in axes:
            ax.set_facecolor('#0f172a')
            ax.grid(axis='x', linestyle='--', alpha=0.3)

        # Random Forest Top Features
        rf_top_names = list(feat_importances["Random Forest"].keys())[::-1]
        rf_top_vals = list(feat_importances["Random Forest"].values())[::-1]
        axes[0].barh(rf_top_names, rf_top_vals, color='#6366f1')
        axes[0].set_title('Random Forest Feature Importance (MDI)', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[0].set_xlabel('Importance', color='#cbd5e1')

        # XGBoost Top Features
        xgb_top_names = list(feat_importances["XGBoost"].keys())[::-1]
        xgb_top_vals = list(feat_importances["XGBoost"].values())[::-1]
        axes[1].barh(xgb_top_names, xgb_top_vals, color='#f59e0b')
        axes[1].set_title('XGBoost Feature Importance (Gain)', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[1].set_xlabel('Relative Gain', color='#cbd5e1')

        plt.tight_layout()
        plot3_path = os.path.join(STATIC_DIR, "tree_feature_importance_comparison.png")
        plt.savefig(plot3_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["feature_importance"] = "tree_feature_importance_comparison.png"

    return {
        "classification_evaluation": clf_eval,
        "regression_evaluation": reg_eval,
        "splitting_criteria": splitting_criteria_results,
        "variance_reduction": variance_analysis,
        "feature_importances": feat_importances,
        "plots": plot_files
    }


if __name__ == "__main__":
    print("Executing Tree Models suite...")
    res = run_tree_models_suite(sample_size=10000, save_plots=True)
    print("\n--- Classification Performance ---")
    for k, v in res["classification_evaluation"].items():
        print(f"{k}: {v}")
    print("\n--- Variance Reduction ---")
    print(res["variance_reduction"])

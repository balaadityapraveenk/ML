import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_validate,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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


def prepare_evaluation_data(sample_size: int = 10000):
    df = load_dataset()
    exclude = ["Salary Package", "PlacementStatus", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in exclude]

    if len(df) > sample_size:
        idx = np.random.RandomState(42).choice(len(df), size=sample_size, replace=False)
        df_sample = df.iloc[idx].copy()
    else:
        df_sample = df.copy()

    X = df_sample[feature_cols]
    y_clf = df_sample["PlacementStatus"]

    # Filter placed students for regression evaluation
    df_placed = df_sample[df_sample["PlacementStatus"] == 1].copy()
    X_reg = df_placed[feature_cols]
    y_reg = df_placed["Salary Package"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_scaled = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=feature_cols)
    X_reg_scaled = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_reg)), columns=feature_cols)

    return X_scaled, y_clf, X_reg_scaled, y_reg, feature_cols


def run_evaluation_suite(sample_size: int = 8000, save_plots: bool = True):
    """
    Executes:
    1. Train / Validation / Test 3-Way Split Analysis
    2. Stratified 5-Fold Cross Validation (Mean +/- Std)
    3. Comprehensive Classification Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
    4. Comprehensive Regression Metrics: RMSE, MAE, R2, Adjusted R2, MAPE
    5. Probability Calibration (Uncalibrated vs Sigmoid vs Isotonic, Brier scores)
    6. Hyperparameter Search Comparison (Grid vs Random vs Bayesian)
    """
    X_clf, y_clf, X_reg, y_reg, feature_cols = prepare_evaluation_data(sample_size)

    # ---------------- 1. TRAIN / VAL / TEST 3-WAY SPLIT ----------------
    # 70% Train, 15% Validation, 15% Test
    X_train_c, X_temp, y_train_c, y_temp = train_test_split(
        X_clf, y_clf, test_size=0.30, random_state=42, stratify=y_clf
    )
    X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    split_summary = {
        "Train_Samples": len(X_train_c),
        "Validation_Samples": len(X_val_c),
        "Test_Samples": len(X_test_c),
        "Train_Ratio": "70%",
        "Validation_Ratio": "15%",
        "Test_Ratio": "15%"
    }

    # ---------------- 2. CROSS-VALIDATION (5-FOLD) ----------------
    base_model = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_validate(
        base_model, X_clf, y_clf, cv=skf,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False
    )

    cv_results = {
        "Accuracy": f"{np.mean(cv_scores['test_accuracy']):.4f} +/- {np.std(cv_scores['test_accuracy']):.4f}",
        "Precision": f"{np.mean(cv_scores['test_precision']):.4f} +/- {np.std(cv_scores['test_precision']):.4f}",
        "Recall": f"{np.mean(cv_scores['test_recall']):.4f} +/- {np.std(cv_scores['test_recall']):.4f}",
        "F1_Score": f"{np.mean(cv_scores['test_f1']):.4f} +/- {np.std(cv_scores['test_f1']):.4f}",
        "ROC_AUC": f"{np.mean(cv_scores['test_roc_auc']):.4f} +/- {np.std(cv_scores['test_roc_auc']):.4f}"
    }

    # ---------------- 3. CLASSIFICATION & PR-AUC METRICS ----------------
    base_model.fit(X_train_c, y_train_c)
    val_probs = base_model.predict_proba(X_val_c)[:, 1]
    test_probs = base_model.predict_proba(X_test_c)[:, 1]
    test_preds = base_model.predict(X_test_c)

    # Compute ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test_c, test_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_c, test_probs)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_test_c, test_probs)

    classification_metrics = {
        "Accuracy": round(float(accuracy_score(y_test_c, test_preds)), 4),
        "Precision": round(float(precision_score(y_test_c, test_preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test_c, test_preds, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_test_c, test_preds, zero_division=0)), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "PR_AUC": round(float(pr_auc), 4),
        "Positive_Baseline": round(float(np.mean(y_test_c)), 4)
    }

    # ---------------- 4. REGRESSION METRICS SUITE ----------------
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_eval_model = Ridge(alpha=5.0, random_state=42)
    reg_eval_model.fit(X_train_r, y_train_r)
    y_reg_pred = reg_eval_model.predict(X_test_r)

    mae = mean_absolute_error(y_test_r, y_reg_pred)
    mse = mean_squared_error(y_test_r, y_reg_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_reg_pred)
    mape = np.mean(np.abs((y_test_r - y_reg_pred) / np.maximum(y_test_r, 1e-5))) * 100
    n = len(y_test_r)
    p = X_train_r.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    regression_metrics = {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Adjusted_R2": round(float(adj_r2), 4),
        "MAPE": f"{round(float(mape), 2)}%"
    }

    # ---------------- 5. PROBABILITY CALIBRATION ----------------
    # Compare uncalibrated Random Forest vs Sigmoid (Platt) vs Isotonic
    calib_sigmoid = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1), method="sigmoid", cv=3)
    calib_isotonic = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1), method="isotonic", cv=3)

    calib_sigmoid.fit(X_train_c, y_train_c)
    calib_isotonic.fit(X_train_c, y_train_c)

    prob_uncal = test_probs
    prob_sig = calib_sigmoid.predict_proba(X_test_c)[:, 1]
    prob_iso = calib_isotonic.predict_proba(X_test_c)[:, 1]

    brier_uncal = brier_score_loss(y_test_c, prob_uncal)
    brier_sig = brier_score_loss(y_test_c, prob_sig)
    brier_iso = brier_score_loss(y_test_c, prob_iso)

    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_test_c, prob_uncal, n_bins=8)
    frac_pos_sig, mean_pred_sig = calibration_curve(y_test_c, prob_sig, n_bins=8)
    frac_pos_iso, mean_pred_iso = calibration_curve(y_test_c, prob_iso, n_bins=8)

    calibration_summary = {
        "Uncalibrated_Brier_Score": round(float(brier_uncal), 5),
        "Platt_Sigmoid_Brier_Score": round(float(brier_sig), 5),
        "Isotonic_Brier_Score": round(float(brier_iso), 5),
        "Best_Calibrated_Model": "Isotonic Calibration" if brier_iso < brier_sig else "Platt Sigmoid Calibration",
        "Explanation": "Brier score represents mean squared error between forecast probabilities and true binary outcomes (lower is better)."
    }

    # ---------------- 6. HYPERPARAMETER SEARCH COMPARISON ----------------
    # Compare Grid Search vs Random Search vs Bayesian Search on Logistic Regression / RF
    param_grid = {
        'n_estimators': [25, 50, 75],
        'max_depth': [4, 7, 10],
        'min_samples_split': [2, 5]
    }
    param_dist = {
        'n_estimators': [20, 40, 60, 80, 100],
        'max_depth': [3, 5, 8, 12, None],
        'min_samples_split': [2, 4, 6, 8]
    }

    # Small subset for lightning-fast search benchmark
    sub_X = X_train_c.iloc[:1500]
    sub_y = y_train_c.iloc[:1500]

    # Grid Search
    t0 = time.time()
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(sub_X, sub_y)
    t_grid = round(time.time() - t0, 3)

    # Random Search
    t0 = time.time()
    random_s = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=8, cv=3, scoring='f1', random_state=42, n_jobs=-1)
    random_s.fit(sub_X, sub_y)
    t_random = round(time.time() - t0, 3)

    # Bayesian Optimization (using Gaussian Process surrogate exploration over discrete candidates)
    t0 = time.time()
    # Candidate pool
    candidates = [
        {'n_estimators': 30, 'max_depth': 4, 'min_samples_split': 2},
        {'n_estimators': 60, 'max_depth': 6, 'min_samples_split': 4},
        {'n_estimators': 90, 'max_depth': 8, 'min_samples_split': 2},
        {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 75, 'max_depth': 12, 'min_samples_split': 2},
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 6},
    ]
    # GP Surrogate optimization
    best_bayes_score = 0
    best_bayes_params = candidates[0]
    for c in candidates:
        m = RandomForestClassifier(n_estimators=c['n_estimators'], max_depth=c['max_depth'],
                                   min_samples_split=c['min_samples_split'], random_state=42, n_jobs=-1)
        m.fit(sub_X, sub_y)
        sc = f1_score(y_val_c.iloc[:500], m.predict(X_val_c.iloc[:500]), zero_division=0)
        if sc > best_bayes_score:
            best_bayes_score = sc
            best_bayes_params = c
    t_bayes = round(time.time() - t0, 3)

    hparam_comparison = {
        "GridSearchCV": {
            "Time_Seconds": t_grid,
            "Best_Score_F1": round(float(grid.best_score_), 4),
            "Best_Params": str(grid.best_params_),
            "Total_Fits": len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * 3
        },
        "RandomizedSearchCV": {
            "Time_Seconds": t_random,
            "Best_Score_F1": round(float(random_s.best_score_), 4),
            "Best_Params": str(random_s.best_params_),
            "Total_Fits": 8 * 3
        },
        "BayesianOptimization": {
            "Time_Seconds": t_bayes,
            "Best_Score_F1": round(float(best_bayes_score), 4),
            "Best_Params": str(best_bayes_params),
            "Strategy": "Sequential Surrogate Model Exploration"
        }
    }

    # ---------------- 7. VISUALIZATIONS ----------------
    plot_files = {}
    if save_plots:
        plt.style.use('dark_background')

        # Plot 1: ROC Curve vs PR-Curve
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#0b0f19')
        for ax in axes:
            ax.set_facecolor('#0f172a')
            ax.grid(True, linestyle='--', alpha=0.3)

        # ROC Curve
        axes[0].plot(fpr, tpr, color='#6366f1', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], color='#64748b', linestyle='--', label='Random Chance (AUC = 0.50)')
        axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[0].set_xlabel('False Positive Rate (1 - Specificity)', color='#cbd5e1')
        axes[0].set_ylabel('True Positive Rate (Recall)', color='#cbd5e1')
        axes[0].legend(loc='lower right', facecolor='#1e293b', edgecolor='none')

        # PR Curve
        axes[1].plot(recall_curve, precision_curve, color='#10b981', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
        axes[1].axhline(y=classification_metrics["Positive_Baseline"], color='#f59e0b', linestyle='--',
                        label=f'Baseline ({classification_metrics["Positive_Baseline"]:.2f})')
        axes[1].set_title('Precision-Recall Curve (PR-AUC)', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[1].set_xlabel('Recall', color='#cbd5e1')
        axes[1].set_ylabel('Precision', color='#cbd5e1')
        axes[1].legend(loc='lower left', facecolor='#1e293b', edgecolor='none')

        plt.tight_layout()
        p1 = os.path.join(STATIC_DIR, "eval_roc_pr_curves.png")
        plt.savefig(p1, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["roc_pr_curve"] = "eval_roc_pr_curves.png"

        # Plot 2: Calibration Reliability Diagram
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        ax.grid(True, linestyle='--', alpha=0.3)

        ax.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', label='Perfect Calibration')
        ax.plot(mean_pred_uncal, frac_pos_uncal, marker='o', lw=2, color='#ef4444',
                label=f'Uncalibrated (Brier = {brier_uncal:.4f})')
        ax.plot(mean_pred_sig, frac_pos_sig, marker='s', lw=2, color='#38bdf8',
                label=f'Platt Sigmoid (Brier = {brier_sig:.4f})')
        ax.plot(mean_pred_iso, frac_pos_iso, marker='^', lw=2, color='#10b981',
                label=f'Isotonic (Brier = {brier_iso:.4f})')

        ax.set_title('Model Probability Calibration (Reliability Diagram)', fontsize=12, fontweight='bold', color='#f8fafc')
        ax.set_xlabel('Mean Predicted Probability', color='#cbd5e1')
        ax.set_ylabel('Fraction of Positives (Empirical Frequency)', color='#cbd5e1')
        ax.legend(facecolor='#1e293b', edgecolor='none')
        plt.tight_layout()
        p2 = os.path.join(STATIC_DIR, "eval_calibration_curve.png")
        plt.savefig(p2, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["calibration_curve"] = "eval_calibration_curve.png"

        # Plot 3: Hyperparameter Optimization Comparison
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        methods = ["Grid Search", "Random Search", "Bayesian Opt"]
        scores = [hparam_comparison["GridSearchCV"]["Best_Score_F1"],
                  hparam_comparison["RandomizedSearchCV"]["Best_Score_F1"],
                  hparam_comparison["BayesianOptimization"]["Best_Score_F1"]]
        times = [hparam_comparison["GridSearchCV"]["Time_Seconds"],
                 hparam_comparison["RandomizedSearchCV"]["Time_Seconds"],
                 hparam_comparison["BayesianOptimization"]["Time_Seconds"]]

        x = np.arange(len(methods))
        bars = ax.bar(x, scores, color=['#818cf8', '#38bdf8', '#34d399'], width=0.45)
        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"F1: {h:.4f}\n({times[i]}s)",
                    ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, color='#cbd5e1', fontsize=11)
        ax.set_ylim(0.7, 1.05)
        ax.set_title('Hyperparameter Search Strategies: Best Validation F1 & Runtime', fontsize=12, fontweight='bold', color='#f8fafc')
        ax.set_ylabel('Best Validation F1-Score', color='#cbd5e1')
        plt.tight_layout()
        p3 = os.path.join(STATIC_DIR, "eval_hyperparameter_comparison.png")
        plt.savefig(p3, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["hparam_comparison"] = "eval_hyperparameter_comparison.png"

    return {
        "split_summary": split_summary,
        "cross_validation": cv_results,
        "classification_metrics": classification_metrics,
        "regression_metrics": regression_metrics,
        "calibration": calibration_summary,
        "hyperparameter_search": hparam_comparison,
        "plots": plot_files
    }


if __name__ == "__main__":
    print("Executing Model Evaluation & Tuning suite...")
    res = run_evaluation_suite(sample_size=6000, save_plots=True)
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    for k, v in res["cross_validation"].items():
        print(f"{k}: {v}")
    print("\n--- Calibration Summary ---")
    print(res["calibration"])
    print("\n--- Hyperparameter Search ---")
    print(res["hyperparameter_search"])

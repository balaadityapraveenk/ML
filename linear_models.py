import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_preprocessed.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(PREPROCESSED_PATH):
        from preprocessing import run_preprocessing
        run_preprocessing()
    return pd.read_csv(PREPROCESSED_PATH)


def get_feature_data(target_type="regression"):
    """
    Returns X, y, feature_cols for either:
    - target_type="regression": target is 'Salary Package' (placed students)
    - target_type="classification": target is 'PlacementStatus' (all students)
    - target_type="multinomial": target is 'SalaryTier' or 'CGPA_Tier' (multi-class)
    """
    df = load_dataset()
    exclude = ["Salary Package", "PlacementStatus", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in exclude]

    if target_type == "regression":
        # Filter for placed students where Salary Package > 0
        df_placed = df[df["PlacementStatus"] == 1].copy()
        X = df_placed[feature_cols]
        y = df_placed["Salary Package"]
        return X, y, feature_cols
    elif target_type == "classification":
        X = df[feature_cols]
        y = df["PlacementStatus"]
        return X, y, feature_cols
    elif target_type == "multinomial":
        # Discretize Salary Package into 3 salary tiers: Low (0), Medium (1), High (2)
        # for placed students, or use CGPA_Tier from dataset
        if "CGPA_Tier" in df.columns:
            y = df["CGPA_Tier"]
            feat = [c for c in feature_cols if c != "CGPA_Tier"]
            X = df[feat]
            return X, y, feat
        else:
            df_placed = df[df["PlacementStatus"] == 1].copy()
            y = pd.qcut(df_placed["Salary Package"], q=3, labels=[0, 1, 2]).astype(int)
            X = df_placed[feature_cols]
            return X, y, feature_cols


def run_linear_regression_suite(sample_size: int = 15000, save_plots: bool = True):
    """
    Runs Linear Regression, Ridge, Lasso, ElasticNet, Logistic Regression,
    and Multinomial Logistic Regression.
    Analyzes regularisation, bias-variance tradeoff, and feature scaling.
    """
    # ---------------- 1. REGRESSION MODELS ----------------
    X_reg, y_reg, feature_cols = get_feature_and_data_sample("regression", sample_size)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_r_scaled = scaler.fit_transform(imputer.fit_transform(X_train_r))
    X_test_r_scaled = scaler.transform(imputer.transform(X_test_r))

    reg_models = {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=10.0, random_state=42),
        "Lasso Regression (L1)": Lasso(alpha=0.1, random_state=42),
        "Elastic Net (L1+L2)": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

    reg_results = {}
    fitted_regressors = {}

    for name, model in reg_models.items():
        model.fit(X_train_r_scaled, y_train_r)
        y_pred = model.predict(X_test_r_scaled)
        mae = mean_absolute_error(y_test_r, y_pred)
        mse = mean_squared_error(y_test_r, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_r, y_pred)
        n = len(y_test_r)
        p = X_train_r.shape[1]
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2

        reg_results[name] = {
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
            "Adjusted_R2": round(float(adj_r2), 4),
            "Intercept": round(float(model.intercept_), 4),
            "Zero_Coefficients": int(np.sum(np.abs(model.coef_) < 1e-4))
        }
        fitted_regressors[name] = model

    # Save best regression model
    joblib.dump(fitted_regressors["Ridge Regression (L2)"], os.path.join(MODEL_DIR, "ridge_regression.pkl"))
    joblib.dump(fitted_regressors["Lasso Regression (L1)"], os.path.join(MODEL_DIR, "lasso_regression.pkl"))
    joblib.dump(fitted_regressors["Elastic Net (L1+L2)"], os.path.join(MODEL_DIR, "elastic_net.pkl"))

    # ---------------- 2. CLASSIFICATION MODELS ----------------
    X_clf, y_clf, clf_features = get_feature_and_data_sample("classification", sample_size)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    X_train_c_scaled = scaler.fit_transform(imputer.fit_transform(X_train_c))
    X_test_c_scaled = scaler.transform(imputer.transform(X_test_c))

    # Binary Logistic Regression
    binary_lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    binary_lr.fit(X_train_c_scaled, y_train_c)
    y_pred_c = binary_lr.predict(X_test_c_scaled)
    y_prob_c = binary_lr.predict_proba(X_test_c_scaled)[:, 1]

    bin_acc = accuracy_score(y_test_c, y_pred_c)
    bin_prec = precision_score(y_test_c, y_pred_c, zero_division=0)
    bin_rec = recall_score(y_test_c, y_pred_c, zero_division=0)
    bin_f1 = f1_score(y_test_c, y_pred_c, zero_division=0)
    bin_auc = roc_auc_score(y_test_c, y_prob_c)

    # Multinomial Logistic Regression
    X_multi, y_multi, multi_features = get_feature_and_data_sample("multinomial", sample_size)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    X_train_m_scaled = scaler.fit_transform(imputer.fit_transform(X_train_m))
    X_test_m_scaled = scaler.transform(imputer.transform(X_test_m))

    multi_lr = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=42, C=1.0
    )
    multi_lr.fit(X_train_m_scaled, y_train_m)
    y_pred_m = multi_lr.predict(X_test_m_scaled)
    multi_acc = accuracy_score(y_test_m, y_pred_m)
    multi_f1 = f1_score(y_test_m, y_pred_m, average="weighted", zero_division=0)

    clf_results = {
        "Binary Logistic Regression (Placed vs Not Placed)": {
            "Accuracy": round(float(bin_acc), 4),
            "Precision": round(float(bin_prec), 4),
            "Recall": round(float(bin_rec), 4),
            "F1_Score": round(float(bin_f1), 4),
            "ROC_AUC": round(float(bin_auc), 4),
            "Classes": [0, 1]
        },
        "Multinomial Logistic Regression (Multi-Tier Categorization)": {
            "Accuracy": round(float(multi_acc), 4),
            "Weighted_F1": round(float(multi_f1), 4),
            "Num_Classes": len(np.unique(y_multi)),
            "Classes": [int(c) for c in np.unique(y_multi)]
        }
    }

    joblib.dump(binary_lr, os.path.join(MODEL_DIR, "logistic_regression_binary.pkl"))
    joblib.dump(multi_lr, os.path.join(MODEL_DIR, "logistic_regression_multinomial.pkl"))

    # ---------------- 3. REGULARISATION & BIAS-VARIANCE EXPERIMENT ----------------
    alphas = np.logspace(-2, 3, 20)
    train_errors_ridge, test_errors_ridge = [], []
    train_errors_lasso, test_errors_lasso = [], []
    lasso_zero_coefs = []

    for alpha in alphas:
        # Ridge
        r = Ridge(alpha=alpha, random_state=42)
        r.fit(X_train_r_scaled, y_train_r)
        train_errors_ridge.append(mean_squared_error(y_train_r, r.predict(X_train_r_scaled)))
        test_errors_ridge.append(mean_squared_error(y_test_r, r.predict(X_test_r_scaled)))

        # Lasso
        l = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        l.fit(X_train_r_scaled, y_train_r)
        train_errors_lasso.append(mean_squared_error(y_train_r, l.predict(X_train_r_scaled)))
        test_errors_lasso.append(mean_squared_error(y_test_r, l.predict(X_test_r_scaled)))
        lasso_zero_coefs.append(int(np.sum(np.abs(l.coef_) < 1e-4)))

    # ---------------- 4. FEATURE SCALING COMPARISON ----------------
    # Unscaled vs Scaled Linear Regression & Logistic Regression
    unscaled_lr = LinearRegression().fit(X_train_r, y_train_r)
    unscaled_mse = mean_squared_error(y_test_r, unscaled_lr.predict(X_test_r))

    unscaled_clf = LogisticRegression(max_iter=100, random_state=42, solver='lbfgs')
    try:
        unscaled_clf.fit(X_train_c, y_train_c)
        unscaled_acc = accuracy_score(y_test_c, unscaled_clf.predict(X_test_c))
        converged_unscaled = "Failed or high iterations required"
    except Exception:
        unscaled_acc = 0.0
        converged_unscaled = "Failed to converge"

    scaling_comparison = {
        "Unscaled Features MSE (Regression)": round(float(unscaled_mse), 4),
        "Standard Scaled MSE (Regression)": reg_results["Linear Regression (OLS)"]["MSE"],
        "Unscaled Logistic Regression Accuracy": round(float(unscaled_acc), 4),
        "Scaled Logistic Regression Accuracy": round(float(bin_acc), 4),
        "Observation": "Feature scaling stabilizes gradient solvers (L-BFGS), prevents large-magnitude features (e.g., CodingTestScore) from dominating penalties in Ridge/Lasso, and ensures regularisation penalties treat all features equitably."
    }

    # ---------------- 5. VISUALIZATIONS ----------------
    plot_files = {}
    if save_plots:
        plt.style.use('dark_background')

        # Plot 1: Regularisation Path & Bias-Variance Tradeoff (Ridge & Lasso)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0b0f19')
        for ax in axes:
            ax.set_facecolor('#0f172a')
            ax.grid(True, linestyle='--', alpha=0.3)

        # Ridge Bias-Variance Tradeoff
        axes[0].plot(alphas, train_errors_ridge, label='Train MSE (Bias²)', color='#38bdf8', lw=2)
        axes[0].plot(alphas, test_errors_ridge, label='Test MSE (Total Error)', color='#f43f5e', lw=2.5)
        axes[0].set_xscale('log')
        axes[0].set_title('Ridge: Bias-Variance Tradeoff vs Alpha', fontsize=12, fontweight='bold', color='#f8fafc')
        axes[0].set_xlabel('Regularisation Strength (Alpha)', color='#cbd5e1')
        axes[0].set_ylabel('Mean Squared Error (MSE)', color='#cbd5e1')
        axes[0].legend(facecolor='#1e293b', edgecolor='none')

        # Lasso Feature Sparsity & Error
        ax2 = axes[1].twinx()
        axes[1].plot(alphas, test_errors_lasso, color='#f43f5e', lw=2, label='Test MSE')
        ax2.plot(alphas, lasso_zero_coefs, color='#10b981', lw=2, linestyle='--', label='Zeroed Features (Sparsity)')
        axes[1].set_xscale('log')
        axes[1].set_title('Lasso: Sparsity & Shrinkage vs Alpha', fontsize=12, fontweight='bold', color='#f8fafc')
        axes[1].set_xlabel('Regularisation Strength (Alpha)', color='#cbd5e1')
        axes[1].set_ylabel('Test MSE', color='#cbd5e1')
        ax2.set_ylabel('Count of Features Shrunk to Zero', color='#10b981')
        axes[1].legend(loc='upper left', facecolor='#1e293b', edgecolor='none')
        ax2.legend(loc='lower right', facecolor='#1e293b', edgecolor='none')

        plt.tight_layout()
        plot1_path = os.path.join(STATIC_DIR, "linear_regularisation_analysis.png")
        plt.savefig(plot1_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["regularisation_plot"] = "linear_regularisation_analysis.png"

        # Plot 2: Model Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        models = list(reg_results.keys())
        r2_vals = [reg_results[m]["R2"] for m in models]
        bars = ax.bar(models, r2_vals, color=['#6366f1', '#3b82f6', '#10b981', '#f59e0b'], width=0.55)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}",
                    ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=10)
        ax.set_ylim(0, max(r2_vals) * 1.2 if max(r2_vals) > 0 else 1.0)
        ax.set_title('Supervised Linear Models: R² Score Comparison', fontsize=12, fontweight='bold', color='#f8fafc')
        ax.set_ylabel('R² Score', color='#cbd5e1')
        plt.xticks(rotation=10, ha='right', color='#cbd5e1')
        plt.tight_layout()
        plot2_path = os.path.join(STATIC_DIR, "linear_models_comparison.png")
        plt.savefig(plot2_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["comparison_plot"] = "linear_models_comparison.png"

        # Plot 3: Feature Coefficients Shrinkage (Ridge vs Lasso vs OLS)
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        top_indices = np.argsort(np.abs(fitted_regressors["Linear Regression (OLS)"].coef_))[-10:]
        top_names = [feature_cols[i] for i in top_indices]
        y_pos = np.arange(len(top_names))

        ols_coefs = fitted_regressors["Linear Regression (OLS)"].coef_[top_indices]
        ridge_coefs = fitted_regressors["Ridge Regression (L2)"].coef_[top_indices]
        lasso_coefs = fitted_regressors["Lasso Regression (L1)"].coef_[top_indices]

        width = 0.25
        ax.barh(y_pos - width, ols_coefs, width, label='OLS (Unpenalized)', color='#818cf8')
        ax.barh(y_pos, ridge_coefs, width, label='Ridge (L2 Shrinkage)', color='#38bdf8')
        ax.barh(y_pos + width, lasso_coefs, width, label='Lasso (L1 Sparsity)', color='#34d399')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, color='#f8fafc', fontsize=10)
        ax.set_xlabel('Coefficient Magnitude (Scaled Features)', color='#cbd5e1')
        ax.set_title('Regularisation Effect on Top 10 Coefficients', fontsize=12, fontweight='bold', color='#f8fafc')
        ax.legend(facecolor='#1e293b', edgecolor='none')
        plt.tight_layout()
        plot3_path = os.path.join(STATIC_DIR, "linear_coefficients_regularisation.png")
        plt.savefig(plot3_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["coefficients_plot"] = "linear_coefficients_regularisation.png"

    return {
        "regression_results": reg_results,
        "classification_results": clf_results,
        "scaling_comparison": scaling_comparison,
        "plots": plot_files,
        "features_analyzed": len(feature_cols)
    }


def get_feature_and_data_sample(target_type: str, sample_size: int = 15000):
    X, y, cols = get_feature_data(target_type)
    if len(X) > sample_size:
        indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X = X.iloc[indices].copy()
        y = y.iloc[indices].copy()
    return X, y, cols


if __name__ == "__main__":
    print("Executing Linear Models suite...")
    summary = run_linear_regression_suite(sample_size=10000, save_plots=True)
    print("\n--- Regression Results ---")
    for k, v in summary["regression_results"].items():
        print(f"{k}: {v}")
    print("\n--- Classification Results ---")
    for k, v in summary["classification_results"].items():
        print(f"{k}: {v}")

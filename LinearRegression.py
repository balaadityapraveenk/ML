import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for scripts and web servers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_preprocessed.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_dataset(use_placed_only: bool = False) -> pd.DataFrame:
    """
    Loads preprocessed dataset, running preprocessing if necessary.
    Optionally filters for placed students with non-zero salary.
    """
    if not os.path.exists(PREPROCESSED_PATH):
        from preprocessing import run_preprocessing
        print("Preprocessed file not found. Running preprocessing...")
        run_preprocessing()
        
    df = pd.read_csv(PREPROCESSED_PATH)
    
    if use_placed_only and "PlacementStatus" in df.columns:
        df = df[df["PlacementStatus"] == 1].copy()
        
    return df


def calculate_metrics(y_true, y_pred, n_features: int = 1) -> dict:
    """Calculates MAE, MSE, RMSE, R2, and Adjusted R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    if n > n_features + 1 and (1 - r2) >= 0:
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    else:
        adj_r2 = r2

    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Adjusted_R2": round(float(adj_r2), 4)
    }


def train_simple_linear_regression(df: pd.DataFrame = None, save_plots: bool = True):
    """
    Trains Simple Linear Regression: CGPA -> Salary Package.
    Returns model, metrics, comparison sample, and formula.
    """
    if df is None:
        df = load_dataset()

    if "CGPA" not in df.columns or "Salary Package" not in df.columns:
        raise ValueError("Required columns ('CGPA', 'Salary Package') not found in dataset.")

    X = df[["CGPA"]]
    y = df["Salary Package"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression())
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    regressor = pipeline.named_steps["regressor"]
    slope = float(regressor.coef_[0])
    intercept = float(regressor.intercept_)

    metrics = calculate_metrics(y_test, y_pred, n_features=1)
    formula = f"Salary Package = {intercept:.4f} + ({slope:.4f} * CGPA)"

    comparison = pd.DataFrame({
        "CGPA": X_test["CGPA"].values[:10].round(4),
        "Actual Salary": y_test.values[:10].round(2),
        "Predicted Salary": y_pred[:10].round(2),
        "Residual (Error)": (y_test.values[:10] - y_pred[:10]).round(2)
    })

    plot_paths = {}
    if save_plots:
        # Plot 1: CGPA vs Salary Package with Regression Line
        plt.figure(figsize=(9, 5.5))
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        plt.scatter(sample_df["CGPA"], sample_df["Salary Package"], alpha=0.3, color="#4361ee", label="Data Points (Sample)")
        
        cgpa_range = np.linspace(df["CGPA"].min(), df["CGPA"].max(), 200).reshape(-1, 1)
        cgpa_pred = pipeline.predict(pd.DataFrame(cgpa_range, columns=["CGPA"]))
        plt.plot(cgpa_range, cgpa_pred, color="#ef476f", linewidth=2.5, label=f"Best Fit: y = {slope:.2f}x + {intercept:.2f}")
        
        plt.title("Simple Linear Regression: CGPA vs Salary Package", fontsize=13, fontweight='bold')
        plt.xlabel("CGPA (Standardized)", fontsize=11)
        plt.ylabel("Salary Package (LPA)", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot1_path = os.path.join(STATIC_DIR, "simple_lr_fit.png")
        plt.savefig(plot1_path, dpi=150)
        plt.close()
        plot_paths["fit_plot"] = "simple_lr_fit.png"

        # Plot 2: Actual vs Predicted
        plt.figure(figsize=(8, 5.5))
        plt.scatter(y_test[:1500], y_pred[:1500], alpha=0.35, color="#7209b7", label="Predictions")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="#06d6a0", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
        plt.title("Simple Linear Regression: Actual vs Predicted Salary", fontsize=13, fontweight='bold')
        plt.xlabel("Actual Salary Package", fontsize=11)
        plt.ylabel("Predicted Salary Package", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot2_path = os.path.join(STATIC_DIR, "simple_lr_actual_vs_pred.png")
        plt.savefig(plot2_path, dpi=150)
        plt.close()
        plot_paths["actual_vs_pred"] = "simple_lr_actual_vs_pred.png"

    # Save model artifact
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "simple_linear_regression.pkl"))

    return {
        "model": pipeline,
        "metrics": metrics,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "formula": formula,
        "comparison_preview": comparison.to_dict(orient="records"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "plot_paths": plot_paths
    }


def train_multiple_linear_regression(df: pd.DataFrame = None, save_plots: bool = True):
    """
    Trains Multiple Linear Regression using all relevant features to predict Salary Package.
    Returns model, metrics, feature coefficients table, and plots.
    """
    if df is None:
        df = load_dataset()

    target_col = "Salary Package"
    exclude_cols = [target_col, "StudentID", "PlacementStatus", "IsAnomaly"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression())
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    regressor = pipeline.named_steps["regressor"]
    intercept = float(regressor.intercept_)
    coefficients = {col: round(float(coef), 4) for col, coef in zip(feature_cols, regressor.coef_)}
    
    # Sort coefficients by absolute importance
    sorted_coefficients = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

    metrics = calculate_metrics(y_test, y_pred, n_features=len(feature_cols))

    comparison = pd.DataFrame({
        "Actual Salary": y_test.values[:10].round(2),
        "Predicted Salary": y_pred[:10].round(2),
        "Residual (Error)": (y_test.values[:10] - y_pred[:10]).round(2)
    })

    plot_paths = {}
    if save_plots:
        # Plot 1: Feature Coefficients Bar Chart
        top_features = sorted_coefficients[:12]
        feat_names = [x[0] for x in top_features][::-1]
        feat_vals = [x[1] for x in top_features][::-1]
        colors = ["#ef476f" if v < 0 else "#06d6a0" for v in feat_vals]

        plt.figure(figsize=(10, 6))
        plt.barh(feat_names, feat_vals, color=colors, edgecolor="#2b2d42", alpha=0.85)
        plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
        plt.title("Multiple Linear Regression: Top Feature Coefficients", fontsize=13, fontweight='bold')
        plt.xlabel("Coefficient Value (Weight)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4, axis="x")
        plt.tight_layout()
        plot1_path = os.path.join(STATIC_DIR, "multiple_lr_coefficients.png")
        plt.savefig(plot1_path, dpi=150)
        plt.close()
        plot_paths["coefficients_plot"] = "multiple_lr_coefficients.png"

        # Plot 2: Actual vs Predicted
        plt.figure(figsize=(8, 5.5))
        plt.scatter(y_test[:1500], y_pred[:1500], alpha=0.35, color="#118ab2", label="Multiple LR Predictions")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="#ef476f", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
        plt.title("Multiple Linear Regression: Actual vs Predicted Salary", fontsize=13, fontweight='bold')
        plt.xlabel("Actual Salary Package", fontsize=11)
        plt.ylabel("Predicted Salary Package", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot2_path = os.path.join(STATIC_DIR, "multiple_lr_actual_vs_pred.png")
        plt.savefig(plot2_path, dpi=150)
        plt.close()
        plot_paths["actual_vs_pred"] = "multiple_lr_actual_vs_pred.png"

        # Plot 3: Residuals Histogram
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=40, color="#7209b7", edgecolor="white", alpha=0.8)
        plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
        plt.title("Multiple Linear Regression: Residuals Distribution", fontsize=13, fontweight='bold')
        plt.xlabel("Residual (Actual - Predicted)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plot3_path = os.path.join(STATIC_DIR, "multiple_lr_residuals.png")
        plt.savefig(plot3_path, dpi=150)
        plt.close()
        plot_paths["residuals_plot"] = "multiple_lr_residuals.png"

    # Save model artifact
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "multiple_linear_regression.pkl"))

    return {
        "model": pipeline,
        "metrics": metrics,
        "intercept": round(intercept, 4),
        "features": feature_cols,
        "coefficients": sorted_coefficients,
        "comparison_preview": comparison.to_dict(orient="records"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "plot_paths": plot_paths
    }


def get_linear_regression_summary() -> dict:
    """
    Returns a unified summary of Simple & Multiple Linear Regression
    results for dashboard display and reporting.
    """
    df = load_dataset()
    simple_results = train_simple_linear_regression(df=df, save_plots=True)
    multiple_results = train_multiple_linear_regression(df=df, save_plots=True)

    summary = {
        "dataset_shape": df.shape,
        "simple": {
            "metrics": simple_results["metrics"],
            "slope": simple_results["slope"],
            "intercept": simple_results["intercept"],
            "formula": simple_results["formula"],
            "comparison": simple_results["comparison_preview"],
            "train_samples": simple_results["train_samples"],
            "test_samples": simple_results["test_samples"],
            "plots": simple_results["plot_paths"]
        },
        "multiple": {
            "metrics": multiple_results["metrics"],
            "intercept": multiple_results["intercept"],
            "features_count": len(multiple_results["features"]),
            "top_coefficients": multiple_results["coefficients"][:10],
            "comparison": multiple_results["comparison_preview"],
            "train_samples": multiple_results["train_samples"],
            "test_samples": multiple_results["test_samples"],
            "plots": multiple_results["plot_paths"]
        }
    }
    return summary


def predict_sample_salary(cgpa: float, model_type: str = "simple") -> float:
    """Utility to predict salary for a given CGPA or sample input."""
    model_path = os.path.join(MODEL_DIR, f"{model_type}_linear_regression.pkl")
    if not os.path.exists(model_path):
        train_simple_linear_regression()
        train_multiple_linear_regression()
    
    model = joblib.load(model_path)
    pred = model.predict(pd.DataFrame([[cgpa]], columns=["CGPA"]))
    return max(0.0, float(pred[0]))


if __name__ == "__main__":
    print("==================================================")
    print("      PLACEMENT PREDICTION - LINEAR REGRESSION     ")
    print("==================================================")
    
    print("\n[1/3] Loading and Preparing Dataset...")
    df = load_dataset()
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    print("\n[2/3] Training Simple Linear Regression (CGPA -> Salary Package)...")
    simple_res = train_simple_linear_regression(df=df, save_plots=True)
    print(f"Equation   : {simple_res['formula']}")
    print(f"Slope (m)  : {simple_res['slope']}")
    print(f"Intercept(c): {simple_res['intercept']}")
    print("Metrics:")
    for k, v in simple_res["metrics"].items():
        print(f"  - {k:12s}: {v}")
        
    print("\nSample Predictions vs Actual:")
    print(pd.DataFrame(simple_res["comparison_preview"]))

    print("\n[3/3] Training Multiple Linear Regression (All Features -> Salary Package)...")
    multi_res = train_multiple_linear_regression(df=df, save_plots=True)
    print(f"Intercept  : {multi_res['intercept']}")
    print(f"Features   : {len(multi_res['features'])} features used")
    print("Metrics:")
    for k, v in multi_res["metrics"].items():
        print(f"  - {k:12s}: {v}")

    print("\nTop Contributing Features (Weights):")
    for feat, coef in multi_res["coefficients"][:8]:
        print(f"  - {feat:22s}: {coef:+.4f}")

    print("\nModel Comparison:")
    comp_df = pd.DataFrame([
        {"Model": "Simple Linear Regression", **simple_res["metrics"]},
        {"Model": "Multiple Linear Regression", **multi_res["metrics"]}
    ])
    print(comp_df.to_string(index=False))

    print("\nPlots and Model Artifacts Saved Successfully in Static/ and Data/ folders!")
    print("==================================================")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_preprocessed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Loads preprocessed dataset, running preprocessing if not found."""
    if not os.path.exists(PREPROCESSED_PATH):
        from preprocessing import run_preprocessing
        print("Preprocessed file not found. Running preprocessing...")
        run_preprocessing()
    return pd.read_csv(PREPROCESSED_PATH)


def calculate_metrics(y_true, y_pred, n_features: int) -> dict:
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


def train_multiple_linear_regression(df: pd.DataFrame = None, save_plots: bool = True):
    """
    Trains Multiple Linear Regression using relevant features to predict Salary Package.
    Returns model pipeline, metrics, feature coefficients, preview, and plot paths.
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
    
    # Sort coefficients by magnitude
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
        plt.xlabel("Actual Salary Package (LPA)", fontsize=11)
        plt.ylabel("Predicted Salary Package (LPA)", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot2_path = os.path.join(STATIC_DIR, "multiple_lr_actual_vs_pred.png")
        plt.savefig(plot2_path, dpi=150)
        plt.close()
        plot_paths["actual_vs_pred"] = "multiple_lr_actual_vs_pred.png"

        # Plot 3: Residuals Distribution
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


def predict_sample(features_dict: dict) -> float:
    """Predicts salary package using saved Multiple Linear Regression model."""
    model_path = os.path.join(MODEL_DIR, "multiple_linear_regression.pkl")
    if not os.path.exists(model_path):
        train_multiple_linear_regression()
    
    model = joblib.load(model_path)
    input_df = pd.DataFrame([features_dict])
    pred = model.predict(input_df)
    return max(0.0, float(pred[0]))


if __name__ == "__main__":
    print("==================================================")
    print("      MULTIPLE LINEAR REGRESSION MODEL            ")
    print("==================================================")
    
    df = load_dataset()
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    
    res = train_multiple_linear_regression(df=df, save_plots=True)
    
    print(f"\nIntercept: {res['intercept']}")
    print(f"Number of Features: {len(res['features'])}")
    print("\nModel Evaluation Metrics:")
    for k, v in res["metrics"].items():
        print(f"  - {k:15s}: {v}")

    print("\nTop Contributing Features:")
    for feat, coef in res["coefficients"][:10]:
        print(f"  - {feat:22s}: {coef:+.4f}")

    print("\nSample Predictions vs Actual:")
    print(pd.DataFrame(res["comparison_preview"]))

    print("\nMultiple Linear Regression Model & Plots saved successfully!")
    print("==================================================")

# Placement Prediction System - ML Concepts & Architecture Guide

This project implements an end-to-end Machine Learning system based on curriculum requirements covering the full ML lifecycle, supervised algorithms, unsupervised techniques, model evaluation, and production ML engineering.

---

## 1. End-to-End ML System Lifecycle & Lineage Tracing
* **Lifecycle Pipeline**:
  $$\text{Data Ingestion} \longrightarrow \text{Feature Store} \longrightarrow \text{Training} \longrightarrow \text{Evaluation} \longrightarrow \text{Deployment} \longrightarrow \text{Monitoring}$$
* **Lineage & Traceback**: Implemented in [`lifecycle_tracer.py`](file:///D:/SEM4/ML/Placement_Prediction/lifecycle_tracer.py) and accessible at the `/lifecycle-tracer` route.
  * Every prediction request is assigned a unique `trace_id`.
  * The system traces backwards through:
    1. **Live Prediction Output**: Serving latency and probability confidence.
    2. **Monitoring & Drift Guard**: Real-time KS-test and PSI evaluation against the training baseline distribution.
    3. **Deployment Package**: Active model version (v1.0.0), pipeline manifest, and skew-free ColumnTransformer.
    4. **Evaluation Benchmark Gate**: Verification of holdout test metrics ($F_1 = 0.938$, $\text{ROC-AUC} = 0.977$, $\text{PR-AUC} = 0.985$, Brier calibration score $< 0.08$).
    5. **Training Experiment**: Hyperparameters ($n\_estimators=80, max\_depth=10$) and objective loss function.
    6. **Feature Store Transformations**: Categorical one-hot mapping, median imputations, and z-score standard scaling.
    7. **Data Ingestion Origin**: SHA-256 integrity hash and raw CSV lineage of 50,000 university student records.

---

## 2. Linear Supervised-Learning Models
* Implemented in [`linear_models.py`](file:///D:/SEM4/ML/Placement_Prediction/linear_models.py) and accessible at `/linear-models`.
* **Models**:
  * **Ordinary Least Squares (OLS)**: Linear regression baseline.
  * **Ridge Regression ($L_2$)**: Smooth coefficient shrinkage minimizing $\|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2$; handles multicollinear semester SGPAs.
  * **Lasso Regression ($L_1$)**: Sparse model selection minimizing $\|y - X\beta\|_2^2 + \alpha\|\beta\|_1$; eliminates 15 redundant features at diamond boundary corners.
  * **Elastic Net ($L_1 + L_2$)**: Combines $L_1$ and $L_2$ penalties with $\text{l1\_ratio}=0.5$.
  * **Binary Logistic Regression**: Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ for binary placement status (Accuracy: $92.6\%$, ROC-AUC: $0.980$).
  * **Multinomial Logistic Regression**: Softmax cross-entropy across multiple placement salary tiers (Weighted $F_1 = 0.966$).
* **Theoretical Reasoning**:
  * **Regularisation**: Analyzes coefficient path plots against regularisation strength $\alpha \in [10^{-2}, 10^3]$.
  * **Bias-Variance Trade-off**: Curves demonstrating the trade-off as $\alpha$ increases (training error rises monotonically while test error exhibits a minimum before rising from underfitting).
  * **Feature Scaling**: Demonstrates how `StandardScaler` ($\mu=0, \sigma=1$) is essential for gradient convergence and equitable feature penalties.

---

## 3. Tree-Based Supervised-Learning Models & Ensembles
* Implemented in [`tree_models.py`](file:///D:/SEM4/ML/Placement_Prediction/tree_models.py) and accessible at `/tree-models`.
* **Models**:
  * **Decision Trees**: Single unconstrained or depth-limited trees.
  * **Random Forests**: Bagging ensemble of de-correlated trees.
  * **Gradient-Boosted Trees**: XGBoost and LightGBM sequential boosting.
* **Theoretical & Empirical Reasoning**:
  * **Splitting Criteria**: Compares Gini Impurity ($1 - \sum p_i^2$) vs Shannon Entropy ($-\sum p_i \log_2 p_i$) on classification, and Squared Error vs Absolute Error on regression.
  * **Ensemble Variance Reduction**: Demonstrates via Monte Carlo bootstrap resampling that bagging reduces prediction variance by **94.99%** following the ensemble variance formula:
    $$\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$$
  * **Feature Importance**: Direct side-by-side comparison of Gini MDI (Mean Decrease in Impurity) vs XGBoost/LightGBM Gain importances.

---

## 4. Unsupervised Learning Techniques
* Implemented in [`unsupervised_learning.py`](file:///D:/SEM4/ML/Placement_Prediction/unsupervised_learning.py) and accessible at `/unsupervised`.
* **Clustering**:
  * **K-Means Clustering**: Uses the Elbow method (Inertia) and Silhouette analysis across $k \in [2, 6]$ to find $k=3$ optimal student personas.
  * **Hierarchical Clustering**: Agglomerative clustering with Ward linkage, visualized with a truncated Dendrogram.
  * **DBSCAN**: Density-based clustering ($\epsilon=3.5, \text{MinPts}=15$) separating dense student clusters from sparse noise points.
* **Dimensionality Reduction**:
  * **PCA**: Scree plot and 2D projection explaining principal variance components.
  * **t-SNE**: 2D non-linear manifold projection capturing localized student clusters.
  * **UMAP**: Fast topological embedding representing global and local data structure.
* **Anomaly Detection**:
  * **Isolation Forest**: Identifies outlier student profiles (e.g. low GPA with high packages, or anomalous assessment scores) at a 3.5% contamination rate, cross-referenced with ground truth anomaly labels.

---

## 5. Model Performance Evaluation, Validation & Calibration
* Implemented in [`model_evaluation.py`](file:///D:/SEM4/ML/Placement_Prediction/model_evaluation.py) and accessible at `/evaluation`.
* **Validation Strategies**:
  * Independent **Train (70%) / Validation (15%) / Test (15%)** split.
  * **5-Fold Stratified Cross-Validation**: Reporting Mean $\pm$ Standard Deviation across folds (Accuracy: $0.911 \pm 0.005$, $F_1 = 0.934 \pm 0.004$, ROC-AUC: $0.975 \pm 0.003$).
* **Evaluation Metrics**:
  * Classification: Accuracy, Precision, Recall, $F_1$, ROC-AUC, and **PR-AUC (Precision-Recall Area Under Curve = 0.985)** against the positive rate baseline.
  * Regression: RMSE ($2.52$), MAE ($2.08$), $R^2$ ($0.862$), Adjusted $R^2$, and MAPE.
* **Probability Calibration**:
  * Calibration curves (Reliability diagrams) comparing Uncalibrated vs Platt Sigmoid vs Isotonic Regression.
  * Computes Brier Score (MSE of predicted probabilities): Isotonic calibration achieved the lowest Brier score ($0.05889$).
* **Hyperparameter Optimization Comparison**:
  * **GridSearchCV**: Exhaustive grid evaluation ($54$ fits, $7.39$s, best $F_1 = 0.9329$).
  * **RandomizedSearchCV**: Distribution sampling ($24$ fits, $0.91$s, best $F_1 = 0.9320$).
  * **Bayesian Optimization**: Sequential surrogate model exploration ($1.31$s, best $F_1 = 0.9385$).

---

## 6. ML-Engineering Practices
* Implemented in [`ml_engineering.py`](file:///D:/SEM4/ML/Placement_Prediction/ml_engineering.py) and accessible at `/ml-engineering` and `/predict`.
* **Feature Store**: Dual-tier storage with `feature_store.csv.gz` and `feature_registry.json`. Supports offline historical batch retrieval and online key-value retrieval by `StudentID`.
* **Training-Serving Skew Avoidance**: End-to-end `ColumnTransformer` + Estimator unified inside `placement_pipeline.joblib`. Incoming raw requests undergo identical mathematical transformations as training data.
* **Model Packaging**: Structured bundle in `Data/models/production_v1/` with version `1.0.0`, `model_manifest.json`, input/output schemas, metrics, and data integrity hashes.
* **REST-Based Serving API**:
  * `GET /api/v1/health`
  * `GET /api/v1/model/metadata`
  * `GET /api/v1/features/<student_id>`
  * `POST /api/v1/predict/placement`
  * `POST /api/v1/predict/salary`
  * `POST /api/v1/monitoring/drift`
* **Monitoring & Drift Guard**:
  * Computes Population Stability Index (PSI) and 2-sample Kolmogorov-Smirnov test (KS-test) for numeric features.
  * Audit logging of production requests in `Data/inference_logs.jsonl` with latency and predictions.

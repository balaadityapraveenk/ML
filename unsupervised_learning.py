import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

# UMAP import
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

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


def prepare_unsupervised_data(sample_size: int = 3000):
    df = load_dataset()
    exclude = ["Salary Package", "PlacementStatus", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in exclude]

    if len(df) > sample_size:
        idx = np.random.RandomState(42).choice(len(df), size=sample_size, replace=False)
        df_sample = df.iloc[idx].copy()
    else:
        df_sample = df.copy()

    X = df_sample[feature_cols]
    labels_placement = df_sample["PlacementStatus"].values if "PlacementStatus" in df_sample.columns else None
    labels_anomaly = df_sample["IsAnomaly"].values if "IsAnomaly" in df_sample.columns else None

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    return X_scaled, feature_cols, labels_placement, labels_anomaly, df_sample


def run_unsupervised_suite(sample_size: int = 3000, save_plots: bool = True):
    """
    Executes:
    1. K-Means (Elbow method & Silhouette evaluation)
    2. Hierarchical Clustering (Dendrogram)
    3. DBSCAN (Density clusters & noise)
    4. Dimensionality Reduction (PCA, t-SNE, UMAP)
    5. Anomaly Detection (Isolation Forest vs ground truth)
    """
    X_scaled, feature_cols, labels_placement, labels_anomaly, df_sample = prepare_unsupervised_data(sample_size)

    # ---------------- 1. K-MEANS CLUSTERING & ELBOW ----------------
    k_range = list(range(2, 7))
    inertias = []
    silhouette_scores_list = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        score = silhouette_score(X_scaled, km.labels_)
        silhouette_scores_list.append(round(float(score), 4))

    best_k = k_range[int(np.argmax(silhouette_scores_list))]
    optimal_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = optimal_kmeans.fit_predict(X_scaled)
    joblib.dump(optimal_kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    kmeans_summary = {
        "tested_k": k_range,
        "inertias": [round(i, 2) for i in inertias],
        "silhouette_scores": silhouette_scores_list,
        "chosen_k": 3,
        "cluster_distribution": pd.Series(kmeans_labels).value_counts().to_dict()
    }

    # ---------------- 2. HIERARCHICAL CLUSTERING ----------------
    # Compute linkage on small subset for fast, readable dendrogram
    dendro_sample_size = min(300, len(X_scaled))
    dendro_data = X_scaled[:dendro_sample_size]
    Z = linkage(dendro_data, method='ward')

    agg_cluster = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg_cluster.fit_predict(X_scaled)

    # ---------------- 3. DENSITY-BASED CLUSTERING (DBSCAN) ----------------
    dbscan = DBSCAN(eps=3.5, min_samples=15)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_db = int(list(dbscan_labels).count(-1))

    dbscan_summary = {
        "eps": 3.5,
        "min_samples": 15,
        "clusters_found": n_clusters_db,
        "noise_points_flagged": n_noise_db,
        "noise_percentage": round((n_noise_db / len(X_scaled)) * 100, 2)
    }

    # ---------------- 4. DIMENSIONALITY REDUCTION (PCA, t-SNE, UMAP) ----------------
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    pca_total_var = round(float(np.sum(pca.explained_variance_ratio_)), 4)
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model.pkl"))

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=750)
    X_tsne = tsne.fit_transform(X_scaled)

    # UMAP
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_scaled)
        joblib.dump(reducer, os.path.join(MODEL_DIR, "umap_model.pkl"))
    else:
        # Fallback to 2D PCA representation if umap library not present
        X_umap = X_pca

    dim_reduction_summary = {
        "pca_variance_ratio": explained_var,
        "pca_total_variance_explained": pca_total_var,
        "techniques": ["PCA (Linear)", "t-SNE (Nonlinear Manifold)", "UMAP (Topological)"]
    }

    # ---------------- 5. ANOMALY DETECTION (ISOLATION FOREST) ----------------
    iso_forest = IsolationForest(contamination=0.035, random_state=42)
    raw_anomaly_preds = iso_forest.fit_predict(X_scaled) # 1: normal, -1: anomaly
    iso_anomalies = np.where(raw_anomaly_preds == -1, 1, 0)
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))

    anomaly_stats = {
        "detected_anomalies": int(np.sum(iso_anomalies)),
        "detection_rate": f"{round((np.sum(iso_anomalies)/len(iso_anomalies))*100, 2)}%"
    }

    if labels_anomaly is not None:
        gt_anomalies = labels_anomaly[:len(iso_anomalies)]
        cm = confusion_matrix(gt_anomalies, iso_anomalies).tolist()
        anomaly_stats["ground_truth_anomalies"] = int(np.sum(gt_anomalies))
        anomaly_stats["confusion_matrix"] = cm

    # ---------------- 6. VISUALIZATIONS ----------------
    plot_files = {}
    if save_plots:
        plt.style.use('dark_background')

        # Plot 1: K-Means Elbow Curve & Silhouette Score
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor('#0b0f19')
        for ax in axes:
            ax.set_facecolor('#0f172a')
            ax.grid(True, linestyle='--', alpha=0.3)

        axes[0].plot(k_range, inertias, marker='o', color='#38bdf8', lw=2.5, markersize=7)
        axes[0].set_title('K-Means Elbow Method (Inertia vs K)', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[0].set_xlabel('Number of Clusters (k)', color='#cbd5e1')
        axes[0].set_ylabel('Inertia (Sum of Squared Distances)', color='#cbd5e1')

        axes[1].plot(k_range, silhouette_scores_list, marker='s', color='#10b981', lw=2.5, markersize=7)
        axes[1].set_title('K-Means Silhouette Scores', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[1].set_xlabel('Number of Clusters (k)', color='#cbd5e1')
        axes[1].set_ylabel('Silhouette Coefficient', color='#cbd5e1')

        plt.tight_layout()
        p1 = os.path.join(STATIC_DIR, "unsupervised_kmeans_elbow.png")
        plt.savefig(p1, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["kmeans_elbow"] = "unsupervised_kmeans_elbow.png"

        # Plot 2: Dendrogram for Hierarchical Clustering
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        dendrogram(Z, p=25, truncate_mode='lastp', ax=ax, color_threshold=15,
                   above_threshold_color='#94a3b8')
        ax.set_title('Hierarchical Clustering: Ward Dendrogram', fontsize=12, fontweight='bold', color='#f8fafc')
        ax.set_xlabel('Cluster Leaf Index / Size', color='#cbd5e1')
        ax.set_ylabel('Euclidean Linkage Distance', color='#cbd5e1')
        plt.tight_layout()
        p2 = os.path.join(STATIC_DIR, "unsupervised_dendrogram.png")
        plt.savefig(p2, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["dendrogram"] = "unsupervised_dendrogram.png"

        # Plot 3: 3-Way Dimensionality Reduction Comparison (PCA vs t-SNE vs UMAP)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#0b0f19')

        colors = ['#6366f1', '#10b981', '#f59e0b']
        cluster_palette = [colors[c % len(colors)] for c in kmeans_labels]

        # PCA
        axes[0].set_facecolor('#0f172a')
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_palette, alpha=0.55, s=16)
        axes[0].set_title(f'PCA 2D (Expl: {pca_total_var*100:.1f}%)', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', color='#cbd5e1')
        axes[0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', color='#cbd5e1')
        axes[0].grid(True, linestyle='--', alpha=0.2)

        # t-SNE
        axes[1].set_facecolor('#0f172a')
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_palette, alpha=0.55, s=16)
        axes[1].set_title('t-SNE Manifold Projection', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[1].set_xlabel('t-SNE 1', color='#cbd5e1')
        axes[1].set_ylabel('t-SNE 2', color='#cbd5e1')
        axes[1].grid(True, linestyle='--', alpha=0.2)

        # UMAP
        axes[2].set_facecolor('#0f172a')
        axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_palette, alpha=0.55, s=16)
        axes[2].set_title('UMAP Topological Embedding', fontsize=11, fontweight='bold', color='#f8fafc')
        axes[2].set_xlabel('UMAP 1', color='#cbd5e1')
        axes[2].set_ylabel('UMAP 2', color='#cbd5e1')
        axes[2].grid(True, linestyle='--', alpha=0.2)

        plt.tight_layout()
        p3 = os.path.join(STATIC_DIR, "unsupervised_dim_reduction.png")
        plt.savefig(p3, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["dim_reduction"] = "unsupervised_dim_reduction.png"

        # Plot 4: Isolation Forest Anomaly Detection Scatter (PCA space)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0f172a')
        ax.grid(True, linestyle='--', alpha=0.2)

        norm_mask = iso_anomalies == 0
        anom_mask = iso_anomalies == 1

        ax.scatter(X_pca[norm_mask, 0], X_pca[norm_mask, 1], c='#3b82f6', alpha=0.4, s=18, label='Normal Profiles')
        ax.scatter(X_pca[anom_mask, 0], X_pca[anom_mask, 1], c='#ef4444', alpha=0.9, s=45, marker='x', label='Isolation Forest Anomalies')
        ax.set_title('Unsupervised Anomaly Detection (Isolation Forest in 2D PCA Space)', fontsize=12, fontweight='bold', color='#f8fafc')
        ax.set_xlabel('PC1', color='#cbd5e1')
        ax.set_ylabel('PC2', color='#cbd5e1')
        ax.legend(facecolor='#1e293b', edgecolor='none')
        plt.tight_layout()
        p4 = os.path.join(STATIC_DIR, "unsupervised_anomaly_detection.png")
        plt.savefig(p4, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        plot_files["anomalies"] = "unsupervised_anomaly_detection.png"

    return {
        "kmeans": kmeans_summary,
        "dbscan": dbscan_summary,
        "dimensionality_reduction": dim_reduction_summary,
        "anomaly_detection": anomaly_stats,
        "plots": plot_files
    }


if __name__ == "__main__":
    print("Executing Unsupervised Learning suite...")
    res = run_unsupervised_suite(sample_size=3000, save_plots=True)
    print("\n--- K-Means Summary ---")
    print(res["kmeans"])
    print("\n--- DBSCAN Summary ---")
    print(res["dbscan"])
    print("\n--- Anomaly Detection Summary ---")
    print(res["anomaly_detection"])

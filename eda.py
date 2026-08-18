import numpy as np
import pandas as pd
from load_data import load_data

def get_eda_summary_data() -> dict:
    df = load_data()
    eda_data = {}
    
    for col in df.columns:
        if col == "StudentID":
            continue
        if df[col].dtype in ["int64", "float64"] and df[col].nunique() > 10:
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                counts, bin_edges = np.histogram(clean_data, bins=15)
                eda_data[col] = {
                    "type": "numerical",
                    "counts": counts.tolist(),
                    "bin_edges": [round(float(x), 2) for x in bin_edges.tolist()],
                    "mean": round(float(clean_data.mean()), 2),
                    "median": round(float(clean_data.median()), 2),
                    "std": round(float(clean_data.std()), 2),
                    "min": round(float(clean_data.min()), 2),
                    "max": round(float(clean_data.max()), 2)
                }
        else:
            counts = df[col].value_counts(dropna=True)
            eda_data[col] = {
                "type": "categorical",
                "labels": [str(x) for x in counts.index.tolist()],
                "counts": counts.values.tolist()
            }
            
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if "StudentID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["StudentID"])
    corr = numeric_df.corr().round(3).fillna(0)
    
    correlation_data = {
        "columns": list(corr.columns),
        "values": corr.values.tolist()
    }
    
    numeric_sample = numeric_df.dropna().sample(n=min(1000, len(numeric_df)), random_state=42).round(4).to_dict(orient="records")
    
    return {
        "eda_data": eda_data,
        "correlation": correlation_data,
        "numeric_sample": numeric_sample
    }


if __name__ == "__main__":
    res = get_eda_summary_data()
    print("Columns in EDA:", list(res["eda_data"].keys()))
    print("Correlation matrix columns:", res["correlation"]["columns"])

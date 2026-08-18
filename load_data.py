import os
import pandas as pd

DATA_PATH = r"D:\SEM4\ML\Programs\placement_predict_50k Dataset.csv"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

def get_column_role(col: str) -> str:
    if col == "StudentID":
        return "Identifier"
    elif col in ["PlacementStatus", "Salary Package"]:
        return "Target (Label)"
    else:
        return "Feature"

def get_data_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },
        "duplicate_counts": {
            col: int(df[col].duplicated().sum())
            for col in df.columns
        },
        "roles": {
            col: get_column_role(col)
            for col in df.columns
        },
        "preview": df.head(10).to_dict(orient="records"),
    }
    return summary


def get_duplicate_count(path: str = DATA_PATH) -> int:
    df = load_data(path)
    duplicate_count = df.duplicated().sum()
    return int(duplicate_count)

if __name__ == "__main__":
    data = load_data()
    print(get_data_summary(), "\n", get_duplicate_count())
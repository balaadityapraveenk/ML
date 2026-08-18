import os
import pandas as pd
from load_data import DATA_PATH

PREPROCESSED_PATH = r"D:\SEM4\ML\Placement_Prediction\Data\placement_predict_preprocessed.csv"

def run_preprocessing(input_path: str = DATA_PATH, output_path: str = PREPROCESSED_PATH) -> dict:
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    df = df.drop(columns=["StudentID"])
    
    imputed_info = {}
    impute_cols = ["Workshops", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore"]
    for col in impute_cols:
        if col in df.columns:
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
            imputed_info[col] = median_val
            
    categorical_cols = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier"]
    encoded_info = {}
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            encoded_info[col] = list(df[col].cat.categories)
            df[col] = df[col].cat.codes
            
    numerical_cols = [
        "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4", 
        "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8", 
        "CGPA", "AttendancePercent", "Internships", "Projects", 
        "Workshops", "Certifications", "Publications", 
        "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", 
        "MockInterviewScore"
    ]
    scaled_info = []
    for col in numerical_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[col] = (df[col] - mean_val) / (std_val if std_val != 0 else 1.0)
            scaled_info.append(col)
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    summary = {
        "initial_shape": initial_shape,
        "final_shape": df.shape,
        "dropped_columns": ["StudentID"],
        "imputed_columns": imputed_info,
        "encoded_columns": encoded_info,
        "scaled_columns": scaled_info,
        "preview": df.head(10).round(4).to_dict(orient="records")
    }
    return summary

if __name__ == "__main__":
    res = run_preprocessing()
    print("Done. Shape:", res["final_shape"])

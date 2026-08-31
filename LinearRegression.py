import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Libraries imported successfully")

df = pd.read_csv('../Programs/placement_predict_preprocessed.csv')

print("First 5 rows")
display(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("CGPA Column exists:", "CGPA" in df.columns)

if "CGPA" not in df.columns or "Salary Package" not in df.columns:
    raise ValueError("CGPA or Salary Package column is missing")

X = df[["CGPA"]]
y = df["Salary Package"]

print("Input features:", X.columns.tolist())
print("Target:", y.name)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["CGPA"],
    df["Salary Package"],
    alpha=0.4
)

plt.title("CGPA vs Salary Package")
plt.xlabel("CGPA")
plt.ylabel("Salary Package")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training sample:", len(X_train))
print("Test sample:", len(X_test))

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ]
)

print("Linear Regression Model created successfully")

model.fit(X_train, y_train)

print("Model trained successfully")

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nActual vs Predicted:")
print(comparison)

mae = mean_absolute_error(y_test, y_pred)
print("\nMAE:", round(mae, 4))

mse = mean_squared_error(y_test, y_pred)
print("MSE:", round(mse, 4))

rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("RMSE:", round(rmse, 4))

r2 = r2_score(y_test, y_pred)
print("R2:", round(r2, 4))
from datasets import load_dataset
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Loading dataset from HuggingFace...")
ds = load_dataset("c01dsnap/CIC-IDS2017")
df = ds["train"].to_pandas()


df.columns = df.columns.str.strip()

print("Columns after strip:", df.columns)

df = df.replace([np.inf, -np.inf], np.nan).dropna()


X = df.drop(columns=["Label"])
y = df["Label"]

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump((X_scaled, y, X.columns), "preprocessed_data.pkl")

print("Preprocessing completed successfully ✔")
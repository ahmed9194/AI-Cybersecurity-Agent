import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Cybersecurity Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔐 AI Cybersecurity Agent")
st.caption("Intelligent Intrusion Detection & Attack Analysis System")

# ================= LOAD MODELS =================
anomaly_model = joblib.load("anomaly_model.pkl")
clf = joblib.load("classifier.pkl")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Choose Data Source",
    ["HuggingFace CIC-IDS2017", "Upload Your Own CSV"]
)

sample_size = st.sidebar.slider("Sample Size", 1000, 10000, 5000)
show_only_anomalies = st.sidebar.checkbox("Show Only Anomalies")
risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.6)

# ================= LOAD DATA =================
if data_source == "HuggingFace CIC-IDS2017":
    from datasets import load_dataset
    ds = load_dataset("c01dsnap/CIC-IDS2017")
    df = ds["train"].to_pandas().sample(sample_size, random_state=42)
else:
    uploaded = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
    if uploaded is None:
        st.warning("Please upload a CSV file to start analysis.")
        st.stop()
    df = pd.read_csv(uploaded)

# ================= DATA VALIDATION =================
st.subheader("🔎 Data Validation")

st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

df = df.replace([np.inf, -np.inf], np.nan)
missing = df.isnull().sum().sum()

if missing > 0:
    st.info(f"Missing values detected: {missing} (will be removed)")
    df = df.dropna()

if df.shape[0] < 100:
    st.warning("Dataset is small, results may be unreliable.")

# ================= PREPROCESS =================
for col in df.select_dtypes(include="object").columns:
    df[col] = pd.factorize(df[col])[0]

# Drop labels if exist
for col in ["Label", " Label", "Class", " Class"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# ================= FEATURE ALIGNMENT =================
expected_features = anomaly_model.n_features_in_
X = df.copy()

if X.shape[1] > expected_features:
    X = X.iloc[:, :expected_features]

if X.shape[1] < expected_features:
    st.error(
        f"Model expects {expected_features} features, "
        f"but input has {X.shape[1]}"
    )
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= PREDICTION =================
anomalies = anomaly_model.predict(X_scaled)
scores = anomaly_model.decision_function(X_scaled)
attacks = clf.predict(X_scaled)

df["Anomaly"] = anomalies
df["Anomaly Score"] = scores
df["Attack Type"] = attacks

# ================= RISK LEVEL =================
df["Risk Level"] = np.where(
    df["Anomaly Score"] < -risk_threshold,
    "HIGH",
    np.where(df["Anomaly Score"] < 0, "MEDIUM", "LOW")
)

# ================= FILTER =================
if show_only_anomalies:
    df = df[df["Anomaly"] == -1]

# ================= KPI METRICS =================
st.subheader("📊 Security Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Flows", len(df))
c2.metric("Anomalies", (df["Anomaly"] == -1).sum())
c3.metric("High Risk", (df["Risk Level"] == "HIGH").sum())
c4.metric("Attack Types", df["Attack Type"].nunique())

# ================= RESULTS TABLE =================
st.subheader("🧾 Detection Results")
st.dataframe(df.head(100), use_container_width=True)

# ================= ATTACK DISTRIBUTION =================
st.subheader("📌 Attack Distribution")
st.bar_chart(df["Attack Type"].value_counts())

# ================= ANOMALY SCORES =================
st.subheader("📈 Anomaly Score Timeline")
st.line_chart(df["Anomaly Score"])

# ================= TOP ANOMALIES =================
st.subheader("🚨 Top Suspicious Network Flows")
top_anomalies = df.sort_values("Anomaly Score").head(10)
st.dataframe(top_anomalies, use_container_width=True)

# ================= AGENT DECISION =================
st.subheader("🤖 Agent Decision & Response")

if (df["Risk Level"] == "HIGH").any():
    st.error("⚠️ Critical Threats Detected")
    st.markdown("""
    **Recommended Actions:**
    - Block suspicious IP addresses
    - Isolate affected hosts
    - Trigger SIEM alerts
    - Start forensic analysis
    """)
else:
    st.success("✅ Network Status: Stable")

# ================= DOWNLOAD =================
st.subheader("⬇️ Export Report")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Detection Report",
    csv,
    "cybersecurity_report.csv",
    "text/csv"
)
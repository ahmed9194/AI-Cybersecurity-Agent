import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_scaled, y, feature_names = joblib.load("preprocessed_data.pkl")

print("Training Anomaly Detection Model...")
anomaly_model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)
anomaly_model.fit(X_scaled)

print("Training Attack Classification Model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(anomaly_model, "anomaly_model.pkl")
joblib.dump(clf, "classifier.pkl")

print("Models trained and saved ✔")

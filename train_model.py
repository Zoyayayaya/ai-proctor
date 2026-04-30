import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# 🔍 Basic sanity check
print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["label"].value_counts())

# 🧠 Explicit feature selection (IMPORTANT)
features = [
    "eye_ratio",
    "head_tilt",
    "body_tilt",
    "looking_up",
    "phone_detected",
    "hand_on_table",
    "duration"
]

X = df[features]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=150, max_depth=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance (VERY useful)
importances = model.feature_importances_
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

# Save model
joblib.dump(model, "cheating_model.pkl")

print("\n✅ Model saved as cheating_model.pkl")
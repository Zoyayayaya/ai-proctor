import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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
y_prob = model.predict_proba(X_test)[:, 1]  # probability of cheating

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve_real.png")
plt.show()

print(f"\nROC AUC: {roc_auc:.4f}")

# Feature importance (VERY useful)
importances = model.feature_importances_
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

# Plot and save confusion matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.savefig("confusion_matrix_real.png")
plt.show()

# Save model
joblib.dump(model, "cheating_model.pkl")

print("\n✅ Model saved as cheating_model.pkl")
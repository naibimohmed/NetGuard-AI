
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# Create folders for models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Training parameters
BATCH_SIZE = 64
RANDOM_STATE = 42


# Load the preprocessed data
print("Loading data...")

X_train = np.load("dataset_pretraitment/X_train.npy")
X_test = np.load("dataset_pretraitment/X_test.npy")

y_train = np.load("dataset_pretraitment/y_train.npy").ravel()
y_test = np.load("dataset_pretraitment/y_test.npy").ravel()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Training and testing features do not match.")

if not set(np.unique(y_train)).issubset({0, 1}):
    raise ValueError("Labels must be binary (0 and 1).")

if not set(np.unique(y_test)).issubset({0, 1}):
    raise ValueError("Labels must be binary (0 and 1).")


# Load the Autoencoder and Encoder trained previously
print("\nLoading saved Autoencoder...")

autoencoder = load_model("models/autoencoder.keras")
encoder = load_model("models/encoder.keras")

print("Models loaded successfully.")


# Extract features from the Autoencoder
def extract_features(data):

    encoded_data = encoder.predict(
        data,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    reconstructed_data = autoencoder.predict(
        data,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    reconstruction_error = np.mean(
        np.square(data - reconstructed_data),
        axis=1
    ).reshape(-1, 1)

    features = np.concatenate(
        [encoded_data, reconstruction_error],
        axis=1
    )

    return features


print("\nExtracting features...")

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

print("New training data:", X_train_features.shape)
print("New testing data:", X_test_features.shape)


# Calculate the class weight for imbalanced data
n_normal = np.sum(y_train == 0)
n_attack = np.sum(y_train == 1)

if n_normal == 0 or n_attack == 0:
    raise ValueError("Training data must contain both classes.")

scale_pos_weight = n_normal / n_attack


# Train the XGBoost classifier
print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

start_time = time.time()

xgb_model.fit(
    X_train_features,
    y_train,
    verbose=False
)

xgb_time = time.time() - start_time

print(f"XGBoost training time: {xgb_time:.2f} seconds")


# Evaluate the model
print("\nEvaluating the model...")

y_pred = xgb_model.predict(X_test_features)
y_proba = xgb_model.predict_proba(X_test_features)[:, 1]

metrics = {
    "model": "Autoencoder + XGBoost",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "classifier_training_time": float(xgb_time)
}

print("\nAutoencoder + XGBoost Results")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))


# Save the trained XGBoost model and metrics
print("\nSaving model...")

xgb_model.save_model("models/xgboost_autoencoder.json")

with open("results/ae_xgb_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")

plt.title("Autoencoder + XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Normal", "Attack"])
plt.yticks([0, 1], ["Normal", "Attack"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    "results/confusion_matrix_AE_XGBoost.png",
    dpi=300
)
plt.close()


print("\nTraining completed successfully!")
print("XGBoost model saved in the models folder.")
print("Evaluation results saved in the results folder.")

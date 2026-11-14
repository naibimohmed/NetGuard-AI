
import os
import json
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier
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
EPOCHS = 30
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


# Use only normal traffic to train the Autoencoder
# In this dataset, 0 represents Normal and 1 represents Attack
X_normal = X_train[y_train == 0]

if len(X_normal) < 2:
    raise ValueError("Not enough normal samples for training.")


# Build the Autoencoder
print("\nBuilding Autoencoder...")

input_dim = X_train.shape[1]

inputs = Input(shape=(input_dim,))

# Encoder
x = Dense(64, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
latent = Dense(16, activation="relu", name="latent_space")(x)

# Decoder
x = Dense(32, activation="relu")(latent)
x = Dense(64, activation="relu")(x)
outputs = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs, outputs)

# Separate model for feature extraction
encoder = Model(inputs, latent)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


# Train the Autoencoder using normal traffic
print("\nTraining Autoencoder...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

start_time = time.time()

history = autoencoder.fit(
    X_normal,
    X_normal,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1
)

ae_time = time.time() - start_time

print(f"Autoencoder training time: {ae_time:.2f} seconds")


# Extract latent features and calculate reconstruction error
print("\nExtracting features...")

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


X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

print("New training data:", X_train_features.shape)
print("New testing data:", X_test_features.shape)


# Train the Random Forest classifier
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start_time = time.time()

rf_model.fit(X_train_features, y_train)

rf_time = time.time() - start_time

print(f"Random Forest training time: {rf_time:.2f} seconds")


# Evaluate the model on the test data
print("\nEvaluating the model...")

y_pred = rf_model.predict(X_test_features)
y_proba = rf_model.predict_proba(X_test_features)[:, 1]

metrics = {
    "model": "Autoencoder + Random Forest",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "autoencoder_training_time": float(ae_time),
    "classifier_training_time": float(rf_time),
    "total_training_time": float(ae_time + rf_time)
}

print("\nAutoencoder + Random Forest Results")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))


# Save the trained models and evaluation results
print("\nSaving models...")

autoencoder.save("models/autoencoder.keras")
encoder.save("models/encoder.keras")

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

with open("results/ae_rf_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")

plt.title("Autoencoder + Random Forest")
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
plt.savefig("results/confusion_matrix_AE_RF.png", dpi=300)
plt.close()


# Plot the Autoencoder training history
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

plt.tight_layout()
plt.savefig("results/training_curve_AE_RF.png", dpi=300)
plt.close()


print("\nTraining completed successfully!")
print("Models saved in the models folder.")
print("Results saved in the results folder.")

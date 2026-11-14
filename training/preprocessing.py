
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# Create folders if they don't exist
os.makedirs("dataset_pretraitment", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Load the training and testing datasets
print("Loading datasets...")

train_df = pd.read_csv("dataset/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("dataset/UNSW_NB15_testing-set.csv")

print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)


# Separate labels from input features
print("\nPreparing features and labels...")

y_train = train_df["label"].values
y_test = test_df["label"].values

columns_to_remove = ["label", "id", "attack_cat"]

X_train = train_df.drop(columns=columns_to_remove, errors="ignore")
X_test = test_df.drop(columns=columns_to_remove, errors="ignore")

# Keep the same feature order in both datasets
X_test = X_test[X_train.columns]


# Encode categorical columns
print("\nEncoding categorical features...")

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

if categorical_columns:
    X_train[categorical_columns] = encoder.fit_transform(
        X_train[categorical_columns].astype(str)
    )

    X_test[categorical_columns] = encoder.transform(
        X_test[categorical_columns].astype(str)
    )


# Convert all features to numeric values
print("Cleaning data...")

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill missing values using training data medians
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


# Scale the features
print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)


# Save the processed datasets
print("\nSaving processed data...")

np.save("dataset_pretraitment/X_train.npy", X_train_scaled)
np.save("dataset_pretraitment/X_test.npy", X_test_scaled)

np.save("dataset_pretraitment/y_train.npy", y_train)
np.save("dataset_pretraitment/y_test.npy", y_test)


# Save preprocessing objects for future predictions
preprocessing = {
    "encoder": encoder,
    "scaler": scaler,
    "categorical_columns": categorical_columns,
    "feature_columns": X_train.columns.tolist(),
    "medians": medians.to_dict()
}

joblib.dump(preprocessing, "models/preprocessing.pkl")


# Display final information
print("\nPreprocessing completed successfully!")

print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nFiles saved:")
print("- X_train.npy")
print("- X_test.npy")
print("- y_train.npy")
print("- y_test.npy")
print("- preprocessing.pkl")

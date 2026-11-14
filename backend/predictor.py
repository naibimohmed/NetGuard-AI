
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier


# Load saved preprocessing
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

preprocessing = joblib.load(MODELS_DIR / "preprocessing.pkl")

encoder = preprocessing["encoder"]
scaler = preprocessing["scaler"]
categorical_columns = preprocessing["categorical_columns"]
feature_columns = preprocessing["feature_columns"]
medians = preprocessing["medians"]


# Load trained models
autoencoder = load_model(MODELS_DIR / "autoencoder.keras")
latent_encoder = load_model(MODELS_DIR / "encoder.keras")

rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")

xgb_model = XGBClassifier()
xgb_model.load_model(str(MODELS_DIR / "xgboost_autoencoder.json"))


def prepare_data(records):

    data = pd.DataFrame(records)

    if data.empty:
        raise ValueError("No data provided.")

    # Remove columns that are not used for prediction
    data = data.drop(
        columns=["id", "attack_cat", "label"],
        errors="ignore"
    )

    missing = [col for col in feature_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data[feature_columns].copy()

    # Convert categorical values using the saved encoder
    if categorical_columns:
        data[categorical_columns] = encoder.transform(
            data[categorical_columns].astype(str)
        )

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(medians)

    # Apply the same scaling used during training
    data = scaler.transform(data)

    return data.astype(np.float32)


def extract_features(data):

    # Get the compressed representation
    encoded_data = latent_encoder.predict(data, verbose=0)

    # Reconstruct the original input
    reconstructed_data = autoencoder.predict(data, verbose=0)

    # Calculate the reconstruction error
    error = np.mean(
        np.square(data - reconstructed_data),
        axis=1
    ).reshape(-1, 1)

    return np.concatenate([encoded_data, error], axis=1)


def get_model(model_name):

    if model_name == "rf":
        return rf_model

    if model_name == "xgboost":
        return xgb_model

    raise ValueError("Choose either 'rf' or 'xgboost'.")


def predict_attack(features, model_name="rf"):

    data = prepare_data([features])
    features = extract_features(data)

    model = get_model(model_name)

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "model": model_name,
        "prediction": prediction,
        "label": "Attack" if prediction == 1 else "Normal",
        "attack_probability": probability
    }


def predict_batch(records, model_name="rf"):

    if not records:
        raise ValueError("The uploaded CSV file is empty.")

    data = prepare_data(records)
    features = extract_features(data)

    model = get_model(model_name)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    results = []

    for i, (prediction, probability) in enumerate(
        zip(predictions, probabilities), start=1
    ):
        results.append({
            "row": i,
            "prediction": int(prediction),
            "label": "Attack" if prediction == 1 else "Normal",
            "attack_probability": float(probability)
        })

    return results

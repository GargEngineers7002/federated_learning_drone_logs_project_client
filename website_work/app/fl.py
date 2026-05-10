import os
import pandas as pd
import io
import numpy as np
import json
import asyncio
from website_work.app.ml_models import (
    preprocess_data,
    run_predictions,
    _load_drone_resources,
)
from keras.models import load_model, Model
from typing import cast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEQ_LENGTH = 50
MODEL_FILENAME = "best_lstm_model.keras"


async def process_job(job_id, uav_model, csv_data):
    # 1. Prediction
    print(f"Running trajectory prediction... {job_id}")

    def _sync_process():
        df = pd.read_csv(io.StringIO(csv_data))
        preprocessed = preprocess_data(df.copy(), uav_model)
        df_clean = (
            df.replace([float("inf"), float("-inf")], 0).ffill().bfill().fillna(0)
        )
        return run_predictions(preprocessed, df_clean, uav_model)

    results = await asyncio.to_thread(_sync_process)
    return results


# =========================================================
# 2. PER-DRONE CONFIGURATION
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# app/ is current_dir. website_work/ is parent. root is grandparent.
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODELS_DIR = os.path.join(BASE_DIR, "website_work", "models")
# TRAINING_DATA_DIR = os.path.join(BASE_DIR, "20Jan") # If needed later

# Each drone maps to:
#   folder:  subfolder name in models/
#   targets: the 3 target column names for that drone
DRONE_CONFIG = {
    "DJI_Matrice_210": {
        "folder": "matrice_210",
        "targets": ["GPS:Long", "GPS:Lat", "GPS:heightMSL"],
    },
    "DJI_Matrice_600": {
        "folder": "matrice_600",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Mavic_2_Zoom": {
        "folder": "mavic_2_zoom",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Mavic_Pro": {
        "folder": "mavic_pro",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Phantom_4": {
        "folder": "Phantom_4",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Phantom_4_Pro_V2": {
        "folder": "Phantom_4_Pro_V2",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
}


async def get_processed_data(uav_model, flight_log):
    # Read CSV
    contents = await flight_log.read()

    def _sync_load_and_preprocess():
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        preprocessed_data = preprocess_data(df.copy(), uav_model)
        resources = _load_drone_resources(uav_model)
        return df, preprocessed_data, resources

    df, preprocessed_data, resources = await asyncio.to_thread(
        _sync_load_and_preprocess
    )

    input_scaler = resources["input_scaler"]
    target_scaler = resources["target_scaler"]
    target_cols = resources["target_cols"]

    # Prepare Input Features
    X_features = preprocessed_data.drop(columns=target_cols, errors="ignore").fillna(0)

    # Scale Inputs
    if input_scaler:
        X_scaled = input_scaler.transform(X_features)
    else:
        X_scaled = X_features.values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # Scale Targets
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0

    ground_truth_raw = df[target_cols].fillna(0).values
    if target_scaler:
        ground_truth_scaled = target_scaler.transform(ground_truth_raw)
    else:
        ground_truth_scaled = ground_truth_raw
    ground_truth_scaled = np.nan_to_num(ground_truth_scaled, nan=0.0)

    if len(X_scaled) < SEQ_LENGTH + 1:
        raise ValueError(
            f"Insufficient data for training. Need at least {SEQ_LENGTH + 1} rows."
        )

    # Create Sliding Windows
    X_sequences = []
    y_targets = []

    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_sequences.append(X_scaled[i : i + SEQ_LENGTH])
        y_targets.append(ground_truth_scaled[i + SEQ_LENGTH])

    return {"x": np.array(X_sequences).tolist(), "y": np.array(y_targets).tolist()}


async def save_global_model_weights(uav_model_name, weights_list):
    config = DRONE_CONFIG.get(uav_model_name)
    if config is None:
        return
    folder = config["folder"]
    model_path = os.path.join(MODELS_DIR, folder, MODEL_FILENAME)

    def _sync_save():
        if os.path.exists(model_path):
            model = load_model(model_path)
            weights = [np.array(w) for w in weights_list]
            model.set_weights(weights)
            model.save(model_path)

    await asyncio.to_thread(_sync_save)


async def train_and_save_model(uav_model_name, csv_str):
    # Load and Train the model and save it to the disk so that when retrieved while inference, it can be used
    def _sync_train():
        df = pd.read_csv(io.StringIO(csv_str))
        preprocessed_data = preprocess_data(df.copy(), uav_model_name)
        resources = _load_drone_resources(uav_model_name)

        input_scaler = resources["input_scaler"]
        target_scaler = resources["target_scaler"]
        target_cols = resources["target_cols"]

        X_features = preprocessed_data.drop(
            columns=target_cols, errors="ignore"
        ).fillna(0)
        X_scaled = (
            input_scaler.transform(X_features) if input_scaler else X_features.values
        )
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        for col in target_cols:
            if col not in df.columns:
                df[col] = 0.0

        ground_truth_raw = df[target_cols].fillna(0).values
        ground_truth_scaled = (
            target_scaler.transform(ground_truth_raw)
            if target_scaler
            else ground_truth_raw
        )
        ground_truth_scaled = np.nan_to_num(np.asarray(ground_truth_scaled), nan=0.0)

        if len(X_scaled) < SEQ_LENGTH + 1:
            print("Insufficient data for training. Skipping.")
            return None

        X_sequences = []
        y_targets = []
        for i in range(len(X_scaled) - SEQ_LENGTH):
            X_sequences.append(X_scaled[i : i + SEQ_LENGTH])
            y_targets.append(ground_truth_scaled[i + SEQ_LENGTH])

        X_train = np.array(X_sequences)
        y_train = np.array(y_targets)

        config = DRONE_CONFIG.get(uav_model_name)
        if config is None:
            return None
        folder = config["folder"]
        model_path = os.path.join(MODELS_DIR, folder, MODEL_FILENAME)

        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist. Skipping training.")
            return None

        from typing import Any

        model: Any = load_model(model_path)

        if model is None:
            print(f"Failed to load the model from {model_path}")
            return None

        # --- Dynamic Feature Alignment ---
        expected_dim = model.input_shape[-1]
        current_dim = X_train.shape[-1]
        if current_dim > expected_dim:
            print(f"[INFO] Training: Slicing features from {current_dim} to {expected_dim}")
            X_train = X_train[:, :, -expected_dim:]
        elif current_dim < expected_dim:
            print(f"[INFO] Training: Padding features from {current_dim} to {expected_dim}")
            padding = np.zeros((X_train.shape[0], X_train.shape[1], expected_dim - current_dim))
            X_train = np.concatenate([X_train, padding], axis=-1)

        # Train
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

        # Save back to disk
        model.save(model_path)

        # Return new weights
        return [w.tolist() for w in model.get_weights()]

    new_weights = await asyncio.to_thread(_sync_train)
    return new_weights


async def send_weights_to_backend(uav_model_name, new_weights):
    import httpx
    from dotenv import load_dotenv
    import json

    load_dotenv()
    backend_url = os.getenv("BACKEND_SERVER_URL", "http://localhost:8000")

    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{backend_url}/api/federated_averaging",
                data={"uav_model": uav_model_name, "weights": json.dumps(new_weights)},
                timeout=60.0,
            )
            print(
                f"[CLIENT] Successfully sent updated weights for {uav_model_name} to backend."
            )
        except Exception as e:
            print(f"[CLIENT] Failed to send federated averaging update: {e}")

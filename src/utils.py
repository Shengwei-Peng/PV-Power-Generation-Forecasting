"""utils"""
import random
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int) -> None:
    """set_seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """calculate_metrics"""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R² Score": r2_score(y_true, y_pred),
    }

def post_process(predictions: np.ndarray) -> np.ndarray:
    """post_process"""
    if predictions.ndim > 1:
        predictions = predictions.ravel()
    predictions = np.maximum(predictions, 0)
    predictions = predictions.round(2)
    return predictions

def train_and_predict(
    model,
    model_name: str,
    dataset: dict[str, dict[str, np.ndarray]],
    upload_template: str | Path,
    output_dir: str | Path,
) -> None:
    """train_and_predict"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / f"{model_name}_model.pkl"
    predictions_file = output_dir / f"{model_name}_pred.csv"

    model.fit(dataset["train"]["X"], dataset["train"]["y"])
    predictions = model.predict(dataset["test"]["X"])

    upload_data = pd.read_csv(upload_template)
    upload_data["答案"] = post_process(predictions)
    upload_data.to_csv(predictions_file, index=False)

    joblib.dump(model, model_file)
    print(f"Predictions stored at: {predictions_file}")
    print(f"Model stored at: {model_file}")

def create_ensemble_submission(
    model_preds: list[str | Path],
    upload_template: str | Path,
    output_file: str | Path
) -> None:
    """create_ensemble_submission"""
    preds = [pd.read_csv(pred_file)["答案"] for pred_file in model_preds]
    ensemble_preds = sum(preds) / len(preds)

    upload_data = pd.read_csv(upload_template)
    upload_data["答案"] = post_process(ensemble_preds)
    upload_data.to_csv(output_file, index=False)
    print(f"Ensemble submission saved to: {output_file}")

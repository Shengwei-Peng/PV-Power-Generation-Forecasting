"""utils"""
import random
from typing import Dict
from tabulate import tabulate

import torch
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

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """calculate_metrics"""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R² Score": r2_score(y_true, y_pred),
    }

def evaluate(target_file: str, prediction_file: str) -> None:
    """evaluate"""
    target_data = pd.read_csv(target_file)
    prediction_data = pd.read_csv(prediction_file)
    merged_data = pd.merge(target_data, prediction_data, on="序號", suffixes=('_true', '_pred'))

    y_true = merged_data['答案_true'].values
    y_pred = merged_data['答案_pred'].values
    metrics = calculate_metrics(y_true, y_pred)

    metrics_table = tabulate([metrics.values()], headers=metrics.keys(), tablefmt="grid")
    print(metrics_table)

"""trainer"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

from .models import Model
from .dataset import get_dataset
from .utils import set_seed


class Trainer:
    """Trainer"""
    def __init__(
        self,
        train_file: Path,
        test_file: Path = None,
        look_back_steps: int = 12,
        scaler_type: str = "minmax",
        random_state: int = 42,
    ) -> None:
        set_seed(random_state)
        self.look_back_steps = look_back_steps
        self.dataset = get_dataset(
            train_file=train_file,
            test_file=test_file,
            look_back_steps=look_back_steps,
            scaler_type=scaler_type
        )
        self.models = {
            "regression":{
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(
                    n_jobs=-1, random_state=random_state
                ),
                "LightGBM": LGBMRegressor(
                    verbose=-1, n_jobs=-1, random_state=random_state
                ),
                "CatBoost": CatBoostRegressor(
                    verbose=0, thread_count=-1, random_state=random_state
                ),
                "NGBoost": NGBRegressor(
                    verbose=False, random_state=random_state
                ),
                "TabNet": TabNetRegressor(),
                "MLP": Model("MLP"),
            },
            "time_series":{
                "LSTM": Model("LSTM", epochs=1),
            },
        }
        self.best_models = {"regression": None, "time_series": None}

    def train(self) -> pd.DataFrame:
        """train"""
        return pd.DataFrame([
            self._find_best_model(model_type)
            for model_type in ["regression", "time_series"]
        ])

    def _find_best_model(
        self,
        model_type: str,
    ) -> Dict[str, Any]:
        best_info = {
            "model": None,
            "name": "",
            "metrics": {"MAE": float("inf")}
        }
        results = []

        with tqdm(self.models[model_type].items(), desc=f"Training {model_type} Models") as pbar:
            for name, model in pbar:
                pbar.set_description(f"Training {name} ({model_type})")

                model.fit(
                    self.dataset[model_type]["train"]["x"], self.dataset[model_type]["train"]["y"]
                )
                y_pred = model.predict(self.dataset[model_type]["train"]["x"])
                y_true = self.dataset[model_type]["train"]["y"]

                metrics = self._calculate_metrics(y_true, y_pred)
                results.append({"Model": name, **metrics})

                if metrics["MAE"] < best_info["metrics"]["MAE"]:
                    best_info.update({
                        "model": model,
                        "name": name,
                        "metrics": metrics,
                    })

        results_df = pd.DataFrame(results).sort_values(by="MAE", ascending=True)

        print(f"\nModel Performance ({model_type}): ")
        print(tabulate(results_df, headers="keys", tablefmt="rounded_grid", showindex=False))

        self.best_models[model_type] = best_info["model"]

        return {
            "Model Type": model_type,
            "Best Model": best_info["name"],
            "MAE": best_info["metrics"]["MAE"],
            "MSE": best_info["metrics"]["MSE"],
            "RMSE": best_info["metrics"]["RMSE"],
            "R² Score": best_info["metrics"]["R² Score"]
        }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R² Score": r2_score(y_true, y_pred),
        }

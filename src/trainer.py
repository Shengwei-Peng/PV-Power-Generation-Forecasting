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

from .models import MLPRegressor
from .dataset import get_dataset, show_data_shapes
from .utils import set_seed


class Trainer:
    """Trainer"""
    def __init__(
        self,
        data_folder: Path,
        look_back_steps: int = 12,
        n_valid_months: int = 2,
        random_state: int = 42,
        combine_data: bool = True,
    ) -> None:
        set_seed(random_state)
        self.dataset = get_dataset(data_folder, look_back_steps, n_valid_months, combine_data)
        self.models = {
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
            "MLP": MLPRegressor(),
        }

    def train(self) -> None:
        """train"""
        results = []
        for data in self.dataset:
            show_data_shapes(data)

            result = self._find_best_model(data["regression"])
            results.append({
                "File": data["file_name"],
                "Best Model": result["model_name"],
                "MAE": result["metrics"]["MAE"],
                "MSE": result["metrics"]["MSE"],
                "RMSE": result["metrics"]["RMSE"],
                "R² Score": result["metrics"]["R² Score"]
            })

        results_df = pd.DataFrame(results)
        average_metrics = results_df.mean(numeric_only=True).to_dict()

        average_row = pd.DataFrame([{
            "File": "Average",
            "Best Model": "-",
            "MAE": round(average_metrics["MAE"], 4),
            "MSE": round(average_metrics["MSE"], 1),
            "RMSE": round(average_metrics["RMSE"], 4),
            "R² Score": round(average_metrics["R² Score"], 4)
        }])
        results_df = pd.concat([results_df, average_row], ignore_index=True)
        results_df = results_df.sort_values(by="MAE", ascending=True)

        print("\nModel Performance Across Datasets:")
        print(tabulate(results_df, headers="keys", tablefmt="pretty", showindex=False))

    def _find_best_model(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        best_info = {
            "model": None,
            "name": "",
            "metrics": {"MAE": float("inf")}
        }

        results = []

        with tqdm(self.models.items(), desc="Training Models", unit="model") as pbar:
            for name, model in pbar:
                pbar.set_description(f"Training {name}")

                model.fit(data["train"]["x"], data["train"]["y"])
                y_pred = model.predict(data["valid"]["x"])
                y_true = data["valid"]["y"]

                metrics = {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "MSE": mean_squared_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "R² Score": r2_score(y_true, y_pred),
                }

                results.append({
                    "Model": name,
                    **metrics,
                })

                if metrics["MAE"] < best_info["metrics"]["MAE"]:
                    best_info.update({
                        "model": model,
                        "name": name,
                        "metrics": metrics,
                    })

        results_df = pd.DataFrame(results).sort_values(by="MAE", ascending=True)

        print("\nModel Performance:")
        print(tabulate(results_df, headers="keys", tablefmt="pretty", showindex=False))
        print(f"\nBest Model: {best_info['name']} with MAE: {best_info['metrics']['MAE']}")

        return {
            "model_name": best_info["name"],
            "model": best_info["model"],
            "metrics": best_info["metrics"]
        }

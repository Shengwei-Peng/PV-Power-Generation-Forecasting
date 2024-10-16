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
from .dataset import get_dataset, show_data_shapes
from .utils import set_seed


class Trainer:
    """Trainer"""
    def __init__(
        self,
        train_folder: Path,
        test_folder: Path,
        look_back_steps: int = 12,
        n_valid_months: int = 2,
        random_state: int = 42,
        combine: bool = True,
    ) -> None:
        set_seed(random_state)
        self.look_back_steps = look_back_steps
        self.dataset = get_dataset(
            train_folder=train_folder,
            test_folder=test_folder,
            look_back_steps=look_back_steps,
            n_valid_months=n_valid_months,
            combine=combine
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
                "LSTM": Model("LSTM"),
            },
        }
        self.best_models = {"regression": None, "time_series": None}

    def train(self) -> None:
        """train"""
        results = [
            self._find_best_model(data, model_type)
            for data in self.dataset
            for model_type in ["regression", "time_series"]
        ]
        self._summarize_results(pd.DataFrame(results))

    def _find_best_model(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        model_type: str,
        ) -> Dict[str, Any]:
        best_info = {
            "model": None,
            "name": "",
            "metrics": {"MAE": float("inf")}
        }
        results = []
        show_data_shapes(data)

        with tqdm(self.models[model_type].items(), desc=f"Training {model_type} Models") as pbar:
            for name, model in pbar:
                pbar.set_description(f"Training {name} ({model_type})")

                model.fit(data[model_type]["train"]["x"], data[model_type]["train"]["y"])
                y_pred = model.predict(data[model_type]["valid"]["x"])
                y_true = data[model_type]["valid"]["y"]

                metrics = self._calculate_metrics(y_true, y_pred)
                results.append({"Model": name, **metrics})


                if metrics["MAE"] < best_info["metrics"]["MAE"]:
                    best_info.update({
                        "model": model,
                        "name": name,
                        "metrics": metrics,
                    })

        results_df = pd.DataFrame(results).sort_values(by="MAE", ascending=True)

        print("\nModel Performance:")
        print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))
        print(f"\nBest Model: {best_info['name']} with MAE: {best_info['metrics']['MAE']}")

        self.best_models[model_type] = best_info["model"]

        return {
            "File": data["file_name"],
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

    def _summarize_results(self, results_df: pd.DataFrame) -> None:
        grouped = results_df.groupby("Model Type")

        for model_type, group in grouped:
            average_metrics = group.mean(numeric_only=True).to_dict()

            average_row = {
                "File": "Average",
                "Model Type": model_type,
                "Best Model": "-",
                "MAE": round(average_metrics["MAE"], 4),
                "MSE": round(average_metrics["MSE"], 1),
                "RMSE": round(average_metrics["RMSE"], 4),
                "R² Score": round(average_metrics["R² Score"], 4),
            }

            group = pd.concat([group, pd.DataFrame([average_row])], ignore_index=True)

            group = group.sort_values(by="MAE", ascending=True)

            print(f"\nModel Performance: {model_type}")
            print(tabulate(group, headers="keys", tablefmt="double_grid", showindex=False))

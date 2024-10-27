"""trainer"""
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .utils import set_seed
from .dataset import Dataset

class Trainer:
    """Trainer"""
    def __init__(
        self,
        dataset: Dataset,
        models: Dict,
        random_state: int = 42,
    ) -> None:
        set_seed(random_state)
        self.dataset = dataset
        self.models = models
        self.best_models = {}

    def train(self) -> pd.DataFrame:
        """train"""
        all_results = []
        for model_type in self.models:
            best_info = {"model": None, "name": "", "metrics": {"MAE": float("inf")}}
            results = []
            train_data = self.dataset[model_type]["train"]

            with tqdm(self.models[model_type].items()) as pbar:
                for name, model in pbar:
                    pbar.set_description(f"Training {name} ({model_type})")

                    model.fit(train_data["x"], train_data["y"])
                    y_pred = model.predict(train_data["x"])
                    y_true = train_data["y"]

                    metrics = self._calculate_metrics(y_true, y_pred)
                    results.append({"Model": name, **metrics})

                    if metrics["MAE"] < best_info["metrics"]["MAE"]:
                        best_info = {"model": model, "name": name, "metrics": metrics}

            results_df = pd.DataFrame(results).sort_values(by="MAE")
            print(f"\nModel Performance ({model_type}): ")
            print(tabulate(results_df, headers="keys", tablefmt="rounded_grid", showindex=False))

            self.best_models[model_type] = best_info["model"]
            all_results.append(
                {"Type": model_type, "Best Model": best_info["name"], **best_info["metrics"]}
            )

        return pd.DataFrame(all_results)

    def save(self, save_dir: Union[Path, str]) -> None:
        """save"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_type, model in self.best_models.items():
            model_name = f"{model_type}_best_model"
            save_path = save_dir / model_name

            if isinstance(model, BaseEstimator):
                joblib.dump(model, save_path.with_suffix(".joblib"))
                print(f"Saved sklearn model: {model_name}.joblib")

            else:
                model.save(save_path.with_suffix(".pt"))
                print(f"Saved PyTorch model: {model_name}.pt")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R² Score": r2_score(y_true, y_pred),
        }

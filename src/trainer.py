"""trainer"""
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.base import BaseEstimator

from .utils import set_seed, calculate_metrics


class Trainer:
    """Trainer"""
    def __init__(self, models: Dict, random_state: int = 42) -> None:
        set_seed(random_state)
        self.models = models
        self.best_models = {}

    def train(self, dataset) -> pd.DataFrame:
        """train"""
        all_results = []
        for model_type in self.models:
            best_info = {"model": None, "name": "", "metrics": {"MAE": float("inf")}}
            results = []
            data = dataset[model_type]

            for name, model in self.models[model_type].items():
                model.fit(data["x"], data["y"])
                y_pred = model.predict(data["x"])
                y_true = data["y"]

                metrics = calculate_metrics(y_true, y_pred)
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

    def predict(self, dataset: Dict[str, Dict[str, np.ndarray]], steps: int = 48) -> pd.DataFrame:
        """predict"""
        prediction_ids, predictions = [], []

        for i, sequence_id in enumerate(dataset["序號"]):
            series = dataset["x"][i:i+1].copy()
            initial_time = pd.to_datetime(str(sequence_id)[:12], format="%Y%m%d%H%M")

            for step in range(steps):
                pred = self.best_models["time_series"].predict(series)[0]
                series = np.roll(series, shift=-1, axis=1)
                series[0, -1] = pred
                predictions.append(pred)

                prediction_ids.append(
                    (initial_time + pd.Timedelta(minutes=10 * step)).strftime("%Y%m%d%H%M")
                    + str(sequence_id)[12:]
                )

        final_preds = [
            round(float(pred), 2)
            for pred in self.best_models["regression"].predict(np.array(predictions))
        ]
        return pd.DataFrame({"序號": prediction_ids, "答案": final_preds})

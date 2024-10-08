"""trainer"""
from pathlib import Path
from typing import Dict, Any, Tuple

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


class Trainer:
    """Trainer"""
    def __init__(
        self,
        data_folder: Path,
        combine_data: bool = True,
        target_column: str = "Power(mW)",
        test_start_time: str = "09:00:00",
        test_end_time: str = "16:59:00",
        window_size: int = 10,
        test_days: int = 200,
        random_state: int = 42,
    ) -> None:
        self.dataset: list = []
        self.data_folder = data_folder
        self.combine_data = combine_data
        self.target_column = target_column
        self.window_size = window_size
        self.test_start_time = pd.to_datetime(test_start_time).time()
        self.test_end_time = pd.to_datetime(test_end_time).time()
        self.test_days = pd.Timedelta(days=test_days)
        self.random_state = random_state
        self.models = {
            "Linear Regression": LinearRegression(),
            "XGBoost": XGBRegressor(
                n_jobs=-1, random_state=self.random_state
            ),
            "LightGBM": LGBMRegressor(
                verbose=-1, n_jobs=-1, random_state=self.random_state
            ),
            "CatBoost": CatBoostRegressor(
                verbose=0, thread_count=-1, random_state=self.random_state
            ),
            "NGBoost": NGBRegressor(
                verbose=False, random_state=self.random_state
            ),
            "TabNet": TabNetRegressor(),
        }

    def pre_process(self) -> None:
        """pre_process"""
        for i, csv_file in enumerate(self.data_folder.glob("*.csv")):
            data = self.load_data(csv_file)
            if self.combine_data:
                if i == 0:
                    combined_data = data
                else:
                    for split in ["train", "test"]:
                        combined_data[split]["X"] = np.concatenate(
                            [combined_data[split]["X"], data[split]["X"]], axis=0
                        )
                        combined_data[split]["y"] = np.concatenate(
                            [combined_data[split]["y"], data[split]["y"]], axis=0
                        )
            else:
                self.dataset.append({
                    "file_name": csv_file.name,
                    "train": data["train"],
                    "test": data["test"]
                })

        if self.combine_data:
            self.dataset.append({
                "file_name": "Combined Data",
                "train": combined_data["train"],
                "test": combined_data["test"]
            })

    def load_data(self, file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Load and preprocess data."""
        raw_data = pd.read_csv(file_path)
        raw_data["DateTime"] = pd.to_datetime(raw_data["DateTime"])

        max_date = raw_data["DateTime"].max()
        min_test_date = max_date - self.test_days
        is_in_test_days = raw_data["DateTime"] >= min_test_date
        is_in_time_range = raw_data["DateTime"].dt.time.between(
            self.test_start_time, self.test_end_time
        )
        test_data = raw_data[is_in_test_days & is_in_time_range].copy()
        train_data = raw_data[~(is_in_test_days & is_in_time_range)].copy()

        for df in [train_data, test_data]:
            df.loc[:, "month"] = df["DateTime"].dt.month
            df.loc[:, "day"] = df["DateTime"].dt.day
            df.loc[:, "hour"] = df["DateTime"].dt.hour
            df.loc[:, "minute"] = df["DateTime"].dt.minute
            df.loc[:, "second"] = df["DateTime"].dt.second

        train_x = train_data.drop(columns=[self.target_column, "DateTime"]).values
        train_y = train_data[self.target_column].values

        test_x = test_data.drop(columns=[self.target_column, "DateTime"]).values
        test_y = test_data[self.target_column].values

        train_x, train_y = self.sliding_window(train_x, train_y)
        test_x, test_y = self.sliding_window(test_x, test_y)

        return {
            "train": {
                "X": train_x,
                "y": train_y.reshape(-1, 1)
            },
            "test": {
                "X": test_x,
                "y": test_y.reshape(-1, 1)
            }
        }

    def sliding_window(
        self,
        features: np.ndarray,
        target: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """sliding_window"""

        num_samples = len(features) - self.window_size
        x = np.empty((num_samples, self.window_size * features.shape[1]))
        y = np.empty(num_samples)

        for i in range(num_samples):
            x[i] = features[i:i + self.window_size].flatten()
            y[i] = target[i + self.window_size - 1]

        return x, y

    def find_best_model(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """find_best_model"""

        best_info = {
            "model": None,
            "name": "",
            "metrics": {"MAE": float("inf")}
        }

        results = []

        with tqdm(self.models.items(), desc="Training Models", unit="model") as pbar:
            for name, model in pbar:
                pbar.set_description(f"Training {name}")

                model.fit(data["train"]["X"], data["train"]["y"])
                y_pred = model.predict(data["test"]["X"])
                y_true = data["test"]["y"]

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

    def train(self) -> None:
        """train"""
        results = []
        for data in self.dataset:
            data_dimensions = [
                ["Training data (X)", data['train']['X'].shape],
                ["Training target (y)", data['train']['y'].shape],
                ["Testing data (X)", data['test']['X'].shape],
                ["Testing target (y)", data['test']['y'].shape]
            ]
            print("\nData Dimensions:")
            print(tabulate(data_dimensions, headers=["Data", "Shape"], tablefmt="pretty"))

            result = self.find_best_model(data)
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

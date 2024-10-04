"""utils"""
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def parse_arguments() -> argparse.Namespace:
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="PV Power Generation Forecast")
    parser.add_argument(
        "--data_folder",
        type=Path,
        required=True,
        help="Path to the folder containing CSV files for data"
    )
    parser.add_argument(
        "--combine_data", 
        action="store_true",
        help="Combine all CSV files in the folder and train a single model"
    )
    return parser.parse_args()

def load_data(
    file_path: str,
    target_column: str = "Power(mW)",
    test_start_time: str = "09:00:00",
    test_end_time: str = "16:59:00",
    test_days: int = 200,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
    """load_data"""
    raw_data = pd.read_csv(file_path)

    raw_data["DateTime"] = pd.to_datetime(raw_data["DateTime"])
    test_start_time = pd.to_datetime(test_start_time).time()
    test_end_time = pd.to_datetime(test_end_time).time()

    max_date = raw_data["DateTime"].max()
    min_test_date = max_date - pd.Timedelta(days=test_days)
    is_in_test_days = raw_data["DateTime"] >= min_test_date

    is_in_time_range = raw_data["DateTime"].dt.time.between(test_start_time, test_end_time)
    test_data = raw_data[is_in_test_days & is_in_time_range].copy()
    train_data = raw_data[~(is_in_test_days & is_in_time_range)].copy()

    for df in [train_data, test_data]:
        df.loc[:, "hour"] = df["DateTime"].dt.hour
        df.loc[:, "minute"] = df["DateTime"].dt.minute
        df.loc[:, "second"] = df["DateTime"].dt.second

    data = {
        "train": {
            "X": train_data.drop(columns=[target_column, "DateTime"]),
            "y": train_data[target_column]
        },
        "test": {
            "X": test_data.drop(columns=[target_column, "DateTime"]),
            "y": test_data[target_column]
        }
    }
    return data

def find_best_model(
    data: Dict[str, Dict[str, pd.DataFrame]], random_state: int = 42
    ) -> Dict[str, Any]:
    """find_best_model"""
    data_dimensions = [
        ["Training data (X)", data['train']['X'].shape],
        ["Training target (y)", data['train']['y'].shape],
        ["Testing data (X)", data['test']['X'].shape],
        ["Testing target (y)", data['test']['y'].shape]
    ]

    print("\nData Dimensions:")
    print(tabulate(data_dimensions, headers=["Data", "Shape"], tablefmt="pretty"))

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=random_state),
        "Lasso Regression": Lasso(random_state=random_state),
        "Support Vector Regression (SVR)": SVR(),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(random_state=random_state),
        "LightGBM": LGBMRegressor(verbose=-1, random_state=random_state),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=random_state),
    }
    best_info = {
        "model": None,
        "name": "",
        "metrics": {"MAE": float("inf")}
    }

    results = []

    with tqdm(models.items(), desc="Training Models", unit="model") as pbar:
        for name, model in pbar:
            pbar.set_description(f"Training {name}")

            model.fit(**data["train"])
            y_pred = model.predict(data["test"]["X"])

            metrics = {
                "MAE": mean_absolute_error(data["test"]["y"], y_pred),
                "MSE": mean_squared_error(data["test"]["y"], y_pred),
                "RMSE": np.sqrt(mean_squared_error(data["test"]["y"], y_pred)),
                "R² Score": r2_score(data["test"]["y"], y_pred),
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

def train(data_folder: Path, combine_data: bool = False) -> None:
    """train"""
    results = []
    if combine_data:
        combined_data = {
            "train": {"X": pd.DataFrame(), "y": pd.Series(dtype=float)},
            "test": {"X": pd.DataFrame(), "y": pd.Series(dtype=float)}
        }

        for csv_file in data_folder.glob("*.csv"):
            data = load_data(csv_file)

            for split in ["train", "test"]:
                combined_data[split]["X"] = pd.concat(
                    [combined_data[split]["X"], data[split]["X"]], ignore_index=True
                )
                combined_data[split]["y"] = pd.concat(
                    [combined_data[split]["y"], data[split]["y"]], ignore_index=True
                ).reset_index(drop=True)

        result = find_best_model(combined_data)
        results.append({
            "File": "Combined Data",
            "Best Model": result["model_name"],
            "MAE": result["metrics"]["MAE"],
            "MSE": result["metrics"]["MSE"],
            "RMSE": result["metrics"]["RMSE"],
            "R² Score": result["metrics"]["R² Score"]
        })

    else:
        for csv_file in data_folder.glob("*.csv"):
            data = load_data(csv_file)
            result = find_best_model(data)

            results.append({
                "File": csv_file.name,
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
        "MAE": average_metrics["MAE"],
        "MSE": average_metrics["MSE"],
        "RMSE": average_metrics["RMSE"],
        "R² Score": average_metrics["R² Score"]
    }])
    results_df = pd.concat([results_df, average_row], ignore_index=True)
    results_df = results_df.sort_values(by="MAE", ascending=True)

    print("\nModel Performance Across Datasets:")
    print(tabulate(results_df, headers="keys", tablefmt="pretty", showindex=False))

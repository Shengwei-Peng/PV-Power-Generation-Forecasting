"""utils"""
from typing import Dict

from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def load_data(
    file_path: str,
    target_column: str = "Power(mW)",
    test_start_time: str = "09:00:00",
    test_end_time: str = "16:59:00",
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
    """load_data"""
    raw_data = pd.read_csv(file_path)

    raw_data["DateTime"] = pd.to_datetime(raw_data["DateTime"])
    test_start_time = pd.to_datetime(test_start_time).time()
    test_end_time = pd.to_datetime(test_end_time).time()

    is_in_time_range = raw_data["DateTime"].dt.time.between(test_start_time, test_end_time)
    test_data = raw_data[is_in_time_range].copy()
    train_data = raw_data[~is_in_time_range].copy()

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

def find_best_model(data: Dict[str, Dict[str, pd.DataFrame]]) -> RegressorMixin:
    """find_best_model"""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Support Vector Regression (SVR)": SVR(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }
    results = []
    best_model = None
    best_mae = float("inf")

    with tqdm(models.items(), desc="Training Models", unit="model") as pbar:
        for name, model in pbar:
            pbar.set_description(f"Training {name}")

            model.fit(**data["train"])

            y_pred = model.predict(data["test"]["X"])

            mae = mean_absolute_error(data["test"]["y"], y_pred)
            results.append({"Model": name, "MAE": mae})

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_model_name = name

    results_df = pd.DataFrame(results).sort_values(by="MAE", ascending=True)

    print("\nModel Performance:")
    print(tabulate(results_df, headers="keys", tablefmt="pretty", showindex=False))

    print(f"\nBest Model: {best_model_name} with MAE: {best_mae}")
    return best_model

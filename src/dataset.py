"""dataset"""
from pathlib import Path
from typing import Dict, Tuple, Union

from tabulate import tabulate
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def create_time_series_data(
    features: np.ndarray,
    look_back_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """create_time_series_data"""
    x = sliding_window_view(
        features, (look_back_steps, features.shape[1])
    )[:-1, 0, :, :]

    y = features[look_back_steps:, :]

    return x, y

def pre_process(
    x: np.ndarray,
    y: np.ndarray,
    look_back_steps: int,
    scaler: Union[MinMaxScaler, StandardScaler],
    fit_scaler: bool = False,
) -> Dict[str, np.ndarray]:
    """pre_process"""

    x = scaler.fit_transform(x) if fit_scaler else scaler.transform(x)
    x_ts, y_ts = create_time_series_data(x, look_back_steps)

    return {"x": x, "y": y, "x_ts": x_ts, "y_ts": y_ts}

def get_dataset(
    train_file: Union[Path, str],
    test_file: Union[Path, str, None] = None,
    look_back_steps: int = 12,
    scaler_type: str = "minmax",
    ) -> Dict[str, Union[str, Dict[str, Dict[str, np.ndarray]]]]:
    """get_dataset"""

    x_columns = [
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
    ]
    y_column = ["Power(mW)"]
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()

    train_data = pd.read_csv(check_file(train_file))
    train_data = pre_process(
        x=train_data[x_columns].to_numpy(),
        y=train_data[y_column].to_numpy(),
        scaler=scaler,
        look_back_steps=look_back_steps,
        fit_scaler=True
    )

    if test_file is not None :
        test_data = pd.read_csv(check_file(test_file))
        test_data = pre_process(
            x=test_data[x_columns].to_numpy(),
            y=test_data[y_column].to_numpy(),
            scaler=scaler,
            look_back_steps=look_back_steps,
        )

        return {
            "regression": {
                "train": {"x": train_data["x"], "y": train_data["y"]},
                "test": {"x": test_data["x"], "y": test_data["y"]},
            },
            "time_series": {
                "train": {"x": train_data["x_ts"], "y": train_data["y_ts"]},
                "test": {"x": test_data["x_ts"], "y": test_data["y_ts"]},
            }
        }

    return {
        "regression": {
            "train": {"x": train_data["x"], "y": train_data["y"]},
        },
        "time_series": {
            "train": {"x": train_data["x_ts"], "y": train_data["y_ts"]},
        }
    }

def check_file(file: Union[Path, str]) -> Path:
    """check_file"""
    file = Path(file)
    if not file.exists() or not file.is_file() or file.suffix != '.csv':
        raise FileNotFoundError(f"CSV file does not exist: {file}")
    return file

def show_data_shapes(dataset: Dict[str, Union[str, Dict[str, Dict[str, np.ndarray]]]]) -> None:
    """show_data_shapes"""
    headers = ["Data Type", "Regression Shape", "Time Series Shape"]
    data_shapes = []

    data_types = ["Train X", "Train Y", "Test X", "Test Y"]

    regression = dataset.get("regression", {})
    time_series = dataset.get("time_series", {})

    regression_shape = (
        regression.get("train", {}).get("x", "-").shape if regression.get("train") else "-",
        regression.get("train", {}).get("y", "-").shape if regression.get("train") else "-",
        regression.get("test", {}).get("x", "-").shape if regression.get("test") else "-",
        regression.get("test", {}).get("y", "-").shape if regression.get("test") else "-"
    )

    time_series_shape = (
        time_series.get("train", {}).get("x", "-").shape if time_series.get("train") else "-",
        time_series.get("train", {}).get("y", "-").shape if time_series.get("train") else "-",
        time_series.get("test", {}).get("x", "-").shape if time_series.get("test") else "-",
        time_series.get("test", {}).get("y", "-").shape if time_series.get("test") else "-"
    )

    for i, data_type in enumerate(data_types):
        data_shapes.append([data_type, regression_shape[i], time_series_shape[i]])

    print("\nData Shapes:")
    print(tabulate(data_shapes, headers=headers, tablefmt="rounded_grid"))

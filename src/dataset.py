"""dataset"""
from pathlib import Path
from typing import Dict, Tuple, Union, List

from tabulate import tabulate
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def extract_serial_data(
    raw_data: pd.DataFrame
    ) -> pd.DataFrame:
    """extract_serial_data"""
    pattern = r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"
    extracted_values = raw_data['Serial'].astype(str).str.extract(pattern).apply(pd.to_numeric)

    raw_data = raw_data.assign(
        year=extracted_values[0],
        month=extracted_values[1],
        day=extracted_values[2],
        hour=extracted_values[3],
        minute=extracted_values[4],
        location_code=extracted_values[5],
    )
    return raw_data

def preprocess_data(
    data: pd.DataFrame,
    x_columns: List[str],
    scaler_type: str = "minmax"
    ) -> pd.DataFrame:
    """preprocess_data"""
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    data[x_columns] = scaler.fit_transform(data[x_columns])
    return data

def split_train_valid_data(
    raw_data: pd.DataFrame,
    n_valid_months: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split_train_valid_data"""
    if n_valid_months > 0:
        unique_months = sorted(raw_data["month"].unique())
        n_valid_months = min(len(unique_months), n_valid_months)
        last_valid_months = unique_months[-n_valid_months:]

        is_valid_set = raw_data["month"].isin(last_valid_months)
        train_data = raw_data[~is_valid_set]
        valid_data = raw_data[is_valid_set]
    else:
        train_data = raw_data
        valid_data = train_data.copy()

    return train_data, valid_data

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

def create_dataset(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    x_columns: List[str],
    y_column: List[str],
    look_back_steps: int
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """create_dataset"""
    train_x = train_data[x_columns].values
    train_y = train_data[y_column].values
    valid_x = valid_data[x_columns].values
    valid_y = valid_data[y_column].values

    train_x_ts, train_y_ts = create_time_series_data(train_x, look_back_steps)
    valid_x_ts, valid_y_ts = create_time_series_data(valid_x, look_back_steps)

    return {
        "time_series": {
            "train": {"x": train_x_ts, "y": train_y_ts},
            "valid": {"x": valid_x_ts, "y": valid_y_ts},
        },
        "regression": {
            "train": {"x": train_x, "y": train_y},
            "valid": {"x": valid_x, "y": valid_y},
        },
    }

def load_data(
    file_path: Union[str, Path],
    look_back_steps: int = 12,
    n_valid_months: int = 2
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """load_data"""
    raw_data = pd.read_csv(file_path)
    raw_data = extract_serial_data(raw_data)

    x_columns = [
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
    ]
    y_column = ["Power(mW)"]
    raw_data = preprocess_data(raw_data, x_columns, "minmax")

    train_data, valid_data = split_train_valid_data(raw_data, n_valid_months)

    return create_dataset(train_data, valid_data, x_columns, y_column, look_back_steps)

def get_dataset(
    data_folder: Union[Path, str],
    look_back_steps: int = 12,
    n_valid_months: int = 2,
    combine_data: bool = True,
    ) -> List[Dict[str, Union[str, Dict[str, Dict[str, np.ndarray]]]]]:
    """get_dataset"""

    if not isinstance(data_folder, Path):
        data_folder = Path(data_folder)

    dataset = []
    combined_data = None

    for csv_file in data_folder.glob("*.csv"):
        data = load_data(csv_file, look_back_steps, n_valid_months)

        if combine_data:
            if combined_data is None:
                combined_data = data
            else:
                for data_type in ["time_series", "regression"]:
                    for split in ["train", "valid"]:
                        combined_data[data_type][split]["x"] = np.concatenate(
                            [combined_data[data_type][split]["x"],
                            data[data_type][split]["x"]], axis=0
                        )
                        combined_data[data_type][split]["y"] = np.concatenate(
                            [combined_data[data_type][split]["y"],
                            data[data_type][split]["y"]], axis=0
                        )
        else:
            dataset.append({
                "file_name": csv_file.name,
                "time_series": data["time_series"],
                "regression": data["regression"]
            })

    if combine_data:
        dataset.append({
            "file_name": "Combined Data",
            "time_series": combined_data["time_series"],
            "regression": combined_data["regression"]
        })

    return dataset

def show_data_shapes(data: dict) -> None:
    """show_data_shapes"""
    headers = ["Data Type", "Regression Shape", "Time Series Shape"]
    data_shapes = []

    data_types = ["Train X", "Train Y", "Valid X", "Valid Y"]

    regression = data.get("regression", {})
    time_series = data.get("time_series", {})

    regression_shape = (
        regression.get("train", {}).get("x", "-").shape if regression.get("train") else "-",
        regression.get("train", {}).get("y", "-").shape if regression.get("train") else "-",
        regression.get("valid", {}).get("x", "-").shape if regression.get("valid") else "-",
        regression.get("valid", {}).get("y", "-").shape if regression.get("valid") else "-"
    )

    time_series_shape = (
        time_series.get("train", {}).get("x", "-").shape if time_series.get("train") else "-",
        time_series.get("train", {}).get("y", "-").shape if time_series.get("train") else "-",
        time_series.get("valid", {}).get("x", "-").shape if time_series.get("valid") else "-",
        time_series.get("valid", {}).get("y", "-").shape if time_series.get("valid") else "-"
    )

    for i, data_type in enumerate(data_types):
        data_shapes.append([data_type, regression_shape[i], time_series_shape[i]])

    print("\nData Shapes:")
    print(tabulate(data_shapes, headers=headers, tablefmt="rounded_grid"))

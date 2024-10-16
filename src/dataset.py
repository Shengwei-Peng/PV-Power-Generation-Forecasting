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
    extracted_values = raw_data["Serial"].astype(str).str.extract(pattern).apply(pd.to_numeric)

    raw_data = raw_data.assign(
        year=extracted_values[0],
        month=extracted_values[1],
        day=extracted_values[2],
        hour=extracted_values[3],
        minute=extracted_values[4],
        location_code=extracted_values[5],
    )
    return raw_data

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

def pre_process(
    x: np.ndarray,
    y: np.ndarray,
    scaler: Union[MinMaxScaler, StandardScaler],
    look_back_steps: int,
    fit_scaler: bool = False,
    ) -> Dict[str, np.ndarray]:
    """pre_process"""

    if fit_scaler:
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)

    x_ts, y_ts = create_time_series_data(x, look_back_steps)

    return {"x": x, "y": y, "x_ts": x_ts, "y_ts": y_ts}

def combine_location_data(
    combined_data: Union[None, Dict[str, Dict[str, np.ndarray]]],
    new_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
    """combine_location_data"""

    if combined_data is None:
        return new_data

    for data_type in ["time_series", "regression"]:
        for split in ["train", "valid"]:
            combined_data[data_type][split]["x"] = np.concatenate(
                [combined_data[data_type][split]["x"], new_data[data_type][split]["x"]], axis=0
            )
            combined_data[data_type][split]["y"] = np.concatenate(
                [combined_data[data_type][split]["y"], new_data[data_type][split]["y"]], axis=0
            )
        if "test" in new_data[data_type]:
            combined_data[data_type]["test"]["x"] = np.concatenate(
                [combined_data[data_type]["test"]["x"], new_data[data_type]["test"]["x"]], axis=0
            )
            combined_data[data_type]["test"]["y"] = np.concatenate(
                [combined_data[data_type]["test"]["y"], new_data[data_type]["test"]["y"]], axis=0
            )

    return combined_data

def load_data(
    train_file_path: Union[str, Path],
    test_file_path: Union[str, Path, None] = None,
    look_back_steps: int = 12,
    n_valid_months: int = 2,
    scaler_type: str = "minmax",
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """load_data"""

    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()

    x_columns = [
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
    ]
    y_column = ["Power(mW)"]

    train_data = pd.read_csv(train_file_path)
    train_data = extract_serial_data(train_data)

    train_data, valid_data = split_train_valid_data(train_data, n_valid_months)

    train_data = pre_process(
        train_data[x_columns], train_data[y_column], scaler, look_back_steps, fit_scaler=True
    )
    valid_data = pre_process(
        valid_data[x_columns], valid_data[y_column], scaler, look_back_steps
    )

    datasets = {
        "regression": {
            "train": {"x": train_data["x"], "y": train_data["y"]},
            "valid": {"x": valid_data["x"], "y": valid_data["y"]}
        },
        "time_series": {
            "train": {"x": train_data["x_ts"], "y": train_data["y_ts"]},
            "valid": {"x": valid_data["x_ts"], "y": valid_data["y_ts"]}
        }
    }
    if test_file_path is not None:
        test_data = pd.read_csv(test_file_path)
        test_data = extract_serial_data(test_data)

        test_processed = pre_process(
            test_data[x_columns], test_data[y_column], scaler, look_back_steps
        )

        datasets["time_series"]["test"] = {"x": test_processed["x_ts"], "y": test_processed["y_ts"]}
        datasets["regression"]["test"] = {"x": test_processed["x"], "y": test_processed["y"]}

    return datasets

def get_dataset(
    train_folder: Union[Path, str],
    test_folder: Union[Path, str, None] = None,
    look_back_steps: int = 12,
    n_valid_months: int = 2,
    combine: bool = True,
    ) -> List[Dict[str, Union[str, Dict[str, Dict[str, np.ndarray]]]]]:
    """get_dataset"""

    dataset = []
    combined_data = None

    train_folder = Path(train_folder) if not isinstance(train_folder, Path) else train_folder

    if test_folder is not None :
        test_folder = Path(test_folder) if not isinstance(test_folder, Path) else test_folder
        files = zip(train_folder.glob("*.csv"), test_folder.glob("*.csv"))
    else:
        files = [(train_file, None) for train_file in train_folder.glob("*.csv")]

    for train_file, test_file in files:
        data = load_data(train_file, test_file, look_back_steps, n_valid_months)

        if combine:
            combined_data = combine_location_data(combined_data, data)

        else:
            dataset.append({
                "file_name": train_file.name,
                "time_series": data["time_series"],
                "regression": data["regression"]
            })

    if combined_data:
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

    data_types = ["Train X", "Train Y", "Valid X", "Valid Y", "Test X", "Test Y"]

    regression = data.get("regression", {})
    time_series = data.get("time_series", {})

    regression_shape = (
        regression.get("train", {}).get("x", "-").shape if regression.get("train") else "-",
        regression.get("train", {}).get("y", "-").shape if regression.get("train") else "-",
        regression.get("valid", {}).get("x", "-").shape if regression.get("valid") else "-",
        regression.get("valid", {}).get("y", "-").shape if regression.get("valid") else "-",
        regression.get("test", {}).get("x", "-").shape if regression.get("test") else "-",
        regression.get("test", {}).get("y", "-").shape if regression.get("test") else "-"
    )

    time_series_shape = (
        time_series.get("train", {}).get("x", "-").shape if time_series.get("train") else "-",
        time_series.get("train", {}).get("y", "-").shape if time_series.get("train") else "-",
        time_series.get("valid", {}).get("x", "-").shape if time_series.get("valid") else "-",
        time_series.get("valid", {}).get("y", "-").shape if time_series.get("valid") else "-",
        time_series.get("test", {}).get("x", "-").shape if time_series.get("test") else "-",
        time_series.get("test", {}).get("y", "-").shape if time_series.get("test") else "-"
    )

    for i, data_type in enumerate(data_types):
        data_shapes.append([data_type, regression_shape[i], time_series_shape[i]])

    print("\nData Shapes:")
    print(tabulate(data_shapes, headers=headers, tablefmt="rounded_grid"))

"""utils"""
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def parse_arguments() -> dict:
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
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n_test_months",
        type=int,
        default=2,
        help="Number of last months to use for the test set (default: 2)"
    )
    parser.add_argument(
        "--look_back_steps",
        type=int,
        default=12,
        help="Number of look-back steps for time series data (default: 12)"
    )
    return vars(parser.parse_args())

def set_seed(seed: int) -> None:
    """set_seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(
    file_path: str,
    look_back_steps: int = 12,
    n_valid_months: int = 2,
    ) -> Dict[str, Dict[str, np.ndarray]]:
    """load_data"""
    raw_data = pd.read_csv(file_path)

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

    x_columns = [
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
    ]
    y_column = [
        "Power(mW)"
    ]

    unique_months = sorted(raw_data["month"].unique())
    if n_valid_months > 0:
        n_valid_months = min(len(unique_months), n_valid_months)
        last_valid_months = unique_months[-n_valid_months:]

        is_valid_set = raw_data["month"].isin(last_valid_months)
        train_data = raw_data[~is_valid_set]
        valid_data  = raw_data[is_valid_set]
    else:
        train_data = raw_data
        valid_data = train_data.copy()

    train_x = train_data[x_columns].values
    train_y = train_data[y_column].values

    valid_x = valid_data[x_columns].values
    valid_y = valid_data[y_column].values

    train_x_ts, train_y_ts = create_time_series_data(train_x, look_back_steps)
    x_valid_ts, y_valid_ts = create_time_series_data(valid_x, look_back_steps)

    return {
        "time_series": {
            "train": {
                "x": train_x_ts,
                "y": train_y_ts
            },
            "valid": {
                "x": x_valid_ts,
                "y": y_valid_ts
            }
        },
        "regression": {
            "train": {
                "x": train_x,
                "y": train_y
            },
            "valid": {
                "x": valid_x,
                "y": valid_y
            }
        },
    }

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

def get_dataset(
    data_folder: Path | str,
    look_back_steps: int = 12,
    n_valid_months: int = 2,
    combine_data: bool = True,
    ) -> list:
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

"""dataset"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import validate_csv


class Dataset:
    """Dataset"""
    def __init__(
        self,
        train_file: Union[Path, str],
        test_file: Union[Path, str, None] = None,
        x_columns: List[str] = None,
        y_column: List[str] = None,
        look_back_steps: int = 12,
        scaler_type: str = "minmax",
    ) -> None:
        self.look_back_steps = look_back_steps
        self.scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        self.x_columns = x_columns or [
            "WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)"
        ]
        self.y_column = y_column or ["Power(mW)"]
        self.dataset = self._load_and_process(train_file, test_file)

    def _load_and_process(self, train_file: Union[Path, str], test_file: Union[Path, str, None]):
        """_load_and_process"""
        train_data = pd.read_csv(validate_csv(train_file))
        train_processed = self._pre_process(
            train_data[self.x_columns].values, train_data[self.y_column].values, fit_scaler=True
        )

        if test_file:
            test_data = pd.read_csv(validate_csv(test_file))
            test_processed = self._pre_process(
                test_data[self.x_columns].values, test_data[self.y_column].values
            )
            return {
                "regression": {
                    "train": {"x": train_processed["x"], "y": train_processed["y"]},
                    "test": {"x": test_processed["x"], "y": test_processed["y"]}
                },
                "time_series": {
                    "train": {"x": train_processed["x_ts"], "y": train_processed["y_ts"]}, 
                    "test": {"x": test_processed["x_ts"], "y": test_processed["y_ts"]}
                }
            }

        return {
            "regression": {"train": {"x": train_processed["x"], "y": train_processed["y"]}},
            "time_series": {"train": {"x": train_processed["x_ts"], "y": train_processed["y_ts"]}}
        }

    def _pre_process(
        self, x: np.ndarray, y: np.ndarray, fit_scaler: bool = False
    ) -> Dict[str, np.ndarray]:
        """_pre_process"""
        x = self.scaler.fit_transform(x) if fit_scaler else self.scaler.transform(x)
        x_ts, y_ts = self._create_time_series_data(x)
        return {"x": x, "y": y, "x_ts": x_ts, "y_ts": y_ts}

    def _create_time_series_data(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_create_time_series_data"""
        x = sliding_window_view(features, (self.look_back_steps, features.shape[1]))[:-1, 0, :, :]
        y = features[self.look_back_steps:, :]
        return x, y

    def __str__(self) -> str:
        def format_structure(data, indent=1):
            lines = []
            indent_space = "  " * indent

            for key, value in data.items():
                if isinstance(value, dict):
                    nested = format_structure(value, indent + 1)
                    lines.append(f'{indent_space}"{key}": {{\n{nested}\n{indent_space}}}')
                else:
                    shape_info = f"{type(value).__name__}{value.shape}"
                    lines.append(f'{indent_space}"{key}": {shape_info}')
            return ",\n".join(lines)

        return f"{{\n{format_structure(self.dataset)}\n}}"

    def __getitem__(self, key: str):
        if key in self.dataset:
            return self.DataProxy(self.dataset[key])
        raise KeyError(f"Key '{key}' not found in dataset.")

    class DataProxy:
        """DataProxy"""
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            if key in self.data:
                value = self.data[key]
                return Dataset.DataProxy(value) if isinstance(value, dict) else value
            raise KeyError(f"Key '{key}' not found.")

        def __repr__(self):
            return repr(self.data)

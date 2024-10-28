"""dataset"""
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    """Dataset"""
    def __init__(self, file_path: Union[Path, str]) -> None:
        self.dataset = self.pre_process(pd.read_csv(file_path))

    def pre_process(self, data: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """pre_process"""
        x = data[
            ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)"]
        ].values
        y = data[["Power(mW)"]].values

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        x_ts = sliding_window_view(x, window_shape=(12, x.shape[1]))[:-1, 0, :, :]
        y_ts = x[12:, :]

        return {
            "regression": {"x": x, "y": y},
            "time_series": {"x": x_ts, "y": y_ts}
        }

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

"""dataset"""
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Dataset:
    """Dataset"""
    def __init__(
        self,
        train_file: Union[Path, str],
        test_file: Optional[Union[Path, str]] = None,
        target_file: Optional[Union[Path, str]] = None,
    ) -> None:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file) if test_file else None
        target = parse_target(pd.read_csv(target_file)) if target_file else None
        self.dataset = self.pre_process(train_data, test_data, target)

    def pre_process(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        target: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """pre_process"""
        raise NotImplementedError(
            "Subclasses must implement 'pre_process' for custom data preprocessing."
        )

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


def resample_data_by_10min(data: pd.DataFrame) -> pd.DataFrame:
    """resample_data_by_10min"""
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data.set_index("DateTime", inplace=True)

    resampled_data = (
        data.groupby("LocationCode")
        .resample("10min")
        .mean()
        .drop(columns="LocationCode")
        .reset_index()
    )

    return resampled_data

def datetime_to_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    """datetime_to_timestamp"""
    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
    data["timestamp"] = data["DateTime"].astype("int64") // 10**9
    data = data.drop(columns=["DateTime"])
    return data

def create_time_series_data(x: np.ndarray, look_back_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """create_time_series_data"""
    x_ts = sliding_window_view(x, window_shape=(look_back_num, x.shape[1]))[:-1, 0, :, :]
    y_ts = x[look_back_num:, :]
    return x_ts, y_ts

def parse_target(target: pd.DataFrame) -> pd.DataFrame:
    """parse_target"""
    target["Year"] = target["序號"].astype(str).str[:4]
    target["Month"] = target["序號"].astype(str).str[4:6]
    target["Day"] = target["序號"].astype(str).str[6:8]
    target["Hour"] = target["序號"].astype(str).str[8:10]
    target["Minute"] = target["序號"].astype(str).str[10:12]
    target["LocationCode"] = target["序號"].astype(str).str[12:14]

    target["Datetime"] = pd.to_datetime(
        target[["Year", "Month", "Day", "Hour", "Minute"]]
        .astype(str)
        .agg("-".join, axis=1),
        format="%Y-%m-%d-%H-%M"
    )

    target["Date"] = target["Datetime"].dt.date
    target = target.drop_duplicates(subset=["Date", "LocationCode"])

    return target[["序號", "Datetime", "LocationCode"]]

def generate_full_data(data, start_time="09:00", end_time="17:00"):
    """generate_full_data"""
    data["DateTime"] = pd.to_datetime(data["DateTime"]).dt.floor("min")
    filled_data = pd.DataFrame()

    for location, group in data.groupby("LocationCode"):
        group["Date"] = group["DateTime"].dt.date
        dates_with_data = group[
            (group["DateTime"].dt.time >= pd.to_datetime(start_time).time()) &
            (group["DateTime"].dt.time <= pd.to_datetime(end_time).time())
        ]["Date"].unique()

        daily_ranges = [
            pd.date_range(f"{date} {start_time}", f"{date} {end_time}", freq="min")
            for date in dates_with_data
        ]

        full_time_index = pd.DatetimeIndex([time for day in daily_ranges for time in day])

        group_filled = (
            group.drop_duplicates(subset=["DateTime"])
            .set_index("DateTime")
            .reindex(full_time_index)
            .assign(LocationCode=location)
            .reset_index()
            .rename(columns={"index": "DateTime"})
        )

        filled_data = pd.concat([filled_data, group_filled], ignore_index=True)

    return filled_data[data.columns]

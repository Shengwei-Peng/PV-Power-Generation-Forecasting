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
    mask = resampled_data["DateTime"].dt.time.between(
        data.index.min().time(), data.index.max().time()
    )

    return resampled_data.loc[mask].reset_index(drop=True)

def encode_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """encode_datetime"""
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data["timestamp"] = data["DateTime"].astype("int64") // 10**9
    data["month"] = data["DateTime"].dt.month
    data["day"] = data["DateTime"].dt.day
    data["hour"] = data["DateTime"].dt.hour
    data["minute"] = data["DateTime"].dt.minute
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
    return target[["序號", "Datetime", "LocationCode"]]

def generate_full_data(
    data: pd.DataFrame, start_time: str="09:00", end_time: str="16:59"
) -> pd.DataFrame:
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

def filter_nan_days(data: pd.DataFrame) -> pd.DataFrame:
    """filter_nan_days"""
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data["Date"] = data["DateTime"].dt.date

    filtered_data = pd.concat(
        [
            day_data for _, group in data.groupby("LocationCode")
            for _, day_data in group.groupby("Date")
            if not day_data.isna().any().any()
        ]
    ).reset_index(drop=True)

    return filtered_data.drop(columns="Date")

def create_samples(data: pd.DataFrame) -> tuple[np.array, np.array]:
    """create_samples"""
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data.set_index("DateTime", inplace=True)

    x_list, y_list = [], []

    for _, group in data.groupby("LocationCode"):
        dates = group.sort_index().index.date
        for i in range(len(dates) - 1):
            first_day_date = dates[i]
            second_day_date = first_day_date + pd.DateOffset(days=1)

            first_day = group.loc[group.index.date == first_day_date]
            second_day = group.loc[group.index.date == second_day_date]

            if not first_day.empty and not second_day.empty:
                x_data = pd.concat([
                    first_day,
                    second_day[second_day.index.time < pd.to_datetime("09:00").time()]
                ])
                y_data = second_day[
                    (second_day.index.time >= pd.to_datetime("09:00").time()) &
                    (second_day.index.time <= pd.to_datetime("16:59").time())
                ]["Power(mW)"]

                x_list.append(x_data.to_numpy())
                y_list.append(y_data.to_numpy())

    return np.array(x_list), np.array(y_list)

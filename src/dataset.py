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

        def __iter__(self):
            return iter(self.data)

        def keys(self):
            """keys"""
            return self.data.keys()

        def items(self):
            """items"""
            return self.data.items()

        def __len__(self):
            return len(self.data)

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
    target["LocationCode"] = target["序號"].astype(str).str[12:14].astype(int)
    target["DateTime"] = pd.to_datetime(
        target["序號"].astype(str).str[:12],
        format="%Y%m%d%H%M"
    )
    return target[["序號", "DateTime", "LocationCode"]]

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

def create_samples(
    data: pd.DataFrame,
    target: pd.DataFrame = None,
    flatten: bool = False,
    subtract_prev: bool = False,
) -> Dict[str, np.ndarray]:
    """create_samples"""
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data["Date"], data["Time"] = data["DateTime"].dt.date, data["DateTime"].dt.time
    data.set_index("DateTime", inplace=True)

    start_time, end_time = pd.to_datetime("09:00").time(), pd.to_datetime("16:59").time()
    x_list, y_list = [], []

    groups = data.groupby("LocationCode")
    if target is not None:
        target["Date"] = target["DateTime"].dt.date
        groups = target.groupby("LocationCode")

    for location, group in groups:
        location_data = data[data["LocationCode"] == location]
        dates = (location_data["Date"] if target is None else group["Date"]).sort_values().unique()

        for date in (dates if target is not None else dates[:-1]):
            first_day_date = date - pd.Timedelta(days=1) if target is not None else date
            second_day_date = date if target is not None else date + pd.Timedelta(days=1)

            first_day = location_data[location_data["Date"] == first_day_date]
            second_day = location_data[location_data["Date"] == second_day_date]
            if first_day.empty or second_day.empty:
                continue

            x_data = pd.concat([first_day, second_day[second_day["Time"] < start_time]])
            prev_y = first_day[
                (first_day["Time"] >= start_time) & (first_day["Time"] <= end_time)
            ]["Power(mW)"]

            if target is None:
                y_data = second_day.loc[
                    (second_day["Time"] >= start_time) & (second_day["Time"] <= end_time),
                    "Power(mW)"
                ].reset_index(drop=True)
                prev_y = prev_y.reset_index(drop=True)
                y = y_data - prev_y if subtract_prev else y_data

            y_list.append(prev_y if target is not None else y)
            x_list.append(x_data.drop(columns=["Date", "Time"]))

    x = np.array(x_list)
    y = np.array(y_list)
    return {
        "X": x.reshape(x.shape[0], -1) if flatten else x, 
        "prev_y" if target is not None else "y": y
    }

"""dataset"""
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

class Dataset:
    """Dataset"""
    def __init__(
        self,
        data_file: Union[Path, str],
        upload_file: Optional[Union[Path, str]] = None,
    ) -> None:
        data = pd.read_csv(data_file)
        upload = pd.read_csv(upload_file) if upload_file else None
        self.dataset = self.pre_process(data, upload)

    def pre_process(
        self,
        data: pd.DataFrame,
        upload: Optional[pd.DataFrame] = None,
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
    data = data.copy()
    data.loc[:, "DateTime"] = pd.to_datetime(data["DateTime"])
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

def merge_external(data: pd.DataFrame, external_file: str) -> pd.DataFrame:
    """merge_external"""
    external = pd.read_csv(external_file)
    data.loc[:, "DateTime"] = pd.to_datetime(data["DateTime"])
    external["datetime"] = pd.to_datetime(external["datetime"])
    cols = [col for col in external.columns if col not in ["datetime", "# stno"]]
    merged = pd.merge(
        data,
        external[["datetime"] + cols],
        left_on="DateTime",
        right_on="datetime",
        how="left"
    )
    return merged.drop(columns=["datetime"])

def add_location_details(data: pd.DataFrame) -> pd.DataFrame:
    """add_location_details"""
    location_details = {
        1: (23.899444, 121.544444, 181, 5),
        2: (23.899722, 121.544722, 175, 5),
        3: (23.899722, 121.545000, 180, 5),
        4: (23.899444, 121.544444, 161, 5),
        5: (23.899444, 121.544722, 208, 5),
        6: (23.899444, 121.544444, 208, 5),
        7: (23.899444, 121.544444, 172, 5),
        8: (23.899722, 121.545000, 219, 3),
        9: (23.899444, 121.544444, 151, 3),
        10: (23.899444, 121.544444, 223, 1),
        11: (23.899722, 121.544722, 131, 1),
        12: (23.899722, 121.544722, 298, 1),
        13: (23.897778, 121.539444, 249, 5),
        14: (23.897778, 121.539444, 197, 5),
        15: (24.009167, 121.617222, 127, 1),
        16: (24.008889, 121.617222, 82, 1),
        17: (23.97512778, 121.613275, 0, 0)
    }

    data[
        ["latitude", "longitude", "orientation", "altitude"]
    ] = data["LocationCode"].apply(lambda x: pd.Series(location_details[x]))

    return data

def create_samples(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_columns: list,
    target_column: list
) -> dict:
    """create_samples"""
    x_list = []
    y_list = []
    has_target = set(target_column).issubset(data.columns)

    data["DateTime"] = pd.to_datetime(data["DateTime"])
    reference_data["DateTime"] = pd.to_datetime(reference_data["DateTime"])

    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        previous_day = row["DateTime"] - pd.Timedelta(days=1)

        past_window = reference_data[
            (reference_data["LocationCode"] == row["LocationCode"]) &
            (reference_data["DateTime"] == previous_day)
        ][feature_columns + target_column]

        if past_window.empty:
            continue

        past_window_flat = past_window.values.flatten()
        current_features = row[feature_columns].values

        x_list.append(np.concatenate([past_window_flat, current_features]))
        if has_target:
            y_list.append(row[target_column])

    x = np.array(x_list).astype(float)

    if has_target:
        y = np.array(y_list).astype(float).ravel()
        return {"X": x, "y": y}

    return {"X": x}

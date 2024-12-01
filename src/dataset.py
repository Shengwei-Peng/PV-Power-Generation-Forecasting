"""dataset"""
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd

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
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data["timestamp"] = data["DateTime"].astype("int64") // 10**9
    data["month"] = data["DateTime"].dt.month
    data["day"] = data["DateTime"].dt.day
    data["hour"] = data["DateTime"].dt.hour
    data["minute"] = data["DateTime"].dt.minute
    return data

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

def merge_external(data: pd.DataFrame, external_data: pd.DataFrame) -> pd.DataFrame:
    """merge_external"""
    data.loc[:, "DateTime"] = pd.to_datetime(data["DateTime"])
    external_data["datetime"] = pd.to_datetime(external_data["datetime"])
    cols = [col for col in external_data.columns if col not in ["datetime"]]
    merged = pd.merge(
        data,
        external_data[["datetime"] + cols],
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

def prepare_external_data(input_dir: str | Path, output_folder: str | Path) -> pd.DataFrame:
    """prepare_external_data"""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    external_data = None
    for file in sorted(Path(input_dir).glob("*.csv")):
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.resample("10min").mean().interpolate("time").fillna(0)
        df.rename(columns={col: f"{file.stem} {col}" for col in df.columns}, inplace=True)
        external_data = (
            df if external_data is None else external_data.merge(df, on="datetime", how="inner")
        )

    external_data.reset_index(inplace=True)
    external_data.to_csv(output_folder / "external_data.csv", index=False)

    return external_data

def merge_csv(input_dirs: list[str | Path], output_folder: str | Path) -> pd.DataFrame:
    """merge_csv"""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_data = []
    for directory in input_dirs:
        for file in sorted(Path(directory).glob("*.csv")):
            df = pd.read_csv(file)
            all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data["DateTime"] = pd.to_datetime(combined_data["DateTime"])
    combined_data.sort_values(by=["LocationCode", "DateTime"], inplace=True)
    combined_data.to_csv(output_folder / "all_data.csv", index=False)

    return combined_data

def get_data(
    date_time: pd.Timestamp,
    location_code: int,
    external_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_columns: list
) -> pd.Series:
    """get_data"""
    features = reference_data[
        (reference_data["LocationCode"] == location_code) &
        (reference_data["DateTime"] == date_time)
    ].copy()

    if features.empty:
        features = external_data[
            (external_data["datetime"] == date_time)
        ].copy()
        if features.empty:
            return None
        features.loc[:, "LocationCode"] = location_code
        features.loc[:, "DateTime"] = date_time
        features.loc[:, "Power(mW)"] = None
        features = encode_datetime(features)
        features = add_location_details(features)

    return features[feature_columns + ["Power(mW)"]].squeeze()

def create_samples(
    data: pd.DataFrame,
    external_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_columns: list[str],
    output_folder: str | Path,
) -> dict:
    """create_samples"""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    data["DateTime"] = pd.to_datetime(data["DateTime"])
    reference_data["DateTime"] = pd.to_datetime(reference_data["DateTime"])
    external_data["datetime"] = pd.to_datetime(external_data["datetime"])

    data = data[
        data["DateTime"].dt.time.between(pd.Timestamp("09:00").time(), pd.Timestamp("16:59").time())
    ]
    x_list, y_list = [], []
    is_train = "Power(mW)" in data.columns
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        date_time = row["DateTime"]
        location_code = row["LocationCode"]

        yesterday_data = get_data(
            date_time - pd.Timedelta(days=1),
            location_code, external_data, reference_data, feature_columns
        )
        if yesterday_data is None:
            continue

        morning_data = get_data(
            date_time.replace(hour=8, minute=50),
            location_code, external_data, reference_data, feature_columns
        )

        current_data = row[feature_columns]
        x = pd.concat(
            [yesterday_data, morning_data, current_data], axis=0
        ).reset_index(drop=True)

        x_list.append(x)
        if is_train:
            y_list.append(row["Power(mW)"])

    if is_train:
        pd.DataFrame(x_list).to_csv(output_folder / "train_x.csv", index=False)
        pd.Series(y_list).to_csv(output_folder / "train_y.csv", index=False)
    else:
        pd.DataFrame(x_list).to_csv(output_folder / "test_x.csv", index=False)

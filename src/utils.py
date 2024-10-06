"""utils"""
import argparse
from pathlib import Path


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
        "--target_column",
        type=str,
        default="Power(mW)",
        help="Name of the target column in the dataset (default: 'Power(mW)')"
    )
    parser.add_argument(
        "--test_start_time",
        type=str,
        default="09:00:00",
        help="Start time for the test data in HH:MM:SS format (default: '09:00:00')"
    )
    parser.add_argument(
        "--test_end_time",
        type=str,
        default="16:59:00",
        help="End time for the test data in HH:MM:SS format (default: '16:59:00')"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for the time series forecasting model (default: 10)"
    )
    parser.add_argument(
        "--test_days",
        type=int,
        default=200,
        help="Number of days to use for testing the model (default: 200)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )

    return vars(parser.parse_args())

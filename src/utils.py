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
    return vars(parser.parse_args())

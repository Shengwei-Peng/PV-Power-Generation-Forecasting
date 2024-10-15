"""utils"""
import random
import argparse
from pathlib import Path
from typing import Dict

import torch
import numpy as np


def parse_arguments() -> Dict:
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="PV Power Generation Forecast")
    parser.add_argument(
        "--train_folder",
        type=Path,
        required=True,
        help="Path to the folder containing CSV files for train data"
    )
    parser.add_argument(
        "--test_folder",
        type=Path,
        default=None,
        help="Path to the folder containing CSV files for data for test data (optional)"
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
        "--n_valid_months",
        type=int,
        default=2,
        help="Number of last months to use for the validation set (default: 2)"
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

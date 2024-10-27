"""utils"""
import random
from pathlib import Path
from typing import Union

import torch
import numpy as np


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

def validate_csv(file: Union[Path, str]) -> Path:
    """validate_csv"""
    file = Path(file)
    if not file.exists() or not file.is_file() or file.suffix != ".csv":
        raise FileNotFoundError(f"CSV file does not exist: {file}")
    return file

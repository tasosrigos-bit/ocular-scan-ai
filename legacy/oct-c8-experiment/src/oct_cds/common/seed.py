from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 1337, deterministic: bool = True) -> int:
    """Seed python, numpy and torch (if installed). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed

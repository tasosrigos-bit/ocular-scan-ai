"""Project-wide configuration for the ocular package.

This module centralises the filesystem layout, the class vocabulary, and the
global random seed, so that every other module and notebook resolves paths and
labels from a single source of truth.

Attributes
----------
ROOT : pathlib.Path
    Repository root, resolved relative to this file.
DATA_DIR : pathlib.Path
    Root of the data tree (``data/``).
REPORTS_DIR : pathlib.Path
    Directory holding the de-identified clinic reports.
TRAINING_DIR : pathlib.Path
    Kermany OCT2017 training split, with one subdirectory per class.
TESTING_DIR : pathlib.Path
    Kermany OCT2017 test split, with one subdirectory per class.
CLASSES : list of str
    The four diagnostic classes in fixed order. This order defines the integer
    label encoding and the column order of every model output.
SEED : int
    Global random seed for reproducible data splits and training.
"""
from pathlib import Path

# Filesystem layout. All paths are absolute and derived from the repository root.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "Reports"
TRAINING_DIR = DATA_DIR / "raw" / "OCT2017" / "train"
TESTING_DIR = DATA_DIR / "raw" / "OCT2017" / "test"

# Diagnostic classes in fixed order. This defines the integer label encoding
# and the column order of every model head.
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Global random seed for reproducible data splits and training.
SEED = 0

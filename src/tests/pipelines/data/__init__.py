"""
Encapsulates some data assets that we use for testing.
"""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).absolute().parent
"""
Directory that this file is in.
"""

TESTING_DATASET_PATH = _PACKAGE_DIR / "testing_dataset.tfrecord"
"""
The location of the small dataset we use for testing.
"""

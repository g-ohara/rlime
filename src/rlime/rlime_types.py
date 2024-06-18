"""This file contains type aliases for the newlime package."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
Rule = tuple[int, ...]
Classifier = Callable[[IntArray | FloatArray], IntArray]


@dataclass
class Dataset:  # pylint: disable=too-many-instance-attributes
    """Dataset class"""

    data: IntArray
    labels: IntArray
    train_idx: IntArray
    train: IntArray
    labels_train: IntArray
    validation_idx: IntArray
    validation: IntArray
    labels_validation: IntArray
    test_idx: IntArray
    test: IntArray
    labels_test: IntArray
    feature_names: list[str]
    categorical_features: list[int]
    categorical_names: dict[int, list[str]]
    ordinal_features: list[int]
    class_target: str
    class_names: list[str]

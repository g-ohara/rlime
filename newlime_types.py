"""This file contains type aliases for the newlime package."""

from typing import Callable

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
Rule = tuple[int, ...]
Classifier = Callable[[IntArray | FloatArray], IntArray]

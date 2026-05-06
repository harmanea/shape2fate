import os
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

PathLike: TypeAlias = str | os.PathLike[str]

Shift: TypeAlias = tuple[float, float]
ShiftLike: TypeAlias = Shift | np.ndarray
Shifts: TypeAlias = tuple[ShiftLike, ShiftLike, ShiftLike] | np.ndarray

PhaseOffsets: TypeAlias = tuple[float, float, float] | np.ndarray
Amplitudes: TypeAlias = tuple[float, float, float] | np.ndarray

ScalarOrArray: TypeAlias = float | ArrayLike

from abc import ABC, abstractmethod

import numpy as np

from .parameters import AcquisitionParameters
from .sim import diffraction_limit


class OTF(ABC):
    """Optical transfer function (OTF) of an optical system."""

    @abstractmethod
    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the OTF at a normalized spatial frequency.

        Args:
            x: Spatial frequency normalized to ``[0, 1]``, where ``0`` is DC (zero frequency) and ``1`` is the maximum
                supported spatial frequency.

        Returns:
            OTF value at ``x``. Values outside ``[0, 1]`` are undefined.
        """
        pass

    @abstractmethod
    def draw(self, size: int | tuple[int, int], x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        """Draw the OTF on a 2D frequency-space grid.

        Args:
            size: Grid size. An integer creates a square grid; a tuple creates a ``(height, width)`` grid.
            x_shift: Shift along the x-axis in frequency space.
            y_shift: Shift along the y-axis in frequency space.

        Returns:
            OTF sampled on the requested grid.
        """
        pass


class ModelOTF(OTF):
    """Exponential approximation of the ideal OTF."""

    def __init__(self, ap: AcquisitionParameters, curvature: float = 0.3) -> None:
        """Initialize the model OTF.

        Args:
            ap: Acquisition parameters of the experiment.
            curvature: Bend of the approximation OTF model function in the ``[0, 1]`` range, where ``1`` is a
                perfect OTF.
        """
        self.image_cutoff = diffraction_limit(ap.na, ap.wavelength, ap.pixel_size, ap.image_size)
        self.curvature = curvature

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x * x)) * self.curvature ** x

    def draw(self, size: int | tuple[int, int], x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        height, width = (size, size) if isinstance(size, int) else size
        X, Y = np.meshgrid(np.hstack([np.arange(height // 2), np.arange(-height // 2, 0)]),
                           np.hstack([np.arange(width // 2), np.arange(-width // 2, 0)]))
        distance_to_origin = np.hypot(X + x_shift, Y + y_shift)
        return self(np.minimum(distance_to_origin / self.image_cutoff, 1))


# TODO: Implement this for a specific example
class MeasuredOTF(OTF):
    """OTF based on measured data, either directly or from a measured PSF."""

    def __init__(self, samples: np.ndarray, image_cutoff: float) -> None:
        """Initialize a measured OTF.

        Args:
            samples: Measured OTF data samples.
            image_cutoff: Cutoff frequency of the OTF.
        """
        self.samples = samples
        self.image_cutoff = image_cutoff

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return np.interp(x, np.linspace(0, 1, len(self.samples)), self.samples)

    def draw(self, size: int | tuple[int, int], x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        height, width = (size, size) if isinstance(size, int) else size
        X, Y = np.meshgrid(np.hstack([np.arange(height // 2), np.arange(-height // 2, 0)]),
                           np.hstack([np.arange(width // 2), np.arange(-width // 2, 0)]))
        distance_to_origin = np.hypot(X + x_shift, Y + y_shift)
        return self(np.minimum(distance_to_origin / self.image_cutoff, 1))

from abc import ABC, abstractmethod

import numpy as np

from .parameters import AcquisitionParameters
from .sim import diffraction_limit


class OTF(ABC):
    """
    The Optical Transfer Function (OTF) of the optical system.
    """

    @abstractmethod
    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the Optical Transfer Function at a given spatial frequency normalized to the range [0, 1], with 0 representing the DC (zero frequency) and 1 corresponding to the maximum spatial frequency at the edge of the function's support. Values outside the range [0, 1] are undefined.
        """
        pass

    @abstractmethod
    def draw(self, size: int | tuple[int, int], x_shift: float = 0, y_shift: float = 0) -> np.ndarray:
        """
        Draw the Optical Transfer Function on a 2D grid in frequency space.

        This method generates and visualizes the OTF over a specified grid size. The grid can be a square or rectangular array, and optional shifts in the x and y directions can be applied.
        """
        pass


class ModelOTF(OTF):
    """
    An exponential approximation of the ideal OTF.
    """

    def __init__(self, ap: AcquisitionParameters, curvature: float = 0.3):
        """
        :param ap: Acquisition parameters of the experiment.
        :param curvature: Bend of the approximation OTF model function in [0, 1] range where 1 is a perfect OTF.
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
    """
    OTF based on measured data, either directly as an OTF or from a measured PSF.
    """

    def __init__(self, samples: np.ndarray, image_cutoff: float):
        """
        :param samples: The measured OTF data samples.
        :param image_cutoff: The cutoff frequency of the OTF.
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

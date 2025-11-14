import numpy as np


def diffraction_limit(na, wavelength, pixel_size, image_size) -> float:
    """
    Calculate the diffraction limit distance in reciprocal space in pixels.

    The diffraction limit (cutoff frequency) is given by the formula:
        cutoff_frequency = (2 * na) / wavelength

    This function converts the limit in physical units to pixels.
    """
    cutoff_frequency = 2 * na / wavelength  # in nanometer^-1 (for incoherent imaging)
    return cutoff_frequency * 1000 * pixel_size * image_size


def illumination_pattern(angle: float, frequency: float, phase_offset: float, amplitude: float, size: int) -> np.ndarray:
    n = size // 2
    Y, X = np.mgrid[-n:n, -n:n]
    ky, kx = np.sin(angle) * frequency, np.cos(angle) * frequency
    return 1 + amplitude * np.cos(2 * np.pi * (X * kx + Y * ky) + phase_offset)


def component_separation_matrix(phase_offset: float = 0, amplitude: float = 1) -> np.ndarray:
    phases = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]) + phase_offset

    M = np.ones((3, 3), dtype=np.complex128)

    M[:, 1] = 0.5 * amplitude * (np.cos(phases) + 1j * np.sin(phases))
    M[:, 2] = 0.5 * amplitude * (np.cos(-phases) + 1j * np.sin(-phases))

    return np.linalg.inv(M)

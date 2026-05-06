import numpy as np
import torch

from .otf import OTF
from .parameters import AcquisitionParameters, ReconstructionParameters
from .sim import diffraction_limit, component_separation_matrix
from .types import Amplitudes, PhaseOffsets, ScalarOrArray, Shifts


def reconstruction_preprocess(images: np.ndarray, rp: ReconstructionParameters) -> np.ndarray:
    """Prepare raw SIM images for reconstruction.

    Args:
        images: Raw SIM image stack.
        rp: Reconstruction parameters containing background and border-fade settings.

    Returns:
        Preprocessed image stack.
    """
    # Subtract background
    images = np.maximum(images - rp.background_intensity, 0)

    # Fade border
    images = fade_border(images, rp.border_fade)

    return images


def fade_border(data: np.ndarray, border_size: int) -> np.ndarray:
    """Fade image borders to black.

    Uses ``sin^2(x)`` mapped from ``[0, px]`` to ``[pi / 2, 0]``. Inspired by fairSIM.

    Args:
        data: Image or stack of images.
        border_size: Number of pixels to use for the gradient fade.

    Returns:
        Copy of ``data`` with faded borders.
    """
    data = data.copy()
    h, w = data.shape[-2:]

    for i in range(border_size):
        factor = np.sin(1 / border_size * np.pi / 2 * i) ** 2

        data[..., i, i:w - i - 1] *= factor  # top
        data[..., i:h - i - 1, w - i - 1] *= factor  # right
        data[..., h - i - 1, i + 1:w - i] *= factor  # bottom
        data[..., i + 1:h - i, i] *= factor  # left

    return data


def cpu_reconstruct(
        images: np.ndarray,
        otf: OTF,
        shifts: Shifts,
        phase_offsets: PhaseOffsets,
        amplitudes: Amplitudes,
        ap: AcquisitionParameters,
        rp: ReconstructionParameters
) -> np.ndarray:
    """Reconstruct one SIM high-resolution image on the CPU.

    Args:
        images: Raw SIM image group containing three phases for each of three orientations.
        otf: Optical transfer function used to weight Fourier components.
        shifts: Estimated frequency shifts for the SIM orientations.
        phase_offsets: Estimated phase offsets for the SIM orientations.
        amplitudes: Estimated modulation amplitudes for the SIM orientations.
        ap: Acquisition parameters for the input images.
        rp: Reconstruction parameters for filtering.

    Returns:
        Reconstructed high-resolution image.
    """
    # Separate components
    fft_images = np.fft.fft2(images)

    separation_matrices = np.array([component_separation_matrix(phase_offsets[i], amplitudes[i])
                                    for i in range(3)])

    components = np.einsum('aij,ajkl->aikl', separation_matrices,
                           fft_images.reshape(3, 3, *fft_images.shape[1:]))

    # Pad components
    padded_components = np.zeros((3, 3, ap.hr_size, ap.hr_size), dtype=np.complex128)
    X, Y = component_padding_matrices(ap.lr_size)
    padded_components[..., Y, X] = components

    # Shift components
    shift_x = np.array([shifts[i][1] for i in range(3)])
    shift_y = np.array([shifts[i][0] for i in range(3)])

    shift_matrix_pos = fourier_shift_matrix(ap.hr_size, ap.hr_size, shift_x, shift_y)
    shift_matrix_neg = fourier_shift_matrix(ap.hr_size, ap.hr_size, -shift_x, -shift_y)

    padded_components[:, 1] = np.fft.fft2(np.fft.ifft2(padded_components[:, 1]) * shift_matrix_pos)
    padded_components[:, 2] = np.fft.fft2(np.fft.ifft2(padded_components[:, 2]) * shift_matrix_neg)

    # Map otf support
    otfs = shifted_otf_images(otf, shifts, ap.hr_size)
    padded_components *= otfs

    # Apply wiener and apodization filters
    wiener = 1 / (np.sum(np.abs(otfs) ** 2, (0, 1)) + rp.w * rp.w)
    apodization = apodization_filter(ap, ap.hr_size, rp.apodization_bend, rp.apodization_cutoff)

    result = np.sum(padded_components, (0, 1)) * wiener * apodization

    return np.real(np.fft.ifft2(result))


class GPUReconstruction:
    """Reusable GPU SIM reconstructor for batched processing."""

    def __init__(
            self,
            otf: OTF,
            shifts: Shifts,
            phase_offsets: PhaseOffsets,
            amplitudes: Amplitudes,
            ap: AcquisitionParameters,
            rp: ReconstructionParameters,
            device: torch.device | str
    ) -> None:
        self.image_size = ap.image_size
        self.device = device

        self.otfs = torch.from_numpy(shifted_otf_images(otf, shifts, ap.hr_size)).to(device)
        self.wiener = 1 / (torch.sum(torch.abs(self.otfs) ** 2, (0, 1)) + rp.w * rp.w)
        self.apodization = torch.from_numpy(
            apodization_filter(ap, ap.hr_size, rp.apodization_bend, rp.apodization_cutoff)
        ).to(device)
        self.separation_matrices = torch.stack([
            torch.from_numpy(component_separation_matrix(phase_offsets[i], amplitudes[i]))
            for i in range(3)
        ]).to(device)
        self.pX, self.pY = component_padding_matrices(ap.lr_size)
        self.shift_matrices = torch.from_numpy(np.array([
            [
                fourier_shift_matrix(ap.hr_size, ap.hr_size, shifts[i][1], shifts[i][0]),
                fourier_shift_matrix(ap.hr_size, ap.hr_size, -shifts[i][1], -shifts[i][0])
            ]
            for i in range(3)
        ])).to(device)

    def reconstruct(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Reconstruct SIM high-resolution images in batches.

        Args:
            data: Raw SIM image stack arranged in consecutive nine-frame SIM groups.
            batch_size: Number of SIM high-resolution images to process per GPU batch.

        Returns:
            Stack of reconstructed high-resolution images.
        """
        data = data.reshape(-1, 3, 3, self.image_size, self.image_size)
        batches = (len(data) - 1) // batch_size + 1

        results = []
        for i in range(batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batch = torch.from_numpy(batch).to(self.device)

            # Extract components
            components = torch.einsum('boqhw,opq->bophw', torch.fft.fft2(batch), self.separation_matrices)

            # Pad to HR size
            padded_components = torch.zeros((len(batch), 3, 3, self.image_size * 2, self.image_size * 2),
                                            dtype=torch.complex128, device=self.device)
            padded_components[..., self.pY, self.pX] = components

            # Shift components
            padded_components[:, :, 1:] = torch.fft.fft2(
                torch.fft.ifft2(padded_components[:, :, 1:]) * self.shift_matrices[None]
            )

            # Map OTF support
            padded_components *= self.otfs[None]

            # Apply Wiener filter denominator and apodization
            fft_results = torch.sum(padded_components, (1, 2)) * self.wiener * self.apodization

            spatial_results = torch.real(torch.fft.ifft2(fft_results)).cpu().numpy()
            results.append(spatial_results)

        return np.concatenate(results)


def component_padding_matrices(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build index matrices for embedding low-resolution components into a high-resolution Fourier grid.

    Args:
        size: Low-resolution image size.

    Returns:
        X and Y index matrices for padding components into a ``2 * size`` Fourier grid.
    """
    X, Y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]),
                       np.hstack([np.arange(size // 2), np.arange(size * 3 // 2, size * 2)]))
    return X, Y


def fourier_shift_matrix(height: int, width: int, x_shift: ScalarOrArray, y_shift: ScalarOrArray) -> np.ndarray:
    """Create a Fourier-domain phase ramp for spatial shifts.

    Args:
        height: Output matrix height.
        width: Output matrix width.
        x_shift: Shift along the x-axis.
        y_shift: Shift along the y-axis.

    Returns:
        Complex phase-ramp matrix.
    """
    x_indices = np.linspace(-0.5, 0.5, width, endpoint=False)
    y_indices = np.linspace(-0.5, 0.5, height, endpoint=False)

    x_shift = np.atleast_1d(x_shift)
    y_shift = np.atleast_1d(y_shift)

    x = np.exp(-1j * 2 * np.pi * x_shift[:, None] * x_indices[None, :])
    y = np.exp(-1j * 2 * np.pi * y_shift[:, None] * y_indices[None, :])

    return (y[:, :, None] * x[:, None, :]).squeeze()


def shifted_otf_images(otf: OTF, shifts: Shifts, image_size: int) -> np.ndarray:
    """Draw OTF support images for all SIM components and orientations.

    Args:
        otf: Optical transfer function to sample.
        shifts: Frequency shifts for the SIM orientations.
        image_size: Size of the high-resolution Fourier grid.

    Returns:
        OTF stack with shape ``(3, 3, image_size, image_size)``.
    """
    otfs = np.zeros((3, 3, image_size, image_size))

    otfs[:, 0] = otf.draw(image_size)
    for i in range(3):
        otfs[i, 1] = otf.draw(image_size, shifts[i][1], shifts[i][0])
        otfs[i, 2] = otf.draw(image_size, -shifts[i][1], -shifts[i][0])

    return otfs


def apodization_filter(
        ap: AcquisitionParameters,
        size: int | None = None,
        bend: float = 0.9,
        cutoff: float = 2.0
) -> np.ndarray:
    """Create a circular apodization filter for SIM reconstruction.

    Args:
        ap: Acquisition parameters used to compute the diffraction limit.
        size: Output filter size. Defaults to the high-resolution reconstruction size.
        bend: Exponent controlling the filter falloff.
        cutoff: Cutoff frequency as a multiple of the diffraction limit.

    Returns:
        Apodization filter image.
    """
    dist_ratio = 1 / (cutoff * diffraction_limit(ap.na, ap.wavelength, ap.pixel_size, ap.image_size))

    if size is None:
        size = ap.hr_size

    Y, X = np.meshgrid(np.fft.fftfreq(size) * size, np.fft.fftfreq(size) * size)
    distance = np.hypot(X, Y) * dist_ratio
    mask = distance < 1

    apo = np.zeros_like(distance)
    apo[mask] = (2 / np.pi) * (np.arccos(distance[mask]) - distance[mask] * np.sqrt(1 - distance[mask] ** 2))
    return apo ** bend

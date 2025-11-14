import warnings

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import optimize

from parameters import ReconstructionParameters, AcquisitionParameters
from sim import component_separation_matrix, diffraction_limit
from otf import OTF
from reconstruction import component_padding_matrices, fourier_shift_matrix


_XX = np.linspace(0, 2 * np.pi, 6, False)
_SEPARATION_MATRICES = np.stack([component_separation_matrix(phase_offset) for phase_offset in _XX])


def check_parameter_estimates(shifts, modulations, rad_rel_tol=0.001, ang_tol_deg=10, mod_tol=0.25, min_mod=0.1, eps=1e-12):
    errors = []

    shifts = np.asarray(shifts)
    modulations = np.asarray(modulations)

    # Check radii
    radii = np.hypot(shifts[:, 0], shifts[:, 1])
    r_med = np.median(radii)
    rel_dev = np.abs(radii - r_med) / max(r_med, eps)

    if np.any(rel_dev > rad_rel_tol):
        errors.append('Estimated illumination frequencies (radii) are not consistent:')

        outlier_index = int(np.argmax(rel_dev)) if int(np.sum(rel_dev > rad_rel_tol)) == 1 else None
        for i, r in enumerate(radii):
            errors.append(f'  - Orientation {i + 1}: {r:.1f}{" <-- outlier" if i == outlier_index else ""}')

    # Check angles
    angles = np.arctan2(shifts[:,0], shifts[:,1])
    angles = [a + np.pi if a < 0 else a for a in angles]

    seps = np.array([np.abs(angles[1] - angles[2]), np.abs(angles[0] - angles[2]), np.abs(angles[0] - angles[1])])
    ang_tol = np.deg2rad(ang_tol_deg)
    failing = [k for k in range(3) if (np.abs(seps[k] - np.pi / 3)  > ang_tol) and (np.abs(seps[k] - 2 * np.pi / 3)  > ang_tol)]

    if failing:
        errors.append('Estimated illumination angles are not consistently spaced:')

        outlier_index = [i for i in range(3) if i not in failing][0] if len(failing) == 2 else None
        for i, a in enumerate(angles):
            errors.append(f'  - Orientation {i + 1}: {np.rad2deg(a):.1f}°{" <-- outlier" if i == outlier_index else ""}')

    # Check modulations
    m_med = np.median(modulations)
    m_dev = np.abs(modulations - m_med)

    if np.any(m_dev > mod_tol) or np.any(modulations < min_mod):
        errors.append('Estimated modulation amplitudes are not consistent or too low.')

        outlier_index = int(np.argmax(m_dev)) if int(np.sum(m_dev > mod_tol)) == 1 else None
        for i, m in enumerate(modulations):
            is_outlier = i == outlier_index
            is_too_low = m < min_mod
            errors.append(f'  - Orientation {i + 1}: {m:.2f}{" <-- " if is_outlier or is_too_low else ""}{"outlier" if is_outlier else ""}{", " if is_outlier and is_too_low else ""}{"too low" if is_too_low else ""}')

    if errors:
        errors = ['SIM illumination may be misaligned or parameter estimation failed.'] + errors
        warnings.warn('\n'.join(errors))


def estimate_parameters(images: np.ndarray, otf: OTF, ap: AcquisitionParameters, rp: ReconstructionParameters,
                        approximate_shifts = None, epsilon: float = 10_000) -> tuple:

    if rp.rl_deconvolution_iterations > 0:
        images = _deconvolve_richardson_lucy(images, rp.rl_deconvolution_iterations, otf)

    fft_images = fft2(images)

    components = np.einsum('aij,ajkl->aikl', component_separation_matrix()[None],
                           fft_images.reshape(3, 3, ap.image_size, ap.image_size))

    # This step may be skipped if we already have approximate shifts from a previous reconstruction
    if approximate_shifts is None:
        minimum_distance = rp.minimum_peak_distance * diffraction_limit(ap.na, ap.wavelength, ap.pixel_size, ap.image_size)
        approximate_shifts = [estimate_integer_shift(components[i][0], components[i][1], minimum_distance, epsilon)
                              for i in range(3)]

    shifts = [estimate_fine_shift(approximate_shifts[i], components[i][0], components[i][1], epsilon=epsilon)
              for i in range(3)]

    phase_offsets = [estimate_phase_offset(fft_images[i * 3:(i + 1) * 3], shifts[i])
                     for i in range(3)]

    # This step is generally not possible to do accurately, so we just hardcode the values, but the estimated values may serve as a quality metric of the reconstruction
    amplitudes = [estimate_amplitude(components[i], shifts[i], otf)
                  for i in range(3)]

    return shifts, phase_offsets, amplitudes


def estimate_integer_shift(zero_component: np.ndarray, first_component: np.ndarray, min_distance: float,
                           epsilon: float = 10_000) -> np.ndarray:
    """Implementation of the method proposed in: https://doi.org/10.1364/BOE.9.005037"""

    size, x, y = _grid(zero_component)

    left = zero_component / (np.abs(zero_component) + epsilon)
    right = first_component / (np.abs(first_component) + epsilon)

    left_spatial_conjugate = np.conj(ifft2(left))
    right_spatial = ifft2(right)

    c = np.abs(fft2(right_spatial * left_spatial_conjugate))
    masked_c = np.where(np.hypot(x, y) > min_distance, c, 0)

    peak = np.unravel_index(np.argmax(masked_c), c.shape)

    return (np.array(peak) + size // 2) % size - size // 2


def estimate_fine_shift(approximate_shift, zero_component: np.ndarray, first_component: np.ndarray,
                        search_range: int = 3, epsilon: float = 10_000):
    """Implementation of the method proposed in: https://doi.org/10.1364/BOE.9.005037"""

    def f(x):
        shifted_component = fft2(ifft2(first_component) * fourier_shift_matrix(*first_component.shape[-2:], x[1], x[0]))
        left = zero_component / (np.abs(zero_component) + epsilon)
        right = shifted_component / (np.abs(shifted_component) + epsilon)
        return -np.abs(np.sum(np.conj(left) * right, axis=(-2, -1)))

    margin = search_range / 2

    result = optimize.minimize(f, approximate_shift,
                               bounds=((approximate_shift[0] - margin, approximate_shift[0] + margin),
                                       (approximate_shift[1] - margin, approximate_shift[1] + margin)),
                               options={'maxls': 50})  # To avoid ABNORMAL_TERMINATION_IN_LNSRCH

    assert result.success, result.message

    return result.x


def estimate_amplitude(components: np.ndarray, shift, otf: OTF):
    """Implementation of the method proposed in: https://doi.org/10.1038/ncomms10980"""
    b0, b1 = _common_region(components[0], components[1], shift, otf)

    b0_spatial = ifft2(b0)
    b1_spatial = ifft2(b1)

    p1 = np.sum(b1_spatial * np.conjugate(b0_spatial)) / np.sum(np.square(np.abs(b0_spatial)))

    return np.abs(p1)


def estimate_phase_offset(fft_images: np.ndarray, shift) -> float:
    """Adapted implementation of the method proposed in: https://doi.org/10.1038/srep37149"""

    size = fft_images.shape[-1]

    components = np.einsum('ijk,klm->ijlm', _SEPARATION_MATRICES, fft_images)

    padded_components = np.zeros((len(_XX), 3, size * 2, size * 2), dtype=np.complex128)
    X, Y = component_padding_matrices(size)
    padded_components[..., Y, X] = components

    I_0 = ifft2(padded_components[:, 0]).real

    padded_components[:, 1] = fft2(ifft2(padded_components[:, 1]) * fourier_shift_matrix(size * 2, size * 2, shift[1], shift[0]))
    padded_components[:, 2] = fft2(ifft2(padded_components[:, 2]) * fourier_shift_matrix(size * 2, size * 2, -shift[1], -shift[0]))
    fft_result = np.sum(padded_components, 1)
    I_SIM_j = ifft2(fft_result).real

    nominator = np.sum(I_SIM_j * I_0, axis=(-2, -1))
    denominator = np.sum(I_SIM_j, axis=(-2, -1)) * np.sum(I_0, axis=(-2, -1))

    yy = nominator / denominator
    yy -= np.mean(yy)  # normalize to remove any y offset

    # noinspection PyTupleAssignmentBalance
    (amplitude, phase), _ = optimize.curve_fit(_sine, _XX, yy, [3 * np.std(yy) / 2 ** 0.5, _XX[np.argmax(yy)]])

    return (np.pi - np.copysign(np.pi / 2, amplitude) - phase) % (2 * np.pi)


def _deconvolve_richardson_lucy(img: np.ndarray, iterations: int, otf: OTF):
    otf_mult = otf.draw(img.shape[-1])
    otf_mult_flipped = np.flip(otf_mult)

    deconv_img = np.copy(img)
    next_img = np.copy(img)

    for i in range(iterations):
        next_img = ifft2(fft2(next_img) * otf_mult)
        next_img = img / next_img
        next_img = ifft2(fft2(next_img) * otf_mult_flipped)
        next_img *= deconv_img

        deconv_img = np.real(next_img)

    return deconv_img


def _grid(zero_component: np.ndarray):
    size = zero_component.shape[-1]
    x, y = np.meshgrid(np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]),
                       np.hstack([np.arange(size // 2), np.arange(-size // 2, 0)]))
    return size, x, y


def _common_region(zero_component: np.ndarray, first_component: np.ndarray, shift, otf: OTF,
                   weight_limit: float = 0.05, min_distance: float = 0.15):
    size, x, y = _grid(zero_component)

    weight0 = otf.draw(size)
    weight1 = otf.draw(size, shift[1], shift[0])

    c0 = np.copy(zero_component)
    c1 = fft2(ifft2(first_component) * fourier_shift_matrix(size, size, shift[1], shift[0]))

    ratio = np.hypot(x, y) / np.hypot(shift[0], shift[1])
    mask = (weight0 > weight_limit) & (weight1 > weight_limit) & (ratio > min_distance) & (ratio < 1 - min_distance)

    c0 = np.where(mask, np.divide(c0, weight0, where=mask), 0)
    c1 = np.where(mask, np.divide(c1, weight0, where=mask), 0)

    return c0, c1


def _sine(x, amplitude, phase):
    return np.sin(x + phase) * amplitude

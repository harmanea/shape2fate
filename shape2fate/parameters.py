from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionParameters:
    """Acquisition parameters for a microscopy image dataset.

    Attributes:
        image_size: Size of one side of the field of view in pixels before reconstruction. Images are assumed to be
            square.
        na: Numerical aperture of the microscope.
        pixel_size: Physical pixel size of the camera sensor in micrometers.
        wavelength: Emission wavelength in nanometers.
        frame_rate: FPS of the acquired movie in Hz. ``None`` represents a single snapshot.
    """
    image_size: int
    na: float
    pixel_size: float
    wavelength: float
    frame_rate: float | None = None

    @property
    def lr_size(self) -> int:
        """Low-resolution image size in pixels."""
        return self.image_size

    @property
    def hr_size(self) -> int:
        """High-resolution reconstruction size in pixels."""
        return self.image_size * 2


@dataclass(frozen=True)
class ReconstructionParameters:
    """Parameters for reconstructing SIM images with a generalized Wiener filter.

    Attributes:
        apodization_cutoff: Cutoff frequency for apodization, given as a multiple of the diffraction limit.
        apodization_bend: Slope of the apodization function.
        background_intensity: Base background intensity level to subtract before reconstruction.
        border_fade: Number of pixels over which the image fades to zero at the borders to reduce edge artifacts.
        minimum_peak_distance: Minimum distance at which a peak is searched for during parameter estimation, given as a
            multiple of the diffraction limit.
        rl_deconvolution_iterations: Number of Richardson-Lucy deconvolution preprocessing iterations.
        wiener_parameter: Regularization parameter for Wiener filter reconstruction.
    """
    apodization_cutoff: float = 2.0
    apodization_bend: float = 0.9
    background_intensity: float = 100
    border_fade: int = 15
    minimum_peak_distance: float = 0.75
    rl_deconvolution_iterations: int = 10
    wiener_parameter: float = 0.05

    @property
    def w(self) -> float:
        """Alias for ``wiener_parameter``."""
        return self.wiener_parameter


@dataclass(frozen=True)
class LinkingParameters:
    """Parameters for linking detected features across microscopy frames.

    Attributes:
        birth_death_cost: Cost of initiating or terminating a track.
        edge_removal_cost: Cost of removing a link between tracklets during the untangling step.
        feature_cost_multiplier: Scaling factor for the feature difference component of the matching cost.
        maximum_distance: Maximum distance between detections considered for linking.
        maximum_skipped_frames: Maximum number of additional frames between detections to still consider for linking.
        minimum_length: Minimum inclusive length of a track retained after linking.
    """
    birth_death_cost: float
    edge_removal_cost: float
    feature_cost_multiplier: float
    maximum_distance: float
    maximum_skipped_frames: int = 1
    minimum_length: int = 6

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionParameters:
    """
    Represents the acquisition parameters for a microscopy image dataset.

    :ivar image_size: Size of one side of the field of view in pixels (before reconstruction, assumes square images).
    :ivar na: Numerical aperture of the microscope.
    :ivar pixel_size: Physical pixel size of the camera sensor in micrometers.
    :ivar wavelength: Emission wavelength in nanometers.
    :ivar frame_rate: FPS of the acquired movie in Hz (1/s). Default is None for single snapshots.
    """
    image_size: int
    na: float
    pixel_size: float
    wavelength: float
    frame_rate: float = None

    @property
    def lr_size(self) -> int:
        return self.image_size

    @property
    def hr_size(self) -> int:
        return self.image_size * 2


@dataclass(frozen=True)
class ReconstructionParameters:
    """
    Represents the parameters for reconstructing SIM images using a generalized Wiener filter.

    :ivar apodization_cutoff: Cutoff frequency for apodization, given as a multiple of the diffraction limit.
    :ivar apodization_bend: Slope of the apodization function.
    :ivar background_intensity: Base background intensity level to be subtracted before reconstruction.
    :ivar border_fade: Number of pixels over which the image will be faded to zero at the borders to reduce edge artifacts.
    :ivar minimum_peak_distance: Minimum distance at which a peak is searched for during parameter estimation, given as a multiple of the diffraction limit.
    :ivar rl_deconvolution_iterations: Number of iterations for Richardson-Lucy deconvolution preprocessing.
    :ivar wiener_parameter: Regularization parameter for Wiener filter reconstruction.
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
        return self.wiener_parameter


@dataclass(frozen=True)
class LinkingParameters:
    """
    Represents the parameters for linking detected features across frames in a microscopy image sequence.

    :ivar birth_death_cost: Cost of initiating or terminating a track.
    :ivar edge_removal_cost: Cost of removing a link between tracklets during the untangling step.
    :ivar feature_cost_multiplier: Scaling factor for the feature difference component of the matching cost.
    :ivar maximum_distance: Maximum distance between detections to be considered for linking.
    :ivar maximum_skipped_frames: Maximum number of additional frames between detections to still be considered for linking.
    :ivar minimum_length: Minimum length of a track to be retained after linking.
    """
    birth_death_cost: float
    edge_removal_cost: float
    feature_cost_multiplier: float
    maximum_distance: float
    maximum_skipped_frames: int = 1
    minimum_length: int = 5

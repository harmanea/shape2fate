import torch

from sim import *
from reconstruction import *
from parameter_estimation import *
from parameters import *
from otf import *

if __name__ == '__main__':
    images = ...

    ap = AcquisitionParameters(image_size=..., na=..., pixel_size=..., wavelength=...)
    rp = ReconstructionParameters()

    otf = ModelOTF(ap)

    images = reconstruction_preprocess(images, rp)

    shifts, phase_offsets, amplitudes = estimate_parameters(images[:9], otf, ap, rp)
    check_parameter_estimates(shifts, amplitudes)

    # Single image reconstruction
    result = cpu_reconstruct(images[:9], otf, shifts, phase_offsets, amplitudes, ap, rp)

    # Batch GPU reconstruction
    gpu_recon = GPUReconstruction(otf, shifts, phase_offsets, amplitudes, ap, rp, torch.device('cuda'))
    result = gpu_recon.reconstruct(lr_images, 2)
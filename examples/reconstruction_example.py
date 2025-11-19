import torch

import shape2fate as s2f
import shape2fate.reconstruction
import shape2fate.parameter_estimation
import shape2fate.parameters
import shape2fate.otf


if __name__ == '__main__':
    images = ...

    ap = s2f.parameters.AcquisitionParameters(image_size=..., na=..., pixel_size=..., wavelength=...)
    rp = s2f.parameters.ReconstructionParameters()

    otf = s2f.otf.ModelOTF(ap)

    images = s2f.reconstruction.reconstruction_preprocess(images, rp)

    shifts, phase_offsets, amplitudes = s2f.parameter_estimation.estimate_parameters(images[:9], otf, ap, rp)
    s2f.parameter_estimation.check_parameter_estimates(shifts, amplitudes)

    # Option 1. Single image reconstruction
    result = s2f.reconstruction.cpu_reconstruct(images[:9], otf, shifts, phase_offsets, amplitudes, ap, rp)

    # Option 2. Batch GPU reconstruction
    gpu_recon = s2f.reconstruction.GPUReconstruction(otf, shifts, phase_offsets, amplitudes, ap, rp, torch.device('cuda'))
    result = gpu_recon.reconstruct(images, 2)

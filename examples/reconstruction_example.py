import os
import zipfile
from dataclasses import asdict

import torch
import numpy as np

import shape2fate as s2f
import shape2fate.reconstruction
import shape2fate.parameter_estimation
import shape2fate.parameters
import shape2fate.otf
import shape2fate.utils

DATA_DIR = "./data"
DATA_URL = "https://zenodo.org/api/records/17484958/files/CME%20tracking%20testing.zip/content"
ZIP_PATH = os.path.join(DATA_DIR, "test_data.zip")


if __name__ == "__main__":
    print("\nRunning shape2fate reconstruction example script ...\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")

    print("Downloading test data ... ")
    s2f.utils.download_file(DATA_URL, ZIP_PATH)
    print("DONE")

    print("Unzipping test data ... ", end="", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print("DONE")

    image_path = os.path.join(DATA_DIR, "CME tracking testing", "RPE1_egfpCLCa_012.nd2")
    print(f"Opening raw example movie: {image_path} ... ", end="", flush=True)
    images = s2f.utils.open_image_file(image_path).astype(np.float64)
    images = images.reshape(-1, 3, 512, 3, 512).transpose(0, 1, 3, 2, 4).reshape(-1, 512, 512)
    print("DONE")
    print(f"   ↳ Shape: {images.shape}")

    ap = s2f.parameters.AcquisitionParameters(
        image_size=512,
        na=1.49,
        pixel_size=0.064395,
        wavelength=526,
        frame_rate=1 / 0.194
    )
    print(f"\nAcquisition parameters:")
    for key, value in asdict(ap).items():
        print(f"   {key}={value}")

    rp = s2f.parameters.ReconstructionParameters()
    print(f"\nReconstruction parameters:")
    for key, value in asdict(rp).items():
        print(f"   {key}={value}")

    otf = s2f.otf.ModelOTF(ap)

    print("\nPreprocessing data ... ", end="", flush=True)
    images = s2f.reconstruction.reconstruction_preprocess(images, rp)
    print("DONE")

    print("Estimating parameters ... ", end="", flush=True)
    shifts, phase_offsets, amplitudes = s2f.parameter_estimation.estimate_parameters(images[:9], otf, ap, rp)
    print("DONE")

    print()
    for i, shift, phase_offset, amplitude in zip(range(3), shifts, phase_offsets, amplitudes):
        print(f"Orientation #{i + 1}:")
        print(f"   Frequency: {np.hypot(*shift):.2f}")
        print(f"   Angle: {np.arctan2(*shift):.2f}")
        print(f"   Phase offset: {phase_offset:.2f}")
        print(f"   Amplitude: {amplitude:.2f}")

    print("\nPerforming parameters check ... ", end="", flush=True)
    s2f.parameter_estimation.check_parameter_estimates(shifts, amplitudes)
    print("DONE")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: GPU ({device})")

        print("Reconstructing ... ", end="", flush=True)
        gpu_recon = s2f.reconstruction.GPUReconstruction(otf, shifts, phase_offsets, amplitudes, ap, rp, device)
        result = gpu_recon.reconstruct(images, 2)

    else:
        print("Using device: CPU")

        print("Reconstructing ... ", end="", flush=True)
        result = np.stack([s2f.reconstruction.cpu_reconstruct(images[i * 9:(i + 1) * 9], otf, shifts, phase_offsets, amplitudes, ap, rp)
                           for i in range(len(images) // 9)])

    print("DONE")

    reconstruction_path = os.path.join(DATA_DIR, "reconstruction.tiff")
    print(f"Saving reconstructed images to: {reconstruction_path} ... ", end="", flush=True)
    s2f.utils.save_tiff_file(result, reconstruction_path)
    print("DONE")

    print("\nAll steps completed successfully.")
    print("Results saved to:")
    print(f"  • {reconstruction_path}")

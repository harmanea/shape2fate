import os
import ssl
import zipfile
from urllib.request import Request, urlopen
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
CERT_URL = "https://pki.cesnet.cz/_media/certs/chain-harica-rsa-ov-crosssigned-root.pem"
CERT_PATH = os.path.join(DATA_DIR, "chain-harica-cross.pem")
DATA_URL = "https://shape2fate.utia.cas.cz/files/endocytosis/shape2fate1.0_107.zip"
ZIP_PATH = os.path.join(DATA_DIR, "test_data.zip")

def download_file(url, dest_path, context=None):
    with urlopen(Request(url), timeout=10, context=context) as resp, open(dest_path, "wb") as f:
        if resp.status != 200:
            raise RuntimeError(f"HTTP error: {resp.status}")

        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            f.write(chunk)


if __name__ == "__main__":
    print("\nRunning shape2fate reconstruction example script ...\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")

    print("Downloading SSL certificate ... ", end="", flush=True)
    download_file(CERT_URL, CERT_PATH)
    print("DONE")

    print("Downloading test data ... ", end="", flush=True)
    context = ssl.create_default_context(cafile=CERT_PATH)
    download_file(DATA_URL, ZIP_PATH, context)
    print("DONE")

    print("Unzipping test data ... ", end="", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print("DONE")

    image_path = os.path.join(DATA_DIR, "data", "Oxford", "20240703", "SHSY5Y_RUSHLAMP_CLCSNAP_107_subset.dv")
    print(f"Opening raw example movie: {image_path} ... ", end="", flush=True)
    images = s2f.utils.open_image_file(image_path).astype(np.float64)
    print("DONE")

    ap = s2f.parameters.AcquisitionParameters(
        image_size=512,
        na=1.5,
        pixel_size=0.0791,
        wavelength=603
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

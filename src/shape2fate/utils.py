import os
import warnings

import numpy as np
import mrcfile
import nd2
import czifile
from PIL import Image

def _open_mrc_file(name: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mrcfile.open(name, permissive=True) as mrc:
            return np.flip(mrc.data, -2)  # Flip y-axis because mrc origin is in the bottom left


def _open_tiff_file(name: str) -> np.ndarray:
    img = Image.open(name)

    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames).squeeze()

def _open_czi_file(name: str) -> np.ndarray:
    return czifile.imread(name).squeeze()


def _open_nd2_file(name: str) -> np.ndarray:
    return nd2.imread(name)

def open_image_file(name: str) -> np.ndarray:
    root, ext = os.path.splitext(name)

    match ext:
        case '.tiff' | '.tif':
            return _open_tiff_file(name)
        case '.mrc' | '.dv' | '.otf':
            return _open_mrc_file(name)
        case '.czi':
            return _open_czi_file(name)
        case '.nd2':
            return _open_nd2_file(name)
        case _:
            raise ValueError(f'"{ext}" is an unrecognized file extension')
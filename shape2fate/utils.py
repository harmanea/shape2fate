import os
import warnings
import importlib

import numpy as np


def _require_import(module_name: str, package_name: str = None):
    if package_name is None:
        package_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        warnings.warn(
            f"Optional IO dependency '{package_name}' is missing. "
            "Install it manually or with `pip install shape2fate[io]`.")
        raise


def _open_mrc_file(name: str) -> np.ndarray:
    mrcfile = _require_import("mrcfile")

    with warnings.catch_warnings():
        # mrcfile can emit warnings for slightly malformed headers; ignore them here
        warnings.simplefilter("ignore")
        with mrcfile.open(name, permissive=True) as mrc:
            return np.flip(mrc.data, -2)  # Flip y-axis because mrc origin is in the bottom left


def _open_tiff_file(name: str) -> np.ndarray:
    Image = _require_import("PIL.Image", "Pillow (PIL)")

    img = Image.open(name)

    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames).squeeze()


def _open_czi_file(name: str) -> np.ndarray:
    czifile = _require_import("czifile")
    return czifile.imread(name).squeeze()


def _open_nd2_file(name: str) -> np.ndarray:
    nd2 = _require_import("nd2")
    return nd2.imread(name)


def open_image_file(name: str) -> np.ndarray:
    _, ext = os.path.splitext(name)

    match ext.lower():
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

import importlib
import os
import warnings
from ssl import SSLContext
from types import ModuleType
from urllib.request import Request, urlopen

import numpy as np
import tqdm

from .types import PathLike


def _require_import(module_name: str, package_name: str | None = None) -> ModuleType:
    if package_name is None:
        package_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        warnings.warn(
            f"Optional IO dependency '{package_name}' is missing. "
            "Install it manually or with `pip install shape2fate[io]`.")
        raise


def _open_mrc_file(name: PathLike) -> np.ndarray:
    mrcfile = _require_import("mrcfile")

    with warnings.catch_warnings():
        # mrcfile can emit warnings for slightly malformed headers; ignore them here
        warnings.simplefilter("ignore")
        with mrcfile.open(name, permissive=True) as mrc:
            return np.flip(mrc.data, -2)  # Flip y-axis because mrc origin is in the bottom left


def _open_tiff_file(name: PathLike) -> np.ndarray:
    Image = _require_import("PIL.Image", "Pillow (PIL)")

    img = Image.open(name)

    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))

    return np.array(frames).squeeze()


def _open_czi_file(name: PathLike) -> np.ndarray:
    czifile = _require_import("czifile")
    return czifile.imread(name).squeeze()


def _open_nd2_file(name: PathLike) -> np.ndarray:
    nd2 = _require_import("nd2")
    return nd2.imread(name)


def open_image_file(name: PathLike) -> np.ndarray:
    """Open a microscopy image file.

    Args:
        name: Path to an image file. Supported extensions are TIFF, MRC, DV, OTF, CZI, and ND2.

    Returns:
        Image data as a NumPy array.

    Raises:
        ValueError: If the file extension is not supported.
        ImportError: If the required optional I/O dependency is missing.
    """
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


def save_tiff_file(images: np.ndarray, name: PathLike) -> None:
    """Save an image or image stack as a TIFF file.

    Args:
        images: Image or image stack to save.
        name: Output TIFF file path.

    Raises:
        ImportError: If Pillow is not installed.
    """
    Image = _require_import("PIL.Image", "Pillow (PIL)")

    if images.ndim == 2:
        Image.fromarray(images).save(name, format="TIFF")

    else:
        pil_images = [Image.fromarray(img) for img in images]
        pil_images[0].save(name, format="TIFF", save_all=True, append_images=pil_images[1:])


def download_file(url: str, dest_path: PathLike, context: SSLContext | None = None, chunk_size: int = 8192) -> None:
    """Download a URL to a local file with a progress bar.

    Args:
        url: URL to download.
        dest_path: Destination file path.
        context: Optional SSL context passed to ``urlopen``.
        chunk_size: Number of bytes to read per response chunk.

    Raises:
        RuntimeError: If the server responds with a non-200 HTTP status.
    """
    with urlopen(Request(url), timeout=10, context=context) as resp, open(dest_path, "wb") as f:
        if resp.status != 200:
            raise RuntimeError(f"HTTP error: {resp.status}")

        total = resp.getheader("Content-Length")
        total = int(total) if total is not None else None

        with tqdm.tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(dest_path)
        ) as pbar:
            for chunk in iter(lambda: resp.read(chunk_size), b""):
                f.write(chunk)
                pbar.update(len(chunk))

import numpy as np
import pandas as pd
import cv2 as cv
import torch

from skimage import feature


def generate_exo_detections(model: torch.nn.Module, device: torch.device, images: np.ndarray) -> pd.DataFrame:
    model.to(device).eval()

    x_coords, y_coords, classes, frames = [], [], [], []
    yy, xx = np.mgrid[:images.shape[-2], :images.shape[-1]]

    for f, img in enumerate(images.copy()):
        with torch.no_grad():
            torch_image = torch.from_numpy(normalize_img(img)).to(device, torch.float32).expand(1, 1, -1, -1)
            predictions = torch.sigmoid(model(torch_image)).squeeze().cpu().numpy()

        for si, si_predictions in enumerate(predictions):
            n_labels, labels = cv.connectedComponents((si_predictions > 0.1).astype(np.uint8))

            for l in range(1, n_labels):
                mask = labels == l

                if np.any(si_predictions[mask] > 0.4 + 0.1 * si):
                    frames.append(f)
                    y_coords.append(yy[mask].mean())
                    x_coords.append(xx[mask].mean())
                    classes.append(si)

    data = {
        'x': np.array(x_coords),
        'y': np.array(y_coords),
        'si': np.array(classes),
        'frame': np.array(frames)
    }

    return pd.DataFrame(data)


def generate_ccp_detections(model: torch.nn.Module, device: torch.device, images: np.ndarray) -> pd.DataFrame:
    model.to(device).eval()

    x_coords, y_coords, classes, frames = [], [], [], []

    for f, img in enumerate(images.copy()):
        with torch.no_grad():
            torch_image = torch.from_numpy(normalize_img(img)).to(device, torch.float32).expand(1, 1, -1, -1)
            predictions = torch.sigmoid(model(torch_image)).squeeze().cpu().numpy()

        peaks = feature.peak_local_max(predictions, threshold_abs=0.1)
        peak_values = predictions[tuple(peaks.T)]

        detections, detection_classes = _merge_adjacent_peaks(peaks, peak_values)

        x_coords.append(detections[:, 1])
        y_coords.append(detections[:, 0])
        classes.append(detection_classes)
        frames.append(np.full(len(detections), f))

    data = {
        'x': np.concatenate(x_coords),
        'y': np.concatenate(y_coords),
        'si': np.concatenate(classes),
        'frame': np.concatenate(frames)
    }

    return pd.DataFrame(data)


def _merge_adjacent_peaks(peaks: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find clusters of detections whose integer coordinates are directly next to each other.
    Multiple such cases are collapsed into a single cluster.
    """

    if len(peaks) == 0:
        return peaks.astype(np.float64), values

    ys = peaks[:, 0].astype(int)
    xs = peaks[:, 1].astype(int)

    y0 = ys.min()
    y1 = ys.max()
    x0 = xs.min()
    x1 = xs.max()

    h = y1 - y0 + 1
    w = x1 - x0 + 1

    mask = np.zeros((h, w), dtype=np.uint8)
    ys_local = ys - y0
    xs_local = xs - x0
    mask[ys_local, xs_local] = 1

    _, labels_img = cv.connectedComponents(mask, connectivity=4)
    det_labels = labels_img[ys_local, xs_local]

    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(det_labels):
        clusters.setdefault(int(lab), []).append(idx)

    merged_coords = []
    merged_values = []
    clustered_mask = np.zeros(len(peaks), dtype=bool)

    for l, inds in clusters.items():
        if l == 0 or len(inds) < 2:
            continue
        inds_arr = np.asarray(inds, dtype=int)
        merged_coords.append(peaks[inds_arr].mean(axis=0))
        merged_values.append(values[inds_arr[0]])
        clustered_mask[inds_arr] = True

    if merged_coords:
        merged_coords = np.vstack(merged_coords)
        merged_values = np.asarray(merged_values)
        out_coords = np.concatenate([merged_coords, peaks[~clustered_mask]], axis=0)
        out_values = np.concatenate([merged_values, values[~clustered_mask]], axis=0)
    else:
        out_coords = peaks
        out_values = values

    return out_coords.astype(np.float64), out_values


def normalize_img(img: np.ndarray) -> np.ndarray:
    return (img - np.mean(img)) / np.std(img)

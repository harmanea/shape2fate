import numpy as np
import pandas as pd
import cv2 as cv
import torch

from scipy import spatial
from skimage import feature


def generate_exo_detections(model: torch.nn.Module, device: torch.device, images: np.ndarray) -> pd.DataFrame:
    model.to(device).eval()

    x_coords, y_coords, classes, frames = [], [], [], []
    yy, xx = np.mgrid[:images.shape[-2], :images.shape[-1]]

    for f, img in enumerate(images.copy()):
        with torch.no_grad():
            torch_image = torch.from_numpy(normalize_img(img)).to(device, torch.float32).expand(1, 1, -1, -1)
            predictions = torch.sigmoid(model(torch_image)).squeeze().cpu().numpy()

        for cls, cls_predictions in enumerate(predictions):
            n_labels, labels = cv.connectedComponents((cls_predictions > 0.1).astype(np.uint8))

            for l in range(1, n_labels):
                mask = labels == l

                if np.any(cls_predictions[mask] > 0.4 + 0.1 * cls):
                    frames.append(f)
                    y_coords.append(yy[mask].mean())
                    x_coords.append(xx[mask].mean())
                    classes.append(cls)

    data = {
        'x': np.array(x_coords),
        'y': np.array(y_coords),
        'cls': np.array(classes),
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

        clusters = _find_clusters(peaks)
        detections, detection_classes = _merge_clusters(clusters, peaks.astype(np.float64), peak_values)

        x_coords.append(detections[:, 1])
        y_coords.append(detections[:, 0])
        classes.append(detection_classes)
        frames.append(np.full(len(detections), f))

    data = {
        'x': np.concatenate(x_coords),
        'y': np.concatenate(y_coords),
        'cls': np.concatenate(classes),
        'frame': np.concatenate(frames)
    }

    return pd.DataFrame(data)


def _find_clusters(detections: np.ndarray) -> list[list[int]]:
    """
    Find clusters of detections whose integer coordinates are directly next to each other.
    Multiple such cases are collapsed into a single cluster.

    :param detections: An array of (y, x) coordinates
    :return: a list of found clusters; each sublist contains indices of detections in that cluster
    """
    distances = spatial.distance.cdist(detections, detections, metric='euclidean')

    clusters: dict[int, set[int]] = {}
    assignments: dict[int, int] = {}

    k = 0
    for i, j in zip(*(distances == 1).nonzero()):
        if i > j:
            continue

        if i in assignments:
            if j in assignments:
                if assignments[i] == assignments[j]:
                    continue

                # Merge clusters
                old_assignment = assignments[j]
                for value in clusters[old_assignment]:
                    clusters[assignments[i]].add(value)
                    assignments[value] = assignments[i]
                clusters.pop(old_assignment)

            else:
                clusters[assignments[i]].add(j)
                assignments[j] = assignments[i]

        elif j in assignments:
            clusters[assignments[j]].add(i)
            assignments[i] = assignments[j]

        else:
            # Add a new cluster
            clusters[k] = {i, j}
            assignments[i] = k
            assignments[j] = k
            k += 1

    return [list(cluster) for cluster in clusters.values()]


def _merge_clusters(clusters: list[list[int]], detections: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if clusters:
        merged_detections = []
        merged_classes = []

        # Keep track of which detections and classes were merged
        merged = np.full(len(detections), False)

        for cluster in clusters:
            merged_detections.append(np.mean(detections[cluster], 0))
            merged_classes.append(classes[cluster[0]])
            merged[cluster] = True

        detections = np.concatenate([np.array(merged_detections), detections[~merged]])
        classes = np.concatenate([np.array(merged_classes), classes[~merged]])

    return detections, classes


def normalize_img(img: np.ndarray) -> np.ndarray:
    return (img - np.mean(img)) / np.std(img)

import os
import sys
import ssl
import zipfile
from urllib.request import Request, urlopen
from dataclasses import asdict

import torch
import numpy as np
import pandas as pd
from scipy import spatial

import shape2fate as s2f
import shape2fate.detection
import shape2fate.linking
import shape2fate.metrics
import shape2fate.models
import shape2fate.parameters
import shape2fate.utils

DATA_DIR = "./data"
CERT_URL = "https://pki.cesnet.cz/_media/certs/chain-harica-rsa-ov-crosssigned-root.pem"
CERT_PATH = os.path.join(DATA_DIR, "chain-harica-cross.pem")
DATA_URL = "https://shape2fate.utia.cas.cz/files/test_data/test_data.zip"
ZIP_PATH = os.path.join(DATA_DIR, "test_data.zip")

MATCHING_THRESHOLD = 5


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
    print("\nRunning shape2fate tracking example script ...\n")

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

    image_path = os.path.join(DATA_DIR, "TIRF488_cam1_1_L.mrc")
    print(f"Opening example movie: {image_path} ... ", end="", flush=True)
    images = s2f.utils.open_image_file(image_path)
    print("DONE")

    print("\nInitializing detection model ... ", end="", flush=True)
    model = s2f.models.UNet(depth=3, start_filters=16, up_mode="nearest")
    print("DONE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(DATA_DIR, "ccp-detector-sandy-wildflower-269.pt")
    print(f"Loading model weights from: {checkpoint_path} ... ", end="", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("DONE")

    print("Running CCP detection on test data ... ", end="", flush=True)
    detections = s2f.detection.generate_ccp_detections(model, device, images)
    print("DONE")

    detections_path = os.path.join(DATA_DIR, "detections.csv")
    print(f"Saving detections to: {detections_path} ... ", end="", flush=True)
    detections.to_csv(detections_path, index=False)
    print("DONE")

    lp = s2f.parameters.LinkingParameters(
        birth_death_cost=5,
        edge_removal_cost=10,
        feature_cost_multiplier=1,
        maximum_distance=7.5,
        maximum_skipped_frames=1,
        minimum_length=6
    )
    print(f"\nLinking parameters:")
    for key, value in asdict(lp).items():
        print(f"   {key}={value:.1f}")

    def distance_function(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
        a_xy = np.array(a[["x", "y"]])
        b_xy = np.array(b[["x", "y"]])
        euclidean_dist = spatial.distance.cdist(a_xy, b_xy, metric="euclidean")

        a_cls = np.array(a["cls"])
        b_cls = np.array(b["cls"])
        cls_dist = np.square(a_cls[:, None] - b_cls[None, :])

        return euclidean_dist + lp.feature_cost_multiplier * cls_dist

    print("\nSetting up MIP linking problem ... ", end="", flush=True)
    linking_graph = s2f.linking.LinkingGraph(detections, distance_function, lp.maximum_distance, lp.birth_death_cost)
    print("DONE")
    print(f"   ↳ {linking_graph.solver.NumConstraints()} constraints, {linking_graph.solver.NumVariables()} variables")

    print("Solving (this can take up to several minutes) ... ", end="", flush=True)
    status = linking_graph.solve()
    print("DONE")

    s2f.linking.print_solver_status(status, linking_graph.solver)
    if status != linking_graph.solver.OPTIMAL:
        print("Linking failed. Aborting.")
        sys.exit(1)

    print("Resolving linking ... ", end="", flush=True)
    tracklets = linking_graph.get_result()
    print("DONE")
    print(f"   ↳ {len(tracklets)} tracklets")

    print("Setting up MIP untangling problem ... ", end="", flush=True)
    untangling_graph = s2f.linking.UntanglingGraph(tracklets, lp.edge_removal_cost)
    print("DONE")
    print(f"   ↳ {untangling_graph.solver.NumConstraints()} constraints, {untangling_graph.solver.NumVariables()} variables")

    print("Solving ... ", end="", flush=True)
    status = untangling_graph.solve()
    print("DONE")

    s2f.linking.print_solver_status(status, untangling_graph.solver)
    if status != untangling_graph.solver.OPTIMAL:
        print("Untangling failed. Aborting.")
        sys.exit(1)

    print("Resolving untangling ... ", end="", flush=True)
    trajectories = untangling_graph.get_result(detections)
    print("DONE")

    print("Filtering trajectories ... ", end="", flush=True)
    filtered_trajectories = trajectories.groupby("particle").filter(lambda t: t.frame.count() >= lp.minimum_length)
    print("DONE")
    print(f"   ↳ {trajectories['particle'].nunique()} trajectories filtered to {filtered_trajectories['particle'].nunique()}")

    trajectories_path = os.path.join(DATA_DIR, "trajectories.csv")
    print(f"Saving trajectories to: {trajectories_path} ... ", end="", flush=True)
    filtered_trajectories.to_csv(trajectories_path, index=False)
    print("DONE")

    print("\nSetting up tracking evaluation ... ", end="", flush=True)
    gts = []
    for i in range(3):
        gt_path = os.path.join(DATA_DIR, f"annotations_{i + 1}.csv")
        gt = pd.read_csv(gt_path)
        gt = gt.rename(columns={"track_id": "particle"})
        gt = gt.groupby("particle").filter(lambda t: (512 < t.y.mean() < 768) and (256 < t.x.mean() < 512))
        gts.append(gt)

    eval_trajectories = filtered_trajectories.groupby("particle").filter(lambda t: (512 < t.y.mean() < 768) and (256 < t.x.mean() < 512))
    print("DONE")

    print("Computing MOTA and MOTP ... ", end="", flush=True)
    motas, motps = [], []
    for gt in gts:
        results = s2f.metrics.mot_metrics(gt, eval_trajectories, MATCHING_THRESHOLD)
        motas.append(results["MOTA"])
        motps.append(1 - results["MOTP"])
    print("DONE")
    print(f"   MOTA:      {np.mean(motas):.2f}")
    print(f"   1 - MOTP:  {np.mean(motps):.2f}")

    print("Computing HOTA metrics ... ", end="", flush=True)
    hotas, detas, assas = [], [], []
    for gt in gts:
        results = s2f.metrics.hota(gt, eval_trajectories, MATCHING_THRESHOLD)
        hotas.append(results["HOTA"])
        detas.append(results["DetA"])
        assas.append(results["AssA"])
    print("DONE")
    print(f"   HOTA:      {np.mean(hotas):.2f}")
    print(f"   DetA:      {np.mean(detas):.2f}")
    print(f"   AssA:      {np.mean(assas):.2f}")

    print("Computing μTIOU ... ", end="", flush=True)
    mtious = []
    for gt in gts:
        result = s2f.metrics.mean_temporal_intersection_over_union(gt, eval_trajectories, MATCHING_THRESHOLD)
        mtious.append(result["mean_tiou"])
    print("DONE")
    print(f"   μTIOU:     {np.mean(mtious):.2f}")

    metrics_path = os.path.join(DATA_DIR, "metrics.txt")
    print(f"Saving metrics to: {metrics_path} ... ", end="", flush=True)

    with open(metrics_path, "w") as f:
        f.write(f"MOTA:      {np.mean(motas):.4f}\n")
        f.write(f"1 - MOTP:  {np.mean(motps):.4f}\n")
        f.write(f"HOTA:      {np.mean(hotas):.4f}\n")
        f.write(f"DetA:      {np.mean(detas):.4f}\n")
        f.write(f"AssA:      {np.mean(assas):.4f}\n")
        f.write(f"μTIOU:     {np.mean(mtious):.4f}\n")
    print("DONE")

    print("\nAll steps completed successfully.")
    print("Results saved to:")
    print(f"  • {detections_path}")
    print(f"  • {trajectories_path}")
    print(f"  • {metrics_path}")

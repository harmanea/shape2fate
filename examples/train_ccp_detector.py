import os
import datetime
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm
from skimage import feature
from scipy import spatial

import shape2fate as s2f
import shape2fate.metrics
import shape2fate.models
import shape2fate.synthetic_data


MODEL_ZOO_DIR = "./model_zoo"

MATCHING_THRESHOLD = 5


def detection_metrics(target_coordinates, predicted_coordinates, target_output, predicted_output):
    n_target = len(target_coordinates)
    n_prediction = len(predicted_coordinates)

    distances = spatial.distance.cdist(target_coordinates.astype(np.float32), predicted_coordinates.astype(np.float32))

    row_ind, col_ind = s2f.metrics.linear_assignment(distances, MATCHING_THRESHOLD)
    mask = (row_ind < n_target) & (col_ind < n_prediction)

    tp = np.count_nonzero(mask)
    fp = np.sum((row_ind >= n_target) & (col_ind < n_prediction))
    fn = np.sum((col_ind >= n_prediction) & (row_ind < n_target))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    detection_distance = np.sum(distances[row_ind[mask], col_ind[mask]]) / tp

    predicted_sis = predicted_output[predicted_coordinates[col_ind[mask]][:, 0], predicted_coordinates[col_ind[mask]][:, 1]]
    target_sis = target_output[target_coordinates[row_ind[mask]][:, 0], target_coordinates[row_ind[mask]][:, 1]]
    si_distance = np.mean(np.abs(predicted_sis - target_sis))

    return {"precision": precision, "recall": recall, "f1": f1, "detection distance": detection_distance, "SI distance": si_distance}


if __name__ == "__main__":
    print("\nRunning shape2fate ccp detector training script ...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16
    cycles = 300
    batches_per_cycle = 100
    learning_rate = batch_size * 1e-5
    optimizer_weight_decay = 0.05
    dataloader_num_workers = 4
    
    training_parameters = {
        "batch_size": batch_size,
        "cycles": cycles,
        "batches_per_cycle": batches_per_cycle,
        "learning_rate": learning_rate,
        "optimizer_weight_decay": optimizer_weight_decay,
        "dataloader_num_workers": dataloader_num_workers,
    }
    print("\nTraining parameters:")
    for key, value in training_parameters.items():
        print(f"   {key}={value}")

    depth = 3
    start_filters = 16
    up_mode = "nearest"

    model_parameters = {
        "depth": depth,
        "start_filters": start_filters,
        "up_mode": up_mode,
    }
    print("\nModel parameters:")
    for key, value in model_parameters.items():
        print(f"   {key}={value}")

    print("\nInitializing synthetic CCP dataset ... ", end="", flush=True)
    dataset = s2f.synthetic_data.SyntheticCCPDataset()
    print("DONE")

    print("Initializing dataloader ... ", end="", flush=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )
    print("DONE")

    print("Initializing model ... ", end="", flush=True)
    model = s2f.models.UNet(depth=depth, start_filters=start_filters, up_mode=up_mode).to(device)
    print("DONE")

    print("Initializing optimizer and loss ... ", end="", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=optimizer_weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    print("DONE")

    losses = []
    metrics = defaultdict(list)

    run_id = f"ccp-detector-{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}"

    print(f"\nStarting training [{run_id}] ...")
    for cycle in (pbar := tqdm.trange(cycles)):
        model.train()
        cycle_loss = 0

        for X, y in itertools.islice(dataloader, batches_per_cycle):
            X, y = X.to(device)[:, None].float(), y.to(device)[:, None].float()

            optimizer.zero_grad(set_to_none=True)
            output = model(X)

            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            cycle_loss += loss.item()

        cycle_loss = cycle_loss / batches_per_cycle
        losses.append(cycle_loss)

        model.eval()
        cycle_metrics = defaultdict(list)

        with torch.no_grad():
            X, y = next(iter(dataloader))
            X, y = X.to(device)[:, None].float(), y.to(device)[:, None].float()

            output = torch.sigmoid(model(X))

            # Compute metrics
            for out, target in zip(output, y):
                out = out.detach().cpu().numpy()[0]
                target = target.detach().cpu().numpy()[0]

                predicted_coordinates = feature.peak_local_max(out, threshold_abs=0.1)
                target_coordinates = feature.peak_local_max(target, threshold_abs=0.1)

                dm = detection_metrics(target_coordinates, predicted_coordinates, target, out)
                for k, v in dm.items():
                    cycle_metrics[k].append(v)

        for k, v in cycle_metrics.items():
            metrics[k].append(np.mean(v))

        metric_summary = {k: values[-1] for k, values in metrics.items()}
        pbar.set_postfix({"loss": cycle_loss, **metric_summary})
    print("DONE")

    print("Training finished.")
    print("\nFinal training metrics:")
    print(f"   loss={losses[-1]:.4f}")
    for key, values in metrics.items():
        print(f"   {key}={values[-1]:.4f}")

    model_path = os.path.join(MODEL_ZOO_DIR, f"{run_id}.pt")
    history_path = os.path.join(MODEL_ZOO_DIR, f"{run_id}-history.csv")
    parameters_path = os.path.join(MODEL_ZOO_DIR, f"{run_id}-parameters.txt")

    history = pd.DataFrame({
        "cycle": np.arange(1, len(losses) + 1),
        "loss": losses,
        **metrics,
    })

    print(f"\nSaving training history to: {history_path} ... ", end="", flush=True)
    history.to_csv(history_path, index=False)
    print("DONE")

    print(f"Saving training parameters to: {parameters_path} ... ", end="", flush=True)
    with open(parameters_path, "w", encoding="utf-8") as f:
        f.write("Training parameters:\n")
        for key, value in training_parameters.items():
            f.write(f"{key}={value}\n")

        f.write("\nModel parameters:\n")
        for key, value in model_parameters.items():
            f.write(f"{key}={value}\n")
    print("DONE")

    print(f"Saving model to: {model_path} ... ", end="", flush=True)
    torch.save(model.state_dict(), model_path)
    print("DONE")

    print("\nAll steps completed successfully.")
    print("Results saved to:")
    print(f"  • {history_path}")
    print(f"  • {parameters_path}")
    print(f"  • {model_path}")

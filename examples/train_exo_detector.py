import os
import datetime
import itertools

import numpy as np
import pandas as pd
import torch
import tqdm

import shape2fate as s2f
import shape2fate.models
import shape2fate.synthetic_data

MODEL_ZOO_DIR = "./model_zoo"

if __name__ == "__main__":
    print("\nRunning shape2fate exocytosis detector training script ...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16
    cycles = 100
    batches_per_cycle = 100
    learning_rate = batch_size * 1e-5
    optimizer_weight_decay = 0.05
    dataloader_num_workers = 8

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
    out_channels = 2

    model_parameters = {
        "depth": depth,
        "start_filters": start_filters,
        "up_mode": up_mode,
        "out_channels": out_channels,
    }
    print("\nModel parameters:")
    for key, value in model_parameters.items():
        print(f"   {key}={value}")

    print("\nInitializing synthetic CCP dataset ... ", end="", flush=True)
    dataset = s2f.synthetic_data.SyntheticExocytosisDataset()
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
    model = s2f.models.UNet(depth=depth, start_filters=start_filters, up_mode=up_mode, out_channels=2).to(device)
    print("DONE")

    print("Initializing optimizer and loss ... ", end="", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=optimizer_weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    print("DONE")

    losses = []
    run_id = f"exo-detector-{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}"

    print(f"\nStarting training [{run_id}] ...")
    model.train()
    for cycle in (pbar := tqdm.trange(cycles)):
        cycle_loss = 0

        for X, y in itertools.islice(dataloader, batches_per_cycle):
            X, y = X.to(device)[:, None].float(), y.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            output = model(X)

            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            cycle_loss += loss.item()

        cycle_loss = cycle_loss / batches_per_cycle
        losses.append(cycle_loss)
        pbar.set_postfix({"loss": cycle_loss})
    print("DONE")

    print("Training finished.")
    print("\nFinal training metrics:")
    print(f"   loss={losses[-1]:.4f}")

    model_path = os.path.join(MODEL_ZOO_DIR, f"{run_id}.pt")
    history_path = os.path.join(MODEL_ZOO_DIR, f"{run_id}-history.csv")
    parameters_path = os.path.join(MODEL_ZOO_DIR, f"{run_id}-parameters.txt")

    history = pd.DataFrame({
        "cycle": np.arange(1, len(losses) + 1),
        "loss": losses,
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

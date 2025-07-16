from criterion import Criterion
from dataset import Dataset
from model import Model
import json
from pathlib import Path
import numpy as np
import torch as tt
from time import time
import matplotlib.pyplot as plt

def step(phase):
    # Prepare model for training or evaluation
    if phase == "train":
        model.train()
        tt.set_grad_enabled(True)
    else:
        model.eval()
        tt.set_grad_enabled(False)

    # Initialize loss and time
    loss_step = 0.0
    time_step = time()

    # Iterate over data loader
    accuracy = [0, 0]

    for data in dl[phase]:
        # Move data to device
        image = data["image"].to(device)
        
        transform_data = data["transform_data"].to(device)

        # Get predictions
        prediction = model(image)

        # Calculate loss
        loss = criterion(prediction, transform_data)

        # Update running loss
        loss_step += loss.item()

        if phase == "train":
            # Backward pass
            loss.backward()

            # Update model
            optimizer.step()
            optimizer.zero_grad()

        accuracy[0] += tt.sum(tt.argmax(prediction, dim=1) == tt.argmax(transform_data, dim=1)).item()
        accuracy[1] += len(transform_data)

    # Log step
    loss_step = loss_step / len(dl[phase])
    log[f"loss_{phase}"].append(loss_step)
    log[f"accuracy_{phase}"].append(accuracy[0] / accuracy[1])
    log[f"time_{phase}"].append(time() - time_step)

if __name__ == "__main__":
    # Define which cross-validation folds to use, or use all
    cross_validations = ["00"]

    # Define device
    device = "cuda" if tt.cuda.is_available() else "cpu"

    # Start cross-validation loop
    for cv in cross_validations:

        # Define name of current run
        run = f"cv{cv}"

        # Define input paths
        path_in_dir_script = Path(__file__).parent
        path_in_dir_data = path_in_dir_script / "../data"
        path_in_file_images = path_in_dir_data / "mnist_images.npy"
        path_in_file_split = path_in_dir_data / "datasplit.npz"
        path_in_file_configuration = path_in_dir_script / f"configuration.json"

        # Define output paths
        path_out_dir = path_in_dir_script / "../out"
        path_out_dir.mkdir(exist_ok=True)
        path_out_file_weights = path_out_dir / "weights" / f"{run}.pt"
        path_out_file_weights.parent.mkdir(parents=True, exist_ok=True)
        path_out_file_logs = path_out_dir / "logs" / f"{run}.json"
        path_out_file_logs.parent.mkdir(parents=True, exist_ok=True)
        path_out_file_plots = path_out_dir / "plots" / f"{run}.png"
        path_out_file_plots.parent.mkdir(parents=True, exist_ok=True)

        # Load configuration
        configuration = json.load(open(path_in_file_configuration, "r"))

        # Load data
        images = np.load(path_in_file_images)
        datasplit = np.load(path_in_file_split)

        # Create log and add the current configuration to it
        log = dict(
            name=run,
            epoch="",
            accuracy_train=[],
            accuracy_val=[],
            loss_train=[],
            loss_val=[],
            time_train=[],
            time_val=[],
            saved_weights=[],
            configuration=configuration,
            device=device,
        )

        # Load datasets
        ds_train = Dataset(
            images=images[datasplit[f"train_{cv}"]],
        )
        ds_val = Dataset(
            images=images[datasplit[f"val_{cv}"]],
        )

        # Load dataloaders
        dl_train = tt.utils.data.dataloader.DataLoader(
            ds_train,
            batch_size=configuration["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        dl_val = tt.utils.data.dataloader.DataLoader(
            ds_val,
            batch_size=configuration["batch_size"],
            shuffle=False,
        )
        dl = dict(train=dl_train, val=dl_val)

        # Load model and move it to device
        model = Model().to(device)

        # Load criterion
        criterion = Criterion()

        # Load optimizer
        optimizer = tt.optim.AdamW(
            model.parameters(),
            lr=configuration["lr"],
        )

        # Start training
        print(f"[Training {run}]")
        for epoch in range(configuration["epochs"]):
            # Train
            step("train")

            # Val
            step("val")

            # Save weights if validation loss is lower than previous
            if epoch == 0 or log["loss_val"][-1] < min(log["loss_val"][:-1]):
                tt.save(model.state_dict(), path_out_file_weights)
                log["saved_weights"].append(True)
            else:
                log["saved_weights"].append(False)

            # Print epoch summary
            print(
                f"[Epoch {epoch}: {(log['time_train'][-1] + log['time_val'][-1]):.2f}s] Train loss: {log['loss_train'][-1]:.5f}, Train acc.: {log['accuracy_train'][-1]:.5f}, Val loss: {log['loss_val'][-1]:.5f}, Val acc.: {log['accuracy_val'][-1]:.5f}",
                end="",
            )
            if log["saved_weights"][-1]:
                print(" [saved]")
            else:
                print("")

            # Save log
            json.dump(log, open(path_out_file_logs, "w"), sort_keys=True, indent=4)

            # Plot progress
            plt.figure()
            plt.plot(range(epoch + 1), log["loss_train"], label="train_loss")
            plt.plot(range(epoch + 1), log["loss_val"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training: {run}")
            plt.legend(loc=1)
            plt.savefig(path_out_file_plots)
            plt.close()
        print(f"[Training {run}] done")
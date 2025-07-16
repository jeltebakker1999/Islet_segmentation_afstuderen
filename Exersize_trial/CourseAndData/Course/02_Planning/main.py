# Import necessary libraries
# Import necessary libraries
import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import sys
import os

# Add the path to the 04_Criterion directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../04_Criterion')))

from criterion import Criterion 

def step():
    # Do the trainy/valy stuff
    pass


if __name__ == "__main__":
    # Get arguments from the command line, such as config file, etc.

    # Define device (cpu, cuda, mps, ...)

    # Define input/output paths

    # Define or load configuration

    # Create log and add the current configuration to it

    # Load datasets

    # Load dataloaders

    # Load model

    # Load criterion
    criterion = Criterion()
    # Load optimizer

    # Start training loop
    epochs = 100
    for epoch in range(100):
        # Train model
        step("train")

        # Validate model
        step("val")

        # Save model if best validation loss

        # Save logs

        # Save plots

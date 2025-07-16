import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

"""Define input paths"""
if __name__ == "__main__":
    path_in_dir_script = Path(__file__).resolve().parent
    path_in_dir_data = path_in_dir_script / "../../Data/Processed"
    path_in_file_data_images = path_in_dir_data / "mnist_images.npy"
    path_in_file_data_labels = path_in_dir_data / "mnist_labels.npy"

    """Load data"""
    mnist_images = np.load(path_in_file_data_images)
    mnist_labels = np.load(path_in_file_data_labels)

    """Visualise data"""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(mnist_images[i], cmap="gray")
        ax.set_title(f"Label: {mnist_labels[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(path_in_dir_script / "visualise_data.png", dpi=300, bbox_inches="tight")

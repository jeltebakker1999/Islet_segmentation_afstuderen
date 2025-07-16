import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":

    """Define input paths"""
    path_in_dir_script = Path(__file__).resolve().parent
    path_in_dir_data = path_in_dir_script / "../../Data/Processed"
    path_in_file_data_images = path_in_dir_data / "mnist_images.npy"
    path_in_file_data_labels = path_in_dir_data / "mnist_labels.npy"

    """Load data"""
    mnist_images = np.load(path_in_file_data_images)
    mnist_labels = np.load(path_in_file_data_labels)

    """Analise statistics of the mnist dataset"""
    nbins = 11

    plt.hist(
        mnist_labels, bins=np.arange(nbins) - 0.5, rwidth=0.5, facecolor="#2ab0ff", edgecolor="#169acf", linewidth=0.5
    )
    plt.xticks(range(nbins - 1))
    plt.title(f"Histogram of the N={len(mnist_labels)} labels of the MNIST dataset")
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_in_dir_script / "histogram_labels.png", dpi=300, bbox_inches="tight")

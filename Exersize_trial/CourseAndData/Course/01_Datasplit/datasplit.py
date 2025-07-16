import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    """Define input paths"""
    path_in_dir_script = Path(__file__).resolve().parent
    path_in_dir_data = path_in_dir_script / "../../Data/Processed"
    path_in_file_data_images = path_in_dir_data / "mnist_images_2.npy"
    path_in_file_data_labels = path_in_dir_data / "mnist_labels.npy"

    """Define output paths"""
    path_out_file_datasplit = path_in_dir_script / "datasplit.npz"

    """Load data"""
    mnist_images = np.load(path_in_file_data_images)
    mnist_labels = np.load(path_in_file_data_labels)

    """Split the data into training and testing, while keeping the distribution of the labels"""
    datasplit = dict()
    idx_train, idx_test, _, _ = train_test_split(
        range(len(mnist_labels)),
        mnist_labels,
        test_size=0.2,
        stratify=mnist_labels,
    )
    datasplit["train"] = idx_train
    datasplit["test"] = idx_test

    
    """Split the training data into training and validation for the 5-fold cross-validation"""
    skf_train_val = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (idx_train, idx_va) in enumerate(skf_train_val.split(datasplit["train"], mnist_labels[datasplit["train"]])):
        datasplit[f"train_{i:02d}"] = idx_train
        datasplit[f"val_{i:02d}"] = idx_va
    
    

    """Plot the histogram of the labels of the training and testing data"""
    nbins = 11
    plt.hist(
        [mnist_labels[datasplit["train"]], mnist_labels[datasplit["test"]]],
        bins=np.arange(nbins) - 0.5,
        rwidth=0.5,
        linewidth=0.5,
        label=["Train", "Val"],
        color=["#2ab0ff", "orange"],
    )
    plt.xticks(range(nbins - 1))
    plt.title(f"Histogram of the N={len(mnist_labels)} labels of the MNIST dataset")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_in_dir_script / "histogram_labels_train_test.png", dpi=300, bbox_inches="tight")
    plt.close()


    """Save the datasplit"""
    np.savez(path_out_file_datasplit, **datasplit)

    

    """Print datasplit information"""
    print("Datasplit information:")
    print(f" - Number of samples: {len(mnist_labels)}")
    print(f" - Number of train samples: {len(datasplit['train'])}")
    print(f" - Number of test samples: {len(datasplit['test'])}")
    print(f" - Number of train_0* samples: {len(datasplit['train_00'])}")
    print(f" - Number of val_0* samples: {len(datasplit['val_00'])}")
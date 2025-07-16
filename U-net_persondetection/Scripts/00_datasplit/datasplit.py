import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

if __name__ == "__main__":

    """Define input paths"""
    path_in_dir_script = Path(__file__).resolve().parent
    path_in_dir_data = path_in_dir_script / "../data"
    path_in_file_data_images = path_in_dir_data / "images.npy"
    path_in_file_data_masks = path_in_dir_data / "masks.npy"

    """Define output paths"""
    path_out_file_datasplit = path_in_dir_script / "datasplit.npz"

    """Load data"""
    images = np.load(path_in_file_data_images)  # (N, H, W) or (N, H, W, C)
    masks = np.load(path_in_file_data_masks)  # (N, H, W), values in {0,1,2,3,4,5}

    """Extract Majority Class Per Mask"""
    unique, counts = np.unique(masks, return_counts=True)
    print(f"Unique values in dataset: {dict(zip(unique, counts))}")  # Debugging check

    # Compute the most common class in each mask (excluding background 0)
    def majority_class(mask):
        labels, counts = np.unique(mask[mask > 0], return_counts=True)
        return labels[np.argmax(counts)] if len(labels) > 0 else 0  # Return 0 if empty

    stratify_labels = np.array([majority_class(mask) for mask in masks])

    """Split the data into training and testing while keeping distribution"""
    datasplit = dict()
    idx_train, idx_test = train_test_split(
        np.arange(len(masks)),
        test_size=0.2,
        stratify=stratify_labels,
    )
    datasplit["train"] = idx_train
    datasplit["test"] = idx_test

    """Split training data into 5-fold cross-validation"""
    skf_train_val = StratifiedKFold(n_splits=5, shuffle=True)
    stratify_train_labels = stratify_labels[idx_train]

    for i, (idx_train_sub, idx_val) in enumerate(
        skf_train_val.split(idx_train, stratify_train_labels)
    ):
        datasplit[f"train_{i:02d}"] = idx_train[idx_train_sub]
        datasplit[f"val_{i:02d}"] = idx_train[idx_val]

    """Plot Histogram of Train/Test Class Distribution"""
    plt.hist(
        [stratify_labels[datasplit["train"]], stratify_labels[datasplit["test"]]],
        bins=np.arange(1, 7) - 0.5,
        rwidth=0.5,
        label=["Train", "Test"],
        color=["#2ab0ff", "orange"],
    )
    plt.xticks(range(1, 6), ["Circle", "Square", "Star", "Moon", "Cross"])
    plt.xlabel("Majority Class in Mask")
    plt.ylabel("Count")
    plt.title(f"Histogram of Majority Mask Class Distribution (N={len(masks)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    """Save the datasplit"""
    np.savez(path_out_file_datasplit, **datasplit)

    """Print datasplit information"""
    print("Datasplit information:")
    print(f" - Total samples: {len(masks)}")
    print(f" - Train samples: {len(datasplit['train'])}")
    print(f" - Test samples: {len(datasplit['test'])}")
    print(f" - Train_00 samples: {len(datasplit['train_00'])}")
    print(f" - Val_00 samples: {len(datasplit['val_00'])}")

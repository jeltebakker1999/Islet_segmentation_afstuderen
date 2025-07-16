from pathlib import Path
import torch as tt
import numpy as np
import sys

class Dataset(tt.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
    ):
        """
        For the purpose of this example, I'm assuming that images are numpy arrays of type np.uint8 with the shape (Batch, Height, Width).
        The images are grayscale images, with values in the range [0, 255].

        For pytorch, we need to convert them to tensors, and the images need to have the shape (Batch, Channels, Height, Width). As they are grayscale images, can just add a channel dimension to the tensor.
        """
        self.images = images  # (Batch, Height, Width)
        self.images = self.images[:, None]  # (Batch, 1, Height, Width)
        self.images = tt.from_numpy(self.images)  # Convert to tensor

        """Normalize images. They are of type np.uint8, so we need to convert them to float first"""
        self.images = self.images.float()
        self.images -= tt.min(self.images.flatten(start_dim=1), dim=1).values[:, None, None, None]
        self.images /= tt.max(self.images.flatten(start_dim=1), dim=1).values[:, None, None, None]

        """The labels are numpy arrays of type np.uint8 with the shape (Batch). We need to convert them to tensors too, and we will also convert them from categorical to one-hot encoding, as we will use a cross-entropy loss. The labels will have the shape (Batch, 10)"""
        self.labels = tt.from_numpy(labels)  # (Batch)
        self.labels = tt.nn.functional.one_hot(self.labels.long(), num_classes=10)  # (Batch, 10)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """Get image and label"""
        image = self.images[index]
        label = self.labels[index]

        """Return data"""
        return dict(image=image, label=label)


if __name__ == "__main__":

    """Define input paths"""
    path_in_dir_script = Path(__file__).parent
    path_in_file_images = path_in_dir_script / "../../Data/Processed/mnist_images.npy"
    path_in_file_labels = path_in_dir_script / "../../Data/Processed/mnist_labels.npy"
    path_in_file_datasplit = path_in_dir_script / "../01_Datasplit/datasplit.npz"

    """Load data"""
    images = np.load(path_in_file_images)
    labels = np.load(path_in_file_labels)
    datasplit = np.load(path_in_file_datasplit)

    """Define dataset"""
    ds_train = Dataset(
        images=images[datasplit["train_00"]],
        labels=labels[datasplit["train_00"]],
    )

    """Define dataloader"""
    dl_train = tt.utils.data.DataLoader(ds_train, batch_size=5, shuffle=True)

    """Iterate over dataloader"""
    for data in dl_train:
        image = data["image"]
        label = data["label"]
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")
        print(f"Image min: {image.min()}, Image max: {image.max()}")
        sys.exit()  # Exit after first iteration

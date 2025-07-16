from pathlib import Path
import torch as tt
import numpy as np
import sys
import torchvision.transforms as transforms
import math
import cv2 as cv
   

class Dataset(tt.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = images[:, None]  # Add channel dimension (N, 1, H, W)
        self.images = tt.from_numpy(self.images).float()
        self.images -= self.images.view(self.images.shape[0], -1).min(dim=1)[0][:, None, None, None]
        self.images /= self.images.view(self.images.shape[0], -1).max(dim=1)[0][:, None, None, None]

        self.masks = masks[:, None]  # Add channel dimension (N, 1, H, W)
        self.masks = tt.from_numpy(self.masks).long()
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        return dict(image=image, mask=mask)

if __name__ == "__main__":
    path_in_dir_script = Path(__file__).parent
    path_in_dir_data = path_in_dir_script / "../../data"
    path_in_file_images = path_in_dir_data / "images.npy"
    path_in_file_masks = path_in_dir_data / "masks.npy"
    path_in_file_datasplit = path_in_dir_data / "datasplit.npy"
    images = np.load(path_in_file_images)
    masks = np.load(path_in_file_masks)
    datasplit = np.load(path_in_file_datasplit)
    ds_train = Dataset(images=images[datasplit["train_00"]])
    dl_train = tt.utils.data.DataLoader(ds_train, batch_size=5, shuffle=True)
    for data in dl_train:
        image = data["image"].unsqueeze(1)
        mask = data["mask"]
        print(f"Image shape: {image.shape},  mask shape: {mask.shape}")
        print(f"Image min: {image.min()}, Image max: {image.max()}")
        sys.exit()
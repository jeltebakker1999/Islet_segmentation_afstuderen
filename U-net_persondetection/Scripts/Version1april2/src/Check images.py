import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch as tt
import math



if __name__ == "__main__":
    # Load images
    path_in_dir_script = Path(__file__).parent
    path_in_dir_data = path_in_dir_script / "../data"
    path_in_file_images = path_in_dir_data / "images.npy"
    path_in_file_masks = path_in_dir_data / "masks.npy"
    print(f"Loading images from: {path_in_file_images}")
    print(f"Loading masks from: {path_in_file_masks}")

    
    images = np.load(path_in_file_images)
    masks = np.load(path_in_file_masks)
    index = np.random.randint(0, len(images))
    index = 0
    image = images[index]  
    mask = masks[index]  
    
    
    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original Image\n(index {index})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Transformed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

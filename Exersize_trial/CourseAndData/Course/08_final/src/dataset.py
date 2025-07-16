from pathlib import Path
import torch as tt
import numpy as np
import sys
import torchvision.transforms as transforms
import math
import cv2 as cv

def create_augmentation_matrix(params, center):
    """
    Creates an augmentation matrix given transformation parameters.
    :param params: [theta, transx, transy, scaling]
    :param center: Center of the original image
    :return: Augmentation matrix M
    """
    theta, tx, ty, scale = params
    cx, cy = center
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Scaling transformation matrix
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Rotation matrix about the center
    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Translation matrix to keep the image centered
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    T2 = np.array([
        [1, 0, cx + tx],
        [0, 1, cy + ty],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Final transformation: Translate -> Rotate -> Scale -> Translate Back
    M = T2 @ R @ S @ T1
    return M[:2]  # Convert to 2x3 for cv2

def generate_random_transformation():
    """
    Generates random transformation parameters.
    :return: theta, transx, transy, scaling
    """
    return np.array([
        np.random.uniform(-math.pi / 2, math.pi / 2),  # theta
        np.random.randint(-15, 16),  # tx
        np.random.randint(-15, 16),  # ty
        np.random.uniform(0.5, 2)  # scale
    ])
def place_on_larger_canvas(image, new_size=(128, 128)):
    """
    Places the image at the center of a larger canvas.
    :param image: Image to be placed (numpy array)
    :param new_size: Tuple specifying the new canvas size (height, width)
    :return: Image on a larger canvas
    """
    
    h, w = image.shape[:2]
    new_h, new_w = new_size
    canvas = np.zeros((new_h, new_w), dtype=image.dtype)
    
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    
    canvas[start_y:start_y + h, start_x:start_x + w] = image
    return canvas
def apply_transformation(image):
    """
    Applies the transformation to the image using nearest neighbor interpolation and places it on a larger canvas.
    :param image: Original image (numpy array)
    :return: Transformed image placed on a larger canvas, transformation parameters
    """
    image = image.squeeze(0).numpy()
    h, w = image.shape[:2]
    new_size = (h * 2, w * 2)  # Define the larger canvas size
    center = (new_size[1] // 2, new_size[0] // 2)  # Center of the new canvas

    # Create a larger canvas and place the original image at its center
    large_canvas = place_on_larger_canvas(image, new_size)

    params = generate_random_transformation()
    M = create_augmentation_matrix(params, center)

    # Warp image using OpenCV
    transformed_image = cv.warpAffine(large_canvas, M, (new_size[1], new_size[0]), flags=cv.INTER_NEAREST)

    transformed_image = np.expand_dims(transformed_image, axis=0)  # [1, H, W]
    
    return tt.from_numpy(transformed_image), params
   

class Dataset(tt.utils.data.Dataset):
    def __init__(self, images):
        self.images = images[:, None]
        self.images = tt.from_numpy(self.images)
        self.images = self.images.float()
        self.images -= tt.min(self.images.flatten(start_dim=1), dim=1).values[:, None, None, None]
        self.images /= tt.max(self.images.flatten(start_dim=1), dim=1).values[:, None, None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        transformed_image, transform_data = apply_transformation(image) # 1 means it takes the full transformation, 0 means it is only rotation, 2 means it is only scaling and translation
        return dict(image=transformed_image, transform_data=transform_data)

if __name__ == "__main__":
    path_in_dir_script = Path(__file__).parent
    path_in_file_images = path_in_dir_script / "../../../Data/Processed/mnist_images.npy"
    path_in_file_datasplit = path_in_dir_script / "../../01_Datasplit/datasplit.npz"
    images = np.load(path_in_file_images)
    datasplit = np.load(path_in_file_datasplit)
    ds_train = Dataset(images=images[datasplit["train_00"]])
    dl_train = tt.utils.data.DataLoader(ds_train, batch_size=5, shuffle=True)
    for data in dl_train:
        image = data["image"].unsqueeze(1)
        transform_data = data["transform_data"]
        print(f"Image shape: {image.shape}, Transform data shape: {transform_data.shape}")
        print(f"Image min: {image.min()}, Image max: {image.max()}")
        sys.exit()
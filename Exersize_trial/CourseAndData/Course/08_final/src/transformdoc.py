import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch as tt
import math

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
        np.random.uniform(0.8, 1.2)  # scale
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

if __name__ == "__main__":
    # Load images
    path_in_file_images = Path("C:/Users/bakke/OneDrive/Bureaublad/Afstuderen/Exersize_trial/CourseAndData/Data/Processed/mnist_images.npy")
    print(f"Loading images from: {path_in_file_images}")

    images = np.load(path_in_file_images)
    index = np.random.randint(0, len(images))
    index = 0
    image = images[index]  # Select the first image
    
    if isinstance(image, tt.Tensor):
        image = image.numpy()
    
    transformed_image, transformed_params = apply_transformation(image)
    
    # Plot the images
    print(transformed_params)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Original Image\n(index {index})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image.squeeze(0).numpy(), cmap="gray")
    plt.title("Transformed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

from pathlib import Path
import torch as tt
import numpy as np
import sys
import torchvision.transforms as transforms
import math
import cv2

def transform_imag_translate(image, t_x, t_y):
    """ Translates an image within the fixed-size canvas. """
    height, width = image.shape[:2]

    # Create a blank canvas of the same size
    translated_image = np.zeros_like(image)

    # Compute new placement while ensuring it stays within bounds
    x_start = max(t_x, 0)
    y_start = max(t_y, 0)
    x_end = min(width, width + t_x)
    y_end = min(height, height + t_y)

    orig_x_start = max(-t_x, 0)
    orig_y_start = max(-t_y, 0)
    orig_x_end = min(width, width - t_x)
    orig_y_end = min(height, height - t_y)

    # Apply the translation
    translated_image[y_start:y_end, x_start:x_end] = image[orig_y_start:orig_y_end, orig_x_start:orig_x_end]

    return translated_image


def transform_imag_rotate(image, theta=0):
    """ Rotates an image without changing the size. """
    theta = math.radians(theta)
    height, width = image.shape[:2]
    
    # Get center of the image
    mid_x, mid_y = width // 2, height // 2

    # Create a blank rotated image
    rotated_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            # Compute relative coordinates
            rel_x = x - mid_x
            rel_y = y - mid_y

            # Apply rotation
            new_x = int(mid_x + (rel_x * math.cos(theta) - rel_y * math.sin(theta)))
            new_y = int(mid_y + (rel_x * math.sin(theta) + rel_y * math.cos(theta)))

            # Check bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_image[new_y, new_x] = image[y, x]

    return rotated_image
 

def transform_imag_scale(image, scale=1.0):
    """ Scales an image while keeping it centered within the fixed canvas. """
    height, width = image.shape[:2]

    # Compute new dimensions
    new_width = max(1, int(width * scale))  # Avoid 0-size images
    new_height = max(1, int(height * scale))

    # Resize image using OpenCV
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas (same size as original)
    canvas = np.zeros_like(image)

    # Compute centering offsets
    x_start = max((width - new_width) // 2, 0)
    y_start = max((height - new_height) // 2, 0)

    x_end = min(x_start + new_width, width)
    y_end = min(y_start + new_height, height)

    # Crop the scaled image if it overflows
    img_x_start = max(0, -((width - new_width) // 2))  # Start in scaled image
    img_y_start = max(0, -((height - new_height) // 2))
    img_x_end = img_x_start + (x_end - x_start)
    img_y_end = img_y_start + (y_end - y_start)

    # Place the resized image within bounds
    canvas[y_start:y_end, x_start:x_end] = scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]

    return canvas


    
def Transform_image(image, transphase):
    # If image is a PyTorch tensor, convert to NumPy and squeeze the channel dimension.
    if isinstance(image, tt.Tensor):
        # image originally is [1, H, W]; squeeze to [H, W]
        image = image.squeeze(0).numpy()

    # Define the transformation parameters
    theta = np.random.randint(-180, 181)  # Rotation in degrees
    tx = np.random.randint(-14, 15)         # Translation in x
    ty = np.random.randint(-14, 15)         # Translation in y
    scale = np.random.uniform(0.5, 2)         # Scaling factor

    # Create a fixed-size canvas only once based on the image dimensions
    height, width = image.shape[:2]
    FIXED_SIZE = (height * 3, width * 3)
    canvas = np.zeros(FIXED_SIZE, dtype=np.float32)
    
    # Center the original image on the fixed canvas
    center_x = FIXED_SIZE[1] // 2 - width // 2
    center_y = FIXED_SIZE[0] // 2 - height // 2
    canvas[center_y:center_y+height, center_x:center_x+width] = image

    # Apply transformations depending on transphase
    if transphase == 1:
        transformed_image = transform_imag_scale(canvas, scale)
        transformed_image = transform_imag_rotate(transformed_image, theta)
        transformed_image = transform_imag_translate(transformed_image, tx, ty)
        transform_data = np.array([theta, tx, ty, scale])
    else:
        transformed_image = transform_imag_scale(canvas, scale)
        transformed_image = transform_imag_translate(transformed_image, tx, ty)
        transform_data = np.array([tx, ty, scale])

    # After transformations, add the channel dimension back (shape becomes [1, H, W])
    transformed_image = np.expand_dims(transformed_image, axis=0)
    # Optionally, convert back to a PyTorch tensor:
    transformed_image = tt.from_numpy(transformed_image)

    return transformed_image, transform_data


   

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
        transformed_image, transform_data = Transform_image(image, 0) # 1 means it takes the full transformation, 0 means it takes the translation and scaling
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
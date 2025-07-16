import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

def apply_transform(image):
    # Calculate the new canvas size
    canvas_width = image.shape[2] * 5
    canvas_height = image.shape[1] * 5
    
    # Apply the padding
    transform = transforms.Compose([
        transforms.Pad((canvas_height - image.shape[1]) // 2),
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    padded_image = transform(image)
    
    # Define transformation parameters
    theta = np.random.uniform(-180, 180)  # Rotation in degrees
    tx = np.random.uniform(-0.5, 0.5) 
    ty = np.random.uniform(-0.5, 0.5)
    scale = np.random.uniform(0.8, 1.2)
    
    # Convert the image to a batch of size 1
    padded_image = padded_image.unsqueeze(0)
    
    # Calculate the transformation matrix
    theta_rad = np.deg2rad(theta)
    transform_matrix = np.array([
        [scale * np.cos(theta_rad), -scale * np.sin(theta_rad), tx],
        [scale * np.sin(theta_rad), scale * np.cos(theta_rad), ty]
    ])
    
    # Convert the transformation matrix to a tensor
    transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32)
    transform_matrix = transform_matrix.unsqueeze(0)  # Add batch dimension 
    
    # Create the affine grid
    grid = F.affine_grid(transform_matrix, padded_image.size(), align_corners=False)
    
    # Apply the grid to the image
    transformed_image = F.grid_sample(padded_image, grid, align_corners=False)
    
    # Remove the batch dimension
    transformed_image = transformed_image.squeeze(0)

    return transformed_image,(theta, tx, ty, scale)

# Load an example image (replace with your own image loading code)
image = torch.rand(3, 28, 28)  # Example image with random values

# Apply the combined transformation
transformed_image,Transform_info = apply_transform(image)

# Convert tensors to numpy arrays for plotting
image_np = image.permute(1, 2, 0).numpy()
transformed_image_np = transformed_image.permute(1, 2, 0).numpy()

# Plot the original and transformed images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(transformed_image_np)
axes[1].set_title('Transformed Image')
axes[1].axis('off')

plt.show()

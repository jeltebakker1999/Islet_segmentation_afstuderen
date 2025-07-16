import cv2 as cv
import numpy as np
from pathlib import Path

def apply_transform(image):
    # Calculate the new canvas size
    canvas_width = image.shape[1] * 2
    canvas_height = image.shape[0] * 2
    
    # Apply the padding
    padded_image = cv.copyMakeBorder(image, 
                                     (canvas_height - image.shape[0]) // 2, 
                                     (canvas_height - image.shape[0]) // 2, 
                                     (canvas_width - image.shape[1]) // 2, 
                                     (canvas_width - image.shape[1]) // 2, 
                                     cv.BORDER_CONSTANT, 
                                     value=[0, 0, 0])
    
    # Define transformation parameters
    theta = np.random.uniform(-180, 180)  # Rotation in degrees
    tx = np.random.uniform(-0.3, 0.3) * padded_image.shape[1]  # Translation in x
    ty = np.random.uniform(-0.3, 0.3) * padded_image.shape[0]  # Translation in y
    scale = np.random.uniform(0.8, 1.2)  # Scaling factor
    
    # Calculate the transformation matrix
    M = cv.getRotationMatrix2D((padded_image.shape[1] // 2, padded_image.shape[0] // 2), theta, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply the transformation
    transformed_image = cv.warpAffine(padded_image, M, (padded_image.shape[1], padded_image.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=[0, 0, 0])
    
    # Normalize transformation parameters
    theta_norm = theta / 180.0
    tx_norm = tx / (0.3 * padded_image.shape[1])
    ty_norm = ty / (0.3 * padded_image.shape[0])
    scale_norm = (scale - 0.8) / 0.4

    return transformed_image, np.array([theta_norm, tx_norm, ty_norm, scale_norm])

def save_transformed_images(images, output_file):
    transformed_images = []
    transformed_params = []

    for image in images:
        transformed_image, params = apply_transform(image)
        transformed_images.append(transformed_image)
        transformed_params.append(params)

    transformed_images = np.array(transformed_images)
    transformed_params = np.array(transformed_params)
    print(transformed_images.shape)
    np.save(output_file, transformed_images)
    np.save(output_file.with_name(output_file.stem + "_params.npy"), transformed_params)

if __name__ == "__main__":
    # Use the full path to the images
    path_in_file_images = Path("C:/Users/bakke/OneDrive/Bureaublad/Afstuderen/Exersize_trial/CourseAndData/Data/Processed/mnist_images.npy")
    
    print(f"Loading images from: {path_in_file_images}")

    images = np.load(path_in_file_images)
    print(images.shape)
    output_file = Path("C:/Users/bakke/OneDrive/Bureaublad/Afstuderen/Exersize_trial/CourseAndData/Data/Processed/mnist_transformedimages.npy")
    
    save_transformed_images(images, output_file)

    print("Transformed images and parameters have been saved.")
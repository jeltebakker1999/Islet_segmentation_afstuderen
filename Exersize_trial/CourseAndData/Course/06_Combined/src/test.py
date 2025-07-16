from criterion import Criterion
from dataset import Dataset, transform_imag_translate, transform_imag_scale
from model import Model
import json
from pathlib import Path
import numpy as np
import torch as tt
import matplotlib.pyplot as plt
import random

def invert_transformation(image, tx, ty, scale):
    """
    Inverts the transformation applied to an image.
    The original transformation (for transphase==0) is:
        1. Scale the image (with a factor 'scale')
        2. Translate the image by (tx, ty)
    To invert this, we first undo the translation (using -tx, -ty) and then undo the scaling (using 1/scale).
    """
    # Undo translation: use negative values of tx and ty.
    inv_translated = transform_imag_translate(image, -int(round(tx)), -int(round(ty)))
    # Undo scaling: apply the inverse scaling factor.
    if scale > 0:
        restored = transform_imag_scale(inv_translated, 1.0/scale)
    else:
        restored = inv_translated
    return restored

def test_single_index(index):
    # Get a single sample from the dataset using the random index.
    sample = ds_test[index]
    # 'image' is the transformed image returned by the Dataset.
    # Add a batch dimension and move to device.
    transformed = sample["image"].unsqueeze(0).to(device)  # shape: [1, 1, H, W]
    
    # Pass the transformed image through the model to get predicted parameters.
    model.eval()
    tt.set_grad_enabled(False)
    prediction = model(transformed)  # shape: [1, 3] -> [tx, ty, scale]
    pred_params = prediction[0].cpu().numpy()
    tx_pred, ty_pred, scale_pred = pred_params

    # Get the transformed image as a NumPy array for plotting.
    transformed_img = transformed[0].cpu().squeeze().numpy()

    # Invert the transformation using the predicted parameters.
    restored_img = invert_transformation(transformed_img, tx_pred, ty_pred, scale_pred)
    h, w = restored_img.shape

    # Define crop size
    crop_size = 28

    # Compute center coordinates
    center_x, center_y = w // 2, h // 2

    # Compute crop boundaries
    x_start = center_x - crop_size // 2
    x_end = center_x + crop_size // 2
    y_start = center_y - crop_size // 2
    y_end = center_y + crop_size // 2
    cropped_img = restored_img[y_start:y_end, x_start:x_end]
    # Retrieve the corresponding original image from the full images array.
    original_img = original_test_images[index]
    if original_img.ndim == 3 and original_img.shape[0] == 1:
        original_img = original_img.squeeze(0)

    # Plot the images side by side.
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title(f"Original Image\n(index {index})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(transformed_img, cmap="gray")
    plt.title("Transformed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cropped_img, cmap="gray")
    plt.title("Restored Image\n(using predicted params, cropped to 28x28)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path_out_dir_test / f"comparison_{index}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # Define device (CPU or GPU)
    device = "cuda" if tt.cuda.is_available() else "cpu"

    # Define input paths (adjust as needed)
    path_in_dir_script = Path(__file__).parent
    path_in_dir_data = path_in_dir_script / "../data"
    path_in_file_images = path_in_dir_data / "mnist_images.npy"
    path_in_file_split = path_in_dir_data / "datasplit.npz"
    path_in_file_log = path_in_dir_script / f"../out/logs/cv00.json"
    path_in_file_weights = path_in_dir_script / f"../out/weights/cv00.pt"

    # Define output directory for test results
    path_out_dir_test = path_in_dir_script / "../out/test/cv00"
    path_out_dir_test.mkdir(exist_ok=True, parents=True)

    # Load configuration (from training log)
    log = json.load(open(path_in_file_log, "r"))
    configuration = log["configuration"]

    # Load the full images array and the test indices.
    images = np.load(path_in_file_images)
    datasplit = np.load(path_in_file_split)
    original_test_images = images[datasplit["test"]]

    # Load test Dataset; note: the Dataset returns only the transformed image and transformation data.
    ds_test = Dataset(images=images[datasplit["test"]])

    # Load test DataLoader (not used for this single-sample test, but kept for consistency)
    dl_test = tt.utils.data.dataloader.DataLoader(
        ds_test,
        batch_size=configuration["batch_size"],
        shuffle=False,
    )

    # Load the model and the trained weights, then move the model to the device.
    model = Model()
    model.load_state_dict(tt.load(path_in_file_weights, map_location=device))
    model.to(device)

    # Choose a random index from the test set.
    rand_idx = random.randint(0, len(ds_test) - 1)

    # Optionally, display the original image separately.
    rand_orig_img = original_test_images[rand_idx]
    if rand_orig_img.ndim == 3 and rand_orig_img.shape[0] == 1:
        rand_orig_img = rand_orig_img.squeeze(0)
    plt.figure(figsize=(4, 4))
    plt.imshow(rand_orig_img, cmap="gray")
    plt.title(f"Random Original Image (index {rand_idx})")
    plt.axis("off")
    plt.savefig(path_out_dir_test / f"random_original_{rand_idx}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Run the test for the selected random index.
    print(f"[Testing cv00] Using sample index: {rand_idx}")
    test_single_index(rand_idx)
    print("[Testing cv00] Done!")

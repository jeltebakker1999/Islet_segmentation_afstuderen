from criterion import Criterion
from dataset import Dataset, create_augmentation_matrix
from model import Model
import json
from pathlib import Path
import numpy as np
import torch as tt
import matplotlib.pyplot as plt
import random
import cv2

def Evaluatedata():
    """
    Evaluates the model predictions on the test set.
    
    :return: Accuracy metrics
    """
    theta_correct = 0
    tx_correct = 0
    ty_correct = 0
    scale_correct = 0
    all_correct = 0
    total = len(ds_test)
    for index in range(len(ds_test)):
        sample = ds_test[index]
        transformed = sample["image"].unsqueeze(0).to(device)  
        reference = sample["transform_data"].cpu().numpy() if isinstance(sample["transform_data"], tt.Tensor) else sample["transform_data"]
        model.eval()
        tt.set_grad_enabled(False)
        prediction = model(transformed) 
        pred_params = prediction[0].detach().cpu().numpy()
        theta = abs(reference[0] - pred_params[0]) <= 1
        tx = abs(reference[1] - pred_params[1]) <= 2
        ty = abs(reference[2] - pred_params[2]) <= 2
        scale = abs(reference[3] - pred_params[3]) <= 0.1
        
        theta_correct += theta
        tx_correct += tx
        ty_correct += ty
        scale_correct += scale
        all_correct += theta and tx and ty and scale
    
    print(f"Theta Accuracy: {100 * theta_correct / total:.2f}%")
    print(f"Translation Accuracy: {100 * (tx_correct + ty_correct) / (2 * total):.2f}%")
    print(f"Scaling Accuracy: {100 * scale_correct / total:.2f}%")
    print(f"Overall Accuracy: {100 * all_correct / total:.2f}%")

def invert_transformation(image, params):
    """
    Applies the inverse transformation to restore the image using the predicted parameters.
    :image: Transformed image 
    :params: Transformation parameters (theta, tx, ty, scale)
    :return: Restored image 
    """
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute the inverse transformation parameters
    inv_params = [-params[0], -params[1], -params[2], 1 / params[3]]
    M_inv = create_augmentation_matrix(inv_params, center)
    
    # Apply inverse transformation using OpenCV
    restored_image = cv2.warpAffine(image, M_inv, (w, h), flags=cv2.INTER_NEAREST)
    
    # Convert back to PyTorch tensor with correct shape
    restored_image = np.expand_dims(restored_image, axis=0)  # [1, H, W]
    return tt.from_numpy(restored_image)

def test_single_index(index):
    sample = ds_test[index]
    transformed = sample["image"].unsqueeze(0).to(device)  
    reference = sample["transform_data"].cpu().numpy() if isinstance(sample["transform_data"], tt.Tensor) else sample["transform_data"]

    model.eval()
    tt.set_grad_enabled(False)
    prediction = model(transformed) 
    pred_params = prediction[0].detach().cpu().numpy()

    # Print transformation data
    print("\n[Transformation Data]")
    print(f"Reference: θ={reference[0]:.2f}, Tx={reference[1]:.2f}, Ty={reference[2]:.2f}, Scale={reference[3]:.2f}")
    print(f"Predicted: θ={pred_params[0]:.2f}, Tx={pred_params[1]:.2f}, Ty={pred_params[2]:.2f}, Scale={pred_params[3]:.2f}")

    transformed_img = transformed[0].cpu().squeeze().numpy()
    restored_img = invert_transformation(transformed_img, pred_params)

    original_img = original_test_images[index].squeeze() if original_test_images[index].ndim == 3 else original_test_images[index]

    # Plot the images
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
    plt.imshow(restored_img.squeeze(0), cmap="gray")
    plt.title("Restored Image\n(using predicted params)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path_out_dir_test / f"comparison_{index}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if tt.cuda.is_available() else "cpu"

    path_in_dir_script = Path(__file__).parent
    path_in_dir_data = path_in_dir_script / "../data"
    path_in_file_images = path_in_dir_data / "mnist_images.npy"
    path_in_file_split = path_in_dir_data / "datasplit.npz"
    path_in_file_log = path_in_dir_script / "../out/logs/cv00.json"
    path_in_file_weights = path_in_dir_script / "../out/weights/cv00.pt"

    path_out_dir_test = path_in_dir_script / "../out/test/cv00"
    path_out_dir_test.mkdir(exist_ok=True, parents=True)

    log = json.load(open(path_in_file_log, "r"))
    configuration = log["configuration"]

    images = np.load(path_in_file_images)
    datasplit = np.load(path_in_file_split)
    original_test_images = images[datasplit["test"]]

    ds_test = Dataset(images=images[datasplit["test"]])

    dl_test = tt.utils.data.dataloader.DataLoader(
        ds_test,
        batch_size=configuration["batch_size"],
        shuffle=False,
    )

    model = Model()
    
    model.load_state_dict(tt.load(path_in_file_weights, map_location=device))
    model.to(device)

    # Evaluate predictions
    #reference_list = [sample["transform_data"].cpu().numpy() if isinstance(sample["transform_data"], tt.Tensor) else sample["transform_data"] for sample in ds_test]
    #pred_params_list = [model(sample["image"].unsqueeze(0).to(device))[0].detach().cpu().numpy() for sample in ds_test]
    Evaluatedata()

    rand_idx = random.randint(0, len(ds_test) - 1)

    rand_orig_img = original_test_images[rand_idx].squeeze() if original_test_images[rand_idx].ndim == 3 else original_test_images[rand_idx]
    
    plt.figure(figsize=(4, 4))
    plt.imshow(rand_orig_img, cmap="gray")
    plt.title(f"Random Original Image (index {rand_idx})")
    plt.axis("off")
    plt.savefig(path_out_dir_test / f"random_original_{rand_idx}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Testing cv00] Using sample index: {rand_idx}")
    test_single_index(rand_idx)
    print("[Testing cv00] Done!")

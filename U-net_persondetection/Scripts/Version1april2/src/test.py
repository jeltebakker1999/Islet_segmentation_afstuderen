import json
from pathlib import Path
import numpy as np
import torch as tt
import matplotlib.pyplot as plt
import random
import cv2
import matplotlib.patches as mpatches  # For legend
from criterion import Criterion
from dataset import Dataset
from model import Model

# Define colors for classes (BGR format for OpenCV)
class_colors = {
    1: (255, 0, 0),    # Circle - Red
    2: (0, 255, 0),    # Square - Green
    3: (0, 0, 255),    # Star - Blue
    4: (255, 255, 0),  # Moon - Yellow
    5: (255, 0, 255),  # Cross - Magenta
}

# Define class labels
class_labels = {
    1: "Circle",
    2: "Square",
    3: "Star",
    4: "Moon",
    5: "Cross",
}

if __name__ == "__main__":
    device = "cuda" if tt.cuda.is_available() else "cpu"

    path_in_dir_script = Path(__file__).parent
    path_in_dir_data = path_in_dir_script / "../../data"
    path_in_file_images = path_in_dir_data / "images.npy"
    path_in_file_split = path_in_dir_data / "datasplit.npz"
    path_in_file_weights = path_in_dir_script / "../out/weights/cv00.pt"

    path_out_dir_test = path_in_dir_script / "../out/test/cv00"
    path_out_dir_test.mkdir(exist_ok=True, parents=True)

    # Load images and test split
    images = np.load(path_in_file_images)
    datasplit = np.load(path_in_file_split)

    test_images = images[datasplit["test"]]

    # Select a random image
    rand_idx = random.randint(0, len(test_images) - 1)
    original_image = test_images[rand_idx]

    # Convert grayscale to BGR for visualization
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Load model
    model = Model().to(device)
    model.load_state_dict(tt.load(path_in_file_weights, map_location=device))
    model.eval()
    tt.set_grad_enabled(False)

    # Preprocess image for model input
    input_tensor = tt.tensor(original_image, dtype=tt.float32).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor /= 255.0 

    # Get prediction from model
    with tt.no_grad():
        prediction = model(input_tensor)  
        predicted_mask = tt.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
        print("Predicted classes unique values:", np.unique(predicted_mask))

    # Create an overlay mask
    overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    for class_id, color in class_colors.items():
        overlay[predicted_mask == class_id] = color

    # Blend overlay with original image
    alpha = 0.5  # Transparency factor
    segmented_image = cv2.addWeighted(original_image_bgr, 1 - alpha, overlay, alpha, 0)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)


    # Add legend
    legend_patches = [mpatches.Patch(color=np.array(color) / 255.0, label=class_labels[class_id])
                      for class_id, color in class_colors.items()]
    
    # Display image
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image[:, :, ::-1])  # Convert BGR to RGB for Matplotlib
    plt.title(f"Predicted Mask (Index {rand_idx})")
    plt.axis("off")
    plt.legend(handles=legend_patches, loc="upper right", fontsize=10, frameon=True)

    # Save output
    output_path = path_out_dir_test / f"predicted_annotated_{rand_idx}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Testing cv00] Using sample index: {rand_idx}")
    print("[Testing cv00] Done!")
    print("Prediction shape:", prediction.shape)  # Should be (1, num_classes, H, W)
    print("Max value in prediction:", prediction.max().item())
    print("Min value in prediction:", prediction.min().item())
    print("Mean value per class:", prediction.mean(dim=(2, 3)).cpu().numpy())  # Average confidence per class
    model_state_dict = tt.load(path_in_file_weights, map_location=device)
    print("Loaded model keys:", model_state_dict.keys())
    print("First layer weights:", model_state_dict[list(model_state_dict.keys())[0]].shape)

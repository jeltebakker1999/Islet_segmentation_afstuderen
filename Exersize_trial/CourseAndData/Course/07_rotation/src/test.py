from criterion import Criterion
from dataset import Dataset, transform_imag_rotate
from model import Model
import json
from pathlib import Path
import numpy as np
import torch as tt
import matplotlib.pyplot as plt
import random





def test_single_index(index):
    sample = ds_test[index]
    transformed = sample["image"].unsqueeze(0).to(device)  
    
    model.eval()
    tt.set_grad_enabled(False)
    prediction = model(transformed) 
    pred_params = prediction[0].cpu().numpy()
    theta = pred_params
    

    transformed_img = transformed[0].cpu().squeeze().numpy()
    restored_img = transform_imag_rotate(transformed_img, -(theta))
    
    original_img = original_test_images[index]
    if original_img.ndim == 3 and original_img.shape[0] == 1:
        original_img = original_img.squeeze(0)

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
    plt.imshow(restored_img, cmap="gray")
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
    path_in_file_log = path_in_dir_script / f"../out/logs/cv00.json"
    path_in_file_weights = path_in_dir_script / f"../out/weights/cv00.pt"
    
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
    
    rand_idx = random.randint(0, len(ds_test) - 1)
    
    rand_orig_img = original_test_images[rand_idx]
    if rand_orig_img.ndim == 3 and rand_orig_img.shape[0] == 1:
        rand_orig_img = rand_orig_img.squeeze(0)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(rand_orig_img, cmap="gray")
    plt.title(f"Random Original Image (index {rand_idx})")
    plt.axis("off")
    plt.savefig(path_out_dir_test / f"random_original_{rand_idx}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"[Testing cv00] Using sample index: {rand_idx}")
    test_single_index(rand_idx)
    print("[Testing cv00] Done!")

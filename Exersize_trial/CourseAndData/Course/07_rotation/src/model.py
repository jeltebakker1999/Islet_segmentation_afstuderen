import torch as tt
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, input_shape=(1, 140, 140)):
        super().__init__()
        # Define the localization network (convolutional layers)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Dynamically calculate the flattened feature size
        with tt.no_grad():
            dummy_input = tt.zeros(1, *input_shape)
            loc_output = self.localization(dummy_input)
            self.flattened_size = loc_output.view(1, -1).shape[1]
        
        # Fully connected layers to predict the 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(3136, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        # Initialize the final FC layer for an identity transformation.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(tt.tensor([1, 0, 0, 0, 1, 0], dtype=tt.float32))

    def forward(self, x):
        batch_size = x.size(0)
        features = self.localization(x)
        features = features.view(batch_size, -1)
        theta = self.fc_loc(features)  # Affine parameters
        theta = theta.view(-1, 2, 3)
        
        # Create an affine grid and sample the input image accordingly.
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        return x_transformed

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial Transformer Network
        self.stn = SpatialTransformer(input_shape=(1, 140, 140))
        
        # Convolutional blocks for feature extraction
        self.conv_block_00 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_03 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final fully connected layer now outputs a single theta value (in degrees)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # Apply the spatial transformer first.
        x = self.stn(x)
        
        # Pass the transformed image through the convolutional blocks.
        x = self.conv_block_00(x)
        x = self.conv_block_01(x)
        x = self.conv_block_02(x)
        x = self.conv_block_03(x)
        
        # Global average pooling and flattening.
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Predict theta in degrees.
        out = self.fc(x)
        return out

if __name__ == "__main__":
    net = Model()
    data = tt.rand(3, 1, 140, 140)  # Batch of 3 images
    output = net(data)
    print("Predicted theta (degrees):", output)

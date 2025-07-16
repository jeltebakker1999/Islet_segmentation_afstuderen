import torch as tt
import torch.nn as nn
import torch.nn.functional as ff

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional blocks (same as before)
        self.conv_block_00 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv_block_03 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.conv_block_04 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        # Global average pooling to compress the spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Single linear layer: mapping from 512 features to 3 outputs
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        hidden = self.conv_block_00(x)
        hidden = self.conv_block_01(hidden)
        hidden = self.conv_block_02(hidden)
        hidden = self.conv_block_03(hidden)
        hidden = self.conv_block_04(hidden)

        hidden = self.global_avg_pool(hidden)
        hidden = hidden.view(hidden.size(0), -1)  # Flatten features
        
        out = self.fc(hidden)  # Output tensor with three values

        # Split the output into translation and scaling components
        tx = out[:, 0]
        ty = out[:, 1]
        scaling = ff.softplus(out[:, 2])  # Ensure scaling is positive
        
        return tt.stack([tx, ty, scaling], dim=1)

if __name__ == "__main__":
    net = Model()
    data = tt.rand(3, 1, 140, 140)  # Batch of 3 images
    output = net.forward(data)
    print(output)  # Outputs a tensor of shape (3, 3)
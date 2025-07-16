import torch as tt
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial Transformer Network
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
        
  
        
        # **Separate Fully Connected Layers for Each Transformation Component**
        self.fc_data = nn.Sequential(
            nn.Linear(12544, 128),
            nn.ReLU(),
            nn.Linear(128, 4) 
        )


    def forward(self, x):
        
        
        # Pass the transformed image through the convolutional blocks.
        x = self.conv_block_00(x)
        x = self.conv_block_01(x)
       

 

        x = x.view(x.size(0), -1)

        
        
        # **Predict Separate Transformation Components**
        out = self.fc_data(x)  # Scalar value for each batch element
        
        
        return out

if __name__ == "__main__":
    net = Model()
    data = tt.rand(3, 1, 140, 140)  # Batch of 3 images
    output = net(data)
    print("Predicted [theta, tx, ty, scale]:\n", output)

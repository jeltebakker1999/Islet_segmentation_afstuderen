import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the layers
        self.conv_block_00 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.dense = nn.Linear(288, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.conv_block_00(x)
        hidden = self.maxpool(hidden)

        hidden = self.conv_block_01(hidden)
        hidden = self.maxpool(hidden)

        hidden = self.conv_block_02(hidden)
        hidden = self.maxpool(hidden)

        hidden = hidden.view(hidden.size(0), -1)  # Flatten for fully connected layer
        hidden = self.dense(hidden)  # Fully connected layer
        out = self.final_activation(hidden)  # Softmax activation
        return out


if __name__ == "__main__":

    net = Model()
    data = tt.rand(3, 1, 28, 28)
    x = net.forward(data)
    print(x)  # 3x10 tensor
    print(x.sum(dim=1))  # 3x1 tensor with values of 1 (sum of probabilities is 1)

import torch.nn as nn
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, prediction, reference):
        loss = F.mse_loss(prediction, reference)
        return loss
import torch as tt
import torch.nn as nn
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, prediction, mask):
        # Ensure mask is of the correct shape [batch_size, height, width]
        mask = mask.squeeze(1)  # Remove any singleton dimensions, if needed

        # Ensure the mask is of type LongTensor for CrossEntropyLoss
        mask = mask.long()

        # CrossEntropyLoss expects raw logits, no need to apply softmax
        loss = F.cross_entropy(prediction, mask)
        return loss

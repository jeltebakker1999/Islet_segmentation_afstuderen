import torch as tt
import torch.nn as nn
import math

class Criterion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, prediction, reference):
        # Multiply by (pi/180) to convert from degrees to radians.
        conversion_factor = math.pi / 180.0
        prediction_rad = prediction * conversion_factor
        reference_rad = reference * conversion_factor
        
        # Compute cosine-based loss
        loss = 1 - tt.cos(prediction_rad - reference_rad)
        
        
        return loss.mean()
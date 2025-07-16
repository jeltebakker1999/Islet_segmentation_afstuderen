import torch as tt
import torch.nn as nn
import math
class Criterion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, prediction, reference):
        
        prediction_rad = prediction[:, 0] 
        reference_rad = reference[:, 0] 

        # Compute individual loss components
        
        loss_translation = (prediction[:, 2] - reference[:, 2])**2 + (prediction[:,1]- reference[:, 1])**2
        loss_scale = (prediction[:, 3] - reference[:, 3])**2
        losstheta = 1 - tt.cos(prediction_rad - reference_rad)

        #weights losses
        w_rot = 10
        w_trans = 1
        w_scale = 2

        # Compute total loss
        total_loss = (
            w_rot* losstheta.mean() +

            w_trans * loss_translation.mean() +
            w_scale * loss_scale.mean()
        )
        
        return total_loss
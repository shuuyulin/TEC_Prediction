import torch.nn as nn
import torch

class RMSELoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class multitaskRegresisonLoss(nn.Module):
    def __init__(self, criterion, weight):
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        
    def forward(self,*args): # (y1, y1_hat), (y2, y2_hat) ...
        losses = [self.criterion(*arg) for arg in args]
        return sum([loss * w for loss, w in zip(losses, self.weight)]) / sum(self.weight[:len(args)]), losses
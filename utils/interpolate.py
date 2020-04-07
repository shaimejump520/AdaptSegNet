import torch
import torch.nn.functional as F
import torch.nn as nn

class Interpolate(nn.Module):
    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode
        self.align_corners =align_corners
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x
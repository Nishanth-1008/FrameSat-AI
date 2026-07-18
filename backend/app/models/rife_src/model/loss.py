import torch.nn as nn

class EPE(nn.Module):
    """Dummy End Point Error class for evaluation."""
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return 0.0

class SOBEL(nn.Module):
    """Dummy Sobel loss class for evaluation."""
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return 0.0

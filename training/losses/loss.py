import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return F.l1_loss(pred, target)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from piq import ssim
            self.use_piq = True
        except ImportError:
            self.use_piq = False
            
    def forward(self, pred, target):
        if self.use_piq:
            from piq import ssim
            # SSIM returns a value between 0 and 1, higher is better
            # We want to minimize loss, so loss = 1 - SSIM
            ssim_val = ssim(pred, target, data_range=1.0)
            return 1.0 - ssim_val
        else:
            raise ImportError("piq package is required for SSIMLoss. Please install it using 'pip install piq'")

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combines L1 and SSIM losses.
        alpha: weight for L1 loss. (1 - alpha) is weight for SSIM loss.
        """
        super().__init__()
        self.alpha = alpha
        self.l1 = L1Loss()
        self.ssim = SSIMLoss()
        
    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        
        combined = (self.alpha * loss_l1) + ((1.0 - self.alpha) * loss_ssim)
        return combined, loss_l1, loss_ssim

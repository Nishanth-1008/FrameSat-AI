import sys
import os
import torch
import numpy as np
from models.base import BaseInterpolator

class RIFEInterpolator(BaseInterpolator):
    """
    Wrapper for the official RIFE v4 model.
    Handles padding requirements and single-channel mapping automatically.
    """
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            elif hasattr(self.model, 'flownet'):
                self.model.flownet.to(self.device)
        return self

    def load_checkpoint(self, path: str):
        """
        Load a single checkpoint file natively.
        Supports both raw state_dicts and wrapped Trainer state dicts.
        """
        from shared.checkpoint import load_checkpoint
        state_dict, _ = load_checkpoint(path, str(self.device))
        
        # Format keys for RIFE flownet
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict[k] = v

        if self.model is None:
            # We import Model directly from our local src directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rife_src_dir = os.path.join(current_dir, "src")
            sys.path.insert(0, rife_src_dir)
            try:
                from RIFE_HDv3 import Model
                self.model = Model()
                self.model.flownet.load_state_dict(new_state_dict, strict=False)
                self.model.eval()
            finally:
                if rife_src_dir in sys.path:
                    sys.path.remove(rife_src_dir)
        else:
            self.model.flownet.load_state_dict(new_state_dict, strict=False)
            self.model.eval()        
    def interpolate(self, t0: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Run inference using the RIFE model.
        """
        h, w = t0.shape
        
        # 1. Convert to RGB (3-channels)
        t0_tensor = torch.from_numpy(t0).float().unsqueeze(0).unsqueeze(0).to(self.device)
        t0_tensor = torch.cat([t0_tensor, t0_tensor, t0_tensor], dim=1)
        
        t2_tensor = torch.from_numpy(t2).float().unsqueeze(0).unsqueeze(0).to(self.device)
        t2_tensor = torch.cat([t2_tensor, t2_tensor, t2_tensor], dim=1)
        
        # 2. Pad to multiple of 128.
        #
        # RIFE's IFNet uses scale_list=[16, 8, 4, 2, 1] and IFBlock.conv0 applies
        # two stride-2 convolutions internally (effective spatial factor of ×4).
        # The maximum downscale encountered before integer convolutions is 16×4=64.
        # To avoid fractional truncation that causes feat/warped_img shape mismatches
        # in the torch.cat inside IFNet.forward(), every spatial dimension must be
        # divisible by 64. We use 128 (= 2×64) as a conservative safe value.
        PAD_MULTIPLE = 128
        ph = int(PAD_MULTIPLE * np.ceil(h / PAD_MULTIPLE))
        pw = int(PAD_MULTIPLE * np.ceil(w / PAD_MULTIPLE))
        pad_h = ph - h
        pad_w = pw - w
        
        if pad_h > 0 or pad_w > 0:
            t0_tensor = torch.nn.functional.pad(t0_tensor, (0, pad_w, 0, pad_h))
            t2_tensor = torch.nn.functional.pad(t2_tensor, (0, pad_w, 0, pad_h))
            
        # 3. Inference
        with torch.no_grad():
            pred_tensor = self.model.inference(t0_tensor, t2_tensor, timestep=0.5)
            
            # Crop padding
            if pad_h > 0 or pad_w > 0:
                pred_tensor = pred_tensor[:, :, :h, :w]
                
        # 4. Convert back to single channel numpy array
        pred_rgb = pred_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_gray = np.mean(pred_rgb, axis=2)
        pred_gray = np.clip(pred_gray, 0.0, 1.0)
        
        return pred_gray

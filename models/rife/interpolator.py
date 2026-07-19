import sys
import os
import torch
import numpy as np
from .base import BaseInterpolator

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

    def load_weights(self, path: str):
        """
        Load weights from the provided directory (expects 'flownet.pkl').
        Supports loading dynamic model architecture from the directory if present.
        """
        # 1. Unload modules to avoid cache conflicts
        for mod in list(sys.modules.keys()):
            if mod.startswith("RIFE_HDv3") or mod.startswith("train_log") or mod.startswith("model.warplayer") or mod.startswith("model.loss"):
                del sys.modules[mod]

        train_log_dir = os.path.abspath(path)
        version_dir = os.path.dirname(train_log_dir)
        
        # 2. Add paths to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rife_src_dir = os.path.join(current_dir, "rife_src")
        
        sys.path.insert(0, rife_src_dir)
        sys.path.insert(0, train_log_dir)
        sys.path.insert(0, version_dir)
        
        try:
            from RIFE_HDv3 import Model
            self.model = Model()
            self.model.load_model(train_log_dir)
            self.model.eval()
        finally:
            # Restore sys.path
            if version_dir in sys.path:
                sys.path.remove(version_dir)
            if train_log_dir in sys.path:
                sys.path.remove(train_log_dir)
            if rife_src_dir in sys.path:
                sys.path.remove(rife_src_dir)

        
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
        
        # 2. Pad to multiple of 32 (required by RIFE)
        tmp = max(32, int(32 * np.ceil(h / 32.0)))
        pad_h = tmp - h
        tmp = max(32, int(32 * np.ceil(w / 32.0)))
        pad_w = tmp - w
        
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

import os
import torch

from .rife_src.RIFE_HDv3 import Model

# Absolute path to this file's directory — used to locate flownet.pkl relative to the module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS_DIR = os.path.join(_HERE, "rife_src", "train_log")


class RIFEModelWrapper:
    """
    Production wrapper for RIFE model.
    Loads pre-trained weights and exposes a clean interpolation API.
    """
    
    def __init__(self, weights_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        self.model.eval()
        
        # Load weights from target directory
        if weights_dir is None:
            # Fallback to backend/app/models/rife_src/train_log
            weights_dir = _DEFAULT_WEIGHTS_DIR
            
        if not os.path.exists(os.path.join(weights_dir, "flownet.pkl")):
            raise FileNotFoundError(f"flownet.pkl weights file not found in: {weights_dir}")
            
        self.model.load_model(weights_dir)
        self.model.eval()
        self.model.device()
        
    def interpolate(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """
        Executes frame interpolation.
        
        Args:
            frame_a: Preprocessed torch.Tensor [1, 3, H, W] on matching device.
            frame_b: Preprocessed torch.Tensor [1, 3, H, W] on matching device.
            
        Returns:
            Interpolated torch.Tensor [1, 3, H, W]
        """
        # Ensure correct device
        frame_a = frame_a.to(self.device)
        frame_b = frame_b.to(self.device)
        
        # RIFE requires inputs to be multiples of 32 in height/width.
        # Since our preprocessor resizes to 384x384 (which is a multiple of 32: 384/32=12),
        # no extra padding/cropping is strictly necessary. We run inference directly.
        with torch.no_grad():
            pred = self.model.inference(frame_a, frame_b, timestep=0.5)
            
        return pred

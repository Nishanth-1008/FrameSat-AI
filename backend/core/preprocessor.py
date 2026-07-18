import cv2
import numpy as np
import torch

class SatellitePreprocessor:
    """
    Production-grade preprocessing engine for satellite weather frames.
    Performs normalization, resizing (384x384), and RIFE-compatible tensor mapping.
    """
    
    def __init__(self, target_height: int = 384, target_width: int = 384):
        self.target_height = target_height
        self.target_width = target_width
        
    def normalize_frame(self, frame: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Rescale raw array values to [0.0, 1.0]."""
        norm_factor = float(max_val - min_val)
        if norm_factor > 0:
            normalized = (frame - min_val) / norm_factor
        else:
            normalized = frame / 255.0
            
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize to the configured resolution (default: 384x384)."""
        if frame.shape[:2] == (self.target_height, self.target_width):
            return frame
            
        resized = cv2.resize(
            frame, 
            (self.target_width, self.target_height), 
            interpolation=cv2.INTER_LINEAR
        )
        return resized

    def to_rife_tensor(self, frame: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Maps a 2D numpy array in range [0, 1] to a RIFE-compatible tensor.
        Shape output: [1, 3, H, W]
        """
        # Convert to tensor: [1, 1, H, W]
        tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
        # Duplicate to 3-channel RGB: [1, 3, H, W]
        tensor = torch.cat([tensor, tensor, tensor], dim=1)
        return tensor

    def preprocess(self, frame: np.ndarray, min_val: float, max_val: float, device: torch.device) -> torch.Tensor:
        """
        Runs the complete preprocessing pipeline.
        
        Args:
            frame: Raw 2D frame array.
            min_val: Min value of the scene sequence.
            max_val: Max value of the scene sequence.
            device: Target torch device (cuda/cpu).
            
        Returns:
            Preprocessed PyTorch tensor ready for model inference.
        """
        norm = self.normalize_frame(frame, min_val, max_val)
        resized = self.resize_frame(norm)
        tensor = self.to_rife_tensor(resized, device)
        return tensor

import os
import cv2
import numpy as np
import torch

class SatellitePostprocessor:
    """
    Production-grade postprocessing engine for satellite weather frames.
    Converts model tensors back into single-channel NumPy arrays and handles saving.
    """
    
    @staticmethod
    def to_numpy_array(tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a RIFE RGB PyTorch tensor [1, 3, H, W] back to a single-channel [0, 1] NumPy array.
        """
        # Remove batch dimension: [3, H, W]
        tensor_np = tensor.squeeze(0).detach().cpu().numpy()
        # Transform transpose to channel last: [H, W, 3]
        tensor_np = tensor_np.transpose(1, 2, 0)
        # Average channels to get grayscale: [H, W]
        gray = np.mean(tensor_np, axis=2)
        # Clip to ensure valid [0.0, 1.0] bounds
        gray = np.clip(gray, 0.0, 1.0)
        return gray
        
    @staticmethod
    def save_as_png(frame: np.ndarray, out_path: str) -> str:
        """
        Saves a [0, 1] float frame as a standard uint8 PNG image with CLAHE contrast enhancement.
        """
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        # Convert float to uint8 [0, 255]
        frame_uint8 = (frame * 255.0).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(frame_uint8)
        
        cv2.imwrite(out_path, enhanced)
        return out_path
        
    def postprocess(self, tensor: torch.Tensor, out_path: str = None) -> dict:
        """
        Runs the full postprocessing pipeline.
        
        Args:
            tensor: Model prediction output tensor.
            out_path: Optional file path to save the generated frame.
            
        Returns:
            Dict containing the prediction array and save status.
        """
        pred_array = self.to_numpy_array(tensor)
        saved_to = None
        
        if out_path:
            saved_to = self.save_as_png(pred_array, out_path)
            
        return {
            "prediction": pred_array,
            "saved_to": saved_to,
            "shape": pred_array.shape,
            "dtype": str(pred_array.dtype)
        }

from abc import ABC, abstractmethod
import cv2
import numpy as np

class BasePreprocessor(ABC):
    """
    Abstract Base Class for all preprocessing strategies.
    """
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a normalized frame array in range [0, 1].
        
        Args:
            frame: Grayscale frame array.
            
        Returns:
            Preprocessed frame array.
        """
        pass

class IdentityPreprocessor(BasePreprocessor):
    """
    Returns the frame unchanged.
    """
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame

class CLAHEPreprocessor(BasePreprocessor):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        # Convert [0, 1] float back to uint8 for cv2.createCLAHE
        frame_uint8 = (frame * 255.0).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        equalized = clahe.apply(frame_uint8)
        
        # Rescale back to [0.0, 1.0] float32
        return equalized.astype(np.float32) / 255.0

class ResizePreprocessor(BasePreprocessor):
    """
    Resizes the frame to the specified dimensions.
    """
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        # cv2.resize expects (width, height)
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return resized

class CompositePreprocessor(BasePreprocessor):
    """
    Chains multiple preprocessors sequentially.
    """
    def __init__(self, preprocessors: list):
        self.preprocessors = preprocessors
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        for p in self.preprocessors:
            frame = p.process(frame)
        return frame

def get_preprocessor_from_config(config: dict) -> BasePreprocessor:
    """
    Factory function to instantiate preprocessors based on configuration dict.
    """
    preprocessors = []
    
    # 1. Check for Resize
    if config.get("resize"):
        h = config["resize"].get("height", 768)
        w = config["resize"].get("width", 768)
        preprocessors.append(ResizePreprocessor(height=h, width=w))
        
    # 2. Check for CLAHE
    if config.get("clahe"):
        clip = config["clahe"].get("clip_limit", 2.0)
        grid = tuple(config["clahe"].get("tile_grid_size", [8, 8]))
        preprocessors.append(CLAHEPreprocessor(clip_limit=clip, tile_grid_size=grid))
        
    if not preprocessors:
        return IdentityPreprocessor()
    elif len(preprocessors) == 1:
        return preprocessors[0]
    else:
        return CompositePreprocessor(preprocessors)

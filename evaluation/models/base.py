from abc import ABC, abstractmethod
import numpy as np

class BaseInterpolator(ABC):
    """
    Abstract base class for all interpolation models evaluated in the framework.
    """
    
    @abstractmethod
    def load_weights(self, path: str):
        """
        Load model weights from the specified path.
        
        Args:
            path: Directory or file path containing the model weights.
        """
        pass
        
    @abstractmethod
    def interpolate(self, t0: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Predict the intermediate frame (t1) given t0 and t2.
        
        Args:
            t0: Normalized input frame array (shape: HxW).
            t2: Normalized input frame array (shape: HxW).
            
        Returns:
            The predicted intermediate frame array (shape: HxW).
        """
        pass

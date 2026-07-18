import os
import h5py
import numpy as np
from .validator import SatelliteValidator, SceneValidationError

class SatelliteDataLoader:
    """
    Production-grade DataLoader for loading local satellite HDF5 files,
    resolving scene triplets, and fetching sequence metadata.
    """
    
    def __init__(self, filepath: str, modality: str = "vis"):
        self.filepath = filepath
        self.modality = modality.lower()
        
        SatelliteValidator.validate_modality(self.modality)
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Satellite HDF5 file not found at: {self.filepath}")
            
        self.h5_file = None
        self.dataset = None
        self.scene_ids = []
        
    def open(self):
        """Open the HDF5 file and map the primary variables."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.filepath, "r")
            
            # Identify main variable key
            keys = list(self.h5_file.keys())
            dataset_key = self.modality if self.modality in keys else keys[0]
            self.dataset = self.h5_file[dataset_key]
            
            # Map scene IDs
            if 'id' in self.h5_file:
                self.scene_ids = [val.decode('utf-8') if isinstance(val, bytes) else val for val in self.h5_file['id'][:]]
            else:
                self.scene_ids = [f"Scene_{i}" for i in range(self.dataset.shape[0])]
                
    def load_scene(self, scene_id: str) -> np.ndarray:
        """
        Loads a full scene array by scene ID.
        
        Returns:
            Numpy array of shape [H, W, Frames]
        """
        self.open()
        scene_idx = SatelliteValidator.validate_scene_exists(self.h5_file, scene_id)
        
        scene_data = self.dataset[scene_idx]
        return scene_data
        
    def get_triplet(self, scene_id: str, t0_idx: int = 23, t1_idx: int = 24, t2_idx: int = 25) -> tuple:
        """
        Loads a frame triplet (t0, t1, t2) from a specified scene ID.
        
        Returns:
            Tuple of (t0, t1, t2) as raw numpy arrays.
        """
        self.open()
        scene_idx = SatelliteValidator.validate_scene_exists(self.h5_file, scene_id)
        
        num_frames = self.dataset.shape[3]
        SatelliteValidator.validate_frame_indices(num_frames, t0_idx, t1_idx, t2_idx)
        
        t0 = self.dataset[scene_idx, :, :, t0_idx]
        t1 = self.dataset[scene_idx, :, :, t1_idx]
        t2 = self.dataset[scene_idx, :, :, t2_idx]
        
        SatelliteValidator.validate_frame_array(t0)
        SatelliteValidator.validate_frame_array(t1)
        SatelliteValidator.validate_frame_array(t2)
        
        return t0, t1, t2
        
    def get_scene_metadata(self, scene_id: str) -> dict:
        """
        Returns scale, offsets, and range metadata for the scene.
        """
        self.open()
        scene_idx = SatelliteValidator.validate_scene_exists(self.h5_file, scene_id)
        scene_data = self.dataset[scene_idx]
        
        metadata = {
            "scene_id": scene_id,
            "modality": self.modality,
            "min_raw_val": float(scene_data.min()),
            "max_raw_val": float(scene_data.max()),
            "shape": scene_data.shape,
            "frames": scene_data.shape[2]
        }
        
        # Pull any dataset attributes if they exist
        if self.dataset.attrs:
            for attr_name, attr_val in self.dataset.attrs.items():
                metadata[attr_name] = attr_val
                
        return metadata
        
    def close(self):
        """Close the open HDF5 file handle."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.dataset = None
            
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

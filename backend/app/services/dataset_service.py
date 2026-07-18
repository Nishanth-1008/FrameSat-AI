import os
import sys
import numpy as np

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

from core.data_loader import SatelliteDataLoader
from core.validator import SatelliteValidator, SceneValidationError

class DatasetService:
    """
    Service providing read-only access to dataset metadata and frame structures
    without performing interpolation.
    """
    
    def __init__(self, datasets_dir: str = None):
        self.datasets_dir = datasets_dir or os.path.join(backend_dir, "..", "evaluation", "datasets")
        
        # Mapping modalities to local files
        self.modality_files = {
            "vis": "SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5",
            "vil": "SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5"
        }

    def _get_loader_for_modality(self, modality: str) -> SatelliteDataLoader:
        """Helper to instantiate loader for modality."""
        mod = modality.lower()
        if mod not in self.modality_files:
            raise ValueError(f"Modality '{modality}' not mapped to a local dataset file.")
            
        filepath = os.path.join(self.datasets_dir, self.modality_files[mod])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"HDF5 dataset file for {modality.upper()} not found at: {filepath}")
            
        return SatelliteDataLoader(filepath=filepath, modality=mod)

    def _find_scene_modality(self, scene_id: str) -> str:
        """Scans available datasets to determine which modality contains the scene ID."""
        for modality in self.modality_files.keys():
            try:
                loader = self._get_loader_for_modality(modality)
                with loader:
                    if scene_id in loader.scene_ids:
                        return modality
            except (FileNotFoundError, ValueError):
                continue
                
        raise SceneValidationError(f"Scene ID '{scene_id}' could not be resolved in any available local datasets.")

    def list_datasets(self) -> list:
        """
        Lists available dataset names and modalities based on local file presence.
        """
        available = []
        for modality, filename in self.modality_files.items():
            path = os.path.join(self.datasets_dir, filename)
            if os.path.exists(path):
                available.append({
                    "modality": modality,
                    "filename": filename,
                    "size_bytes": os.path.getsize(path)
                })
        return available

    def list_scenes(self, dataset: str = "sevir", modality: str = "vis") -> list:
        """
        Lists all scene IDs for a specific dataset modality.
        """
        loader = self._get_loader_for_modality(modality)
        with loader:
            return loader.scene_ids

    def get_scene(self, scene_id: str) -> dict:
        """
        Retrieves detailed metadata for a given scene ID.
        """
        modality = self._find_scene_modality(scene_id)
        loader = self._get_loader_for_modality(modality)
        with loader:
            return loader.get_scene_metadata(scene_id)

    def get_scene_frames(self, scene_id: str) -> dict:
        """
        Retrieves frame dimensions, counts, and spatial sizes for a scene.
        """
        meta = self.get_scene(scene_id)
        return {
            "scene_id": scene_id,
            "modality": meta["modality"],
            "total_frames": meta["frames"],
            "height": meta["shape"][0],
            "width": meta["shape"][1]
        }

    def get_frame(self, scene_id: str, frame_index: int) -> np.ndarray:
        """
        Loads and returns a single normalized frame array in range [0, 1].
        """
        modality = self._find_scene_modality(scene_id)
        loader = self._get_loader_for_modality(modality)
        
        with loader:
            meta = loader.get_scene_metadata(scene_id)
            scene_data = loader.load_scene(scene_id)
            
        num_frames = scene_data.shape[2]
        if frame_index < 0 or frame_index >= num_frames:
            raise IndexError(f"Frame index {frame_index} out of range for scene (0 to {num_frames-1}).")
            
        raw_frame = scene_data[:, :, frame_index]
        
        # Normalize to [0, 1] using scene bounds
        d_min = meta["min_raw_val"]
        d_max = meta["max_raw_val"]
        norm_factor = float(d_max - d_min) if d_max > d_min else 255.0
        
        norm_frame = (raw_frame - d_min) / norm_factor
        return np.clip(norm_frame, 0.0, 1.0).astype(np.float32)

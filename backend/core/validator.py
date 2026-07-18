import numpy as np
import h5py

class SceneValidationError(Exception):
    """Custom exception raised for validation failures in the interpolation pipeline."""
    pass

class SatelliteValidator:
    """
    Validates satellite datasets, scenes, and individual frame arrays.
    """
    
    SUPPORTED_MODALITIES = {"vis", "vil", "ir107", "goes_c13"}
    
    @staticmethod
    def validate_modality(modality: str):
        """Verifies modality type is supported."""
        if modality.lower() not in SatelliteValidator.SUPPORTED_MODALITIES:
            raise SceneValidationError(
                f"Modality '{modality}' is not supported. Choose from: {list(SatelliteValidator.SUPPORTED_MODALITIES)}"
            )

    @staticmethod
    def validate_scene_exists(h5_file: h5py.File, scene_id: str):
        """Checks if a given scene ID exists in the loaded dataset."""
        if 'id' not in h5_file:
            raise SceneValidationError("No scene 'id' variable found in the HDF5 file.")
        
        # Decode bytes-based ID array from HDF5
        scene_ids = [val.decode('utf-8') if isinstance(val, bytes) else val for val in h5_file['id'][:]]
        if scene_id not in scene_ids:
            raise SceneValidationError(f"Scene ID '{scene_id}' not found in the dataset.")
        
        return scene_ids.index(scene_id)

    @staticmethod
    def validate_frame_indices(num_frames: int, t0_idx: int, t1_idx: int, t2_idx: int):
        """Verifies indices are within the available frame counts and ordered chronologically."""
        for idx_name, idx_val in [("t0", t0_idx), ("t1", t1_idx), ("t2", t2_idx)]:
            if idx_val < 0 or idx_val >= num_frames:
                raise SceneValidationError(
                    f"Frame index {idx_name} ({idx_val}) is out of bounds for scene (0 to {num_frames-1})."
                )
                
        if not (t0_idx < t1_idx < t2_idx):
            raise SceneValidationError(
                f"Frame indices must be strictly sequential (t0={t0_idx} < t1={t1_idx} < t2={t2_idx})."
            )

    @staticmethod
    def validate_frame_array(frame: np.ndarray, expected_dim: int = 2):
        """Verifies arrays are non-empty and possess the correct dimensional counts."""
        if frame is None:
            raise SceneValidationError("Frame array cannot be None.")
            
        if not isinstance(frame, np.ndarray):
            raise SceneValidationError("Frame must be a numpy ndarray.")
            
        if frame.size == 0:
            raise SceneValidationError("Frame array is empty.")
            
        if len(frame.shape) != expected_dim:
            raise SceneValidationError(
                f"Invalid frame dimensions. Expected {expected_dim}D shape, got {len(frame.shape)}D (shape={frame.shape})."
            )
            
        if np.isnan(frame).any():
            raise SceneValidationError("Frame array contains NaN values.")

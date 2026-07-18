import os
import sys
import torch
import numpy as np

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

from core.data_loader import SatelliteDataLoader
from core.preprocessor import SatellitePreprocessor
from core.postprocessor import SatellitePostprocessor
from app.models.rife_wrapper import RIFEModelWrapper
from core.validator import SatelliteValidator, SceneValidationError

class InterpolationService:
    """
    Orchestrating service that serves as the public API to the interpolation backend.
    Uses Dependency Injection to allow components to be easily replaced.
    """
    
    def __init__(
        self,
        loader: SatelliteDataLoader = None,
        model: RIFEModelWrapper = None,
        preprocessor: SatellitePreprocessor = None,
        postprocessor: SatellitePostprocessor = None
    ):
        self.loader = loader
        self.model = model
        self.preprocessor = preprocessor or SatellitePreprocessor(target_height=384, target_width=384)
        self.postprocessor = postprocessor or SatellitePostprocessor()

    def _init_default_loader(self, modality: str):
        """Initializes default data loader if none injected."""
        if self.loader is not None and self.loader.modality == modality.lower():
            return self.loader
            
        # Default local datasets directory mapping
        default_path = os.path.join(backend_dir, "..", "evaluation", "datasets", "SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5")
        if modality.lower() == "vil":
            default_path = os.path.join(backend_dir, "..", "evaluation", "datasets", "SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5")
            
        return SatelliteDataLoader(filepath=default_path, modality=modality)

    def _init_default_model(self):
        """Initializes default RIFE wrapper if none injected."""
        if self.model is not None:
            return self.model
            
        default_weights = os.path.join(backend_dir, "app", "models", "rife_src", "train_log")
        return RIFEModelWrapper(weights_dir=default_weights)

    def list_scenes(self, modality: str = "vis") -> list:
        """
        Lists all scene IDs available in the dataset.
        """
        loader = self._init_default_loader(modality)
        with loader:
            return loader.scene_ids

    def get_scene(self, scene_id: str, modality: str = "vis") -> dict:
        """
        Gets detailed metadata for a given scene ID.
        """
        loader = self._init_default_loader(modality)
        with loader:
            return loader.get_scene_metadata(scene_id)

    def interpolate(
        self,
        scene_id: str,
        modality: str,
        frame_before: int,
        frame_after: int,
        out_path: str = None
    ) -> dict:
        """
        Interpolates a single intermediate frame.
        """
        loader = self._init_default_loader(modality)
        model = self._init_default_model()
        
        with loader:
            # 1. Load frame triplet and metadata
            t0, gt, t2 = loader.get_triplet(scene_id, t0_idx=frame_before, t1_idx=(frame_before+frame_after)//2, t2_idx=frame_after)
            metadata = loader.get_scene_metadata(scene_id)
            
        min_val = metadata["min_raw_val"]
        max_val = metadata["max_raw_val"]
        
        # 2. Preprocess
        t0_tensor = self.preprocessor.preprocess(t0, min_val, max_val, model.device)
        t2_tensor = self.preprocessor.preprocess(t2, min_val, max_val, model.device)
        
        # 3. Model Inference
        pred_tensor = model.interpolate(t0_tensor, t2_tensor)
        
        # 4. Postprocess
        result = self.postprocessor.postprocess(pred_tensor, out_path=out_path)
        
        return {
            "prediction": result["prediction"],
            "saved_to": result["saved_to"],
            "metadata": {
                "scene_id": scene_id,
                "modality": modality,
                "frame_before": frame_before,
                "frame_after": frame_after,
                "min_raw_val": min_val,
                "max_raw_val": max_val
            }
        }

    def interpolate_sequence(
        self,
        scene_id: str,
        modality: str,
        frame_before: int,
        frame_after: int,
        num_frames: int,
        out_dir: str = None
    ) -> list:
        """
        Interpolates multiple intermediate frames recursively.
        
        Returns:
            List of dicts representing the generated sequence.
        """
        if num_frames <= 0:
            raise ValueError("Number of frames must be positive.")
            
        loader = self._init_default_loader(modality)
        model = self._init_default_model()
        
        with loader:
            t0, _, t2 = loader.get_triplet(scene_id, t0_idx=frame_before, t1_idx=(frame_before+frame_after)//2, t2_idx=frame_after)
            metadata = loader.get_scene_metadata(scene_id)
            
        min_val = metadata["min_raw_val"]
        max_val = metadata["max_raw_val"]
        
        t0_tensor = self.preprocessor.preprocess(t0, min_val, max_val, model.device)
        t2_tensor = self.preprocessor.preprocess(t2, min_val, max_val, model.device)
        
        # Determine depth needed: 2^depth - 1 >= num_frames
        depth = 1
        while (2**depth - 1) < num_frames:
            depth += 1
            
        # Recursive subdivision helper
        def subdivide(tensor_a: torch.Tensor, tensor_b: torch.Tensor, current_depth: int) -> list:
            if current_depth == 0:
                return []
                
            mid = model.interpolate(tensor_a, tensor_b)
            
            left = subdivide(tensor_a, mid, current_depth - 1)
            right = subdivide(mid, tensor_b, current_depth - 1)
            
            return left + [mid] + right
            
        # Generate full sequence of size (2^depth - 1)
        full_sequence = subdivide(t0_tensor, t2_tensor, depth)
        
        # Downsample/slice list to match target num_frames evenly
        indices = np.round(np.linspace(0, len(full_sequence) - 1, num_frames)).astype(int)
        sliced_sequence = [full_sequence[idx] for idx in indices]
        
        # Postprocess each frame
        results = []
        for i, tensor in enumerate(sliced_sequence):
            out_path = os.path.join(out_dir, f"seq_frame_{i}.png") if out_dir else None
            post_res = self.postprocessor.postprocess(tensor, out_path=out_path)
            
            results.append({
                "frame_index": i,
                "prediction": post_res["prediction"],
                "saved_to": post_res["saved_to"]
            })
            
        return results

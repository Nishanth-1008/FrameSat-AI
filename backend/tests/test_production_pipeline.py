import os
import sys
import torch
import numpy as np

# Ensure root backend directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from core.data_loader import SatelliteDataLoader
from core.preprocessor import SatellitePreprocessor
from core.postprocessor import SatellitePostprocessor
from app.models.rife_wrapper import RIFEModelWrapper

def main():
    print("================ VERIFYING PRODUCTION PIPELINE ================")
    
    # 1. Paths
    dataset_path = os.path.join(backend_dir, "..", "evaluation", "datasets", "SEVIR_VIS_STORMEVENTS_2018_0101_0131.h5")
    weights_path = os.path.join(backend_dir, "app", "models", "rife_src", "train_log")
    output_path = os.path.join(backend_dir, "tests", "outputs", "prod_test_prediction.png")
    
    # Clean up old output
    if os.path.exists(output_path):
        os.remove(output_path)

    # 2. Initialize modules
    print("Initializing DataLoader and Model...")
    loader = SatelliteDataLoader(filepath=dataset_path, modality="vis")
    preprocessor = SatellitePreprocessor(target_height=384, target_width=384)
    postprocessor = SatellitePostprocessor()
    engine = RIFEModelWrapper(weights_dir=weights_path)
    
    # 3. Load triplet
    print("Loading frame triplet...")
    loader.open()
    scene_id = loader.scene_ids[0]
    t0, gt, t2 = loader.get_triplet(scene_id, t0_idx=23, t1_idx=24, t2_idx=25)
    metadata = loader.get_scene_metadata(scene_id)
    loader.close()
    
    print(f"Loaded Scene ID: {scene_id}")
    print(f"Metadata: {metadata}")
    
    # 4. Preprocess
    print("Running Preprocessing...")
    min_val = metadata["min_raw_val"]
    max_val = metadata["max_raw_val"]
    
    t0_tensor = preprocessor.preprocess(t0, min_val, max_val, engine.device)
    t2_tensor = preprocessor.preprocess(t2, min_val, max_val, engine.device)
    
    print(f"Preprocessed tensor shape: {t0_tensor.shape}")
    assert t0_tensor.shape == (1, 3, 384, 384), f"Unexpected preprocessed shape: {t0_tensor.shape}"
    
    # 5. Run Model Inference
    print("Running Interpolation Model...")
    pred_tensor = engine.interpolate(t0_tensor, t2_tensor)
    print(f"Inference complete. Prediction shape: {pred_tensor.shape}")
    assert pred_tensor.shape == (1, 3, 384, 384), f"Unexpected model prediction shape: {pred_tensor.shape}"
    
    # 6. Postprocess and Save
    print("Running Postprocessing...")
    result = postprocessor.postprocess(pred_tensor, out_path=output_path)
    
    print(f"Postprocessing result keys: {list(result.keys())}")
    print(f"Saved prediction image to: {result['saved_to']}")
    
    assert os.path.exists(output_path), "Failed to save prediction PNG!"
    assert result["shape"] == (384, 384), f"Unexpected final prediction shape: {result['shape']}"
    
    print("================ VERIFICATION SUCCESSFUL ================")

if __name__ == "__main__":
    main()

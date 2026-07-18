#!/usr/bin/env python3
"""
End-to-end inference smoke pipeline script.
Loads two real GOES-19 frames, validates them, preprocesses them,
runs RIFE inference, postprocesses the output, and saves it as a PNG.
Logs resolutions and times.
"""

import os
import sys
import time
import torch

try:
    # Attempt absolute imports with backend prefix first
    from backend.core.goes19_provider import GOES19Provider
    from backend.core.preprocessor import SatellitePreprocessor
    from backend.core.postprocessor import SatellitePostprocessor
    from backend.core.validator import SatelliteValidator
    from backend.app.models.rife_wrapper import RIFEModelWrapper
except ImportError:
    # Fallback to direct imports if backend directory is directly on sys.path (e.g. during some test runs)
    from core.goes19_provider import GOES19Provider
    from core.preprocessor import SatellitePreprocessor
    from core.postprocessor import SatellitePostprocessor
    from core.validator import SatelliteValidator
    from app.models.rife_wrapper import RIFEModelWrapper


def run_pipeline():
    print("================ STARTING INFERENCE SMOKE PIPELINE ================")

    # 1. Initialize Provider
    cache_dir = os.path.join("datasets", "cache", "goes19_cache")
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"GOES-19 cache directory not found at: {cache_dir}")

    provider = GOES19Provider(data_dir=cache_dir)
    if not provider.scene_ids or len(provider.scene_ids) < 2:
        raise ValueError(
            f"Expected at least 2 distinct GOES-19 scenes in cache, found {len(provider.scene_ids)}"
        )

    scene_a_id = provider.scene_ids[0]
    scene_b_id = provider.scene_ids[1]
    print(f"Loading frames from scene A ({scene_a_id}) and scene B ({scene_b_id})...")

    # Load 2D frame arrays
    frame_a = provider.load_scene(scene_a_id)[:, :, 0]
    frame_b = provider.load_scene(scene_b_id)[:, :, 0]

    input_res = frame_a.shape
    print(f"Loaded frame A shape: {frame_a.shape}")
    print(f"Loaded frame B shape: {frame_b.shape}")

    # 2. Validate Inputs
    print("Validating inputs...")
    SatelliteValidator.validate_frame_array(frame_a)
    SatelliteValidator.validate_frame_array(frame_b)
    print("Validation successful.")

    # 3. Preprocess
    print("Initializing model wrapper and preprocessor...")
    preprocessor = SatellitePreprocessor(target_height=384, target_width=384)
    postprocessor = SatellitePostprocessor()
    engine = RIFEModelWrapper()

    print(f"Running preprocessing on device: {engine.device}...")
    t_pre_start = time.perf_counter()

    meta_a = provider.get_scene_metadata(scene_a_id)
    meta_b = provider.get_scene_metadata(scene_b_id)
    min_val = min(meta_a["min_raw_val"], meta_b["min_raw_val"])
    max_val = max(meta_a["max_raw_val"], meta_b["max_raw_val"])

    t0_tensor = preprocessor.preprocess(frame_a, min_val, max_val, engine.device)
    t2_tensor = preprocessor.preprocess(frame_b, min_val, max_val, engine.device)
    t_preprocess = time.perf_counter() - t_pre_start

    print(f"Preprocessed tensor shape: {t0_tensor.shape}")

    # 4. Inference
    print("Running frame interpolation inference...")
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        pred_tensor = engine.interpolate(t0_tensor, t2_tensor)
    t_inference = time.perf_counter() - t_inf_start
    print(f"Inference complete. Prediction shape: {pred_tensor.shape}")

    # 5. Postprocess and Save
    out_dir = os.path.join("backend", "tests", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_smoke_prediction.png")

    print(f"Postprocessing and saving to {out_path}...")
    result = postprocessor.postprocess(pred_tensor, out_path=out_path)

    print("\n================ METRICS ================")
    print(f"Input Resolution:    {input_res[0]}x{input_res[1]}")
    print(f"Preprocessing Time:  {t_preprocess:.4f} seconds")
    print(f"Inference Time:      {t_inference:.4f} seconds")
    print(f"Output Resolution:   {result['shape'][0]}x{result['shape'][1]}")
    print(f"Saved Image:         {result['saved_to']}")
    print("=========================================\n")

    return out_path, result["shape"]


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)

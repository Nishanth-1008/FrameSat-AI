import os
import sys
import numpy as np

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from app.services.dataset_service import DatasetService

def main():
    print("================ VERIFYING DATASET SERVICE ================")
    
    service = DatasetService()
    
    # 1. Test listing datasets
    print("Listing datasets...")
    datasets = service.list_datasets()
    print(f"Available local datasets: {datasets}")
    assert len(datasets) > 0, "No local datasets detected! Make sure files exist."
    
    # 2. Test listing scenes
    print("Listing scenes for VIS...")
    vis_scenes = service.list_scenes(dataset="sevir", modality="vis")
    print(f"Available VIS scenes count: {len(vis_scenes)}")
    assert len(vis_scenes) > 0, "No VIS scenes listed!"
    
    # 3. Test get scene details
    target_scene = vis_scenes[0]
    print(f"Fetching metadata for scene: {target_scene}...")
    meta = service.get_scene(target_scene)
    print(f"Metadata properties: {meta}")
    assert meta["scene_id"] == target_scene, "Scene ID mismatch!"
    assert meta["modality"] == "vis", "Modality mismatch!"
    
    # 4. Test get scene frames metadata
    print(f"Fetching frames metadata for scene: {target_scene}...")
    frames_meta = service.get_scene_frames(target_scene)
    print(f"Frames Metadata: {frames_meta}")
    assert frames_meta["total_frames"] == 49, f"Expected 49 frames, got {frames_meta['total_frames']}"
    assert frames_meta["height"] == 768, f"Expected 768 height, got {frames_meta['height']}"
    
    # 5. Test get frame
    frame_idx = 10
    print(f"Loading frame {frame_idx} for scene {target_scene}...")
    frame = service.get_frame(target_scene, frame_index=frame_idx)
    print(f"Frame shape: {frame.shape}, Data range: [{frame.min():.2f}, {frame.max():.2f}], dtype: {frame.dtype}")
    assert frame.shape == (768, 768), f"Unexpected frame shape: {frame.shape}"
    assert frame.min() >= 0.0 and frame.max() <= 1.0, f"Frame values outside [0, 1] range: [{frame.min()}, {frame.max()}]"
    
    print("================ DATASET SERVICE VERIFICATION SUCCESSFUL ================")

if __name__ == "__main__":
    main()

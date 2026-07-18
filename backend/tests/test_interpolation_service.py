import os
import sys

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from app.services.interpolation_service import InterpolationService

def main():
    print("================ VERIFYING INTERPOLATION SERVICE ================")
    
    service = InterpolationService()
    
    # 1. Test listing scenes
    print("Listing scenes...")
    scenes = service.list_scenes(modality="vis")
    print(f"Available scenes count: {len(scenes)}")
    print(f"First 3 Scene IDs: {scenes[:3]}")
    assert len(scenes) > 0, "No scenes returned!"
    
    target_scene = scenes[0]
    
    # 2. Test get scene details
    print(f"Fetching metadata for scene: {target_scene}...")
    metadata = service.get_scene(target_scene, modality="vis")
    print(f"Metadata properties: {metadata}")
    assert metadata["scene_id"] == target_scene, "Scene ID mismatch!"
    
    # 3. Test single interpolation
    print("Running single frame interpolation...")
    out_file = os.path.join(current_dir, "outputs", "service_single_test.png")
    if os.path.exists(out_file):
        os.remove(out_file)
        
    res_single = service.interpolate(
        scene_id=target_scene,
        modality="vis",
        frame_before=23,
        frame_after=25,
        out_path=out_file
    )
    print(f"Single prediction shape: {res_single['prediction'].shape}")
    print(f"Saved single frame to: {res_single['saved_to']}")
    assert res_single["prediction"].shape == (384, 384), "Invalid output shape!"
    assert os.path.exists(out_file), "Single frame PNG file not created!"
    
    # 4. Test sequence interpolation
    print("Running sequence interpolation (3 frames)...")
    out_dir = os.path.join(current_dir, "outputs", "service_seq_test")
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    res_seq = service.interpolate_sequence(
        scene_id=target_scene,
        modality="vis",
        frame_before=20,
        frame_after=24,
        num_frames=3,
        out_dir=out_dir
    )
    print(f"Sequence frames generated: {len(res_seq)}")
    assert len(res_seq) == 3, f"Expected 3 frames, got {len(res_seq)}"
    for idx, r in enumerate(res_seq):
        print(f"  Frame {idx}: saved to {r['saved_to']} (shape={r['prediction'].shape})")
        assert r["prediction"].shape == (384, 384), "Frame shape mismatch in sequence!"
        assert os.path.exists(r["saved_to"]), f"Sequence frame PNG {idx} not created!"
        
    print("================ SERVICE VERIFICATION SUCCESSFUL ================")

if __name__ == "__main__":
    main()

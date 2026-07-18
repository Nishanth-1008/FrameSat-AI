import os
import sys

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from core.goes19_provider import GOES19Provider

def main():
    print("================ VERIFYING GOES-19 DATA PROVIDER ================")
    
    provider = GOES19Provider()
    
    # 1. Test listing scenes
    print("Listing scenes...")
    scenes = provider.scene_ids
    print(f"Available scenes count: {len(scenes)}")
    print(f"Scene IDs: {scenes}")
    assert len(scenes) > 0, "No scenes listed in GOES-19 provider!"
    
    target_scene = scenes[0]
    
    # 2. Test fetching metadata
    print(f"Fetching metadata for scene: {target_scene}...")
    meta = provider.get_scene_metadata(target_scene)
    print(f"Metadata: {meta}")
    assert meta["scene_id"] == target_scene, "Scene ID mismatch in metadata!"
    assert meta["modality"] == "goes_c13", "Modality mismatch in metadata!"
    assert meta["shape"] == (384, 384, 49), f"Unexpected scene shape: {meta['shape']}"
    
    # 3. Test triplet extraction
    print(f"Extracting frame triplet for scene: {target_scene}...")
    t0, t1, t2 = provider.get_triplet(target_scene)
    print(f"Triplet shapes: t0={t0.shape}, t1={t1.shape}, t2={t2.shape}")
    assert t0.shape == (384, 384), "Triplet frame shape is incorrect!"
    assert t1.shape == (384, 384), "Triplet frame shape is incorrect!"
    assert t2.shape == (384, 384), "Triplet frame shape is incorrect!"
    
    print("================ GOES-19 PROVIDER VERIFICATION SUCCESSFUL ================")

if __name__ == "__main__":
    main()

import os
import sys
from fastapi.testclient import TestClient

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from app.main import app

def main():
    print("================ VERIFYING REST API ENDPOINTS ================")
    
    client = TestClient(app)
    
    # 1. Test GET /api/v1/datasets
    print("Testing GET /api/v1/datasets...")
    resp = client.get("/api/v1/datasets")
    print(f"Status Code: {resp.status_code}")
    print(f"Body: {resp.json()}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    datasets = resp.json()["datasets"]
    assert len(datasets) > 0, "No datasets returned!"
    
    # 2. Test GET /api/v1/datasets/sevir/scenes
    print("Testing GET /api/v1/datasets/sevir/scenes...")
    resp = client.get("/api/v1/datasets/sevir/scenes?modality=vis")
    print(f"Status Code: {resp.status_code}")
    scenes = resp.json()["scenes"]
    print(f"Scenes count: {len(scenes)}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(scenes) > 0, "No scenes returned!"
    
    target_scene = scenes[0]
    
    # 3. Test GET /api/v1/scenes/{scene_id}
    print(f"Testing GET /api/v1/scenes/{target_scene}...")
    resp = client.get(f"/api/v1/scenes/{target_scene}")
    print(f"Status Code: {resp.status_code}")
    print(f"Properties: {resp.json().keys()}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    
    # 4. Test GET /api/v1/scenes/{scene_id}/frames
    print(f"Testing GET /api/v1/scenes/{target_scene}/frames...")
    resp = client.get(f"/api/v1/scenes/{target_scene}/frames")
    print(f"Status Code: {resp.status_code}")
    print(f"Body: {resp.json()}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    
    # 5. Test POST /api/v1/interpolate
    print("Testing POST /api/v1/interpolate...")
    payload = {
        "scene_id": target_scene,
        "modality": "vis",
        "frame_before": 23,
        "frame_after": 25
    }
    resp = client.post("/api/v1/interpolate", json=payload)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error detail: {resp.json()}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    
    # 6. Test POST /api/v1/interpolate-sequence
    print("Testing POST /api/v1/interpolate-sequence...")
    payload_seq = {
        "scene_id": target_scene,
        "modality": "vis",
        "frame_before": 20,
        "frame_after": 24,
        "num_frames": 2
    }
    resp = client.post("/api/v1/interpolate-sequence", json=payload_seq)
    print(f"Status Code: {resp.status_code}")
    print(f"Body size: {len(resp.json())}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(resp.json()) == 2, "Expected 2 generated frames"
    
    print("================ REST API VERIFICATION SUCCESSFUL ================")

if __name__ == "__main__":
    main()
